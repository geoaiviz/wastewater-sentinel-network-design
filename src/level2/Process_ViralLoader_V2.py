#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process_ViralLoader_V2_FullPanel.py
-----------------------------------
Extended clinical-data loader with one key upgrade:


Existing function calls remain supported:
  - load_county_clinical(file_path, disease=..., county_shp=..., county_pop_csv=..., metric=..., value_kind=...)
  - load_wwtp_clinical_from_metrics(file_path, disease=..., wwtp_shp=..., metric=..., value_kind=...)

Expected long-format columns in clinical CSVs (case-insensitive, fuzzy ok):
  "Region Level","Region Name","Pathogen Name","Event Onset Date",
  (optional) "Age Group","Case Count","Hospitalized Count"

Population sources (for 'rate' only):
  • Counties: COCounty.shp (COUNTY, US_FIPS) + CO_County_Population_FIPS5.csv (County_FIPS, Total_Population)
  • WWTPs:   WWTP_Select.shp (wwtp, pop_served)

Outputs
  • Aggregate (Flu/COVID/RSV-all-ages):
        County:  ['week','County','clinical_rate']   # 'rate' per 100k or raw when value_kind='raw'
        WWTP:    ['week','wwtp','clinical_rate']
  • RSV by age (counts only):
        County:  ['week','County','Age Group','count']
        WWTP:    ['week','wwtp','Age Group','count']

Requires: pandas, numpy, geopandas, matplotlib (for viz)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Callable, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger("Process_ViralLoader_V2")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Fixed clinical file mapping (adjust if paths change)
# -----------------------------------------------------------------------------
COVID_FILE = "../ZoneSelection/Input/Viral/NewFormat/20251006_COVID_agg.csv"
FLU_FILE   = "../ZoneSelection/Input/Viral/NewFormat/20250926_Flu_agg.csv"
RSV_FILE   = "../ZoneSelection/Input/Viral/NewFormat/20250922_RSV_agg.csv"

_FILE_MAP = {
    "COVID": COVID_FILE,
    "INFLUENZA": FLU_FILE,
    "FLU": FLU_FILE,
    "RSV": RSV_FILE,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _infer_col(preferred: str, candidates) -> Optional[str]:
    low = preferred.lower()
    for c in candidates:
        if str(c).lower() == low:
            return c
    for c in candidates:
        if low in str(c).lower():
            return c
    return None

def _infer_age_col(cols, hint="Age Group") -> Optional[str]:
    c = _infer_col(hint, cols)
    if c is None:
        for k in cols:
            s = str(k).lower()
            if "age" in s and "group" in s:
                return k
    return c

def _norm_disease(s: str) -> str:
    s = str(s).strip().lower()
    if s in {"covid", "sars-cov-2", "sars cov 2", "sars_cov_2", "sarscov2"}:
        return "COVID"
    if s in {"influenza", "flu"}:
        return "Influenza"
    if s in {"rsv"}:
        return "RSV"
    return s.title() if s else s

def _norm_region_level(s: str) -> str:
    s = str(s).strip().lower()
    if "county" in s:
        return "county"
    if "sewer" in s or "wwtp" in s or "plant" in s or "utility" in s:
        return "sewershed"
    return s

def _read_table(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)

def _to_monday(x: pd.Series) -> pd.Series:
    dt = pd.to_datetime(x, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")

def _drop_entities_with_any_nan(df: pd.DataFrame, entity_col: str, value_col: str = "clinical_rate") -> pd.DataFrame:
    """
    Remove any entity that *ever* has NaN in value_col.
    Used to drop entities with missing denominators for rates.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["_entity_norm"] = out[entity_col].astype(str).str.strip().str.lower()

    bad_entities = (
        out.loc[out[value_col].isna(), "_entity_norm"]
        .dropna()
        .unique()
        .tolist()
    )

    if bad_entities:
        out = out[~out["_entity_norm"].isin(bad_entities)].copy()

    out.drop(columns=["_entity_norm"], inplace=True, errors="ignore")
    return out


def _load_entity_universe_from_shp(shp_path: str, field_name: str) -> List[str]:
    """Return master entity list (normalized) from shapefile."""
    if not (shp_path and os.path.exists(shp_path)):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if field_name not in gdf.columns:
        raise KeyError(f"Field '{field_name}' not found in {shp_path}. Available: {list(gdf.columns)}")

    s = (
        gdf[field_name]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return sorted(pd.unique(s).tolist())


def _enforce_full_week_panel(
    df: pd.DataFrame,
    *,
    week_col: str = "week",
    entity_col: str = "region",
    value_col: str = "value_raw",
    freq: str = "W-MON",
    start_week=None,
    end_week=None,
    fill_value=0.0,
    entity_universe: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ensure every entity has a row for every week in [min..max] (or explicit start/end).
    Uses entity_universe if provided (master list from shapefile).
    """
    if df.empty:
        # If empty but universe+date range exist, we *could* build a panel,
        # but in this pipeline df shouldn't be empty in normal runs.
        return df.copy()

    out = df.copy()
    out[week_col] = pd.to_datetime(out[week_col], errors="coerce")

    # Choose the master calendar
    wmin = pd.to_datetime(start_week) if start_week is not None else out[week_col].min()
    wmax = pd.to_datetime(end_week) if end_week is not None else out[week_col].max()
    all_weeks = pd.date_range(wmin, wmax, freq=freq)

    # Build full MultiIndex (entity × week)
    if entity_universe is None:
        entities = out[entity_col].astype(str).str.strip().str.lower().unique()
    else:
        entities = pd.Series(entity_universe).astype(str).str.strip().str.lower().unique()

    full_idx = pd.MultiIndex.from_product([entities, all_weeks], names=[entity_col, week_col])

    # Reindex
    out[entity_col] = out[entity_col].astype(str).str.strip().str.lower()
    out = out.set_index([entity_col, week_col]).sort_index()
    out = out.reindex(full_idx).reset_index()

    # Fill missing values
    if fill_value is not None:
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(float(fill_value))

    return out


def _build_county_pop_lookup(
    county_shp: str,
    county_pop_csv: str,
    *,
    county_name_field: str = "COUNTY",   #
    county_fips_field: str = "US_FIPS",
) -> Dict[Tuple[str, str], float]:
    """Return {('county', county_name_lower): population} by joining shapefile to pop CSV (via FIPS)."""
    if not (county_shp and os.path.exists(county_shp)):
        raise FileNotFoundError(f"County shapefile not found: {county_shp}")
    if not (county_pop_csv and os.path.exists(county_pop_csv)):
        raise FileNotFoundError(f"County population CSV not found: {county_pop_csv}")

    gdf = gpd.read_file(county_shp)

    shp_name = county_name_field
    shp_fips = county_fips_field
    if shp_name not in gdf.columns:
        # fallback for other states/files
        shp_name = _infer_col("COUNTY", gdf.columns) or _infer_col("LABEL", gdf.columns) or shp_name
    if shp_fips not in gdf.columns:
        shp_fips = _infer_col("US_FIPS", gdf.columns) or shp_fips

    if shp_name not in gdf.columns or shp_fips not in gdf.columns:
        raise KeyError(f"County shapefile must include '{shp_name}' and '{shp_fips}'.")

    pdf = pd.read_csv(county_pop_csv)
    pop_fips = _infer_col("County_FIPS", pdf.columns) or "County_FIPS"
    pop_val  = _infer_col("Total_Population", pdf.columns) or "Total_Population"
    if pop_fips not in pdf.columns or pop_val not in pdf.columns:
        raise KeyError("County population CSV must include 'County_FIPS' and 'Total_Population'.")

    gdf["_US_FIPS"] = gdf[shp_fips].astype(str).str.zfill(5)
    gdf["_COUNTY"]  = gdf[shp_name].astype(str).str.strip().str.lower()

    pdf["_County_FIPS"] = pdf[pop_fips].astype(str).str.zfill(5)
    pdf["_pop"] = pd.to_numeric(pdf[pop_val], errors="coerce")

    joined = gdf.merge(pdf[["_County_FIPS", "_pop"]],
                       left_on="_US_FIPS", right_on="_County_FIPS", how="left")
    miss = int(joined["_pop"].isna().sum())
    if miss > 0:
        logger.warning(f"[build_county_pop_lookup] Missing population for {miss} counties (by FIPS).")

    pop_map = {
        ("county", row["_COUNTY"]): float(row["_pop"])
        for _, row in joined.iterrows()
        if pd.notna(row["_pop"])
    }
    logger.info(f"[build_county_pop_lookup] Loaded {len(pop_map)} counties with population.")
    return pop_map


def _build_wwtp_pop_lookup(wwtp_shp: str) -> Dict[Tuple[str, str], float]:
    """Return {('sewershed', wwtp_lower): pop_served} from WWTP shapefile."""
    if not (wwtp_shp and os.path.exists(wwtp_shp)):
        raise FileNotFoundError(f"WWTP shapefile not found: {wwtp_shp}")

    gdf = gpd.read_file(wwtp_shp)
    name_col = _infer_col("wwtp", gdf.columns) or "wwtp"
    pop_col  = _infer_col("pop_served", gdf.columns) or "pop_served"
    if name_col not in gdf.columns or pop_col not in gdf.columns:
        raise KeyError("WWTP shapefile must include 'wwtp' and 'pop_served'.")

    gdf["_wwtp"] = gdf[name_col].astype(str).str.strip().str.lower()
    gdf["_pop"]  = pd.to_numeric(gdf[pop_col], errors="coerce").fillna(0.0)

    pop_map = {
        ("sewershed", row["_wwtp"]): float(row["_pop"])
        for _, row in gdf.iterrows()
        if float(row["_pop"]) > 0
    }
    logger.info(f"[build_wwtp_pop_lookup] Loaded {len(pop_map)} WWTPs with population.")
    return pop_map


# -----------------------------------------------------------------------------
# Internal extractors (aggregate all-ages)
# -----------------------------------------------------------------------------

def _extract_from_long(
    file_path: str,
    *,
    region_level: str,                  # "county" or "sewershed"
    metric: str = "hospitalization",    # "case" or "hospitalization"
    value_kind: str = "rate",           # "rate" or "raw"
    pop_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    entity_universe: Optional[List[str]] = None,
    date_col_hint: str = "Event Onset Date",
    agg_func: str | Callable = "sum",
    min_population: float = 1.0,
) -> pd.DataFrame:
    """
    Internal aggregate (all-ages) loader for new long-format files.
    Aggregates by Event Onset Date → Monday; sums within (week, region).
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Clinical file not found: {file_path}")

    df = _read_table(file_path)
    cols = list(df.columns)

    c_level = _infer_col("Region Level", cols) or "Region Level"
    c_name  = _infer_col("Region Name", cols)  or "Region Name"
    c_date  = _infer_col(date_col_hint, cols) or _infer_col("Date", cols) or _infer_col("Week", cols) or date_col_hint
    c_case  = _infer_col("Case Count", cols) or _infer_col("Cases", cols)
    c_hosp  = _infer_col("Hospitalized Count", cols) or _infer_col("Hospitalizations", cols)

    for req in (c_level, c_name, c_date):
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' missing in {file_path}. Have: {cols}")

    m = metric.lower().strip()
    if m not in {"case", "hospitalization"}:
        raise ValueError("metric must be 'case' or 'hospitalization'")

    if m == "case":
        series = pd.to_numeric(df[c_case], errors="coerce") if (c_case in df.columns) else np.nan
    else:
        series = pd.to_numeric(df[c_hosp], errors="coerce") if (c_hosp in df.columns) else np.nan

    tmp = pd.DataFrame({
        "week": _to_monday(df[c_date]),
        "region_level": df[c_level].map(_norm_region_level),
        "region": df[c_name].astype(str).str.strip().str.lower(),
        "value_raw": pd.to_numeric(series, errors="coerce"),
    })

    tmp = tmp[tmp["region_level"] == region_level]
    tmp = tmp.groupby(["week", "region"], as_index=False)["value_raw"].agg(agg_func)

    # Build the complete entity-by-week panel and fill missing observations with zero.
    tmp = _enforce_full_week_panel(
        tmp,
        week_col="week",
        entity_col="region",
        value_col="value_raw",
        freq="W-MON",
        start_week=None,
        end_week=None,
        fill_value=0.0,
        entity_universe=entity_universe,
    )

    kind = value_kind.lower().strip()
    if kind == "rate":
        if pop_lookup is None:
            logger.warning("[_extract_from_long] value_kind='rate' but population lookup is None; rates will be NaN.")
            tmp["value"] = np.nan
        else:
            keys = list(zip([region_level] * len(tmp), tmp["region"]))
            pops = np.array([float(pop_lookup.get(k, np.nan)) for k in keys], dtype=float)
            pops = np.where((pops >= min_population), pops, np.nan)
            tmp["value"] = (tmp["value_raw"] / pops) * 1e5
            missing = int(np.isnan(pops).sum())
            if missing:
                logger.warning(f"[_extract_from_long] Missing population for {missing} {region_level} rows; rates set to NaN.")
    elif kind == "raw":
        tmp["value"] = tmp["value_raw"]
    else:
        raise ValueError("value_kind must be 'rate' or 'raw'")

    return tmp[["week", "region", "value"]].sort_values(["region", "week"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# RSV-only: age-group extractor (counts only)
# -----------------------------------------------------------------------------

def _extract_rsv_by_age(
    file_path: str,
    *,
    region_level: str,                # "county" or "sewershed"
    metric: str = "hospitalization",  # enforced; RSV age analysis uses hospitalizations
    date_col_hint: str = "Event Onset Date",
    age_col_hint: str = "Age Group",
    agg_func: str | Callable = "sum",
) -> pd.DataFrame:
    """
    Age-specific RSV extractor (COUNTS ONLY). Ignores rates/denominators.
    Returns columns: ['week','region','age_group','count']
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Clinical file not found: {file_path}")

    df = _read_table(file_path)
    cols = list(df.columns)

    c_level = _infer_col("Region Level", cols) or "Region Level"
    c_name  = _infer_col("Region Name", cols)  or "Region Name"
    c_date  = _infer_col(date_col_hint, cols) or _infer_col("Date", cols) or _infer_col("Week", cols) or date_col_hint
    c_hosp  = _infer_col("Hospitalized Count", cols) or _infer_col("Hospitalizations", cols)
    c_age   = _infer_age_col(cols, age_col_hint)

    for req in (c_level, c_name, c_date, c_hosp):
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' missing in {file_path}. Have: {cols}")

    base = pd.DataFrame({
        "week": _to_monday(df[c_date]),
        "region_level": df[c_level].map(_norm_region_level),
        "region": df[c_name].astype(str).str.strip().str.lower(),
        "age_group": (df[c_age].astype(str).str.strip() if (c_age in df.columns) else "All"),
        "value_raw": pd.to_numeric(df[c_hosp], errors="coerce"),
    })

    base = base[base["region_level"] == region_level]
    tmp = base.groupby(["week", "region", "age_group"], as_index=False)["value_raw"].agg(agg_func)

    out = tmp.rename(columns={"value_raw": "count"})
    return out.sort_values(["region", "age_group", "week"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Public API (aggregate all-ages for Flu/COVID/RSV)
# -----------------------------------------------------------------------------

def load_county_clinical(
    file_path: str,
    *,
    disease: str = "COVID",
    county_shp = "../ZoneSelection/Input/Census/COCounty.shp",
    county_pop_csv: str,
    metric: str = "hospitalization",   # or "case" (COVID only)
    value_kind: str = "rate",          # "rate" (per 100k) or "raw"
) -> pd.DataFrame:
    """
    County-level aggregate (all ages). Returns ['week','County','clinical_rate'].
    Uses county_shp (field COUNTY) as the MASTER list to build a full panel.
    """
    county_universe = _load_entity_universe_from_shp(county_shp, "COUNTY")

    pop_lookup = _build_county_pop_lookup(county_shp, county_pop_csv) if value_kind.lower() == "rate" else None

    tmp = _extract_from_long(
        file_path,
        region_level="county",
        metric=metric,
        value_kind=value_kind,
        pop_lookup=pop_lookup,
        entity_universe=county_universe,   # enforce the complete county-by-week panel
    )

    out = pd.DataFrame({
        "week": tmp["week"],
        "County": tmp["region"],           # lowercase at this point
        "clinical_rate": tmp["value"],
    })

    # Drop counties with ANY NaN in rate (missing denom)
    if value_kind.lower() == "rate":
        out = _drop_entities_with_any_nan(out, entity_col="County", value_col="clinical_rate")

    # Make County pretty AFTER dropping
    out["County"] = out["County"].astype(str).str.strip().str.title()
    out = out.sort_values(["County", "week"]).reset_index(drop=True)

    if value_kind.lower() == "rate":
        logger.info(f"[load_county_clinical] CLEANED -> {len(out['County'].unique())} counties • {len(out)} rows • rate per 100k.")
    else:
        logger.info(f"[load_county_clinical] {len(out['County'].unique())} counties • {len(out)} rows • raw counts/sum.")

    return out

def _pick_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_wwtp_shapes(wwtp_shapefile):
    wwtp_pop_map = {}
    if not (wwtp_shapefile and os.path.exists(wwtp_shapefile)):
        return {}

    gdf = gpd.read_file(wwtp_shapefile)

    name_col = _pick_first_existing_column(
        gdf, ["wwtp","WWTP","Utility","UTILITY","Name","NAME","UtilityName","UTIL_NAME", "sewershed"]
    )
    if name_col is None:
        raise ValueError("WWTP shapefile missing a recognizable name column.")

    pop_col = _pick_first_existing_column(
        gdf, ["pop_served","POP_SERVED","PopServed","POP","Population","SERVPOPSUM","POP_SERV","SERV_POP"]
    )
    if pop_col is None:
        gdf["__POP_TMP__"] = 0.0
        pop_col = "__POP_TMP__"

    # Normalize
    gdf["_wwtp_norm"] = gdf[name_col].astype(str).str.strip().str.lower()
    gdf["_pop_num"]   = pd.to_numeric(gdf[pop_col], errors="coerce").fillna(0.0).astype(float)

    # Ensure that the normalized output contains one canonical WWTP column.
    if "wwtp" in gdf.columns:
        # Preserve the original source value under an explicit name.
        gdf = gdf.rename(columns={"wwtp": "wwtp_raw"})

    # Build a clean GeoDataFrame with unique column names
    wwtp_gdf = gdf.rename(columns={"_wwtp_norm": "wwtp", "_pop_num": "pop_served"})[
        ["wwtp", "pop_served", "geometry"]
    ].copy()

    # Final guard in case anything else duplicated
    wwtp_gdf = wwtp_gdf.loc[:, ~wwtp_gdf.columns.duplicated(keep="last")]

    # Sum multipart plant polygons when present.
    # wwtp_gdf = wwtp_gdf.groupby("wwtp", as_index=False).agg({"pop_served":"sum", "geometry":"unary_union"})

    wwtp_pop_map = wwtp_gdf.groupby("wwtp")["pop_served"].sum().to_dict()
    return wwtp_pop_map


def load_wwtp_clinical_from_metrics(
    file_path: str,
    *,
    disease: str = "COVID",
    wwtp_shp = "../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp",
    metric: str = "hospitalization",   # or "case"
    value_kind: str = "rate",          # "rate" or "raw"
) -> pd.DataFrame:
    """
    WWTP/sewershed-level aggregate (all ages). Returns ['week','wwtp','clinical_rate'].
    Uses wwtp_shp (field wwtp) as the MASTER list to build a full panel.
    """
    wwtp_universe = _load_entity_universe_from_shp(wwtp_shp, "wwtp")

    pop_lookup = _build_wwtp_pop_lookup(wwtp_shp) if value_kind.lower() == "rate" else None

    tmp = _extract_from_long(
        file_path,
        region_level="sewershed",
        metric=metric,
        value_kind=value_kind,
        pop_lookup=pop_lookup,
        entity_universe=wwtp_universe,     # enforce the complete WWTP-by-week panel
    )

    out = pd.DataFrame({
        "week": tmp["week"],
        "wwtp": tmp["region"],             # lowercase at this point
        "clinical_rate": tmp["value"],
    })

    # Drop WWTPs with ANY NaN in rate (missing denom)
    if value_kind.lower() == "rate":
        out = _drop_entities_with_any_nan(out, entity_col="wwtp", value_col="clinical_rate")

    # Make names pretty AFTER dropping
    out["wwtp"] = out["wwtp"].astype(str).str.strip().str.title()
    out = out.sort_values(["wwtp", "week"]).reset_index(drop=True)

    if value_kind.lower() == "rate":
        logger.info(f"[load_wwtp_clinical_from_metrics] CLEANED -> {len(out['wwtp'].unique())} WWTPs • {len(out)} rows • rate per 100k.")
    else:
        logger.info(f"[load_wwtp_clinical_from_metrics] {len(out['wwtp'].unique())} WWTPs • {len(out)} rows • raw counts/sum.")

    return out


# -----------------------------------------------------------------------------
# RSV-only: public API (age-group counts)
# -----------------------------------------------------------------------------

def load_county_rsv_by_age(*, file_path: str) -> pd.DataFrame:
    """Return weekly RSV hospitalization counts by County and Age Group (counts only)."""
    tmp = _extract_rsv_by_age(file_path, region_level="county")
    out = (
        tmp.rename(columns={"region": "County", "age_group": "Age Group"})
           .assign(County=lambda d: d["County"].str.title())
           .sort_values(["County", "Age Group", "week"])
           .reset_index(drop=True)
    )
    return out

def load_wwtp_rsv_by_age(*, file_path: str) -> pd.DataFrame:
    """Return weekly RSV hospitalization counts by WWTP and Age Group (counts only)."""
    tmp = _extract_rsv_by_age(file_path, region_level="sewershed")
    out = (
        tmp.rename(columns={"region": "wwtp", "age_group": "Age Group"})
           .assign(wwtp=lambda d: d["wwtp"].str.title())
           .sort_values(["wwtp", "Age Group", "week"])
           .reset_index(drop=True)
    )
    return out


# -----------------------------------------------------------------------------
# Fixed-source wrappers (same as V2)
# -----------------------------------------------------------------------------

def _resolve_file_for_disease(disease: str) -> str:
    key = _norm_disease(disease).upper()
    if key not in _FILE_MAP:
        raise ValueError(f"Unsupported disease '{disease}'. Must be one of: {list(_FILE_MAP.keys())}.")
    return _FILE_MAP[key]

def load_county_clinical_fixed(
    *,
    disease: str,
    county_shp: str,
    county_pop_csv: str,
    metric: str = "hospitalization",
    value_kind: str = "rate",
) -> pd.DataFrame:
    file_path = _resolve_file_for_disease(disease)
    return load_county_clinical(
        file_path,
        disease=disease,
        county_shp=county_shp,
        county_pop_csv=county_pop_csv,
        metric=metric,
        value_kind=value_kind,
    )

def load_wwtp_clinical_from_metrics_fixed(
    *,
    disease: str,
    wwtp_shp: str,
    metric: str = "hospitalization",
    value_kind: str = "rate",
) -> pd.DataFrame:
    file_path = _resolve_file_for_disease(disease)
    return load_wwtp_clinical_from_metrics(
        file_path,
        disease=disease,
        wwtp_shp=wwtp_shp,
        metric=metric,
        value_kind=value_kind,
    )


# -----------------------------------------------------------------------------
# Optional panel verification.
# -----------------------------------------------------------------------------

def verify_full_panel(df: pd.DataFrame, entity_col: str, week_col: str = "week") -> None:
    d = df.copy()
    d[week_col] = pd.to_datetime(d[week_col], errors="coerce")
    nE = int(d[entity_col].nunique())
    nW = int(d[week_col].nunique())
    expected = nE * nW
    actual = int(len(d))
    logger.info(f"[verify_full_panel] {entity_col} unique={nE} | weeks={nW} | expected rows={expected} | actual rows={actual}")
    if expected != actual:
        raise AssertionError("Panel is NOT complete: missing entity-week rows.")

def weekly_county_rates_to_monthly(
    df_weekly,
    county_shp,
    county_pop_csv,
    week_col="week",
    county_col="County",
    rate_col="clinical_rate",
    extra_group_cols=["disease"],
    month_col="month",
):
    """
    Convert weekly county rate (per 100k) data to monthly rate (per 100k).

    Steps:
    1. Convert weekly rate -> implied weekly counts
    2. Sum counts within month
    3. Convert back to monthly rate per 100k
    """

    df = df_weekly.copy()

    df[week_col] = pd.to_datetime(df[week_col])
    df[county_col] = df[county_col].astype(str).str.strip()

    # --- Build population lookup ---
    pop_lookup = _build_county_pop_lookup(county_shp, county_pop_csv)

    df["_county_norm"] = df[county_col].str.lower()
    keys = list(zip(["county"] * len(df), df["_county_norm"]))
    pops = np.array([float(pop_lookup.get(k, np.nan)) for k in keys])

    df["_pop"] = pops

    # Weekly rate -> weekly implied count
    df["_count"] = df[rate_col] * df["_pop"] / 1e5

    # Create month column
    df[month_col] = df[week_col].dt.to_period("M").dt.to_timestamp()

    group_cols = [month_col, "_county_norm"] + extra_group_cols

    monthly = (
        df.groupby(group_cols, as_index=False)
        .agg(
            _count_sum=("_count", "sum"),
            _pop_first=("_pop", "first")
        )
    )

    # Convert back to monthly rate per 100k
    monthly[rate_col] = monthly["_count_sum"] / monthly["_pop_first"] * 1e5

    # Clean output
    monthly[county_col] = monthly["_county_norm"].str.title()

    final_cols = [month_col, county_col] + extra_group_cols + [rate_col]
    monthly = monthly[final_cols]

    monthly = monthly.sort_values(final_cols[:2]).reset_index(drop=True)

    return monthly

# if __name__ == "__main__":
#     # Example usage (raw counts -> best for “missing count should be 0”)
#     # df_c = load_county_clinical_from_fixed = load_county_clinical_fixed(
#     #     disease="COVID",
#     #     county_shp="../ZoneSelection/Input/Geo/COCounty.shp",
#     #     county_pop_csv="../ZoneSelection/Input/Geo/CO_County_Population_FIPS5.csv",
#     #     metric="case",
#     #     value_kind="raw",
#     # )
#     # verify_full_panel(df_c, "County")
#     #
#     # df_w = load_wwtp_clinical_from_metrics_fixed(
#     #     disease="COVID",
#     #     wwtp_shp="../ZoneSelection/Input/Geo/WWTP_Select.shp",
#     #     metric="case",
#     #     value_kind="raw",
#     # )
#     # verify_full_panel(df_w, "wwtp")
#
#     import pandas as pd
#     from Process_ViralLoader_V2 import (
#         load_county_clinical_fixed
#     )
#
#     # ---------------------------------------------------
#     # Paths (update if needed)
#     # ---------------------------------------------------
#     county_shp = "../ZoneSelection/Input/Census/COCounty.shp"
#     county_pop_csv = "../ZoneSelection/Input/Census/CO_County_Population_FIPS5.csv"
#
#     # ---------------------------------------------------
#     # Load case rate (per 100k) for each disease
#     # ---------------------------------------------------
#     df_covid = load_county_clinical_fixed(
#         disease="COVID",
#         county_shp=county_shp,
#         county_pop_csv=county_pop_csv,
#         metric="hospitalization",
#         value_kind="rate"
#     )
#     df_covid["disease"] = "COVID"
#
#     df_flu = load_county_clinical_fixed(
#         disease="FLU",
#         county_shp=county_shp,
#         county_pop_csv=county_pop_csv,
#         metric="hospitalization",
#         value_kind="rate"
#     )
#     df_flu["disease"] = "Influenza"
#
#     df_rsv = load_county_clinical_fixed(
#         disease="RSV",
#         county_shp=county_shp,
#         county_pop_csv=county_pop_csv,
#         metric="hospitalization",
#         value_kind="rate"
#     )
#     df_rsv["disease"] = "RSV"
#
#     # ---------------------------------------------------
#     # Combine
#     # ---------------------------------------------------
#     df_all = pd.concat([df_covid, df_flu, df_rsv], ignore_index=True)
#
#     df_all = df_all.sort_values(["County", "disease", "week"]).reset_index(drop=True)
#
#     # ---------------------------------------------------
#     # Save
#     # ---------------------------------------------------
#     # ---------------------------------------------------
#     # Save WEEKLY
#     # ---------------------------------------------------
#     weekly_out = "../ZoneSelection/Outfile/county_case_rates_all_diseases.csv"
#     df_all.to_csv(weekly_out, index=False)
#     print("Saved weekly:", weekly_out)
#
#     # ---------------------------------------------------
#     # Build + Save MONTHLY (from weekly rates)
#     # ---------------------------------------------------
#     monthly_out = "../ZoneSelection/Outfile/county_case_rates_all_diseases_monthly.csv"
#
#     df_monthly = weekly_county_rates_to_monthly(
#         df_weekly=df_all,
#         county_shp=county_shp,
#         county_pop_csv=county_pop_csv,
#         week_col="week",
#         county_col="County",
#         rate_col="clinical_rate",
#         extra_group_cols=["disease"],
#         month_col="month",
#     )
#
#     df_monthly.to_csv(monthly_out, index=False)
#     print("Saved monthly:", monthly_out)


if __name__ == "__main__":
    import os
    import pandas as pd

    # ---------------------------------------------------
    # Paths
    # ---------------------------------------------------
    county_shp = "../ZoneSelection/Input/Census/COCounty.shp"
    county_pop_csv = "../ZoneSelection/Input/Census/CO_County_Population_FIPS5.csv"
    wwtp_shp = "../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp"

    out_dir = "../ZoneSelection/Outfile"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------
    # Load weekly hospitalization rates
    # Same logic as Analysis_ODDiffusionRisk_newdata_clinicalonly.py
    # ---------------------------------------------------

    county_rates = []

    for disease_label, disease_loader_name in [
        ("COVID", "COVID"),
        ("Flu", "Influenza"),
        ("RSV", "RSV"),
    ]:
        df = load_county_clinical_fixed(
            disease=disease_loader_name,
            county_shp=county_shp,
            county_pop_csv=county_pop_csv,
            metric="hospitalization",
            value_kind="rate",
        )
        df["disease"] = disease_label
        county_rates.append(df)

    county_weekly = pd.concat(county_rates, ignore_index=True)
    county_weekly = county_weekly.sort_values(
        ["County", "disease", "week"]
    ).reset_index(drop=True)

    county_weekly_out = os.path.join(
        out_dir, "county_hospitalization_rates_all_diseases_weekly.csv"
    )
    county_weekly.to_csv(county_weekly_out, index=False)
    print("Saved weekly county hospitalization rates:", county_weekly_out)

    # ---------------------------------------------------
    # Monthly county mean
    # This matches the simple monthly mean logic used in diffusion output
    # ---------------------------------------------------
    county_weekly["year_month"] = pd.to_datetime(county_weekly["week"]).dt.to_period("M").astype(str)

    county_monthly = (
        county_weekly
        .groupby(["County", "disease", "year_month"], as_index=False)["clinical_rate"]
        .mean()
    )

    county_monthly_out = os.path.join(
        out_dir, "county_hospitalization_rates_all_diseases_monthly_mean.csv"
    )
    county_monthly.to_csv(county_monthly_out, index=False)
    print("Saved monthly county hospitalization rates:", county_monthly_out)

    # ---------------------------------------------------
    # Load weekly WWTP hospitalization rates
    # Same logic as diffusion summary clinical_rate_wwtp_{Disease}
    # ---------------------------------------------------

    wwtp_rates = []

    for disease_label, disease_loader_name in [
        ("COVID", "COVID"),
        ("Flu", "Influenza"),
        ("RSV", "RSV"),
    ]:
        df = load_wwtp_clinical_from_metrics_fixed(
            disease=disease_loader_name,
            wwtp_shp=wwtp_shp,
            metric="hospitalization",
            value_kind="rate",
        )
        df["disease"] = disease_label
        wwtp_rates.append(df)

    wwtp_weekly = pd.concat(wwtp_rates, ignore_index=True)
    wwtp_weekly = wwtp_weekly.sort_values(
        ["wwtp", "disease", "week"]
    ).reset_index(drop=True)

    wwtp_weekly_out = os.path.join(
        out_dir, "wwtp_hospitalization_rates_all_diseases_weekly.csv"
    )
    wwtp_weekly.to_csv(wwtp_weekly_out, index=False)
    print("Saved weekly WWTP hospitalization rates:", wwtp_weekly_out)

    # ---------------------------------------------------
    # Monthly WWTP mean
    # This matches diffusion monthly mean output style
    # ---------------------------------------------------
    wwtp_weekly["year_month"] = pd.to_datetime(wwtp_weekly["week"]).dt.to_period("M").astype(str)

    wwtp_monthly = (
        wwtp_weekly
        .groupby(["wwtp", "disease", "year_month"], as_index=False)["clinical_rate"]
        .mean()
    )

    wwtp_monthly_out = os.path.join(
        out_dir, "wwtp_hospitalization_rates_all_diseases_monthly_mean.csv"
    )
    wwtp_monthly.to_csv(wwtp_monthly_out, index=False)
    print("Saved monthly WWTP hospitalization rates:", wwtp_monthly_out)
