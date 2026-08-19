"""Central preprocessing utilities for weekly clinical and wastewater inputs.

Weekly time series are standardized to Monday and duplicate records are
collapsed by mean aggregation. Viral-load channels are optional and were not
used in the reported Level 2 case-only implementation.
"""

from __future__ import annotations
import os
from typing import Optional, Callable, Dict
import geopandas as gpd
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _to_monday(x: pd.Series) -> pd.Series:
    """Convert any datetime-like series to the Monday of that ISO week."""
    dt = pd.to_datetime(x, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def _infer_col(preferred: str, candidates: list[str]) -> Optional[str]:
    """Return an existing column from candidates that exactly matches (case-insensitive)
    or contains the preferred token; otherwise None."""
    low = preferred.lower()
    for c in candidates:
        if str(c).lower() == low:
            return c
    for c in candidates:
        if low in str(c).lower():
            return c
    return None


# -----------------------------------------------------------------------------
# County-level clinical
# -----------------------------------------------------------------------------

def preprocess_county_covid_clinical(
    file_path: str,
    county_col: str = "County",
    date_candidates: tuple[str, ...] = ("week", "Week", "Date", "date", "week_end"),
    rate_col: Optional[str] = None,
    agg_func: str | Callable = "mean",
) -> pd.DataFrame:
    """
    Load county-level COVID clinical data and output a tidy weekly frame with Monday anchoring.

    Output columns: ['week','County','clinical_rate']
      • 'week' is Monday of week
      • County is lowercased/stripped string
      • Duplicate (week, County) rows are aggregated by `agg_func` (default mean)

    The function is permissive about the date column by searching in `date_candidates`.
    """
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame(columns=["week", "County", "clinical_rate"])

    # Read CSV or Excel
    df = pd.read_excel(file_path) if not file_path.lower().endswith(".csv") else pd.read_csv(file_path)

    # Resolve columns
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None or county_col not in df.columns:
        raise ValueError("Bad county COVID clinical file/columns: need a date col and 'County'.")

    if rate_col is None:
        # Try to guess a numeric rate/count column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        hints = [c for c in num_cols if any(k in str(c).lower() for k in ["rate", "per100", "per_100", "case"]) ]
        rate_col = hints[0] if hints else (num_cols[0] if num_cols else None)
    if rate_col is None or rate_col not in df.columns:
        raise ValueError("Could not infer the COVID clinical rate column.")

    out = df[[date_col, county_col, rate_col]].rename(
        columns={date_col: "week", county_col: "County", rate_col: "clinical_rate"}
    )

    out["week"] = _to_monday(out["week"])  # Monday-anchored
    out["County"] = out["County"].astype(str).str.strip().str.lower()

    out = (
        out.groupby(["week", "County"], as_index=False, sort=False)["clinical_rate"]
           .agg(agg_func)
           .sort_values(["County", "week"])
           .reset_index(drop=True)
    )
    return out


def preprocess_flu_rsv_hospitalization_data(
    file_path: str,
    fips_lookup: Dict[str, str] | None,
    pathogen: str,                              # "Influenza" or "RSV"
    date_col: str = "Date",
    county_col: str = "County",
    pathogen_col: str = "pathogen",
    hosp_rate_col: str = "Hospitalized_CaseCount_r100Kutil",
    clip_max: float = 10.0,
) -> pd.DataFrame:
    df = pd.read_excel(file_path) if not file_path.lower().endswith(".csv") else pd.read_csv(file_path)

    req = {date_col, county_col, pathogen_col, hosp_rate_col}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s) {missing} in {file_path}")

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["week"] = _to_monday(dt)
    df["County"] = df[county_col].astype(str).str.strip().str.lower()

    # Try to get a CountyFIPS value:
    fips_col_candidates = ["CountyFIPS","county_fips","FIPS","GEOID","geoid"]
    existing_fips_col = next((c for c in fips_col_candidates if c in df.columns), None)
    if existing_fips_col:
        df["CountyFIPS"] = df[existing_fips_col].astype(str).str.zfill(5)
        has_fips = True
    elif fips_lookup:
        df["CountyFIPS"] = df["County"].map(fips_lookup).astype(str).str.zfill(5)
        has_fips = df["CountyFIPS"].notna().any()
        if not has_fips:
            df.drop(columns=["CountyFIPS"], errors="ignore", inplace=True)
    else:
        has_fips = False  # no mapping and no column; proceed without FIPS

    want = str(pathogen).strip().lower()
    if want not in {"influenza", "rsv"}:
        raise ValueError("pathogen must be 'Influenza' or 'RSV'")
    df = df[df[pathogen_col].astype(str).str.strip().str.lower() == want].copy()

    df[hosp_rate_col] = pd.to_numeric(df[hosp_rate_col], errors="coerce")
    df["real_value"] = df[hosp_rate_col]
    df["norm_rate"] = df[hosp_rate_col].clip(0, clip_max) / float(clip_max)

    num_cols = df.select_dtypes(include=[np.number]).columns
    group_keys = ["week", "County"] + (["CountyFIPS"] if has_fips else [])
    out = (
        df.groupby(group_keys, as_index=False)[num_cols]
          .mean()
          .sort_values(["County", "week"]).reset_index(drop=True)
    )
    return out



# -----------------------------------------------------------------------------
# County-level WVAL (weighted viral load per county)
# -----------------------------------------------------------------------------

def preprocess_county_weighted_wval(
    csv_path: str,
    value_col: Optional[str] = None,
    fips_to_name: Optional[Dict[str, str]] = None,
    agg_func: str | Callable = "mean",
) -> pd.DataFrame:
    """
    Returns ['week','County','county_wval'] averaged within (week, County).
    Accepts either 'County' or 'county_fips' in the CSV and normalizes 'week' to Monday.
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=["week", "County", "county_wval"])

    cw = pd.read_csv(csv_path)

    # Date column
    date_col = None
    for c in ("week", "Week", "date", "Date", "week_end"):
        if c in cw.columns:
            date_col = c
            break
    if not date_col:
        raise ValueError("county_wval_csv missing a date/week column")

    # County column: accept 'County' or 'county_fips'
    if "County" in cw.columns:
        cw["County"] = cw["County"].astype(str).str.strip().str.lower()
    elif "county_fips" in cw.columns:
        cw["county_fips"] = cw["county_fips"].astype(str).str.strip().str.zfill(5)
        if fips_to_name:
            cw["County"] = cw["county_fips"].map(fips_to_name).fillna(cw["county_fips"])
        else:
            cw["County"] = cw["county_fips"]
        cw["County"] = cw["County"].astype(str).str.strip().str.lower()
    else:
        raise ValueError("county_wval_csv must contain a 'County' or 'county_fips' column")

    # Value column inference if needed
    if value_col is None:
        num_cols = [c for c in cw.columns if pd.api.types.is_numeric_dtype(cw[c])]
        hints = [c for c in num_cols if any(k in c.lower() for k in ["wval", "viral", "ww", "index", "norm"])]
        value_col = hints[0] if hints else (num_cols[0] if num_cols else None)
    if not value_col or value_col not in cw.columns:
        raise ValueError("Could not infer county WVAL value column")

    tmp = cw[[date_col, "County", value_col]].rename(
        columns={date_col: "week", value_col: "county_wval"}
    )
    tmp["week"] = _to_monday(tmp["week"])  # Monday-anchored

    out = (
        tmp.groupby(["week", "County"], as_index=False, sort=False)["county_wval"]
           .agg(agg_func)
           .sort_values(["County", "week"]).reset_index(drop=True)
    )
    return out


# -----------------------------------------------------------------------------
# WWTP-level viral load (WVAL)
# -----------------------------------------------------------------------------

def preprocess_viral_load(
    file_path: str,
    value_col: str = "ww_index_normed_ln_lin",
    wwtp_name_col: str = "wwtp_name",
    week_end_col: str = "week_end",
    agg_func: str | Callable = "mean",
) -> pd.DataFrame:
    """
    Load WWTP viral load (WVAL) from an Excel/CSV with columns containing
    the WWTP name and a week/date column (default: 'wwtp_name', 'week_end').

    Output columns: ['week','wwtp','ww_index_normed_ln_lin','viral_value','real_value']
    with Monday anchoring and mean aggregation within (week, wwtp).
    """
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame(columns=["week", "wwtp", value_col])

    df = pd.read_excel(file_path) if not file_path.lower().endswith(".csv") else pd.read_csv(file_path)

    # Resolve columns if naming differs slightly
    if week_end_col not in df.columns:
        week_end_col = _infer_col("week_end", list(df.columns)) or _infer_col("week", list(df.columns)) or week_end_col
    if wwtp_name_col not in df.columns:
        wwtp_name_col = _infer_col("wwtp_name", list(df.columns)) or _infer_col("wwtp", list(df.columns)) or wwtp_name_col

    if week_end_col not in df.columns or wwtp_name_col not in df.columns:
        raise KeyError("Required columns for WWTP WVAL not found (need week/week_end and wwtp/wwtp_name).")

    if value_col not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        hints = [c for c in num_cols if any(k in c.lower() for k in ["wval", "viral", "ww", "index", "norm"]) ]
        if hints:
            value_col = hints[0]
        else:
            raise KeyError("Unable to infer WVAL value column.")

    out = df[[week_end_col, wwtp_name_col, value_col]].rename(
        columns={week_end_col: "week", wwtp_name_col: "wwtp", value_col: "ww_index_normed_ln_lin"}
    )

    out["week"] = _to_monday(out["week"])  # Monday-anchored
    out["wwtp"] = out["wwtp"].astype(str).str.strip().str.lower()

    out = (
        out.groupby(["week", "wwtp"], as_index=False)["ww_index_normed_ln_lin"].agg(agg_func)
           .sort_values(["wwtp", "week"]).reset_index(drop=True)
    )

    # Convenience normalized columns (optional)
    out["viral_value"] = out["ww_index_normed_ln_lin"].clip(0, 10) / 10.0
    out["real_value"]  = out["ww_index_normed_ln_lin"]
    return out


# -----------------------------------------------------------------------------
# WWTP-level clinical allocation metrics (case / hospitalization)
# -----------------------------------------------------------------------------

def preprocess_wwtp_clinical_alloc(
    file_path: str,
    disease: str = "COVID",                 # e.g., "COVID" (handles synonyms)
    target: str = "case",                   # "case" | "hospitalization"
    treat_missing_as_zero: bool = True,     # fill missing rate(s) with 0 after pivot
    return_both: bool = False,              # if True → returns both case_rate & hosp_rate
    keep_coverage_cols: bool = True,        # include coverage columns if present
    # Resolve input columns case-insensitively, with substring matching as fallback.
    week_col: str = "week",
    utility_col: str = "Utility",
    disease_col: str = "disease",
    metric_col: str = "metric",
    value_col: str = "wwtp_value",
    covered_pop_col: str = "covered_pop",
    total_pop_col: str = "total_pop",
    coverage_pct_col: str = "coverage_pct",
    num_counties_used_col: str = "num_counties_used",
) -> pd.DataFrame:
    """
    Reads a WWTP clinical metrics file (CSV/XLSX) with long-format rows like:
      week, Utility, disease, metric, wwtp_value, [covered_pop, total_pop, coverage_pct, num_counties_used]

    When return_both == False: → ['week','wwtp','disease','rate', (optional coverage cols)]
    When return_both == True:  → ['week','wwtp','disease','case_rate','hosp_rate', (optional coverage cols)]
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)

    # Resolve core columns
    wk  = _infer_col(week_col, df.columns) or week_col
    uti = _infer_col(utility_col, df.columns) or utility_col
    dis = _infer_col(disease_col, df.columns) or disease_col
    met = _infer_col(metric_col, df.columns) or metric_col
    val = _infer_col(value_col, df.columns) or value_col
    for req in (wk, uti, dis, met, val):
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' missing. Available: {list(df.columns)}")

    # Normalize identifiers
    df = df.copy()
    df["wwtp"] = df[uti].astype(str).str.strip().str.lower()
    df["week"] = _to_monday(df[wk])

    # Disease filter with synonyms
    synonyms = {
        "covid": {"covid", "sars-cov-2", "sars_cov_2", "sarscov2", "sars cov 2"},
        "influenza": {"influenza", "flu"},
        "rsv": {"rsv"},
    }
    want = str(disease).strip().lower()
    cand = {want}
    for k, vals in synonyms.items():
        if want == k or want in vals:
            cand |= vals | {k}
    df[dis] = df[dis].astype(str).str.strip().str.lower()
    df = df[df[dis].isin(cand)]
    if df.empty:
        return pd.DataFrame(columns=(
            ["week", "wwtp", "disease", "rate"] if not return_both
            else ["week", "wwtp", "disease", "case_rate", "hosp_rate"]
        ))

    # Metric → normalized labels
    def _norm_metric(s: str) -> Optional[str]:
        s = str(s).strip().lower()
        if ("hosp" in s) or ("hospital" in s) or s.startswith("hosp_rate"):
            return "hosp_rate"
        if ("case" in s) or s.startswith("case_rate"):
            return "case_rate"
        return None

    df["metric_norm"] = df[met].map(_norm_metric)
    df = df[df["metric_norm"].isin(["case_rate", "hosp_rate"])].copy()
    if df.empty:
        return pd.DataFrame(columns=(
            ["week", "wwtp", "disease", "rate"] if not return_both
            else ["week", "wwtp", "disease", "case_rate", "hosp_rate"]
        ))

    # Values → numeric
    df["value"] = pd.to_numeric(df[val], errors="coerce")

    # Coverage column discovery (optional)
    cov_want = [covered_pop_col, total_pop_col, coverage_pct_col, num_counties_used_col]
    cov_map: Dict[str, str] = {}
    if keep_coverage_cols:
        for name in cov_want:
            found = _infer_col(name, df.columns)
            if found is not None:
                cov_map[name] = found

    # Pivot metrics to wide (mean across duplicates)
    wide = (
        df.pivot_table(index=["week", "wwtp", dis], columns="metric_norm", values="value", aggfunc="mean")
          .reset_index()
          .rename_axis(None, axis=1)
          .rename(columns={dis: "disease"})
    )

    # Attach coverage (mean across duplicates)
    if keep_coverage_cols and len(cov_map) > 0:
        cov_cols_actual = list(cov_map.values())
        cov_df = df[["week", "wwtp"] + cov_cols_actual].copy()
        for c in cov_cols_actual:
            cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
        cov_agg = cov_df.groupby(["week", "wwtp"], as_index=False).mean()
        back_names = {v: k for (k, v) in cov_map.items()}
        cov_agg = cov_agg.rename(columns=back_names)
        wide = wide.merge(cov_agg, on=["week", "wwtp"], how="left")

    # Missing policy for rates
    if treat_missing_as_zero:
        if "case_rate" in wide.columns:
            wide["case_rate"] = wide["case_rate"].fillna(0.0)
        if "hosp_rate" in wide.columns:
            wide["hosp_rate"] = wide["hosp_rate"].fillna(0.0)

    # Final selection
    if return_both:
        if "case_rate" not in wide.columns:
            wide["case_rate"] = np.nan
        if "hosp_rate" not in wide.columns:
            wide["hosp_rate"] = np.nan
        base_cols = ["week", "wwtp", "disease", "case_rate", "hosp_rate"]
    else:
        if target not in {"case", "hospitalization"}:
            raise ValueError("target must be 'case' or 'hospitalization'")
        rate_col = "case_rate" if target == "case" else "hosp_rate"
        if rate_col not in wide.columns:
            wide[rate_col] = np.nan
        if rate_col == "hosp_rate":
            wide["rate"] = np.where(wide["hosp_rate"].notna(), wide["hosp_rate"], wide.get("case_rate", np.nan))
        else:
            wide["rate"] = wide["case_rate"]
        base_cols = ["week", "wwtp", "disease", "rate"]

    cov_canon = [covered_pop_col, total_pop_col, coverage_pct_col, num_counties_used_col]
    cov_keep = [c for c in cov_canon if (keep_coverage_cols and (c in wide.columns))]
    cols_final = [c for c in (base_cols + cov_keep) if c in wide.columns]

    out = wide[cols_final].sort_values(["wwtp", "week"]).reset_index(drop=True)
    return out



def load_county_clinical(disease, covid_xlsx=None, covid_rate_col=None, flu_rsv_rate_col=None,flu_rsv_xlsx=None, agg_func="mean"):
    """
    Returns ['week','County','clinical_rate'] with Monday-anchored week
    and duplicates aggregated by agg_func (default: mean).
    """
    disease_u = str(disease).upper()

    if disease_u == "COVID":
        if not covid_xlsx or not os.path.exists(covid_xlsx):
            return pd.DataFrame(columns=["week","County","clinical_rate"])

        df = pd.read_excel(covid_xlsx)
        date_col = next((c for c in ["week","Week","Date","date","week_end"] if c in df.columns), None)
        if not date_col or "County" not in df.columns or not covid_rate_col or covid_rate_col not in df.columns:
            raise ValueError("Bad COVID county clinical file/columns")

        out = df[[date_col,"County",covid_rate_col]].rename(
            columns={date_col:"week", covid_rate_col:"clinical_rate"}
        )
        out["week"]   = _to_monday(out["week"])
        out["County"] = out["County"].astype(str).str.strip().str.lower()

        out = (out.groupby(["week","County"], as_index=False, sort=False)["clinical_rate"]
                  .agg(agg_func)
                  .sort_values(["County","week"])
                  .reset_index(drop=True))
        return out

    elif disease_u in {"INFLUENZA","RSV"}:
        out = preprocess_flu_rsv_hospitalization_data(
            file_path=flu_rsv_xlsx, fips_lookup={}, pathogen=("Influenza" if disease_u=="INFLUENZA" else "RSV"),
            date_col="Date", county_col="County", pathogen_col="pathogen",
            # Use the configured hospitalization-rate field; raw counts are not
            # the clinical outcome in the reported analysis.
            hosp_rate_col=flu_rsv_rate_col,
        )[["week","County","norm_rate"]].rename(columns={"norm_rate":"clinical_rate"})

        # Defensive re-agg (the preprocessor already averages; this makes behavior uniform)
        out = (out.groupby(["week","County"], as_index=False, sort=False)["clinical_rate"]
                  .agg(agg_func)
                  .sort_values(["County","week"])
                  .reset_index(drop=True))
        return out

    else:
        raise ValueError("Unsupported disease")


def load_county_weighted_wval(csv_path, value_col=None, fips_to_name=None, agg_func="mean"):
    """
    Returns ['week','County','county_wval'] with Monday-anchored week and duplicates averaged.
    Accepts either 'County' or 'county_fips' in the CSV. If only county_fips is present,
    it is zero-padded and mapped to names if fips_to_name is provided.
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=["week","County","county_wval"])

    cw = pd.read_csv(csv_path)

    # date column
    date_col = next((c for c in ["week","Week","date","Date","week_end"] if c in cw.columns), None)
    if not date_col:
        raise ValueError("county_wval_csv missing a date/week column")

    # county column: accept 'County' or 'county_fips'
    if "County" in cw.columns:
        cw["County"] = cw["County"].astype(str).str.strip().str.lower()
    elif "county_fips" in cw.columns:
        cw["county_fips"] = cw["county_fips"].astype(str).str.strip().str.zfill(5)
        if fips_to_name:
            cw["County"] = cw["county_fips"].map(fips_to_name).fillna(cw["county_fips"])
        else:
            cw["County"] = cw["county_fips"]
        cw["County"] = cw["County"].astype(str).str.strip().str.lower()
    else:
        raise ValueError("county_wval_csv must contain a 'County' or 'county_fips' column")

    # value column
    if value_col is None:
        num_cols = [c for c in cw.columns if pd.api.types.is_numeric_dtype(cw[c])]
        hints    = [c for c in num_cols if any(k in c.lower() for k in ["wval","viral","ww"])]
        value_col = hints[0] if hints else (num_cols[0] if num_cols else None)
    if not value_col or value_col not in cw.columns:
        raise ValueError("Could not infer county WVAL value column")

    tmp = cw[[date_col, "County", value_col]].rename(
        columns={date_col: "week", value_col: "county_wval"}
    )
    tmp["week"] = _to_monday(tmp["week"])

    out = (tmp.groupby(["week","County"], as_index=False, sort=False)["county_wval"]
              .agg(agg_func)
              .sort_values(["County","week"])
              .reset_index(drop=True))
    return out


def load_wwtp_wval(xlsx_path, agg_func="mean"):
    """
    Returns ['week','wwtp','ww_index_normed_ln_lin'] with Monday-anchored week and duplicates averaged.
    """
    if not xlsx_path or not os.path.exists(xlsx_path):
        return pd.DataFrame(columns=["week","wwtp","ww_index_normed_ln_lin"])

    df = preprocess_viral_load(xlsx_path)

    # Defensive re-agg in case upstream file had residual duplicates
    out = (df.groupby(["week","wwtp"], as_index=False, sort=False)["ww_index_normed_ln_lin"]
             .agg(agg_func)
             .sort_values(["wwtp","week"])
             .reset_index(drop=True))
    return out


def load_wwtp_clinical_from_metrics(metrics_csv, disease, target, treat_missing_as_zero, agg_func="mean"):
    """
    Returns ['week','wwtp','clinical_rate'] with Monday-anchored week and duplicates averaged.
    """
    if not metrics_csv or not os.path.exists(metrics_csv):
        return pd.DataFrame(columns=["week","wwtp","clinical_rate"])

    wide = preprocess_wwtp_clinical_alloc(
        file_path=metrics_csv, disease=disease, target=target,
        treat_missing_as_zero=treat_missing_as_zero,
        return_both=False, keep_coverage_cols=False,
    )
    if wide.empty:
        return pd.DataFrame(columns=["week","wwtp","clinical_rate"])

    out = wide.rename(columns={"rate":"clinical_rate"})[["week","wwtp","clinical_rate"]].copy()

    # Defensive re-agg
    out = (out.groupby(["week","wwtp"], as_index=False, sort=False)["clinical_rate"]
             .agg(agg_func)
             .sort_values(["wwtp","week"])
             .reset_index(drop=True))
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
        gdf, ["wwtp","WWTP","Utility","UTILITY","Name","NAME","UtilityName","UTIL_NAME"]
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

    # Aggregate multiple polygons belonging to the same plant.
    # wwtp_gdf = wwtp_gdf.groupby("wwtp", as_index=False).agg({"pop_served":"sum", "geometry":"unary_union"})

    wwtp_pop_map = wwtp_gdf.groupby("wwtp")["pop_served"].sum().to_dict()
    return wwtp_pop_map


# -------------------------------------------------------------------------
# Generic county-level syndromic preprocessor (wrapper)
# Returns a tidy frame with: ['week','CountyFIPS','norm_rate','real_value']
# -------------------------------------------------------------------------
def preprocess_syndromic_data(
    file_path: str,
    fips_lookup: dict | None = None,
    rate_col: str | None = None,
    pathogen: str | None = None,     # "Influenza" or "RSV" when the file contains a 'pathogen' column
    clip_max: float = 10.0           # used to create a 0–1 normalized 'norm_rate'
) -> pd.DataFrame:
    """
    Generic loader for county-level syndromic datasets.

    Behavior:
      • If the file has a 'pathogen' column → treated as Flu/RSV hospitalization input.
        - If `pathogen` is provided, filter to that pathogen; otherwise, infer if only one exists.
        - Uses preprocess_flu_rsv_hospitalization_data(...).
      • Otherwise → treated as a COVID-like county clinical file (cases or hospitalization),
        using preprocess_county_covid_clinical(...).
        - `rate_col` should be the numeric rate column to use (e.g., 'County_cases_3dayavg_r100Kutil'
          or 'County_hosp_3dayavg_r100Kutil').

    Output columns are standardized to:
        ['week','CountyFIPS','norm_rate','real_value']
    """
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame(columns=["week", "CountyFIPS", "norm_rate", "real_value"])

    # Read once to inspect columns (final parsing happens in the called helpers)
    peek = pd.read_excel(file_path) if not file_path.lower().endswith(".csv") else pd.read_csv(file_path)
    cols_lower = {c.lower(): c for c in peek.columns}

    # ---- Branch 1: Flu/RSV hospitalization (has 'pathogen' column) ----
    if "pathogen" in cols_lower:
        # Decide which pathogen to use
        if pathogen is None:
            unique_pathogens = (
                peek[cols_lower["pathogen"]].astype(str).str.strip().str.lower().unique().tolist()
            )
            if len(unique_pathogens) == 1:
                pathogen = "Influenza" if unique_pathogens[0] in {"influenza", "flu"} else "RSV"
            else:
                # If ambiguous, default to Influenza and print a hint
                print("[preprocess_syndromic_data] Multiple pathogens present; defaulting to Influenza. "
                      "Pass pathogen='RSV' if needed.")
                pathogen = "Influenza"

        out = preprocess_flu_rsv_hospitalization_data(
            file_path=file_path,
            fips_lookup=(fips_lookup or {}),
            pathogen=pathogen,
            # If caller passes a custom column name, use it; else default used inside helper
            hosp_rate_col=(rate_col if rate_col else "Hospitalized_CaseCount_r100Kutil"),
            clip_max=clip_max,
        )

        # Ensure CountyFIPS exists (helper may already provide it)
        if "CountyFIPS" not in out.columns:
            if fips_lookup:
                out["CountyFIPS"] = (out["County"].map(fips_lookup).astype(str).str.zfill(5))
            else:
                out["CountyFIPS"] = np.nan

        # Standardize output
        keep = ["week", "CountyFIPS", "norm_rate", "real_value"]
        for k in keep:
            if k not in out.columns:
                out[k] = np.nan
        return out[keep].drop_duplicates().reset_index(drop=True)

    # ---- Branch 2: COVID-like county clinical (cases or hospitalization) ----
    # Here `rate_col` should be provided (we'll attempt a best-effort guess if missing).
    if rate_col is None:
        # Try to guess a numeric rate-like column
        num_cols = [c for c in peek.columns if pd.api.types.is_numeric_dtype(peek[c])]
        hints = [c for c in num_cols if any(k in str(c).lower() for k in ["rate", "per100", "per_100", "case", "hosp"])]
        rate_col = hints[0] if hints else None

    out = preprocess_county_covid_clinical(
        file_path=file_path,
        county_col=("County" if "county" in cols_lower else list(cols_lower.values())[0]),
        rate_col=rate_col,
        agg_func="mean",
    )  # → ['week','County','clinical_rate']

    # Normalize and attach FIPS
    out["real_value"] = pd.to_numeric(out["clinical_rate"], errors="coerce")
    out["norm_rate"]  = out["real_value"].clip(0, clip_max) / float(clip_max)

    if fips_lookup:
        out["CountyFIPS"] = out["County"].map(fips_lookup).astype(str).str.zfill(5)
    else:
        # Best-effort: if a CountyFIPS column exists in the raw file, reuse it
        if "countyfips" in cols_lower:
            out["CountyFIPS"] = peek[cols_lower["countyfips"]].astype(str).str.zfill(5)
        elif "fips" in cols_lower:
            out["CountyFIPS"] = peek[cols_lower["fips"]].astype(str).str.zfill(5)
        else:
            out["CountyFIPS"] = np.nan

    keep = ["week", "CountyFIPS", "norm_rate", "real_value"]
    return out[keep].drop_duplicates().reset_index(drop=True)

