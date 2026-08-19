# -*- coding: utf-8 -*-
"""
Analysis_ODDiffusionRisk_newdata_FileGen.py

Single export module for the clinical-only OD-risk pipeline.

Key rule
--------
`weekly_results` is the canonical source for `total_in` and `total_out`.
All weekly risk files and monthly summary files receive those same canonical
trip values, so the same WWTP-week cannot contain conflicting trip counts.

Trip units remain the units supplied by the weekly OD files: mean daily,
outside-adjusted trip volume for the corresponding weekly period. Monthly
traffic summaries report the average, minimum, and maximum across the weekly
mean-daily values assigned to each calendar month.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:  # optional; CSV export does not require geopandas
    gpd = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return safe.strip("._") or "sewershed"


def _write_csv(df: pd.DataFrame, out_csv: str) -> str:
    _ensure_dir(os.path.dirname(out_csv))
    df.to_csv(out_csv, index=False)
    print(f"[export] CSV -> {out_csv}")
    return out_csv


def _canon_wwtp(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def _canon_week(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _in_window(week, start_week=None, end_week=None) -> bool:
    wk = pd.to_datetime(week).normalize()
    if start_week is not None and wk < pd.to_datetime(start_week).normalize():
        return False
    if end_week is not None and wk > pd.to_datetime(end_week).normalize():
        return False
    return True


def _normalize_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    out = mat.copy()
    out.index = out.index.astype(str).str.strip().str.lower()
    out.columns = out.columns.astype(str).str.strip().str.zfill(5)
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


# -----------------------------------------------------------------------------
# Canonical weekly trip tables
# -----------------------------------------------------------------------------

def build_weekly_trip_tables(
    weekly_results: dict,
    start_week=None,
    end_week=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build canonical weekly WWTP totals and weekly county-WWTP edge tables.

    Returns
    -------
    weekly_totals:
        week, wwtp, total_in, total_out, outflow_source

    weekly_edges:
        week, direction, wwtp, county_fips, trips_mean_daily

    Notes
    -----
    `ensure_total_out_column()` should be called upstream first. The totals
    exported here are taken directly from each weekly result DataFrame, rather
    than recalculated separately inside each disease export.
    """
    total_pieces = []
    edge_pieces = []

    for week, result_df in sorted(weekly_results.items(), key=lambda x: pd.to_datetime(x[0])):
        if not _in_window(week, start_week, end_week):
            continue
        if result_df is None or len(result_df) == 0:
            continue

        week_dt = pd.to_datetime(week).normalize()
        base = result_df.copy()
        if "wwtp" not in base.columns:
            if str(base.index.name).lower() == "wwtp":
                base = base.reset_index()
            else:
                raise KeyError(f"Weekly result for {week} has no 'wwtp' key.")

        base["wwtp"] = _canon_wwtp(base["wwtp"])
        for col in ["total_in", "total_out"]:
            if col not in base.columns:
                raise KeyError(
                    f"Weekly result for {week} has no '{col}'. "
                    "Call ensure_total_out_column() before exporting."
                )
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

        # Guard against accidental duplicate site rows.
        totals = (
            base.groupby("wwtp", as_index=False)[["total_in", "total_out"]]
            .sum()
        )
        totals.insert(0, "week", week_dt)
        # pandas may propagate DataFrame-valued attrs from result_df; clear them
        # before concatenation to avoid ambiguous attr comparisons.
        totals.attrs = {}

        mat_out = getattr(result_df, "attrs", {}).get("wwtp_to_county")
        true_out_available = isinstance(mat_out, pd.DataFrame) and not mat_out.empty
        if true_out_available:
            mat_out_norm = _normalize_matrix(mat_out)
            true_out_available = float(mat_out_norm.to_numpy().sum()) > 0.0
        totals["outflow_source"] = (
            "wwtp_to_county" if true_out_available else "inflow_fallback_or_existing_total_out"
        )
        total_pieces.append(totals)

        mat_in = getattr(result_df, "attrs", {}).get("county_to_wwtp")
        if isinstance(mat_in, pd.DataFrame) and not mat_in.empty:
            mi = _normalize_matrix(mat_in)
            edges_in = (
                mi.rename_axis(index="wwtp", columns="county_fips")
                .stack()
                .rename("trips_mean_daily")
                .reset_index()
            )
            edges_in.insert(0, "direction", "county_to_wwtp")
            edges_in.insert(0, "week", week_dt)
            edges_in.attrs = {}
            edge_pieces.append(edges_in)

        if true_out_available:
            edges_out = (
                mat_out_norm.rename_axis(index="wwtp", columns="county_fips")
                .stack()
                .rename("trips_mean_daily")
                .reset_index()
            )
            edges_out.insert(0, "direction", "wwtp_to_county")
            edges_out.insert(0, "week", week_dt)
            edges_out.attrs = {}
            edge_pieces.append(edges_out)

    if not total_pieces:
        raise ValueError("No weekly trip records were available in the requested window.")

    weekly_totals = pd.concat(total_pieces, ignore_index=True)
    weekly_totals["week"] = _canon_week(weekly_totals["week"])
    weekly_totals["wwtp"] = _canon_wwtp(weekly_totals["wwtp"])
    weekly_totals = weekly_totals.sort_values(["week", "wwtp"]).reset_index(drop=True)

    duplicate_totals = weekly_totals.duplicated(["week", "wwtp"], keep=False)
    if duplicate_totals.any():
        raise ValueError("Duplicate canonical WWTP-week rows remain in weekly trip totals.")

    if edge_pieces:
        weekly_edges = pd.concat(edge_pieces, ignore_index=True)
        weekly_edges["week"] = _canon_week(weekly_edges["week"])
        weekly_edges["wwtp"] = _canon_wwtp(weekly_edges["wwtp"])
        weekly_edges["county_fips"] = weekly_edges["county_fips"].astype(str).str.zfill(5)
        weekly_edges["trips_mean_daily"] = pd.to_numeric(
            weekly_edges["trips_mean_daily"], errors="coerce"
        ).fillna(0.0)
        weekly_edges = weekly_edges.sort_values(
            ["week", "direction", "wwtp", "county_fips"]
        ).reset_index(drop=True)
    else:
        weekly_edges = pd.DataFrame(
            columns=["week", "direction", "wwtp", "county_fips", "trips_mean_daily"]
        )

    return weekly_totals, weekly_edges


def build_monthly_mean_table(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Mean of weekly values grouped by the Monday week date's calendar month."""
    df = weekly_df.copy()
    if "week" not in df.columns:
        raise KeyError("Expected a 'week' column for monthly aggregation.")
    df["week"] = _canon_week(df["week"])
    df["year_month"] = df["week"].dt.to_period("M").astype(str)
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month

    group_cols = ["wwtp", "year", "month", "year_month"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"year", "month"}]

    monthly = (
        df.groupby(group_cols, as_index=False, dropna=False)[numeric_cols]
        .mean()
        .sort_values(["wwtp", "year", "month"])
        .reset_index(drop=True)
    )
    return monthly


def build_monthly_trip_stats(weekly_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize canonical weekly trip values by WWTP and calendar month.

    The input `total_in` and `total_out` values are weekly mean-daily trip
    volumes. Therefore, the monthly minimum and maximum are the lowest and
    highest *weekly mean-daily* values within the month, not daily extrema.
    """
    required = {"week", "wwtp", "total_in", "total_out"}
    missing = required.difference(weekly_totals.columns)
    if missing:
        raise KeyError(f"Weekly trip table is missing columns: {sorted(missing)}")

    df = weekly_totals[["week", "wwtp", "total_in", "total_out"]].copy()
    df["week"] = _canon_week(df["week"])
    df["wwtp"] = _canon_wwtp(df["wwtp"])
    df["total_in"] = pd.to_numeric(df["total_in"], errors="coerce")
    df["total_out"] = pd.to_numeric(df["total_out"], errors="coerce")
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month
    df["year_month"] = df["week"].dt.to_period("M").astype(str)

    stats = (
        df.groupby(["wwtp", "year", "month", "year_month"], as_index=False)
        .agg(
            total_in_mean=("total_in", "mean"),
            total_in_min=("total_in", "min"),
            total_in_max=("total_in", "max"),
            total_out_mean=("total_out", "mean"),
            total_out_min=("total_out", "min"),
            total_out_max=("total_out", "max"),
            n_weeks=("week", "nunique"),
        )
        .sort_values(["wwtp", "year", "month"])
        .reset_index(drop=True)
    )
    return stats


def _attach_canonical_trips(
    df: pd.DataFrame,
    weekly_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Replace any disease-table trip columns with canonical WWTP-week values."""
    out = df.copy()
    if out.empty:
        return out
    if "wwtp" not in out.columns or "week" not in out.columns:
        raise KeyError("Risk export requires both 'wwtp' and 'week'.")

    out["wwtp"] = _canon_wwtp(out["wwtp"])
    out["week"] = _canon_week(out["week"])
    out = out.drop(columns=["total_in", "total_out", "outflow_source"], errors="ignore")

    canonical = weekly_totals[
        ["week", "wwtp", "total_in", "total_out", "outflow_source"]
    ].copy()
    out = out.merge(canonical, on=["week", "wwtp"], how="left", validate="many_to_one")

    missing = out[["total_in", "total_out"]].isna().any(axis=1)
    if missing.any():
        examples = out.loc[missing, ["week", "wwtp"]].head(10).to_dict("records")
        raise ValueError(f"Risk rows could not be matched to canonical trip totals: {examples}")
    return out



# -----------------------------------------------------------------------------
# Monthly county-source tables by sewershed
# -----------------------------------------------------------------------------

def build_monthly_county_source_table(
    weekly_edges: pd.DataFrame,
    weekly_totals: pd.DataFrame,
    county_lookup: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Build monthly inbound county-source values for each sewershed.

    The output preserves county-level sources and deliberately does not contain
    `total_in` or `total_out`. It reports the monthly average, minimum, and
    maximum across weekly mean-daily inbound trip values assigned to the month
    of each Monday week date.

    Missing county-WWTP edges in an otherwise available WWTP-week are filled
    with zero before the monthly mean is calculated. This guarantees that the
    county-source values sum to the canonical monthly `total_in`.
    """
    required_edges = {"week", "direction", "wwtp", "county_fips", "trips_mean_daily"}
    missing_edges = required_edges.difference(weekly_edges.columns)
    if missing_edges:
        raise KeyError(f"Weekly edge table is missing columns: {sorted(missing_edges)}")

    required_totals = {"week", "wwtp", "total_in"}
    missing_totals = required_totals.difference(weekly_totals.columns)
    if missing_totals:
        raise KeyError(f"Weekly total table is missing columns: {sorted(missing_totals)}")

    inbound = weekly_edges.loc[
        weekly_edges["direction"].eq("county_to_wwtp"),
        ["week", "wwtp", "county_fips", "trips_mean_daily"],
    ].copy()
    if inbound.empty:
        return pd.DataFrame(
            columns=[
                "wwtp", "county_fips", "county_name", "year", "month",
                "year_month", "trips_to_sewershed_mean_daily",
                "trips_to_sewershed_min_weekly_mean_daily",
                "trips_to_sewershed_max_weekly_mean_daily", "n_weeks",
            ]
        )

    inbound["week"] = _canon_week(inbound["week"])
    inbound["wwtp"] = _canon_wwtp(inbound["wwtp"])
    inbound["county_fips"] = inbound["county_fips"].astype(str).str.zfill(5)
    inbound["trips_mean_daily"] = pd.to_numeric(
        inbound["trips_mean_daily"], errors="coerce"
    ).fillna(0.0)

    # Collapse any accidental duplicate county-WWTP-week records first.
    inbound = (
        inbound.groupby(["week", "wwtp", "county_fips"], as_index=False)["trips_mean_daily"]
        .sum()
    )
    inbound["year_month"] = inbound["week"].dt.to_period("M").astype(str)

    site_weeks = weekly_totals[["week", "wwtp"]].copy()
    site_weeks["week"] = _canon_week(site_weeks["week"])
    site_weeks["wwtp"] = _canon_wwtp(site_weeks["wwtp"])
    site_weeks["year_month"] = site_weeks["week"].dt.to_period("M").astype(str)
    site_weeks = site_weeks.drop_duplicates(["week", "wwtp"])

    # All source counties observed for a sewershed within each month.
    site_month_counties = inbound[["wwtp", "year_month", "county_fips"]].drop_duplicates()

    # Complete week x county grid for every sewershed-month. Missing links are
    # real zero contributions relative to the canonical total for that week.
    grid = site_weeks.merge(
        site_month_counties,
        on=["wwtp", "year_month"],
        how="inner",
        validate="many_to_many",
    )
    grid = grid.merge(
        inbound[["week", "wwtp", "county_fips", "trips_mean_daily"]],
        on=["week", "wwtp", "county_fips"],
        how="left",
        validate="one_to_one",
    )
    grid["trips_mean_daily"] = grid["trips_mean_daily"].fillna(0.0)
    grid["year"] = grid["week"].dt.year
    grid["month"] = grid["week"].dt.month

    monthly = (
        grid.groupby(
            ["wwtp", "county_fips", "year", "month", "year_month"],
            as_index=False,
        )
        .agg(
            trips_to_sewershed_mean_daily=("trips_mean_daily", "mean"),
            trips_to_sewershed_min_weekly_mean_daily=("trips_mean_daily", "min"),
            trips_to_sewershed_max_weekly_mean_daily=("trips_mean_daily", "max"),
            n_weeks=("week", "nunique"),
        )
        .sort_values(["wwtp", "year", "month", "county_fips"])
        .reset_index(drop=True)
    )

    # Remove county-month rows that are zero across the whole month only after
    # calculating the mean, so zero weeks remain part of the denominator.
    monthly = monthly.loc[
        monthly["trips_to_sewershed_mean_daily"].abs() > 0
    ].copy()

    lookup = {str(k).zfill(5): v for k, v in (county_lookup or {}).items()}
    monthly["county_name"] = monthly["county_fips"].map(lookup)

    cols = [
        "wwtp", "county_fips", "county_name", "year", "month", "year_month",
        "trips_to_sewershed_mean_daily",
        "trips_to_sewershed_min_weekly_mean_daily",
        "trips_to_sewershed_max_weekly_mean_daily", "n_weeks",
    ]
    return monthly[cols]


def export_monthly_county_sources_by_sewershed(
    monthly_sources: pd.DataFrame,
    out_dir: str,
) -> Dict[str, object]:
    """
    Export one monthly county-source CSV per sewershed plus a combined table.

    These files contain county-level inbound sources only; they intentionally
    exclude WWTP-level `total_in` and `total_out` columns.
    """
    _ensure_dir(out_dir)
    combined_path = _write_csv(
        monthly_sources,
        os.path.join(out_dir, "sewershed_county_sources_monthly_mean.csv"),
    )

    by_sewershed_dir = os.path.join(out_dir, "by_sewershed")
    _ensure_dir(by_sewershed_dir)
    manifest_rows = []

    for wwtp, sub in monthly_sources.groupby("wwtp", sort=True):
        filename = f"county_sources_{_safe_filename(wwtp)}.csv"
        path = _write_csv(
            sub.reset_index(drop=True),
            os.path.join(by_sewershed_dir, filename),
        )
        manifest_rows.append({"wwtp": wwtp, "file": path})

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = _write_csv(
        manifest,
        os.path.join(out_dir, "sewershed_county_source_file_manifest.csv"),
    )
    return {
        "combined_csv": combined_path,
        "by_sewershed_dir": by_sewershed_dir,
        "manifest_csv": manifest_path,
    }


# -----------------------------------------------------------------------------
# Generic exports retained for compatibility
# -----------------------------------------------------------------------------

def export_county_case_rate(
    county_df: pd.DataFrame,
    out_dir: str,
    *,
    county_key: str = "county_fips",
    case_rate_col: str = "case_rate",
    extra_cols: Optional[Tuple[str, ...]] = None,
    filename: str = "county_case_rate.csv",
    county_geom_path: Optional[str] = None,
    county_geom_key: str = "GEOID",
) -> Dict[str, str]:
    extra_cols = extra_cols or tuple()
    cols = [county_key, case_rate_col, *extra_cols]
    missing = [c for c in [county_key, case_rate_col] if c not in county_df.columns]
    if missing:
        raise KeyError(f"County table is missing required columns: {missing}")
    out = county_df.loc[:, [c for c in cols if c in county_df.columns]].copy()
    out[case_rate_col] = pd.to_numeric(out[case_rate_col], errors="coerce")
    csv_path = _write_csv(out, os.path.join(out_dir, filename))
    outputs = {"csv": csv_path}

    if county_geom_path:
        if gpd is None:
            raise RuntimeError("geopandas is required for shapefile export.")
        geom = gpd.read_file(county_geom_path)
        geom[county_geom_key] = geom[county_geom_key].astype(str)
        out[county_key] = out[county_key].astype(str)
        joined = geom.merge(out, left_on=county_geom_key, right_on=county_key, how="left")
        shp_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".shp")
        joined.to_file(shp_path)
        outputs["shp"] = shp_path
    return outputs


def export_sewershed_trips(
    od_df: pd.DataFrame,
    out_dir: str,
    *,
    from_col: str = "from_id",
    to_col: str = "to_id",
    trips_col: str = "trips",
    time_col: Optional[str] = None,
    sewershed_id_prefix: Optional[str] = None,
    aggregate_per_sewershed: bool = True,
) -> Dict[str, str]:
    required = [from_col, to_col, trips_col]
    missing = [c for c in required if c not in od_df.columns]
    if missing:
        raise KeyError(f"OD table is missing required columns: {missing}")

    out_cols = ([time_col] if time_col and time_col in od_df.columns else []) + required
    edges = od_df[out_cols].copy()
    edges[trips_col] = pd.to_numeric(edges[trips_col], errors="coerce").fillna(0.0)
    outputs = {
        "edges_csv": _write_csv(edges, os.path.join(out_dir, "trips_od_edges.csv"))
    }
    if not aggregate_per_sewershed:
        return outputs

    tmp = edges.copy()
    tmp[from_col] = tmp[from_col].astype(str)
    tmp[to_col] = tmp[to_col].astype(str)
    time_groups = [time_col] if time_col and time_col in tmp.columns else []

    if sewershed_id_prefix:
        to_mask = tmp[to_col].str.startswith(sewershed_id_prefix)
        from_mask = tmp[from_col].str.startswith(sewershed_id_prefix)
        inbound = (
            tmp.loc[to_mask]
            .groupby(time_groups + [to_col], as_index=False)[trips_col]
            .sum()
            .rename(columns={to_col: "sewershed_id", trips_col: "trips_in"})
        )
        outbound = (
            tmp.loc[from_mask]
            .groupby(time_groups + [from_col], as_index=False)[trips_col]
            .sum()
            .rename(columns={from_col: "sewershed_id", trips_col: "trips_out"})
        )
        agg = inbound.merge(outbound, on=time_groups + ["sewershed_id"], how="outer")
        agg[["trips_in", "trips_out"]] = agg[["trips_in", "trips_out"]].fillna(0.0)
    else:
        agg = (
            tmp.groupby(time_groups + [to_col], as_index=False)[trips_col]
            .sum()
            .rename(columns={to_col: "dest_id", trips_col: "trips_to_dest"})
        )

    outputs["agg_csv"] = _write_csv(agg, os.path.join(out_dir, "trips_by_sewershed.csv"))
    return outputs


def export_import_export_risk(
    risk_df: pd.DataFrame,
    out_dir: str,
    *,
    sewershed_key: str = "sewershed_id",
    import_col: str = "import_risk",
    export_col: str = "export_risk",
    total_col: Optional[str] = "ie_risk",
    extra_cols: Optional[Tuple[str, ...]] = None,
    filename: str = "sewershed_import_export_risk.csv",
    sewershed_geom_path: Optional[str] = None,
    sewershed_geom_key: str = "sewershed_id",
) -> Dict[str, str]:
    extra_cols = extra_cols or tuple()
    cols = [sewershed_key, import_col, export_col]
    if total_col:
        cols.append(total_col)
    cols.extend(extra_cols)
    out = risk_df.loc[:, [c for c in cols if c in risk_df.columns]].copy()
    csv_path = _write_csv(out, os.path.join(out_dir, filename))
    outputs = {"csv": csv_path}

    if sewershed_geom_path:
        if gpd is None:
            raise RuntimeError("geopandas is required for shapefile export.")
        geom = gpd.read_file(sewershed_geom_path)
        geom[sewershed_geom_key] = geom[sewershed_geom_key].astype(str)
        out[sewershed_key] = out[sewershed_key].astype(str)
        joined = geom.merge(out, left_on=sewershed_geom_key, right_on=sewershed_key, how="left")
        shp_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".shp")
        joined.to_file(shp_path)
        outputs["shp"] = shp_path
    return outputs


def export_monthly_arcgis_tables_mean_only(
    weekly_summary_csv: str,
    output_dir: str,
) -> str:
    """Compatibility wrapper: weekly summary -> monthly mean table."""
    df = pd.read_csv(weekly_summary_csv, parse_dates=["week"])
    monthly = build_monthly_mean_table(df)
    return _write_csv(monthly, os.path.join(output_dir, "wwtp_monthly_mean.csv"))


# -----------------------------------------------------------------------------
# Clinical-only pipeline export orchestrator
# -----------------------------------------------------------------------------

def export_clinical_arcgis_outputs(
    *,
    weekly_results: dict,
    county_rates: Dict[str, pd.DataFrame],
    disease_import_dfs: Dict[str, pd.DataFrame],
    disease_export_dfs: Dict[str, pd.DataFrame],
    weekly_summary_csv: str,
    county_boundary_fp: Optional[str] = None,
    out_dir: str = "outputs/arcgis_exports",
    start_week=None,
    end_week=None,
) -> Dict[str, object]:
    """
    Export all clinical-only ArcGIS CSVs from one canonical trip source.

    Important consistency guarantee
    -------------------------------
    - `wwtp_trips_weekly.csv`
    - `import_risk_weekly_<disease>.csv`
    - `export_risk_weekly_<disease>.csv`
    - `wwtp_weekly_risk_components_2024.csv`
    - `monthly/wwtp_monthly_mean.csv`
    - `monthly/wwtp_monthly_traffic_stats.csv`

    all use the same canonical weekly `total_in` and `total_out` values.
    """
    _ensure_dir(out_dir)
    monthly_dir = os.path.join(out_dir, "monthly")
    _ensure_dir(monthly_dir)

    outputs: Dict[str, object] = {}

    weekly_totals, weekly_edges = build_weekly_trip_tables(
        weekly_results=weekly_results,
        start_week=start_week,
        end_week=end_week,
    )

    outputs["weekly_trips"] = _write_csv(
        weekly_totals,
        os.path.join(out_dir, "wwtp_trips_weekly.csv"),
    )
    outputs["weekly_edges"] = _write_csv(
        weekly_edges,
        os.path.join(out_dir, "trips_county_wwtp_weekly.csv"),
    )

    monthly_trips = build_monthly_mean_table(
        weekly_totals.drop(columns=["outflow_source"], errors="ignore")
    )
    outputs["monthly_trips"] = _write_csv(
        monthly_trips,
        os.path.join(monthly_dir, "wwtp_trips_monthly_mean.csv"),
    )

    # Explicit traffic statistics requested for interpretation. The mean/min/max
    # are calculated across weekly mean-daily trip values within each month.
    monthly_trip_stats = build_monthly_trip_stats(weekly_totals)
    outputs["monthly_trip_stats"] = _write_csv(
        monthly_trip_stats,
        os.path.join(monthly_dir, "wwtp_monthly_traffic_stats.csv"),
    )

    # Monthly county sources for each sewershed. These preserve the county
    # contribution rows and do not contain total_in or total_out.
    county_lookup = {}
    if county_boundary_fp:
        if gpd is None:
            raise RuntimeError("geopandas is required to read county names from the boundary file.")
        county_geom = gpd.read_file(county_boundary_fp)
        if {"US_FIPS", "LABEL"}.issubset(county_geom.columns):
            county_geom["US_FIPS"] = county_geom["US_FIPS"].astype(str).str.zfill(5)
            county_lookup = (
                county_geom[["US_FIPS", "LABEL"]]
                .drop_duplicates("US_FIPS")
                .set_index("US_FIPS")["LABEL"]
                .to_dict()
            )

    monthly_county_sources = build_monthly_county_source_table(
        weekly_edges=weekly_edges,
        weekly_totals=weekly_totals,
        county_lookup=county_lookup,
    )
    source_dir = os.path.join(monthly_dir, "county_sources")
    outputs["monthly_county_sources"] = export_monthly_county_sources_by_sewershed(
        monthly_county_sources,
        source_dir,
    )

    # QA: county-source monthly means summed across counties must reproduce
    # the canonical monthly total_in for the same sewershed-month.
    source_sum = (
        monthly_county_sources.groupby(["wwtp", "year_month"], as_index=False)
        ["trips_to_sewershed_mean_daily"]
        .sum()
        .rename(columns={
            "trips_to_sewershed_mean_daily": "county_source_sum_mean_daily"
        })
    )
    source_check = monthly_trips[["wwtp", "year_month", "total_in"]].merge(
        source_sum,
        on=["wwtp", "year_month"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    source_check["diff_total_in"] = (
        source_check["total_in"] - source_check["county_source_sum_mean_daily"]
    )
    outputs["county_source_total_in_check"] = _write_csv(
        source_check,
        os.path.join(out_dir, "county_source_vs_total_in_check.csv"),
    )
    max_source_diff = source_check["diff_total_in"].abs().max(skipna=True)
    print(f"[QA] county-source maximum |total_in difference| = {max_source_diff}")
    if (source_check["_merge"] != "both").any() or (
        pd.notna(max_source_diff) and max_source_diff > 1e-6
    ):
        raise ValueError(
            "County-source monthly values do not reproduce monthly total_in. "
            "See county_source_vs_total_in_check.csv."
        )

    # County clinical-rate exports.
    county_output_paths = {}
    for disease, cdf in county_rates.items():
        if not isinstance(cdf, pd.DataFrame) or cdf.empty:
            continue
        out = cdf.copy()
        if "week" in out.columns:
            out["week"] = _canon_week(out["week"])
        path = _write_csv(
            out,
            os.path.join(out_dir, f"county_clinical_rate_{disease}.csv"),
        )
        county_output_paths[disease] = path
    outputs["county_rates"] = county_output_paths

    # Risk exports with canonical trip columns attached.
    import_paths = {}
    export_paths = {}
    corrected_import = {}
    corrected_export = {}

    for disease, imp_df in disease_import_dfs.items():
        if not isinstance(imp_df, pd.DataFrame) or imp_df.empty:
            continue
        out = _attach_canonical_trips(imp_df, weekly_totals)
        corrected_import[disease] = out
        import_paths[disease] = _write_csv(
            out,
            os.path.join(out_dir, f"import_risk_weekly_{disease}.csv"),
        )
        _write_csv(
            build_monthly_mean_table(out.drop(columns=["outflow_source"], errors="ignore")),
            os.path.join(monthly_dir, f"import_risk_monthly_{disease}.csv"),
        )

    for disease, exp_df in disease_export_dfs.items():
        if not isinstance(exp_df, pd.DataFrame) or exp_df.empty:
            continue
        out = _attach_canonical_trips(exp_df, weekly_totals)
        corrected_export[disease] = out
        export_paths[disease] = _write_csv(
            out,
            os.path.join(out_dir, f"export_risk_weekly_{disease}.csv"),
        )
        _write_csv(
            build_monthly_mean_table(out.drop(columns=["outflow_source"], errors="ignore")),
            os.path.join(monthly_dir, f"export_risk_monthly_{disease}.csv"),
        )

    outputs["import_risk"] = import_paths
    outputs["export_risk"] = export_paths

    # Correct the main weekly summary with the same canonical trip source.
    summary = pd.read_csv(weekly_summary_csv, parse_dates=["week"])
    summary["wwtp"] = _canon_wwtp(summary["wwtp"])
    summary["week"] = _canon_week(summary["week"])
    summary = summary.drop(columns=["total_in", "total_out", "outflow_source"], errors="ignore")
    summary = summary.merge(
        weekly_totals,
        on=["week", "wwtp"],
        how="left",
        validate="many_to_one",
    )

    if summary[["total_in", "total_out"]].isna().any().any():
        raise ValueError("The weekly summary contains WWTP-week rows without canonical trips.")

    weekly_components_path = os.path.join(out_dir, "wwtp_weekly_risk_components_2024.csv")
    outputs["weekly_components"] = _write_csv(summary, weekly_components_path)

    monthly = build_monthly_mean_table(
        summary.drop(columns=["outflow_source"], errors="ignore")
    )

    # Keep the existing monthly means for all components, while appending
    # interpretable trip minimum/maximum fields and the number of weekly records.
    # `total_in` and `total_out` remain the monthly averages for backward
    # compatibility; explicit mean aliases are added for clarity.
    trip_stats_for_merge = monthly_trip_stats[[
        "wwtp", "year", "month", "year_month",
        "total_in_mean", "total_in_min", "total_in_max",
        "total_out_mean", "total_out_min", "total_out_max", "n_weeks",
    ]].copy()
    monthly = monthly.merge(
        trip_stats_for_merge,
        on=["wwtp", "year", "month", "year_month"],
        how="left",
        validate="one_to_one",
    )
    outputs["monthly_all_components"] = _write_csv(
        monthly,
        os.path.join(monthly_dir, "wwtp_monthly_mean.csv"),
    )

    # QA table: the monthly trip fields in both monthly files must be identical.
    monthly_check = monthly[["wwtp", "year_month", "total_in", "total_out"]].merge(
        monthly_trips[["wwtp", "year_month", "total_in", "total_out"]],
        on=["wwtp", "year_month"],
        how="outer",
        suffixes=("_all_components", "_trip_file"),
        indicator=True,
    )
    monthly_check["diff_total_in"] = (
        monthly_check["total_in_all_components"] - monthly_check["total_in_trip_file"]
    )
    monthly_check["diff_total_out"] = (
        monthly_check["total_out_all_components"] - monthly_check["total_out_trip_file"]
    )
    outputs["monthly_trip_consistency"] = _write_csv(
        monthly_check,
        os.path.join(out_dir, "monthly_trip_consistency_check.csv"),
    )

    max_in = monthly_check["diff_total_in"].abs().max(skipna=True)
    max_out = monthly_check["diff_total_out"].abs().max(skipna=True)
    print(f"[QA] monthly maximum |total_in difference|  = {max_in}")
    print(f"[QA] monthly maximum |total_out difference| = {max_out}")

    # Each disease-specific monthly risk table must carry the same trip values
    # as the monthly all-component table.
    risk_monthly_checks = {}
    for disease, exp_df in corrected_export.items():
        exp_monthly = build_monthly_mean_table(
            exp_df.drop(columns=["outflow_source"], errors="ignore")
        )
        check = monthly[["wwtp", "year_month", "total_in", "total_out"]].merge(
            exp_monthly[["wwtp", "year_month", "total_in", "total_out"]],
            on=["wwtp", "year_month"],
            how="outer",
            suffixes=("_monthly_summary", f"_export_{disease}"),
            indicator=True,
        )
        check["diff_total_in"] = (
            check["total_in_monthly_summary"] - check[f"total_in_export_{disease}"]
        )
        check["diff_total_out"] = (
            check["total_out_monthly_summary"] - check[f"total_out_export_{disease}"]
        )
        check_path = _write_csv(
            check,
            os.path.join(out_dir, f"monthly_trip_consistency_check_{disease}.csv"),
        )
        risk_monthly_checks[disease] = check_path
        disease_max_in = check["diff_total_in"].abs().max(skipna=True)
        disease_max_out = check["diff_total_out"].abs().max(skipna=True)
        print(
            f"[QA] {disease} monthly max trip differences: "
            f"in={disease_max_in}, out={disease_max_out}"
        )
        if (check["_merge"] != "both").any() or (
            pd.notna(disease_max_in) and disease_max_in > 1e-6
        ) or (
            pd.notna(disease_max_out) and disease_max_out > 1e-6
        ):
            raise ValueError(
                f"Monthly trip consistency failed for {disease}. See {check_path}."
            )
    outputs["monthly_risk_trip_consistency"] = risk_monthly_checks

    return outputs
