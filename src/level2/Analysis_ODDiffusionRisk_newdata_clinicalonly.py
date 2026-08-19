"""Create mobility-weighted clinical-risk inputs for Level 1 and Level 2.

The reported case study uses weekly hospitalization rates and authorized OD
flows. Wastewater viral-load preprocessing remains optional.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import GeoJson, GeoJsonTooltip, FeatureGroup, LayerControl
import branca.colormap as cm
import matplotlib.pyplot as plt
import re

from AnalysisViz_ODFlowPure import compute_weekly_flows
from Process_ViralLoader import preprocess_viral_load

from Process_ViralLoader_V2 import (
    load_county_clinical_fixed,
    load_wwtp_clinical_from_metrics_fixed,
)

from Analysis_ODDiffusionRisk_newdata_FileGen import (
    export_clinical_arcgis_outputs,
)

# FileGen outputs include:
#   - weekly and monthly import/export risk tables with canonical trip fields
#   - monthly WWTP means plus monthly trip mean/min/max and n_weeks
#   - one county-source monthly CSV per sewershed, retaining county rows and
#     reporting county-source mean/min/max across weekly mean-daily values
FLOW_MAX = 1500000


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _minmax_norm(series, vmin=None, vmax=None):
    s = pd.to_numeric(series, errors="coerce")
    if vmin is None:
        vmin = np.nanquantile(s, 0.01)
    if vmax is None:
        vmax = np.nanquantile(s, 0.99)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return pd.Series(np.nan, index=s.index), 0.0, 1.0
    out = (s - vmin) / (vmax - vmin)
    return out.clip(0, 1), float(vmin), float(vmax)



def ensure_total_out_column(weekly_results, prefer_true_out=True):
    """
    Robustly ensure each weekly_results[week] has numeric:
      - total_in: from county_to_wwtp sum(axis=1)
      - total_out:
          * from wwtp_to_county if present and nonzero (prefer_true_out)
          * else fallback to symmetric assumption total_out = total_in
    """
    import pandas as pd

    for week, df in weekly_results.items():
        if df is None or len(df) == 0:
            continue
        if "wwtp" not in df.columns:
            # if upstream changed and df has wwtp as index
            if df.index.name and str(df.index.name).lower() == "wwtp":
                df = df.reset_index()
                weekly_results[week] = df
            else:
                # cannot fix without a WWTP key
                continue

        # canonical WWTP key
        wwtp_series = df["wwtp"].astype(str).str.strip().str.lower()

        # Total inbound flow from the county-to-WWTP matrix
        mat_in = getattr(df, "attrs", {}).get("county_to_wwtp", None)
        total_in = None
        if isinstance(mat_in, pd.DataFrame) and not mat_in.empty:
            mat_in2 = mat_in.copy()
            mat_in2.index = mat_in2.index.astype(str).str.strip().str.lower()
            total_in = mat_in2.sum(axis=1)

        # write total_in
        if total_in is not None:
            df["total_in"] = wwtp_series.map(total_in.to_dict()).fillna(0.0).astype(float)
        else:
            if "total_in" not in df.columns:
                df["total_in"] = 0.0

        # try true outflow if available
        total_out = None
        if prefer_true_out:
            mat_out = getattr(df, "attrs", {}).get("wwtp_to_county", None)
            if isinstance(mat_out, pd.DataFrame) and not mat_out.empty:
                mat_out2 = mat_out.copy()
                mat_out2.index = mat_out2.index.astype(str).str.strip().str.lower()
                total_out = mat_out2.sum(axis=1)

        # fallback or guard against all-zero "true out"
        if total_out is None or float(pd.to_numeric(pd.Series(total_out), errors="coerce").fillna(0).sum()) == 0.0:
            if total_in is not None:
                total_out = total_in
            else:
                total_out = pd.Series(0.0, index=wwtp_series.unique())

        # write total_out
        df["total_out"] = wwtp_series.map(total_out.to_dict()).fillna(0.0).astype(float)

        weekly_results[week] = df

    return weekly_results




def compute_import_export_multi(
    weekly_results,
    disease_datasets,
    viral_datasets,
    output_folder, prefix,
    wwtp_shapefile,
    county_boundary_fp,
    start_week=None,
    end_week=None,
    inflow_min_trips: float = 0.0,
    outflow_min_trips: float = 0.0,
    import_risk_min: float = None,
    export_risk_min: float = None,
    county_clinical_rates: dict = None,
    wwtp_clinical_rates: dict = None,
):
    os.makedirs(output_folder, exist_ok=True)

    # Load geometries
    wwtp_gdf = gpd.read_file(wwtp_shapefile).to_crs(epsg=4326)
    wwtp_gdf["wwtp"] = wwtp_gdf["wwtp"].astype(str).str.strip().str.lower()
    wwtp_gdf["centroid"] = wwtp_gdf.geometry.centroid
    wwtp_gdf["centroid_x"] = wwtp_gdf["centroid"].x
    wwtp_gdf["centroid_y"] = wwtp_gdf["centroid"].y
    wwtp_gdf = wwtp_gdf[~wwtp_gdf["wwtp"].str.contains("historic", case=False)]

    county_gdf = gpd.read_file(county_boundary_fp).to_crs(epsg=4326)
    county_gdf["US_FIPS"] = county_gdf["US_FIPS"].astype(str).str.zfill(5)
    county_gdf["_label_lower"] = county_gdf["LABEL"].astype(str).str.strip().str.lower()

    if start_week:
        start_week = pd.to_datetime(start_week)
    if end_week:
        end_week = pd.to_datetime(end_week)

    disease_import_dfs = {disease: [] for disease in viral_datasets.keys()}
    disease_export_dfs = {disease: [] for disease in viral_datasets.keys()}

    # Pooled global clinical normalization range
    # ------------------------------------------------------------
    # For direct comparison across import/export pathways AND across
    # COVID, Flu, and RSV, all hospitalization-rate values are pooled
    # across diseases, spatial units (county + WWTP/sewershed), and weeks.
    # The same 1st--99th percentile range is then used to normalize all
    # clinical burden terms before applying the bounded modifier:
    #     risk = normalized_flow * (1 + clinical_norm)
    # This makes the clinical modifier comparable across diseases and
    # between county-side import and WWTP-side export calculations.
    clinical_norm_pieces = []

    if county_clinical_rates:
        for disease, cdf in county_clinical_rates.items():
            if isinstance(cdf, pd.DataFrame) and "clinical_rate" in cdf.columns:
                clinical_norm_pieces.append(pd.to_numeric(cdf["clinical_rate"], errors="coerce"))

    if wwtp_clinical_rates:
        for disease, wdf in wwtp_clinical_rates.items():
            if isinstance(wdf, pd.DataFrame) and "clinical_rate" in wdf.columns:
                clinical_norm_pieces.append(pd.to_numeric(wdf["clinical_rate"], errors="coerce"))

    if clinical_norm_pieces:
        pooled_clinical_rates = pd.concat(clinical_norm_pieces, ignore_index=True).dropna()
        pooled_vmin = np.nanquantile(pooled_clinical_rates, 0.01) if len(pooled_clinical_rates) else np.nan
        pooled_vmax = np.nanquantile(pooled_clinical_rates, 0.99) if len(pooled_clinical_rates) else np.nan
    else:
        pooled_vmin, pooled_vmax = np.nan, np.nan

    if not np.isfinite(pooled_vmin) or not np.isfinite(pooled_vmax) or pooled_vmin == pooled_vmax:
        pooled_vmin, pooled_vmax = 0.0, 1.0

    print(
        f"[clinical norm] pooled all diseases + county/WWTP 1st--99th percentile range = "
        f"{pooled_vmin:.6g}, {pooled_vmax:.6g}"
    )

    for disease in viral_datasets.keys():
        print(f"Building map for {disease}...")

        weeks = sorted(weekly_results.keys())
        if start_week:
            weeks = [w for w in weeks if pd.to_datetime(w) >= start_week]
        if end_week:
            weeks = [w for w in weeks if pd.to_datetime(w) <= end_week]

        county_rate_df = county_clinical_rates.get(disease) if county_clinical_rates else None
        wwtp_rate_df = wwtp_clinical_rates.get(disease) if wwtp_clinical_rates else None
        vmin, vmax = pooled_vmin, pooled_vmax

        for week in weeks:
            m = folium.Map(location=[39.0, -105.5], zoom_start=7, tiles="CartoDB positron")

            mat = weekly_results[week].attrs.get("county_to_wwtp")
            if mat is None:
                continue

            mat = mat.copy()
            mat.columns = mat.columns.astype(str).str.zfill(5)
            mat.index = mat.index.astype(str).str.strip().str.lower()

            week_date = pd.to_datetime(week)
            week_monday = week_date - pd.to_timedelta(week_date.weekday(), unit="D")

            # ---------------- COUNTY clinical layer ----------------
            if county_rate_df is not None:
                wk = county_rate_df[county_rate_df["week"] == week_monday].copy()
                wk["_label_lower"] = wk["County"].astype(str).str.strip().str.lower()
                county_map = county_gdf.merge(
                    wk[["_label_lower", "clinical_rate"]],
                    on="_label_lower", how="left"
                )

                county_map["norm_rate"] = (county_map["clinical_rate"] - vmin) / (vmax - vmin + 1e-9)
                county_map["norm_rate"] = county_map["norm_rate"].clip(0, 1)

                rate_colormap = cm.linear.Greens_09.scale(0, 1)
                fg_county = FeatureGroup(name=f"{disease} – County clinical rate (per 100k)", show=True)
                GeoJson(
                    county_map.drop(columns=["_label_lower"], errors="ignore"),
                    style_function=lambda f: {
                        "fillColor": rate_colormap(f["properties"]["norm_rate"])
                        if not pd.isna(f["properties"]["norm_rate"]) else "lightgrey",
                        "color": "black", "weight": 0.5, "fillOpacity": 0.8
                    },
                    tooltip=GeoJsonTooltip(
                        fields=["LABEL", "clinical_rate"],
                        aliases=["County", "Rate / 100k"]
                    ),
                ).add_to(fg_county)
                m.add_child(fg_county)
                rate_colormap.caption = f"{disease} – County clinical rate"
                m.add_child(rate_colormap)

            # ---------------- WWTP inflow/outflow polygons ----------------
            total_inflow_map = mat.sum(axis=1).to_dict()

            # robust outflow mapping: trust df["total_out"] if nonzero, else fallback to symmetric
            tmp = weekly_results[week].copy()
            tmp["wwtp"] = tmp["wwtp"].astype(str).str.strip().str.lower()
            if "total_out" not in tmp.columns:
                tmp["total_out"] = 0.0

            outflow_map = tmp.set_index("wwtp")["total_out"].to_dict()

            # apply maps
            wwtp_gdf["total_in"] = wwtp_gdf["wwtp"].map(total_inflow_map).fillna(0.0)
            wwtp_gdf["total_out"] = wwtp_gdf["wwtp"].map(outflow_map).fillna(0.0)

            # if outflow is all zero, do symmetric fallback
            if float(pd.to_numeric(wwtp_gdf["total_out"], errors="coerce").fillna(0.0).sum()) == 0.0:
                wwtp_gdf["total_out"] = wwtp_gdf["total_in"].copy()


            # Inflow layer
            wwtp_in_gdf = wwtp_gdf[wwtp_gdf["total_in"] >= float(inflow_min_trips)].copy()
            inflow_colormap = cm.linear.Blues_09.scale(0, FLOW_MAX)
            fg_inflow = FeatureGroup(name=f"{disease} – Trips to sewershed", show=True)
            if not wwtp_in_gdf.empty:
                GeoJson(
                    wwtp_in_gdf.drop(columns=["centroid"], errors="ignore"),
                    style_function=lambda f: {
                        "fillColor": inflow_colormap(f["properties"]["total_in"]),
                        "color": "black", "weight": 0.5, "fillOpacity": 0.9
                    },
                    tooltip=GeoJsonTooltip(
                        fields=["wwtp", "total_in"],
                        aliases=["WWTP", "Trips to WWTP"]
                    ),
                ).add_to(fg_inflow)
            m.add_child(fg_inflow)
            inflow_colormap.caption = f"{disease} – Trips to sewershed"
            m.add_child(inflow_colormap)

            # Outflow layer
            wwtp_out_gdf = wwtp_gdf[wwtp_gdf["total_out"] >= float(outflow_min_trips)].copy()
            outflow_colormap = cm.linear.Blues_09.scale(0, FLOW_MAX)
            fg_outflow = FeatureGroup(name=f"{disease} – Trips from sewershed", show=False)
            if not wwtp_out_gdf.empty:
                GeoJson(
                    wwtp_out_gdf.drop(columns=["centroid"], errors="ignore"),
                    style_function=lambda f: {
                        "fillColor": outflow_colormap(f["properties"]["total_out"]),
                        "color": "black", "weight": 0.5, "fillOpacity": 0.9
                    },
                    tooltip=GeoJsonTooltip(
                        fields=["wwtp", "total_out"],
                        aliases=["WWTP", "Trips from WWTP"]
                    ),
                ).add_to(fg_outflow)
            m.add_child(fg_outflow)

            # Import-risk calculation
            if county_rate_df is not None:
                wk_rates = county_rate_df[county_rate_df["week"] == week_monday].copy()
                wk_rates["_label_lower"] = wk_rates["County"].astype(str).str.strip().str.lower()
                wk_rates = county_gdf.merge(
                    wk_rates[["_label_lower", "clinical_rate"]],
                    on="_label_lower", how="left"
                )[["US_FIPS", "clinical_rate"]]
                wk_rates["US_FIPS"] = wk_rates["US_FIPS"].astype(str).str.zfill(5)
                county_rate_map = wk_rates.set_index("US_FIPS")["clinical_rate"]
                norm_case = ((county_rate_map - vmin) / (vmax - vmin + 1e-9)).clip(0, 1)
                norm_case = norm_case.fillna(0.0)

            else:
                norm_case = pd.Series(index=mat.columns, dtype=float)

            case_mask = norm_case.notna().astype(int).reindex(mat.columns).fillna(0)
            inflow_with_data = (mat * case_mask).sum(axis=1)
            total_inflow = mat.sum(axis=1).replace(0, np.nan)
            coverage_percent = (inflow_with_data / total_inflow).fillna(0) * 100

            flow_flat = mat.stack()
            norm_flow = (flow_flat - 0) / (FLOW_MAX - 0 + 1e-9)
            risk_df = norm_flow.reset_index()
            risk_df.columns = ["wwtp", "CountyFIPS", "norm_flow"]
            risk_df["norm_case"] = risk_df["CountyFIPS"].map(norm_case)
            risk_df["risk_boost"] = (1 + risk_df["norm_case"]) * risk_df["norm_flow"]

            import_df = risk_df.groupby("wwtp").agg({"risk_boost": "sum"}).reset_index()

            # Interpretable components for ArcGIS / diagnostics:
            # total_in_norm = mobility-flow component
            # clinical_rate_inflow_wt_norm = inflow-weighted county clinical component
            # risk_boost = total_in_norm * (1 + clinical_rate_inflow_wt_norm)
            total_in_series = mat.sum(axis=1)
            norm_case_aligned = norm_case.reindex(mat.columns).fillna(0.0)
            clin_num = (mat * norm_case_aligned.values).sum(axis=1)
            clin_den = total_in_series.replace(0, np.nan)
            clinical_rate_inflow_wt_norm = (clin_num / clin_den).fillna(0.0)

            import_df["total_in"] = import_df["wwtp"].map(total_in_series.to_dict()).fillna(0.0)
            import_df["flow_in_norm"] = import_df["total_in"] / (FLOW_MAX + 1e-9)
            import_df["clinical_rate_inflow_wt_norm"] = import_df["wwtp"].map(
                clinical_rate_inflow_wt_norm.to_dict()
            ).fillna(0.0)

            import_df["inflow_coverage_percent"] = import_df["wwtp"].map(coverage_percent)
            import_df["week"] = week
            disease_import_dfs[disease].append(import_df)

            fg_import = FeatureGroup(name=f"{disease} – Import risk (mobility-weighted)", show=True)
            rmin, rmax = 0, 2
            for _, row in import_df.iterrows():
                cov = row["inflow_coverage_percent"]
                rb = row["risk_boost"]
                if (import_risk_min is not None) and (pd.notna(rb)) and (rb < import_risk_min):
                    continue
                match = wwtp_gdf.loc[wwtp_gdf["wwtp"] == row["wwtp"], ["centroid_y", "centroid_x"]]
                if match.empty:
                    continue
                lat, lon = match.iloc[0]
                color = cm.linear.PuRd_09.scale(rmin, rmax)(rb) if cov > 0 else "lightgrey"
                radius = 0 if pd.isna(rb) else 5 + ((rb - rmin) / (rmax - rmin + 1e-6)) * 15
                if radius <= 0:
                    continue
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color="black",
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    weight=0.3,
                    popup=(f"{row['wwtp']}<br>"
                           f"Import risk: {rb:.3f}<br>"
                           f"Coverage (trips with rate data): {cov:.1f}%")
                ).add_to(fg_import)
            m.add_child(fg_import)
            risk_colormap = cm.linear.PuRd_09.scale(rmin, rmax)
            risk_colormap.caption = f"{disease} – Risk (mobility-weighted)"
            m.add_child(risk_colormap)

            # Export-risk calculation using total outbound flow
            if wwtp_rate_df is not None:
                wk_w = wwtp_rate_df[wwtp_rate_df["week"] == week_monday].copy()
                wk_w["wwtp"] = wk_w["wwtp"].astype(str).str.strip().str.lower()
                wwtp_case_map = wk_w.set_index("wwtp")["clinical_rate"]
                norm_wwtp_case = ((wwtp_case_map - pooled_vmin) / (pooled_vmax - pooled_vmin + 1e-9)).clip(0, 1)
                norm_wwtp_case = norm_wwtp_case.fillna(0.0)
            else:
                wwtp_case_map = pd.Series(dtype=float)
                norm_wwtp_case = pd.Series(dtype=float)

            # Use the canonical trip totals already calculated in weekly_results.
            # Do not remap total_out through the shapefile: that can create
            # inconsistencies when site names are duplicated, filtered, or unmatched.
            export_df = weekly_results[week].copy()
            export_df["wwtp"] = export_df["wwtp"].astype(str).str.strip().str.lower()
            export_df["total_in"] = pd.to_numeric(
                export_df.get("total_in", 0.0), errors="coerce"
            ).fillna(0.0)
            export_df["total_out"] = pd.to_numeric(
                export_df.get("total_out", 0.0), errors="coerce"
            ).fillna(0.0)

            export_df["clinical_rate_real"] = export_df["wwtp"].map(wwtp_case_map.to_dict())
            export_df["clinical_rate_norm"] = export_df["wwtp"].map(norm_wwtp_case.to_dict())

            export_df["norm_out"] = (export_df["total_out"] - 0) / (FLOW_MAX - 0 + 1e-9)
            export_df["flow_out_norm"] = export_df["norm_out"]
            export_df["export_risk"] = (1 + export_df["clinical_rate_norm"]) * export_df["norm_out"]

            export_df["week"] = week
            disease_export_dfs[disease].append(export_df)

            fg_export = FeatureGroup(name=f"{disease} – Export risk (clinical×mobility)", show=False)
            rmin, rmax = 0, 2
            for _, row in export_df.iterrows():
                if pd.isna(row["clinical_rate_norm"]):
                    continue
                er = row["export_risk"]
                if (export_risk_min is not None) and (pd.notna(er)) and (er < export_risk_min):
                    continue
                match = wwtp_gdf.loc[wwtp_gdf["wwtp"] == row["wwtp"], ["centroid_y", "centroid_x"]]
                if match.empty:
                    continue
                lat, lon = match.iloc[0]
                color = cm.linear.PuRd_09.scale(rmin, rmax)(er)
                radius = 0 if pd.isna(er) else 5 + ((er - rmin) / (rmax - rmin + 1e-6)) * 15
                if radius <= 0:
                    continue
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color="black",
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    weight=0.3,
                    popup=(f"{row['wwtp']}<br>"
                           f"Export risk: {er:.3f}<br>"
                           f"WWTP clinical rate (raw): {row['clinical_rate_real']:.3f}")
                ).add_to(fg_export)
            m.add_child(fg_export)

            m.add_child(LayerControl(collapsed=False))
            out_fp = os.path.join(output_folder, f"{prefix}risk_summary_{disease}_{week}.html")
            m.save(out_fp)
            print(f"Saved {disease} map to {out_fp}")

    # merge to DataFrames
    for disease in disease_import_dfs:
        rows = []
        for df in disease_import_dfs[disease]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                rows.extend(df.to_dict(orient="records"))
        disease_import_dfs[disease] = pd.DataFrame.from_records(rows)

    for disease in disease_export_dfs:
        rows = []
        for df in disease_export_dfs[disease]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                rows.extend(df.to_dict(orient="records"))
        disease_export_dfs[disease] = pd.DataFrame.from_records(rows)

    return disease_import_dfs, disease_export_dfs


# ------------------------------------------------------------
# Weekly summary (now adds clinical rates to the CSV)
# ------------------------------------------------------------
def save_weekly_risk_summary(
    weekly_results,
    disease_import_dfs,
    disease_export_dfs,
    wwtp_shapefile,
    output_csv,
    start_week=None,
    end_week=None,
    county_boundary_fp=None,
    county_clinical_rates: dict = None,   # {"COVID": county df}
    wwtp_clinical_rates: dict = None,     # {"COVID": wwtp df}
):
    """
    Save per-WWTP weekly summary including:
      - inflow, outflow
      - import risk (COVID, Flu, RSV)
      - export risk (COVID, Flu, RSV)
      - WVAL (real value)
      - pop served (from WWTP shapefile)
      - clinical_rate_wwtp_{Disease}
      - clinical_rate_county_inflow_wt_{Disease}
    """
    # Load WWTP geometry + pop served
    wwtp_gdf = gpd.read_file(wwtp_shapefile).to_crs(epsg=4326)
    wwtp_gdf["wwtp"] = (
        wwtp_gdf["wwtp"].astype(str).str.strip().str.lower()
    )
    wwtp_pop = wwtp_gdf.set_index("wwtp")["pop_served"]

    # For inflow-weighted county rates we need a county LABEL<->FIPS map
    if county_boundary_fp is not None:
        county_gdf = gpd.read_file(county_boundary_fp).to_crs(epsg=4326)
        county_gdf["US_FIPS"] = county_gdf["US_FIPS"].astype(str).str.zfill(5)
        county_gdf["_label_lower"] = (
            county_gdf["LABEL"].astype(str).str.strip().str.lower()
        )
        label_to_fips = county_gdf.set_index("_label_lower")["US_FIPS"].to_dict()
    else:
        county_gdf = None
        label_to_fips = {}

    if start_week:
        start_week = pd.to_datetime(start_week)
    if end_week:
        end_week = pd.to_datetime(end_week)

    all_rows = []

    for week in sorted(weekly_results.keys()):
        week_dt = pd.to_datetime(week)
        if start_week and week_dt < start_week:
            continue
        if end_week and week_dt > end_week:
            continue

        base_df = weekly_results[week].copy()
        base_df["wwtp"] = (
            base_df["wwtp"].astype(str).str.strip().str.lower()
        )

        # -----------------------------
        # Base info per WWTP-week
        # -----------------------------
        summary = base_df[["wwtp", "total_in", "total_out"]].copy()
        summary["week"] = week
        summary["pop_served"] = summary["wwtp"].map(wwtp_pop)

        # -----------------------------
        # Import risks
        # -----------------------------
        for disease, imp_df in disease_import_dfs.items():
            if not isinstance(imp_df, pd.DataFrame) or imp_df.empty:
                continue
            imp_cols = [
                "wwtp", "risk_boost", "flow_in_norm",
                "clinical_rate_inflow_wt_norm", "inflow_coverage_percent"
            ]
            imp_cols = [c for c in imp_cols if c in imp_df.columns]
            imp_week = (
                imp_df[pd.to_datetime(imp_df["week"]) == week_dt][imp_cols]
                .copy()
                .rename(columns={
                    "risk_boost": f"import_risk_{disease}",
                    "flow_in_norm": f"flow_in_norm_{disease}",
                    "clinical_rate_inflow_wt_norm": f"clinical_rate_inflow_wt_norm_{disease}",
                    "inflow_coverage_percent": f"inflow_coverage_percent_{disease}",
                })
            )
            imp_week["wwtp"] = (
                imp_week["wwtp"].astype(str).str.strip().str.lower()
            )
            summary = summary.merge(imp_week, on="wwtp", how="left")

        # -----------------------------
        # Export risks + WVAL (real)
        # -----------------------------
        for disease, exp_df in disease_export_dfs.items():
            if not isinstance(exp_df, pd.DataFrame) or exp_df.empty:
                continue
            exp_cols = [
                "wwtp", "export_risk", "clinical_rate_real",
                "clinical_rate_norm", "flow_out_norm"
            ]
            exp_cols = [c for c in exp_cols if c in exp_df.columns]
            exp_week = (
                exp_df[pd.to_datetime(exp_df["week"]) == week_dt][exp_cols]
                .copy()
                .rename(
                    columns={
                        "export_risk": f"export_risk_{disease}",
                        "clinical_rate_real": f"export_signal_clinical_rate_{disease}",
                        "clinical_rate_norm": f"clinical_rate_wwtp_norm_{disease}",
                        "flow_out_norm": f"flow_out_norm_{disease}",
                    }
                )
            )
            exp_week["wwtp"] = (
                exp_week["wwtp"].astype(str).str.strip().str.lower()
            )
            summary = summary.merge(exp_week, on="wwtp", how="left")

        # -----------------------------
        # WWTP clinical rate per disease
        # -----------------------------
        if wwtp_clinical_rates:
            for disease, wdf in wwtp_clinical_rates.items():
                if not isinstance(wdf, pd.DataFrame) or wdf.empty:
                    continue
                wk = wdf[wdf["week"] == pd.to_datetime(week)].copy()
                if wk.empty:
                    continue
                wk["wwtp"] = (
                    wk["wwtp"].astype(str).str.strip().str.lower()
                )
                wk = wk[["wwtp", "clinical_rate"]].rename(
                    columns={"clinical_rate": f"clinical_rate_wwtp_{disease}"}
                )
                summary = summary.merge(wk, on="wwtp", how="left")

        # -----------------------------
        # Inflow-weighted county clinical rate per disease
        # -----------------------------
        mat = weekly_results[week].attrs.get("county_to_wwtp")
        if (county_clinical_rates is not None) and (county_gdf is not None) and (mat is not None):
            mat = mat.copy()
            mat.columns = mat.columns.astype(str).str.zfill(5)
            mat.index = mat.index.astype(str).str.strip().str.lower()

            for disease, cdf in county_clinical_rates.items():
                if not isinstance(cdf, pd.DataFrame) or cdf.empty:
                    continue

                wk = cdf[cdf["week"] == pd.to_datetime(week)].copy()
                if wk.empty:
                    continue

                wk["_label_lower"] = (
                    wk["County"].astype(str).str.strip().str.lower()
                )
                wk["FIPS"] = wk["_label_lower"].map(label_to_fips)
                wk = wk.dropna(subset=["FIPS"])
                wk["FIPS"] = wk["FIPS"].astype(str).str.zfill(5)

                # Series: index = FIPS, value = clinical_rate
                rate_by_fips = wk.set_index("FIPS")["clinical_rate"]

                # align to matrix columns
                aligned = rate_by_fips.reindex(mat.columns)

                # inflow-weighted average per WWTP
                inflow = mat.sum(axis=1).replace(0, np.nan)
                numer = (mat * aligned.values).sum(axis=1)
                inflow_wt_rate = (numer / inflow)
                inflow_wt_rate.name = (
                    f"clinical_rate_county_inflow_wt_{disease}"
                )

                # Normalize the WWTP key and assign the expected index name.
                inflow_wt_rate.index = (
                    inflow_wt_rate.index.astype(str)
                    .str.strip()
                    .str.lower()
                )
                inflow_wt_rate.index.name = "wwtp"

                # Turn the Series into a 2-col DataFrame with guaranteed ['wwtp', metric_name]
                tmp_merge = inflow_wt_rate.reset_index(name=inflow_wt_rate.name)

                # Ensure summary has canonical 'wwtp' key
                if "wwtp" not in summary.columns:
                    if summary.index.name == "wwtp":
                        summary = summary.reset_index()
                    else:
                        raise KeyError("Expected 'wwtp' column in summary before merge.")
                summary["wwtp"] = (
                    summary["wwtp"].astype(str).str.strip().str.lower()
                )

                summary = summary.merge(tmp_merge, on="wwtp", how="left")

        all_rows.append(summary)

    # -----------------------------
    # Concatenate all weeks and save
    # -----------------------------
    final_df = pd.concat(all_rows, ignore_index=True)

    # File naming with time window
    start_str = start_week.strftime("%Y%m%d") if start_week is not None else "full"
    end_str = end_week.strftime("%Y%m%d") if end_week is not None else "full"
    base, ext = os.path.splitext(output_csv)
    output_csv_with_weeks = f"{base}_{start_str}_{end_str}{ext}"

    os.makedirs(os.path.dirname(output_csv_with_weeks), exist_ok=True)
    final_df.to_csv(output_csv_with_weeks, index=False)
    print(f" Saved weekly summary to {output_csv_with_weeks}")
    return output_csv_with_weeks


def plot_yoy_monthly_changes(
    summary_csv,
    output_dir,
    group_field="wwtp",
    disease="COVID",
    start_month=1,
    end_month=6,
    min_years=2,
    sharey=True
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(summary_csv, parse_dates=["week"])
    if "week" not in df.columns or group_field not in df.columns:
        print("Required columns missing in summary CSV.")
        return

    imp_col = f"import_risk_{disease}"
    exp_col = f"export_risk_{disease}"
    missing_cols = [c for c in [imp_col, exp_col] if c not in df.columns]
    if missing_cols:
        print(f"Columns not found: {missing_cols}")
        return

    use = df[[group_field, "week", imp_col, exp_col]].copy()
    use["year"] = use["week"].dt.year
    use["month"] = use["week"].dt.month
    use = use[use["month"].between(start_month, end_month)]

    monthly = (use.groupby([group_field, "year", "month"], as_index=False)[[imp_col, exp_col]]
               .mean()
               .sort_values([group_field, "year", "month"]))

    for g, sub in monthly.groupby(group_field):
        has_imp = sub[imp_col].notna().any()
        has_exp = sub[exp_col].notna().any()
        if not (has_imp or has_exp):
            continue

        piv_imp = sub.pivot(index="month", columns="year", values=imp_col).sort_index() if has_imp else None
        piv_exp = sub.pivot(index="month", columns="year", values=exp_col).sort_index() if has_exp else None

        def _valid_cols(p):
            return [y for y in (p.columns if p is not None else []) if p[y].notna().any()]

        years_imp = _valid_cols(piv_imp)
        years_exp = _valid_cols(piv_exp)
        years_all = sorted(set(years_imp) | set(years_exp))
        if len(years_all) < min_years:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=sharey)
        axes = list(axes)

        ax = axes[0]
        if piv_imp is not None and years_imp:
            for y in years_imp:
                ax.plot(range(start_month, end_month + 1),
                        piv_imp.reindex(range(start_month, end_month + 1))[y],
                        marker="o", label=str(y))
        ax.set_title(f"Import risk — {disease}")
        ax.set_xlabel("Month")
        ax.set_xticks(list(range(start_month, end_month + 1)))
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Risk (normalized units)")

        ax = axes[1]
        if piv_exp is not None and years_exp:
            for y in years_exp:
                ax.plot(range(start_month, end_month + 1),
                        piv_exp.reindex(range(start_month, end_month + 1))[y],
                        marker="o", label=str(y))
        ax.set_title(f"Export risk — {disease}")
        ax.set_xlabel("Month")
        ax.set_xticks(list(range(start_month, end_month + 1)))
        ax.grid(True, alpha=0.3)
        ax.legend(title="Year", ncol=2, fontsize=9)

        fig.suptitle(f"YOY monthly risk (Jan–Jun) — {group_field}: {g}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_fp = os.path.join(output_dir, f"yoy_risk_{disease}_{group_field}_{_safe_filename(g)}.png")
        plt.savefig(out_fp, dpi=150)
        plt.close()
        print("Saved:", out_fp)



def run_OD_diffusion():
    weekly_od_dir = "../ZoneSelection/Outfile/ODData/Weekly/"
    wwtp_shapefile = "../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp"
    county_boundary_fp = "../ZoneSelection/Input/Census/COCounty.shp"
    county_pop_csv = "../ZoneSelection/Input/Census/CO_County_Population_FIPS5.csv"

    viral_fp_covid = "../ZoneSelection/Input/Viral/NWSS WVAL.xlsx"
    viral_fp_flua = "../ZoneSelection/Input/Viral/NWSS FluA WVAL.xlsx"
    viral_fp_rsv = "../ZoneSelection/Input/Viral/NWSS RSV WVAL.xlsx"

    start_week = "2024-01-01"
    end_week = "2024-12-30"

    diffusion_output_dir = "outputs/risk_maps"
    summary_output_dir = "outputs"
    arcgis_output_dir = "outputs/arcgis_exports"
    os.makedirs(diffusion_output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(arcgis_output_dir, exist_ok=True)

    # Weekly ODData/Weekly is produced by Process_ODData_Aggr.py from the
    # fixed daily county-WWTP outputs.
    weekly_results, fips_lookup = compute_weekly_flows(weekly_od_dir)
    weekly_results = ensure_total_out_column(weekly_results, prefer_true_out=True)

    viral_df_covid = preprocess_viral_load(viral_fp_covid)
    viral_df_flua = preprocess_viral_load(viral_fp_flua)
    viral_df_rsv = preprocess_viral_load(viral_fp_rsv)

    county_rates = {
        "COVID": load_county_clinical_fixed(
            disease="COVID",
            county_shp=county_boundary_fp,
            county_pop_csv=county_pop_csv,
            metric="hospitalization",
            value_kind="rate",
        ),
        "Flu": load_county_clinical_fixed(
            disease="Influenza",
            county_shp=county_boundary_fp,
            county_pop_csv=county_pop_csv,
            metric="hospitalization",
            value_kind="rate",
        ),
        "RSV": load_county_clinical_fixed(
            disease="RSV",
            county_shp=county_boundary_fp,
            county_pop_csv=county_pop_csv,
            metric="hospitalization",
            value_kind="rate",
        ),
    }

    wwtp_rates = {
        "COVID": load_wwtp_clinical_from_metrics_fixed(
            disease="COVID",
            wwtp_shp=wwtp_shapefile,
            metric="hospitalization",
            value_kind="rate",
        ),
        "Flu": load_wwtp_clinical_from_metrics_fixed(
            disease="Influenza",
            wwtp_shp=wwtp_shapefile,
            metric="hospitalization",
            value_kind="rate",
        ),
        "RSV": load_wwtp_clinical_from_metrics_fixed(
            disease="RSV",
            wwtp_shp=wwtp_shapefile,
            metric="hospitalization",
            value_kind="rate",
        ),
    }

    disease_import_dfs, disease_export_dfs = compute_import_export_multi(
        weekly_results=weekly_results,
        disease_datasets={},
        viral_datasets={"COVID": viral_df_covid, "Flu": viral_df_flua, "RSV": viral_df_rsv},
        output_folder=diffusion_output_dir,
        prefix="hosp_",
        wwtp_shapefile=wwtp_shapefile,
        county_boundary_fp=county_boundary_fp,
        start_week=start_week,
        end_week=end_week,
        inflow_min_trips=50,
        outflow_min_trips=50,
        import_risk_min=0.00,
        export_risk_min=0.00,
        county_clinical_rates=county_rates,
        wwtp_clinical_rates=wwtp_rates,
    )

    # Generate the weekly all-component summary once.
    output_csv_with_weeks = save_weekly_risk_summary(
        weekly_results=weekly_results,
        disease_import_dfs=disease_import_dfs,
        disease_export_dfs=disease_export_dfs,
        wwtp_shapefile=wwtp_shapefile,
        output_csv="outputs/csv/weekly_top10_summary_multi.csv",
        start_week=start_week,
        end_week=end_week,
        county_boundary_fp=county_boundary_fp,
        county_clinical_rates=county_rates,
        wwtp_clinical_rates=wwtp_rates,
    )

    plot_yoy_monthly_changes(
        summary_csv=output_csv_with_weeks,
        output_dir="outputs/plots_yoy_risk",
        group_field="wwtp",
        disease="COVID",
        start_month=1,
        end_month=6,
        sharey=True,
    )

    # FileGen is now the single export path. It attaches canonical weekly
    # total_in/total_out values to every disease file, creates matching monthly
    # risk files, adds monthly trip mean/min/max summaries, and retains
    # county-source monthly tables for every sewershed.
    export_clinical_arcgis_outputs(
        weekly_results=weekly_results,
        county_rates=county_rates,
        disease_import_dfs=disease_import_dfs,
        disease_export_dfs=disease_export_dfs,
        weekly_summary_csv=output_csv_with_weeks,
        county_boundary_fp=county_boundary_fp,
        out_dir=arcgis_output_dir,
        start_week=start_week,
        end_week=end_week,
    )

    print(
        f"Pipeline complete: maps -> {diffusion_output_dir}, "
        f"summary -> {output_csv_with_weeks}, ArcGIS -> {arcgis_output_dir}"
    )


if __name__ == "__main__":
    run_OD_diffusion()
