"""
AnalysisViz_Level1BaselineComparison.py

Level-1 baseline comparison helpers for sentinel WWTP network selection.

Purpose
-------
Compare non-mobility Level-1 baseline portfolios against the Ours Level-1
selection using the SAME evaluation framework:

  1) Population-only baseline
  2) Spatial coverage baseline (true spatial optimization; avoids BG overlap)
  3) Census metro/micro-area baseline
  4) Ours Level-1 selection from the main network-selection logic

This module also adds spatial / EJ diagnostics, because a portfolio can look
strong on total coverage but still be spatially concentrated. In particular,
small subnetworks / singleton reserves can improve EJ and peripheral-area
representation even when they do not maximize one simple metric.

Expected inputs
---------------
features : DataFrame indexed by wwtp_clean, with columns such as:
    - wwtp
    - pop_served
    - pop_covered_by_od
    - od_volume_total
    - area_reached

strategy_orders : dict[str, list[str]]
    Each strategy is an ordered list of WWTP names. Names are canonicalized to
    lowercase internally. The final row of each cumulative table is used as the
    strategy-level portfolio summary.

BG-link cumulative metrics are delegated to AnalysisViz_CumulativeCoverageV1.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None

from AnalysisViz_CumulativeCoverageV1 import (
    build_unique_cumulative_table_bg,
    load_bg_link_window,
)


# -----------------------------------------------------------------------------
# Canonicalization and baseline order construction
# -----------------------------------------------------------------------------


def _canon(x) -> str:
    return str(x).strip().lower()


def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by canonical WWTP name."""
    feat = features.copy()
    feat.index = feat.index.map(_canon)
    if "wwtp" not in feat.columns:
        feat["wwtp"] = feat.index
    # Keep first duplicate, if any.
    feat = feat[~feat.index.duplicated(keep="first")]
    return feat


def _rank_desc(s: pd.Series) -> pd.Series:
    """Rank larger values first; lower rank is better."""
    return pd.to_numeric(s, errors="coerce").rank(ascending=False, method="min")


def build_greedy_unique_area_order_bg(
    features: pd.DataFrame,
    bg_link_dir: str,
    start_date: Optional[str],
    end_date: Optional[str],
    total_N: int = 21,
    direction: str = "Destination",
    min_weight: float = 0.0,
    weight_from: str = "Volume",
    tie_break_cols: Sequence[str] = ("area_reached", "pop_served", "od_volume_total"),
) -> List[str]:
    """
    Build a true spatial-optimization baseline using greedy unique BG area.

    This differs from the simple Area-only baseline. Area-only ranks sites by
    their site-level OD-linked area and can double-count the same BG/region.
    Greedy unique-area instead selects the site with the largest *new* BG area
    gain at each step. Once a BG has been covered by a selected site, adding
    another site connected to the same BG gives no additional spatial gain.

    The BG link is treated as binary for the purpose of spatial optimization:
    if a selected WWTP is linked to a BG in the analysis window, that BG can be
    spatially covered. Use `min_weight` to suppress very weak links if needed.
    The final portfolio is still evaluated by the standard BG-union metrics in
    evaluate_level1_strategy_orders(), so all strategies share the same scoring
    framework after selection.
    """
    feat = _prepare_features(features)
    N = int(total_N)
    if N <= 0:
        return []

    links = load_bg_link_window(
        bg_link_dir=bg_link_dir,
        start_date=start_date,
        end_date=end_date,
        direction=direction,  # type: ignore[arg-type]
    )
    if links is None or links.empty:
        # Fallback to Area-only ranking if BG links are unavailable.
        if "area_reached" in feat.columns:
            return (
                pd.to_numeric(feat["area_reached"], errors="coerce")
                .fillna(0.0)
                .sort_values(ascending=False, kind="mergesort")
                .head(N)
                .index
                .tolist()
            )
        return feat.index.tolist()[:N]

    links = links.copy()
    links["wwtp_clean"] = links["wwtp_clean"].map(_canon)
    links["bg_fips"] = links["bg_fips"].astype(str).str.strip()

    if weight_from in links.columns and float(min_weight) > 0:
        links[weight_from] = pd.to_numeric(links[weight_from], errors="coerce").fillna(0.0)
        links = links[links[weight_from] > float(min_weight)].copy()

    if links.empty or "Area" not in links.columns:
        if "area_reached" in feat.columns:
            return (
                pd.to_numeric(feat["area_reached"], errors="coerce")
                .fillna(0.0)
                .sort_values(ascending=False, kind="mergesort")
                .head(N)
                .index
                .tolist()
            )
        return feat.index.tolist()[:N]

    # One BG area value; max is robust to duplicated weekly rows.
    bg_area = pd.to_numeric(links["Area"], errors="coerce").fillna(0.0).groupby(links["bg_fips"]).max()

    # Candidate BG sets by WWTP. Keep only sites known in features.
    site_to_bgs: Dict[str, set] = {}
    for w, sub in links.groupby("wwtp_clean"):
        if w in feat.index:
            site_to_bgs[w] = set(sub["bg_fips"].dropna().astype(str).tolist())

    if not site_to_bgs:
        if "area_reached" in feat.columns:
            return (
                pd.to_numeric(feat["area_reached"], errors="coerce")
                .fillna(0.0)
                .sort_values(ascending=False, kind="mergesort")
                .head(N)
                .index
                .tolist()
            )
        return feat.index.tolist()[:N]

    # Deterministic tie-breaker: site-level area, then pop, then mobility.
    tie = pd.Series(0.0, index=feat.index, dtype=float)
    scale = 1.0
    for c in tie_break_cols:
        if c not in feat.columns:
            continue
        vals = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
        denom = vals.max() - vals.min()
        norm = (vals - vals.min()) / denom if denom > 0 else vals * 0.0
        tie = tie + scale * norm
        scale *= 0.01

    selected: List[str] = []
    covered: set = set()
    remaining = set(site_to_bgs.keys())

    while remaining and len(selected) < N:
        best_site = None
        best_gain = -1.0
        best_tie = -np.inf
        for w in remaining:
            new_bgs = site_to_bgs.get(w, set()) - covered
            gain = float(bg_area.reindex(list(new_bgs)).fillna(0.0).sum()) if new_bgs else 0.0
            t = float(tie.get(w, 0.0))
            if (gain > best_gain) or (gain == best_gain and t > best_tie) or (gain == best_gain and t == best_tie and (best_site is None or w < best_site)):
                best_site = w
                best_gain = gain
                best_tie = t
        if best_site is None:
            break
        selected.append(best_site)
        covered |= site_to_bgs.get(best_site, set())
        remaining.remove(best_site)

    # Fill any remaining slots by simple area ranking so the portfolio has N sites.
    if len(selected) < N:
        selected_set = set(selected)
        if "area_reached" in feat.columns:
            filler_order = (
                pd.to_numeric(feat["area_reached"], errors="coerce")
                .fillna(0.0)
                .sort_values(ascending=False, kind="mergesort")
                .index
                .tolist()
            )
        else:
            filler_order = feat.index.tolist()
        for w in filler_order:
            if w not in selected_set:
                selected.append(w)
                selected_set.add(w)
            if len(selected) >= N:
                break

    return selected[:N]



def build_spatial_geometry_greedy_order(
    features: pd.DataFrame,
    sewersheds_gdf=None,
    total_N: int = 21,
) -> List[str]:
    """
    Non-mobility spatial baseline.

    This baseline does NOT use OD links, BG mobility reach, commute volume, or
    viral signals. It selects WWTPs greedily by the largest additional area of
    the WWTP/service-boundary geometry not already covered by selected sites.
    If sewershed/service polygons are unavailable, it falls back to population
    ordering so the pipeline can still run.
    """
    feat = _prepare_features(features)
    N = int(total_N)
    if N <= 0:
        return []

    if sewersheds_gdf is None or getattr(sewersheds_gdf, "empty", True):
        return _full_population_order(feat)[:N] if "pop_served" in feat.columns else feat.index.tolist()[:N]

    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        return _full_population_order(feat)[:N] if "pop_served" in feat.columns else feat.index.tolist()[:N]

    sg = sewersheds_gdf.copy()
    name_col = None
    for c in ["wwtp_clean", "wwtp", "WWTP", "facility", "Facility", "name", "Name"]:
        if c in sg.columns:
            name_col = c
            break
    if name_col is None or "geometry" not in sg.columns:
        return _full_population_order(feat)[:N] if "pop_served" in feat.columns else feat.index.tolist()[:N]

    sg["wwtp_clean"] = sg[name_col].map(_canon)
    sg = sg[sg["wwtp_clean"].isin(feat.index)].copy()
    sg = sg[sg.geometry.notna() & (~sg.geometry.is_empty)].copy()
    if sg.empty:
        return _full_population_order(feat)[:N] if "pop_served" in feat.columns else feat.index.tolist()[:N]

    try:
        if sg.crs is None:
            pass
        elif getattr(sg.crs, "is_geographic", False):
            sg = sg.to_crs("EPSG:5070")
        else:
            sg = sg.to_crs(sg.crs)
    except Exception:
        pass

    # Dissolve multipart rows per WWTP into one geometry.
    geoms = {}
    try:
        for w, sub in sg.groupby("wwtp_clean"):
            geom = sub.geometry.unary_union
            if geom is not None and (not geom.is_empty):
                geoms[w] = geom
    except Exception:
        for _, r in sg.iterrows():
            w = r["wwtp_clean"]
            geom = r.geometry
            if geom is not None and (not geom.is_empty) and w not in geoms:
                geoms[w] = geom

    if not geoms:
        return _full_population_order(feat)[:N] if "pop_served" in feat.columns else feat.index.tolist()[:N]

    # Deterministic non-mobility tie-breaker: served population, then site name.
    pop = pd.to_numeric(feat.get("pop_served", pd.Series(0.0, index=feat.index)), errors="coerce").fillna(0.0)

    selected: List[str] = []
    selected_set = set()
    remaining = set(geoms.keys())
    current_union = None

    while remaining and len(selected) < N:
        best_site = None
        best_gain = -1.0
        best_pop = -np.inf
        for w in remaining:
            geom = geoms.get(w)
            if geom is None or geom.is_empty:
                gain = 0.0
            elif current_union is None:
                gain = float(geom.area)
            else:
                try:
                    gain = float(geom.difference(current_union).area)
                except Exception:
                    gain = 0.0
            p = float(pop.get(w, 0.0))
            if (gain > best_gain) or (gain == best_gain and p > best_pop) or (gain == best_gain and p == best_pop and (best_site is None or w < best_site)):
                best_site = w
                best_gain = gain
                best_pop = p
        if best_site is None:
            break
        selected.append(best_site)
        selected_set.add(best_site)
        geom = geoms.get(best_site)
        if geom is not None and (not geom.is_empty):
            if current_union is None:
                current_union = geom
            else:
                try:
                    current_union = current_union.union(geom)
                except Exception:
                    pass
        remaining.remove(best_site)

    # Fill any remaining slots by population only. No mobility fields are used.
    for w in _full_population_order(feat):
        if len(selected) >= N:
            break
        if w not in selected_set:
            selected.append(w)
            selected_set.add(w)

    return selected[:N]




def make_level1_baseline_orders(
    features: pd.DataFrame,
    total_N: int = 21,
    include_ej_only: bool = False,
    ej_scores: Optional[pd.Series] = None,
    # Optional inputs for the non-mobility spatial baseline
    bg_link_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    direction: str = "Destination",
    greedy_unique_area: bool = True,
    greedy_min_weight: float = 0.0,
    # Optional inputs for Census metro/micropolitan baseline
    sewersheds_gdf=None,
    cbsa_shp: Optional[str] = None,
    include_cbsa_baseline: bool = True,
    include_legacy_mobility_area_baselines: bool = False,
) -> Dict[str, List[str]]:
    """
    Build reviewer-facing Level-1 baseline portfolios.

    Main-text baselines are intentionally non-mobility alternatives because
    mobility is part of Ours Level 1:
      - Population-only: largest sewershed population served.
      - Spatial coverage: at each step picks the site that adds the most
        previously uncovered WWTP/service-boundary area.
      - Census metro/micro-area: one representative WWTP per official Census
        CBSA, then fill by a simple non-mobility site score.

    Legacy mobility/area-only baselines can be enabled for SI/internal checks,
    but are disabled by default.
    """
    feat = _prepare_features(features)
    N = int(total_N)

    orders: Dict[str, List[str]] = {}

    if "pop_served" in feat.columns:
        score = pd.to_numeric(feat["pop_served"], errors="coerce").fillna(0.0)
        orders["Population-only"] = score.sort_values(ascending=False, kind="mergesort").head(N).index.tolist()

    if greedy_unique_area and sewersheds_gdf is not None:
        orders["Spatial coverage"] = build_spatial_geometry_greedy_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            total_N=N,
        )

    if include_cbsa_baseline and sewersheds_gdf is not None and cbsa_shp:
        cbsa_order = build_cbsa_metro_micro_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
            total_N=N,
        )
        if cbsa_order:
            orders["Census metro/micro-area"] = cbsa_order

    if include_legacy_mobility_area_baselines:
        if "area_reached" in feat.columns:
            score = pd.to_numeric(feat["area_reached"], errors="coerce").fillna(0.0)
            orders["Area-only (SI)"] = score.sort_values(ascending=False, kind="mergesort").head(N).index.tolist()

        mobility_cols = [c for c in ["od_volume_total", "pop_covered_by_od"] if c in feat.columns]
        if mobility_cols:
            rank_mat = pd.concat([_rank_desc(feat[c]) for c in mobility_cols], axis=1)
            score_rank = rank_mat.mean(axis=1, skipna=True)
            orders["Mobility-activity-only (SI)"] = score_rank.sort_values(kind="mergesort").head(N).index.tolist()

    if include_ej_only and ej_scores is not None and len(ej_scores) > 0:
        ej = pd.to_numeric(ej_scores, errors="coerce")
        ej.index = ej.index.map(_canon)
        ej = ej.reindex(feat.index)
        if ej.notna().any():
            orders["EJ-only"] = ej.sort_values(ascending=False, kind="mergesort").head(N).index.tolist()

    return orders


# -----------------------------------------------------------------------------
# Spatial / EJ diagnostics
# -----------------------------------------------------------------------------


def load_dominant_county_map(utility_county_csv: Optional[str]) -> Dict[str, str]:
    """
    Load dominant county label by WWTP from utility_county_adj.csv-like table.

    Expected columns include Utility, county_name, and optionally PctOfUtility.
    Returns {wwtp_clean: county_name_titlecase}.
    """
    if not utility_county_csv or not os.path.exists(utility_county_csv):
        return {}
    try:
        tbl = pd.read_csv(utility_county_csv)
    except Exception:
        return {}

    util_col = next((c for c in tbl.columns if c.lower() in {"utility", "wwtp", "wwtp_id", "name"}), None)
    county_col = next((c for c in tbl.columns if c.lower() in {"county_name", "county", "countyname"}), None)
    if util_col is None or county_col is None:
        return {}

    tbl = tbl.copy()
    tbl["_util"] = tbl[util_col].map(_canon)
    tbl["_county"] = tbl[county_col].astype(str).str.strip().str.title()

    pct_col = next((c for c in tbl.columns if c.lower() in {"pctofutility", "pct", "percent"}), None)
    if pct_col is not None:
        tbl["_pct"] = pd.to_numeric(tbl[pct_col], errors="coerce").fillna(0.0)
        tbl = tbl.sort_values(["_util", "_pct"], ascending=[True, False])
    else:
        tbl = tbl.sort_values(["_util"])

    tbl = tbl.drop_duplicates("_util", keep="first")
    return dict(zip(tbl["_util"], tbl["_county"]))


# -----------------------------------------------------------------------------
# CBSA metro/micropolitan baseline and diagnostics
# -----------------------------------------------------------------------------


def load_cbsa_site_assignments(sewersheds_gdf=None, cbsa_shp: Optional[str] = None) -> pd.DataFrame:
    """
    Assign each WWTP centroid to a Census Core Based Statistical Area (CBSA).

    This supports a reviewer-facing simple baseline based on official Census
    metropolitan / micropolitan statistical areas. CBSA LSAD codes are usually:
      - M1 = Metropolitan Statistical Area
      - M2 = Micropolitan Statistical Area

    Returns a DataFrame indexed by wwtp_clean with:
      cbsa_fips, cbsa_name, cbsa_type, in_cbsa

    If the CBSA shapefile is unavailable, returns an empty DataFrame so the main
    workflow can continue without failing.
    """
    if gpd is None or sewersheds_gdf is None or not cbsa_shp or not os.path.exists(cbsa_shp):
        return pd.DataFrame()

    try:
        cbsa = gpd.read_file(cbsa_shp)
    except Exception as e:
        print(f"[CBSA] Could not read CBSA shapefile: {cbsa_shp} ({e})")
        return pd.DataFrame()

    if cbsa.empty or "geometry" not in cbsa.columns:
        return pd.DataFrame()

    sg = sewersheds_gdf.copy()
    if "wwtp" not in sg.columns:
        sg = sg.reset_index().rename(columns={"index": "wwtp"})

    try:
        if sg.crs is None:
            # Most project shapefiles have a CRS. If not, avoid an unsafe join.
            print("[CBSA] Sewershed GeoDataFrame has no CRS; skipping CBSA baseline.")
            return pd.DataFrame()
        if cbsa.crs is None:
            print("[CBSA] CBSA shapefile has no CRS; skipping CBSA baseline.")
            return pd.DataFrame()
        sg2 = sg.to_crs(cbsa.crs)
    except Exception as e:
        print(f"[CBSA] CRS conversion failed; skipping CBSA baseline ({e})")
        return pd.DataFrame()

    pts = sg2.copy()
    pts["geometry"] = pts.geometry.centroid
    pts["wwtp_clean"] = pts["wwtp"].map(_canon)

    cbsa_cols = [c for c in ["CBSAFP", "GEOID", "NAME", "NAMELSAD", "LSAD", "MEMI"] if c in cbsa.columns]
    cbsa_use = cbsa[cbsa_cols + ["geometry"]].copy()

    try:
        joined = gpd.sjoin(
            pts[["wwtp_clean", "geometry"]],
            cbsa_use,
            how="left",
            predicate="within",
        )
    except Exception:
        # Older geopandas versions may not support predicate=.
        joined = gpd.sjoin(
            pts[["wwtp_clean", "geometry"]],
            cbsa_use,
            how="left",
            op="within",
        )

    if joined.empty:
        return pd.DataFrame()

    def _cbsa_type(lsad) -> str:
        x = str(lsad).strip().upper()
        if x == "M1":
            return "Metro"
        if x == "M2":
            return "Micro"
        if x and x != "NAN":
            return "CBSA"
        return "Outside CBSA"

    out = pd.DataFrame({
        "wwtp_clean": joined["wwtp_clean"].astype(str),
        "cbsa_fips": joined.get("CBSAFP", joined.get("GEOID", pd.Series(index=joined.index, dtype=object))).astype(str),
        "cbsa_name": joined.get("NAMELSAD", joined.get("NAME", pd.Series(index=joined.index, dtype=object))).astype(str),
        "cbsa_lsad": joined.get("LSAD", pd.Series(index=joined.index, dtype=object)).astype(str),
    })
    out["cbsa_type"] = out["cbsa_lsad"].map(_cbsa_type)
    out.loc[out["cbsa_fips"].isin(["nan", "None", ""]), "cbsa_fips"] = np.nan
    out.loc[out["cbsa_name"].isin(["nan", "None", ""]), "cbsa_name"] = np.nan
    out["in_cbsa"] = out["cbsa_fips"].notna()

    # One row per WWTP. If a centroid lies on a boundary and joins multiple
    # polygons, keep the first deterministically.
    out = out.drop_duplicates("wwtp_clean", keep="first").set_index("wwtp_clean")
    return out


def _minmax_series(s: pd.Series) -> pd.Series:
    """Simple 0-1 normalization with all-zero fallback."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=x.index)
    return (x - lo) / (hi - lo)



def _site_centroid_xy_from_sewersheds(sewersheds_gdf) -> pd.DataFrame:
    """
    Return projected centroid coordinates for WWTPs.

    Used only for non-mobility spatial filling in the CBSA-representative
    baseline. It does not use BG reach, OD volume, or any mobility-derived
    field.
    """
    if gpd is None or sewersheds_gdf is None:
        return pd.DataFrame(columns=["x", "y"])
    try:
        sg = sewersheds_gdf.copy()
        if "wwtp" not in sg.columns:
            sg = sg.reset_index().rename(columns={"index": "wwtp"})
        if "geometry" not in sg.columns or sg.empty:
            return pd.DataFrame(columns=["x", "y"])
        if sg.crs is None:
            return pd.DataFrame(columns=["x", "y"])
        sgp = sg.to_crs(epsg=3857)
        cent = sgp.geometry.centroid
        out = pd.DataFrame({
            "wwtp_clean": sgp["wwtp"].map(_canon),
            "x": cent.x,
            "y": cent.y,
        }).drop_duplicates("wwtp_clean", keep="first").set_index("wwtp_clean")
        return out
    except Exception:
        return pd.DataFrame(columns=["x", "y"])


def _farthest_first_fill_order(
    candidates: Sequence[str],
    selected: Sequence[str],
    xy: pd.DataFrame,
    tie_score: Optional[pd.Series] = None,
) -> List[str]:
    """
    Spatial-spread filler order for non-mobility baselines.

    Iteratively adds the candidate farthest from the currently selected set.
    If no selected sites have coordinates, starts from the candidate with the
    highest average distance to all other candidates. Population may be used
    only as a tie-breaker, not as the primary refill criterion.
    """
    cand = [str(c) for c in candidates if str(c)]
    cand = [c for c in dict.fromkeys(cand)]
    chosen = [str(s) for s in selected if str(s)]
    xy2 = xy.copy() if xy is not None else pd.DataFrame(columns=["x", "y"])
    if xy2.empty:
        if tie_score is not None and len(tie_score) > 0:
            return [c for c in tie_score.reindex(cand).fillna(0.0).sort_values(ascending=False, kind="mergesort").index.astype(str).tolist()]
        return cand

    def _dist_to_set(site: str, selected_sites: List[str]) -> float:
        if site not in xy2.index:
            return -np.inf
        sx, sy = float(xy2.loc[site, "x"]), float(xy2.loc[site, "y"])
        selected_sites = [s for s in selected_sites if s in xy2.index]
        if not selected_sites:
            others = [c for c in cand if c != site and c in xy2.index]
            if not others:
                return 0.0
            dx = xy2.loc[others, "x"].astype(float).values - sx
            dy = xy2.loc[others, "y"].astype(float).values - sy
            return float(np.nanmean(np.sqrt(dx * dx + dy * dy)))
        dx = xy2.loc[selected_sites, "x"].astype(float).values - sx
        dy = xy2.loc[selected_sites, "y"].astype(float).values - sy
        return float(np.nanmin(np.sqrt(dx * dx + dy * dy)))

    remaining = [c for c in cand if c not in set(chosen)]
    out: List[str] = []
    while remaining:
        rows = []
        for c in remaining:
            d = _dist_to_set(c, chosen + out)
            t = float(tie_score.get(c, 0.0)) if tie_score is not None and c in tie_score.index else 0.0
            rows.append((c, d, t))
        rows = sorted(rows, key=lambda z: (z[1], z[2], z[0]), reverse=True)
        best = rows[0][0]
        out.append(best)
        remaining.remove(best)
    return out


def build_cbsa_metro_micro_order(
    features: pd.DataFrame,
    sewersheds_gdf=None,
    cbsa_shp: Optional[str] = None,
    total_N: int = 21,
    return_details: bool = False,
):
    """
    Build a transparent Census metro/micro representative baseline.

    Selection logic:
      1) Assign WWTPs to official Census CBSA polygons by centroid.
      2) For each Metro/Micro CBSA, select one population-representative WWTP.
      3) Order CBSA representatives by CBSA type and aggregate CBSA score.
      4) If fewer than N representatives are available, fill remaining slots
         using non-mobility spatial spread:
           4a) first from outside-CBSA sites,
           4b) then from all remaining unselected sites if needed.

    If return_details=True, returns (order, detail_df), where detail_df records
    whether each selected site entered as:
      - cbsa_representative
      - farthest_outside_cbsa
      - farthest_remaining
    """
    feat = _prepare_features(features)
    N = int(total_N)
    if N <= 0:
        return ([], pd.DataFrame()) if return_details else []

    cbsa = load_cbsa_site_assignments(sewersheds_gdf, cbsa_shp=cbsa_shp)
    if cbsa.empty:
        return ([], pd.DataFrame()) if return_details else []

    # Simple non-mobility score only for choosing a representative within each CBSA
    # and as a tie-breaker. It is not used as a global refill rule.
    site_score = _minmax_series(feat["pop_served"] if "pop_served" in feat.columns else pd.Series(0.0, index=feat.index))

    joined = feat.copy()
    joined.index = joined.index.astype(str)
    joined = joined.join(cbsa, how="left")
    joined["site_score"] = site_score.reindex(joined.index).fillna(0.0)
    joined["in_cbsa"] = joined["in_cbsa"].fillna(False).astype(bool)

    selected: List[str] = []
    seen = set()
    detail_rows = []

    # Representatives: one highest-scoring site per Metro/Micro CBSA.
    in_cbsa = joined[joined["in_cbsa"] & joined["cbsa_fips"].notna()].copy()
    if not in_cbsa.empty:
        cbsa_total_score = in_cbsa.groupby("cbsa_fips")["site_score"].sum().rename("cbsa_total_score")
        in_cbsa = in_cbsa.join(cbsa_total_score, on="cbsa_fips")

        rep_rows = []
        for cbsa_id, sub in in_cbsa.groupby("cbsa_fips", sort=False):
            sub = sub.sort_values(["site_score"], ascending=False, kind="mergesort")
            rep = sub.iloc[0]
            rep_rows.append(rep)

        reps = pd.DataFrame(rep_rows)
        if not reps.empty:
            # Metro first, then Micro, then other CBSA; larger aggregate score first.
            def _type_order(x):
                x = str(x).upper()
                if x == "M1":
                    return 0
                if x == "M2":
                    return 1
                return 2

            reps["_type_order"] = reps["cbsa_lsad"].map(_type_order)
            reps = reps.sort_values(
                ["_type_order", "cbsa_total_score", "site_score"],
                ascending=[True, False, False],
                kind="mergesort",
            )

            for w, row in reps.iterrows():
                w = str(w)
                if w in feat.index and w not in seen:
                    selected.append(w)
                    seen.add(w)
                    detail_rows.append({
                        "order": len(selected),
                        "wwtp_clean": w,
                        "selection_source": "cbsa_representative",
                        "selection_phase": "CBSA representative",
                        "cbsa_fips": row.get("cbsa_fips", ""),
                        "cbsa_name": row.get("cbsa_name", ""),
                        "cbsa_lsad": row.get("cbsa_lsad", ""),
                        "in_cbsa": True,
                        "site_score": float(row.get("site_score", np.nan)),
                    })
                if len(selected) >= N:
                    out_df = pd.DataFrame(detail_rows)
                    return (selected[:N], out_df) if return_details else selected[:N]

    xy = _site_centroid_xy_from_sewersheds(sewersheds_gdf)

    # Neutral reserve: if CBSA representatives are fewer than N, fill outside-CBSA
    # sites first using geographic spread, then any remaining unselected sites by
    # the same spread rule. This keeps the baseline an official-region baseline,
    # not a mobility or health-risk baseline.
    outside = joined[(~joined["in_cbsa"])].index.astype(str).tolist()
    outside = [w for w in outside if w not in seen and w in feat.index]
    fill1 = _farthest_first_fill_order(outside, selected, xy=xy, tie_score=site_score)
    for w in fill1:
        if w not in seen:
            selected.append(w)
            seen.add(w)
            detail_rows.append({
                "order": len(selected),
                "wwtp_clean": w,
                "selection_source": "farthest_outside_cbsa",
                "selection_phase": "Outside-CBSA farthest-first refill",
                "cbsa_fips": "",
                "cbsa_name": "",
                "cbsa_lsad": "",
                "in_cbsa": False,
                "site_score": float(site_score.get(w, np.nan)),
            })
        if len(selected) >= N:
            out_df = pd.DataFrame(detail_rows)
            return (selected[:N], out_df) if return_details else selected[:N]

    remaining = [w for w in feat.index.astype(str).tolist() if w not in seen]
    fill2 = _farthest_first_fill_order(remaining, selected, xy=xy, tie_score=site_score)
    for w in fill2:
        if w not in seen:
            selected.append(w)
            seen.add(w)
            row = joined.loc[w] if w in joined.index else pd.Series(dtype=object)
            detail_rows.append({
                "order": len(selected),
                "wwtp_clean": w,
                "selection_source": "farthest_remaining",
                "selection_phase": "Remaining farthest-first refill",
                "cbsa_fips": row.get("cbsa_fips", ""),
                "cbsa_name": row.get("cbsa_name", ""),
                "cbsa_lsad": row.get("cbsa_lsad", ""),
                "in_cbsa": bool(row.get("in_cbsa", False)) if len(row) else False,
                "site_score": float(site_score.get(w, np.nan)),
            })
        if len(selected) >= N:
            break

    out_df = pd.DataFrame(detail_rows)
    return (selected[:N], out_df) if return_details else selected[:N]


def _resolve_local_path(path_like: Optional[str]) -> Optional[str]:
    """Resolve a local path robustly from the current script location."""
    if not path_like:
        return None
    raw = str(path_like)
    candidates = []
    # as given
    candidates.append(raw)
    # relative to current working directory
    candidates.append(os.path.normpath(os.path.join(os.getcwd(), raw)))
    # relative to this script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.normpath(os.path.join(script_dir, raw)))
    except Exception:
        pass
    # de-duplicate while preserving order
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            if os.path.exists(cand):
                return cand
    return raw

def _graph_site_metadata(G) -> pd.DataFrame:
    """
    Extract component size, centroid, and node-level counties from the network graph.
    """
    if G is None:
        return pd.DataFrame()

    try:
        import networkx as nx  # local import to avoid hard dependency when unused
    except Exception:
        return pd.DataFrame()

    comps = list(nx.connected_components(G))
    node_to_compid = {}
    node_to_compsize = {}
    for cid, comp in enumerate(comps, start=1):
        for n in comp:
            node_to_compid[n] = cid
            node_to_compsize[n] = len(comp)

    rows = []
    for n in G.nodes:
        attrs = G.nodes[n]
        nm = attrs.get("wwtp") or str(n)
        counties_in = attrs.get("counties_inflow_from", []) or []
        counties_out = attrs.get("counties_outflow_to", []) or []
        counties = sorted({str(x).zfill(5) for x in list(counties_in) + list(counties_out)})
        c = attrs.get("centroid")
        rows.append({
            "wwtp_clean": _canon(nm),
            "component_id": node_to_compid.get(n, np.nan),
            "component_size": node_to_compsize.get(n, np.nan),
            "x": getattr(c, "x", np.nan),
            "y": getattr(c, "y", np.nan),
            "county_fips_union": ";".join(counties),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates("wwtp_clean", keep="first").set_index("wwtp_clean")
    return out


def _mean_nearest_neighbor_km(xy: np.ndarray) -> float:
    """Mean nearest-neighbor distance among selected points, assuming meters."""
    if xy.shape[0] < 2:
        return np.nan
    dmins = []
    for i in range(xy.shape[0]):
        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d[i] = np.inf
        dmins.append(float(np.nanmin(d)))
    return float(np.nanmean(dmins) / 1000.0)


def compute_spatial_ej_stats(
    selected_order: Sequence[str],
    features: pd.DataFrame,
    G=None,
    ej_scores: Optional[pd.Series] = None,
    utility_county_csv: Optional[str] = None,
    metro_counties: Optional[Iterable[str]] = None,
    small_component_max_size: int = 2,
    sewersheds_gdf=None,
    cbsa_shp: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute portfolio spatial and EJ diagnostics.

    These columns are designed to explain WHY adding peripheral singleton/small
    subnetworks can be useful: it may improve EJ/high-burden representation and
    reduce spatial concentration, even if a simple coverage baseline wins one
    metric.
    """
    feat = _prepare_features(features)
    sel = [_canon(s) for s in selected_order if _canon(s) in feat.index]
    sel = list(dict.fromkeys(sel))
    sub = feat.reindex(sel)

    meta = _graph_site_metadata(G)
    county_map = load_dominant_county_map(utility_county_csv)

    # Counties represented: prefer graph county FIPS union; fallback to dominant county names.
    county_tokens = set()
    dominant_counties = []
    for w in sel:
        if not meta.empty and w in meta.index:
            raw = str(meta.loc[w].get("county_fips_union", ""))
            county_tokens.update([x for x in raw.split(";") if x])
        if w in county_map:
            dominant_counties.append(county_map[w])
    if not county_tokens:
        county_tokens = set(dominant_counties)

    # Component representation.
    n_components = np.nan
    n_total_components = np.nan
    component_representation_frac = np.nan
    n_singleton_sites = np.nan
    n_small_component_sites = np.nan
    n_large_component_sites = np.nan
    mean_nn_km = np.nan
    if not meta.empty:
        n_total_components = float(meta["component_id"].dropna().nunique()) if "component_id" in meta.columns else np.nan
        msel = meta.reindex(sel)
        n_components = float(msel["component_id"].dropna().nunique())
        component_representation_frac = (n_components / n_total_components) if n_total_components and not np.isnan(n_total_components) else np.nan
        comp_size = pd.to_numeric(msel["component_size"], errors="coerce")
        n_singleton_sites = float((comp_size == 1).sum())
        n_small_component_sites = float((comp_size <= int(small_component_max_size)).sum())
        n_large_component_sites = float((comp_size > int(small_component_max_size)).sum())
        xy = msel[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().values
        mean_nn_km = _mean_nearest_neighbor_km(xy)

    # Metro representation by dominant county label, if available.
    if metro_counties is None:
        # Default Colorado Front Range / major metro-linked counties. Safe fallback:
        # ignored when dominant county map is unavailable.
        metro_counties = {
            "Denver", "Arapahoe", "Jefferson", "Adams", "Boulder", "Broomfield",
            "Douglas", "El Paso", "Larimer", "Weld", "Pueblo",
        }
    metro_set = {str(c).strip().title() for c in metro_counties}
    n_metro_sites = float(sum(1 for c in dominant_counties if c in metro_set)) if dominant_counties else np.nan

    # Official Census CBSA representation. This directly addresses reviewer
    # questions about using Metropolitan/Micropolitan Statistical Areas as a
    # simpler network-design proxy.
    n_cbsa_represented = np.nan
    n_metro_cbsa_represented = np.nan
    n_micro_cbsa_represented = np.nan
    n_outside_cbsa_sites = np.nan
    cbsa_site_frac = np.nan
    cbsa_assign = load_cbsa_site_assignments(sewersheds_gdf=sewersheds_gdf, cbsa_shp=cbsa_shp)
    if cbsa_assign is not None and not cbsa_assign.empty and len(sel) > 0:
        csel = cbsa_assign.reindex(sel)
        in_cbsa = csel["in_cbsa"].fillna(False).astype(bool) if "in_cbsa" in csel.columns else pd.Series(False, index=csel.index)
        n_cbsa_represented = float(csel.loc[in_cbsa, "cbsa_fips"].dropna().nunique()) if "cbsa_fips" in csel.columns else np.nan
        ctype = csel.get("cbsa_type", pd.Series(index=csel.index, dtype=object)).astype(str)
        n_metro_cbsa_represented = float(csel.loc[in_cbsa & (ctype == "Metro"), "cbsa_fips"].dropna().nunique()) if "cbsa_fips" in csel.columns else np.nan
        n_micro_cbsa_represented = float(csel.loc[in_cbsa & (ctype == "Micro"), "cbsa_fips"].dropna().nunique()) if "cbsa_fips" in csel.columns else np.nan
        n_outside_cbsa_sites = float((~in_cbsa).sum())
        cbsa_site_frac = float(in_cbsa.sum() / len(sel)) if len(sel) else np.nan

    # EJ metrics.
    ej_mean = np.nan
    ej_median = np.nan
    ej_pop_weighted_mean = np.nan
    ej_weighted_pop_score_frac = np.nan
    high_ej_sites_selected = np.nan
    high_ej_sites_selected_frac_of_all_high_ej = np.nan
    high_ej_pop_served_frac = np.nan

    if ej_scores is not None and len(ej_scores) > 0:
        ej = pd.to_numeric(ej_scores, errors="coerce")
        ej.index = ej.index.map(_canon)
        ej_all = ej.reindex(feat.index)
        ej_sel = ej.reindex(sel)
        if ej_sel.notna().any():
            ej_mean = float(ej_sel.mean(skipna=True))
            ej_median = float(ej_sel.median(skipna=True))
            pop_all = pd.to_numeric(feat.get("pop_served", pd.Series(0.0, index=feat.index)), errors="coerce").fillna(0.0)
            pop_sel = pop_all.reindex(sel).fillna(0.0)
            denom_pop = float(pop_sel[ej_sel.notna()].sum())
            if denom_pop > 0:
                ej_pop_weighted_mean = float((ej_sel.fillna(0.0) * pop_sel).sum() / denom_pop)

            all_score_pop = float((ej_all.fillna(0.0) * pop_all).sum())
            sel_score_pop = float((ej_sel.fillna(0.0) * pop_sel).sum())
            if all_score_pop > 0:
                ej_weighted_pop_score_frac = sel_score_pop / all_score_pop

            q75 = float(ej_all.quantile(0.75)) if ej_all.notna().any() else np.nan
            if not np.isnan(q75):
                high_all = set(ej_all[ej_all >= q75].dropna().index.tolist())
                high_sel = [w for w in sel if w in high_all]
                high_ej_sites_selected = float(len(high_sel))
                high_ej_sites_selected_frac_of_all_high_ej = (len(high_sel) / len(high_all)) if high_all else np.nan
                high_pop_all = float(pop_all.reindex(list(high_all)).fillna(0.0).sum())
                high_pop_sel = float(pop_all.reindex(high_sel).fillna(0.0).sum())
                high_ej_pop_served_frac = high_pop_sel / high_pop_all if high_pop_all > 0 else np.nan

    out = {
        "n_selected": float(len(sel)),
        "n_counties_represented": float(len(county_tokens)),
        "n_dominant_counties_represented": float(len(set(dominant_counties))) if dominant_counties else np.nan,
        "n_components_represented": n_components,
        "n_total_components": n_total_components,
        "component_representation_frac": component_representation_frac,
        "n_singleton_sites": n_singleton_sites,
        "n_small_component_sites": n_small_component_sites,
        "n_large_component_sites": n_large_component_sites,
        "small_component_site_frac": (n_small_component_sites / len(sel)) if len(sel) and not np.isnan(n_small_component_sites) else np.nan,
        "mean_nearest_neighbor_km": mean_nn_km,
        "n_metro_sites": n_metro_sites,
        "metro_site_frac": (n_metro_sites / len(sel)) if len(sel) and not np.isnan(n_metro_sites) else np.nan,
        "non_metro_site_frac": (1.0 - (n_metro_sites / len(sel))) if len(sel) and not np.isnan(n_metro_sites) else np.nan,
        "n_cbsa_represented": n_cbsa_represented,
        "n_metro_cbsa_represented": n_metro_cbsa_represented,
        "n_micro_cbsa_represented": n_micro_cbsa_represented,
        "n_outside_cbsa_sites": n_outside_cbsa_sites,
        "cbsa_site_frac": cbsa_site_frac,
        "ej_mean_selected": ej_mean,
        "ej_median_selected": ej_median,
        "ej_pop_weighted_mean_selected": ej_pop_weighted_mean,
        "ej_weighted_pop_score_frac": ej_weighted_pop_score_frac,
        "high_ej_sites_selected": high_ej_sites_selected,
        "high_ej_sites_selected_frac_of_all_high_ej": high_ej_sites_selected_frac_of_all_high_ej,
        "high_ej_pop_served_frac": high_ej_pop_served_frac,
    }
    return out



def compute_mobility_county_contribution_stats(
    selected_order: Sequence[str],
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    direction: str = "Destination",
    union_mode: str = "cap_sum",
    weight_from: str = "Volume",
    bg_weight_mode: str = "intensity_exp",
    tau: float = 2.0,
    eps_pop: float = 100.0,
    county_meaningful_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    County-level mobility-reach and site-contribution diagnostics.

    Important interpretation:
    - BG population/area metrics are already unique mobility-linked reach.
    - This function therefore does NOT recompute another unique BG metric.
    - Instead, it summarizes whether the unique BG reach is distributed across
      county-level public-health units and whether selected sites make balanced
      marginal contributions.

    County metrics are derived from BG FIPS prefixes in the BG-link files:
      county_fips = first five digits of bg_fips.

    Main county metric:
      mobility_counties_contributing_frac_5pct =
        fraction of BG-link counties for which selected sites cover at least
        5% of that county's mobility-linked BG area.

    Site evenness metric:
      site_marginal_area_evenness =
        normalized effective number of sites contributing unique BG area.
        Values near 1 mean many selected sites add distinct mobility reach;
        lower values mean the portfolio is dominated by a few sites.
    """
    sel = [_canon(s) for s in selected_order]
    sel = [s for s in dict.fromkeys(sel) if s]
    if not sel:
        return {}

    links = load_bg_link_window(bg_link_dir, start_date, end_date, direction=direction)  # type: ignore[arg-type]
    if links is None or links.empty or "bg_fips" not in links.columns or "wwtp_clean" not in links.columns:
        return {}

    links = links.copy()
    links["bg_fips"] = links["bg_fips"].astype(str).str.strip()
    links["county_fips"] = links["bg_fips"].str.zfill(12).str[:5]
    links["wwtp_clean"] = links["wwtp_clean"].astype(str).str.strip().str.lower()

    if weight_from not in links.columns:
        return {}
    for c in ["Volume", "Area", "Population", weight_from]:
        if c in links.columns:
            links[c] = pd.to_numeric(links[c], errors="coerce").fillna(0.0)

    # BG masses and county universe from the mobility-linked BG data.
    bg_pop = links.groupby("bg_fips")["Population"].max() if "Population" in links.columns else pd.Series(dtype=float)
    bg_area = links.groupby("bg_fips")["Area"].max() if "Area" in links.columns else pd.Series(dtype=float)
    bg_county = links.groupby("bg_fips")["county_fips"].first()

    if bg_area.empty:
        return {}

    county_area_total = bg_area.groupby(bg_county).sum()
    n_counties_total = float((county_area_total > 0).sum())
    if n_counties_total <= 0:
        return {}

    # Build BG weights using the same logic as build_unique_cumulative_table_bg.
    w_df = (
        links.groupby(["wwtp_clean", "bg_fips"], as_index=False)[weight_from]
        .sum()
        .rename(columns={weight_from: "w"})
    )

    mode = str(bg_weight_mode).strip().lower()
    if str(union_mode).lower() == "binary":
        w_df["wi"] = 1.0
    elif mode == "share":
        bg_total_w = w_df.groupby("bg_fips")["w"].sum().rename("bg_total_w")
        w_df = w_df.merge(bg_total_w, on="bg_fips", how="left")
        w_df["wi"] = np.where(w_df["bg_total_w"] > 0, w_df["w"] / w_df["bg_total_w"], 0.0)
    elif mode in ("intensity_cap", "intensity_exp"):
        pop_df = bg_pop.rename("bg_pop").reset_index()
        w_df = w_df.merge(pop_df, on="bg_fips", how="left")
        w_df["bg_pop"] = pd.to_numeric(w_df["bg_pop"], errors="coerce").fillna(0.0)
        denom = float(tau) * (w_df["bg_pop"].values + float(eps_pop))
        denom = np.where(denom > 0, denom, np.nan)
        x = w_df["w"].values / denom
        if mode == "intensity_cap":
            w_df["wi"] = np.minimum(1.0, np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0))
        else:
            w_df["wi"] = 1.0 - np.exp(-np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        w_df["wi"] = pd.to_numeric(w_df["wi"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        return {}

    wwtp_to_bgwi: Dict[str, Dict[str, float]] = {}
    for wwtp, sub in w_df.groupby("wwtp_clean"):
        wwtp_to_bgwi[str(wwtp)] = dict(zip(sub["bg_fips"].astype(str), sub["wi"].astype(float)))

    covered = pd.Series(0.0, index=bg_area.index, dtype=float)
    site_area_deltas: List[float] = []
    site_pop_deltas: List[float] = []

    for wwtp in sel:
        bgwi = wwtp_to_bgwi.get(wwtp, {})
        if not bgwi:
            site_area_deltas.append(0.0)
            site_pop_deltas.append(0.0)
            continue

        covered_prev = covered.copy()
        bgs = [b for b in bgwi.keys() if b in covered.index]
        if not bgs:
            site_area_deltas.append(0.0)
            site_pop_deltas.append(0.0)
            continue
        wi = pd.Series(bgwi, dtype=float).reindex(bgs).fillna(0.0)

        umode = str(union_mode).lower()
        if umode == "binary":
            covered.loc[bgs] = np.maximum(covered.loc[bgs].values, (wi.values > 0).astype(float))
        elif umode == "prob_union":
            covered.loc[bgs] = 1.0 - (1.0 - covered.loc[bgs].values) * (1.0 - wi.values)
        else:
            covered.loc[bgs] = np.minimum(1.0, covered.loc[bgs].values + wi.values)

        delta_cov = (covered - covered_prev).clip(lower=0.0)
        site_area_deltas.append(float((bg_area.reindex(delta_cov.index).fillna(0.0) * delta_cov).sum()))
        if not bg_pop.empty:
            site_pop_deltas.append(float((bg_pop.reindex(delta_cov.index).fillna(0.0) * delta_cov).sum()))
        else:
            site_pop_deltas.append(0.0)

    # County-level mobility-linked area coverage fraction.
    county_area_covered = (bg_area.reindex(covered.index).fillna(0.0) * covered).groupby(bg_county.reindex(covered.index)).sum()
    county_area_cov_frac = (county_area_covered / county_area_total).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    threshold = float(county_meaningful_threshold)
    n_counties_reached = float((county_area_cov_frac > 0).sum())
    n_counties_contributing = float((county_area_cov_frac >= threshold).sum())

    # Effective-number evenness for marginal site contributions.
    def _evenness(vals: Sequence[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            return np.nan
        shares = arr / arr.sum()
        eff_n = 1.0 / np.sum(np.square(shares))
        denom = float(len(sel))
        return float(eff_n / denom) if denom > 0 else np.nan

    return {
        "n_mobility_counties_total": n_counties_total,
        "n_mobility_counties_reached": n_counties_reached,
        "mobility_counties_reached_frac": n_counties_reached / n_counties_total if n_counties_total > 0 else np.nan,
        "n_mobility_counties_contributing_5pct": n_counties_contributing,
        "mobility_counties_contributing_frac_5pct": n_counties_contributing / n_counties_total if n_counties_total > 0 else np.nan,
        "county_mobility_area_coverage_mean": float(county_area_cov_frac.mean()) if len(county_area_cov_frac) else np.nan,
        "county_mobility_area_coverage_median": float(county_area_cov_frac.median()) if len(county_area_cov_frac) else np.nan,
        "county_mobility_area_coverage_min": float(county_area_cov_frac.min()) if len(county_area_cov_frac) else np.nan,
        "site_marginal_area_evenness": _evenness(site_area_deltas),
        "site_marginal_pop_evenness": _evenness(site_pop_deltas),
        "site_marginal_area_top1_frac": float(max(site_area_deltas) / sum(site_area_deltas)) if sum(site_area_deltas) > 0 else np.nan,
        "site_marginal_area_top3_frac": float(sum(sorted(site_area_deltas, reverse=True)[:3]) / sum(site_area_deltas)) if sum(site_area_deltas) > 0 else np.nan,
    }



def _load_county_name_lookup(county_shp: Optional[str] = None) -> Dict[str, str]:
    """Best-effort county FIPS -> county name lookup for compact regional plots."""
    out: Dict[str, str] = {}
    cpath = _resolve_local_path(county_shp) if "_resolve_local_path" in globals() else county_shp
    if cpath and os.path.exists(str(cpath)) and gpd is not None:
        try:
            cg = gpd.read_file(cpath)
            cols_lower = {str(c).lower(): c for c in cg.columns}
            name_col = cols_lower.get("name") or cols_lower.get("namelsad") or cols_lower.get("county") or cols_lower.get("county_nam")
            if "geoid" in cols_lower:
                fips = cg[cols_lower["geoid"]].astype(str).str.zfill(5)
            elif "statefp" in cols_lower and "countyfp" in cols_lower:
                fips = cg[cols_lower["statefp"]].astype(str).str.zfill(2) + cg[cols_lower["countyfp"]].astype(str).str.zfill(3)
            elif "fips" in cols_lower:
                fips = cg[cols_lower["fips"]].astype(str).str.zfill(5)
            else:
                fips = pd.Series(dtype=str)
            if name_col is not None and len(fips) == len(cg):
                names = cg[name_col].astype(str).str.replace(" County", "", regex=False).str.strip().str.title()
                out = dict(zip(fips, names))
        except Exception:
            out = {}
    return out


def _county_area_coverage_fraction_for_order(
    selected_order: Sequence[str],
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    direction: str = "Destination",
    union_mode: str = "cap_sum",
    weight_from: str = "Volume",
    bg_weight_mode: str = "intensity_exp",
    tau: float = 2.0,
    eps_pop: float = 100.0,
) -> pd.Series:
    """
    Return county-level mobility-linked BG area coverage fractions for one portfolio.
    Each county receives equal weight later, so large metro counties do not dominate.
    """
    sel = [_canon(s) for s in selected_order]
    sel = [s for s in dict.fromkeys(sel) if s]
    if not sel:
        return pd.Series(dtype=float)

    links = load_bg_link_window(bg_link_dir, start_date, end_date, direction=direction)  # type: ignore[arg-type]
    if links is None or links.empty or "bg_fips" not in links.columns or "wwtp_clean" not in links.columns:
        return pd.Series(dtype=float)

    links = links.copy()
    links["bg_fips"] = links["bg_fips"].astype(str).str.strip()
    links["county_fips"] = links["bg_fips"].str.zfill(12).str[:5]
    links["wwtp_clean"] = links["wwtp_clean"].astype(str).str.strip().str.lower()

    if weight_from not in links.columns:
        return pd.Series(dtype=float)
    for c in ["Volume", "Area", "Population", weight_from]:
        if c in links.columns:
            links[c] = pd.to_numeric(links[c], errors="coerce").fillna(0.0)

    bg_pop = links.groupby("bg_fips")["Population"].max() if "Population" in links.columns else pd.Series(dtype=float)
    bg_area = links.groupby("bg_fips")["Area"].max() if "Area" in links.columns else pd.Series(dtype=float)
    bg_county = links.groupby("bg_fips")["county_fips"].first()
    if bg_area.empty:
        return pd.Series(dtype=float)

    county_area_total = bg_area.groupby(bg_county).sum()
    county_area_total = county_area_total[county_area_total > 0]
    if county_area_total.empty:
        return pd.Series(dtype=float)

    w_df = (
        links.groupby(["wwtp_clean", "bg_fips"], as_index=False)[weight_from]
        .sum()
        .rename(columns={weight_from: "w"})
    )

    mode = str(bg_weight_mode).strip().lower()
    if str(union_mode).lower() == "binary":
        w_df["wi"] = 1.0
    elif mode == "share":
        bg_total_w = w_df.groupby("bg_fips")["w"].sum().rename("bg_total_w")
        w_df = w_df.merge(bg_total_w, on="bg_fips", how="left")
        w_df["wi"] = np.where(w_df["bg_total_w"] > 0, w_df["w"] / w_df["bg_total_w"], 0.0)
    elif mode in ("intensity_cap", "intensity_exp"):
        pop_df = bg_pop.rename("bg_pop").reset_index()
        w_df = w_df.merge(pop_df, on="bg_fips", how="left")
        w_df["bg_pop"] = pd.to_numeric(w_df["bg_pop"], errors="coerce").fillna(0.0)
        denom = float(tau) * (w_df["bg_pop"].values + float(eps_pop))
        denom = np.where(denom > 0, denom, np.nan)
        x = w_df["w"].values / denom
        if mode == "intensity_cap":
            w_df["wi"] = np.minimum(1.0, np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0))
        else:
            w_df["wi"] = 1.0 - np.exp(-np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        w_df["wi"] = pd.to_numeric(w_df["wi"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        return pd.Series(dtype=float)

    wwtp_to_bgwi: Dict[str, Dict[str, float]] = {}
    for wwtp, sub in w_df.groupby("wwtp_clean"):
        wwtp_to_bgwi[str(wwtp)] = dict(zip(sub["bg_fips"].astype(str), sub["wi"].astype(float)))

    covered = pd.Series(0.0, index=bg_area.index, dtype=float)
    for wwtp in sel:
        bgwi = wwtp_to_bgwi.get(wwtp, {})
        if not bgwi:
            continue
        bgs = [b for b in bgwi.keys() if b in covered.index]
        if not bgs:
            continue
        wi = pd.Series(bgwi, dtype=float).reindex(bgs).fillna(0.0)
        umode = str(union_mode).lower()
        if umode == "binary":
            covered.loc[bgs] = np.maximum(covered.loc[bgs].values, (wi.values > 0).astype(float))
        elif umode == "prob_union":
            covered.loc[bgs] = 1.0 - (1.0 - covered.loc[bgs].values) * (1.0 - wi.values)
        else:
            covered.loc[bgs] = np.minimum(1.0, covered.loc[bgs].values + wi.values)

    county_area_covered = (bg_area.reindex(covered.index).fillna(0.0) * covered).groupby(bg_county.reindex(covered.index)).sum()
    county_frac = (county_area_covered / county_area_total).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    return county_frac.reindex(county_area_total.index).fillna(0.0)


def plot_level1_regional_area_reach_stacked(
    strategy_orders: Mapping[str, Sequence[str]],
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    out_path: str,
    direction: str = "Destination",
    union_mode: str = "cap_sum",
    weight_from: str = "Volume",
    bg_weight_mode: str = "intensity_exp",
    tau: float = 2.0,
    eps_pop: float = 100.0,
    county_shp: Optional[str] = None,
    top_n_regions: int = 8,
) -> None:
    """
    Stacked version of the county-balanced area-reach metric.

    Each county contributes coverage_fraction / number_of_counties to the total
    bar height. Therefore the total bar height equals the mean county-level
    mobility-linked area reach, not the population-weighted statewide total.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    desired = ["Population-only", "Spatial coverage", "Census metro/micro-area", "Ours Level 1"]
    alias = {
        "Spatial greedy": "Spatial coverage",
        "Greedy unique-area": "Spatial coverage",
        "Integrated Level 1": "Ours Level 1",
        "Ours L1": "Ours Level 1",
        "Existing network": "Existing 20",
        "Current network": "Existing 20",
        "Existing": "Existing 20",
    }

    cov_by_strategy: Dict[str, pd.Series] = {}
    for strategy, order in strategy_orders.items():
        sname = alias.get(str(strategy), str(strategy))
        if sname not in desired:
            continue
        cov = _county_area_coverage_fraction_for_order(
            selected_order=order,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            union_mode=union_mode,
            weight_from=weight_from,
            bg_weight_mode=bg_weight_mode,
            tau=tau,
            eps_pop=eps_pop,
        )
        if not cov.empty:
            cov_by_strategy[sname] = cov

    if not cov_by_strategy:
        print("[Level1][regional stack] skipped: no county coverage data")
        return

    counties = sorted(set().union(*[set(s.index) for s in cov_by_strategy.values()]))
    mat = pd.DataFrame({k: v.reindex(counties).fillna(0.0) for k, v in cov_by_strategy.items()}).T
    mat = mat.reindex([s for s in desired if s in mat.index])

    # Contribution to county-balanced score: each county has equal weight.
    n_counties = max(len(mat.columns), 1)
    contrib = mat / float(n_counties)

    # Keep the most visible counties and group the rest to avoid an unreadable legend.
    region_order = contrib.sum(axis=0).sort_values(ascending=False).index.tolist()
    top_regions = region_order[:int(top_n_regions)]
    plot_df = contrib[top_regions].copy()
    other_cols = [c for c in contrib.columns if c not in top_regions]
    if other_cols:
        plot_df["Other counties"] = contrib[other_cols].sum(axis=1)

    name_lookup = _load_county_name_lookup(county_shp)
    labels = [name_lookup.get(str(c), str(c)) for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=(5.35, 3.05))
    bottom = np.zeros(len(plot_df), dtype=float)
    x = np.arange(len(plot_df))
    for col, lab in zip(plot_df.columns, labels):
        vals = plot_df[col].values * 100.0
        ax.bar(x, vals, bottom=bottom * 100.0, width=0.58, label=lab)
        bottom += plot_df[col].values

    ax.set_xticks(x)
    ax.set_xticklabels([_short_strategy_label(s) for s in plot_df.index], fontsize=7.6)
    ax.set_ylabel("County-balanced\narea reach score", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_title("Regional composition of area reach", fontsize=8.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.8, title="County contribution", title_fontsize=7.0)
    fig.tight_layout(pad=0.55)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Level1][regional stack] saved: {out_path}")


def _derive_cbsa_metro_counties(
    county_shp: Optional[str] = None,
    cbsa_shp: Optional[str] = None,
) -> Set[str]:
    """
    Best-effort derivation of county FIPS belonging to metro/micro CBSA polygons.

    Used only for a Level-1 non-metro/peripheral reach diagnostic. This avoids
    needing a new health-region file in the current revision.
    """
    if gpd is None:
        return set()

    county_candidates = [
        county_shp,
        "../ZoneSelection/Input/Census/COCounty.shp",
        "../ZoneSelection/Input/Census/tl_2024_us_county/tl_2024_us_county.shp",
    ]
    cbsa_candidates = [
        cbsa_shp,
        "../ZoneSelection/Input/Census/tl_2024_us_cbsa/tl_2024_us_cbsa.shp",
    ]

    county_path = None
    for pth in county_candidates:
        rp = _resolve_local_path(pth) if "_resolve_local_path" in globals() else pth
        if rp and os.path.exists(str(rp)):
            county_path = str(rp)
            break

    cbsa_path = None
    for pth in cbsa_candidates:
        rp = _resolve_local_path(pth) if "_resolve_local_path" in globals() else pth
        if rp and os.path.exists(str(rp)):
            cbsa_path = str(rp)
            break

    if not county_path or not cbsa_path:
        return set()

    try:
        counties = gpd.read_file(county_path)
        cbsa = gpd.read_file(cbsa_path)
        if counties.empty or cbsa.empty:
            return set()

        # Colorado counties only when using national county file.
        cols_lower = {str(c).lower(): c for c in counties.columns}
        if "statefp" in cols_lower:
            counties = counties[counties[cols_lower["statefp"]].astype(str).str.zfill(2) == "08"].copy()

        if counties.crs is not None and cbsa.crs is not None:
            cbsa = cbsa.to_crs(counties.crs)

        # County FIPS.
        cols_lower = {str(c).lower(): c for c in counties.columns}
        if "geoid" in cols_lower:
            counties["_county_fips"] = counties[cols_lower["geoid"]].astype(str).str.zfill(5)
        elif "statefp" in cols_lower and "countyfp" in cols_lower:
            counties["_county_fips"] = (
                counties[cols_lower["statefp"]].astype(str).str.zfill(2)
                + counties[cols_lower["countyfp"]].astype(str).str.zfill(3)
            )
        elif "fips" in cols_lower:
            counties["_county_fips"] = counties[cols_lower["fips"]].astype(str).str.zfill(5)
        else:
            return set()

        # Use county centroids for a conservative county-to-CBSA assignment.
        pts = counties.copy()
        try:
            pts["geometry"] = counties.geometry.representative_point()
        except Exception:
            pts["geometry"] = counties.geometry.centroid

        joined = gpd.sjoin(
            pts[["_county_fips", "geometry"]],
            cbsa[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        return set(joined["_county_fips"].astype(str).str.zfill(5).tolist())
    except Exception as e:
        print(f"[Level1][nonmetro] Could not derive CBSA counties: {e}")
        return set()


def compute_cbsa_nonmetro_reach_stats(
    selected_order: Sequence[str],
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    direction: str = "Destination",
    union_mode: str = "cap_sum",
    weight_from: str = "Volume",
    bg_weight_mode: str = "intensity_exp",
    tau: float = 2.0,
    eps_pop: float = 100.0,
    cbsa_shp: Optional[str] = None,
    county_shp: Optional[str] = None,
) -> Dict[str, float]:
    """
    CBSA-based peripheral/non-metro reach diagnostic.

    This is the main Level-1 spatial/rural metric. It asks how much of the
    mobility-linked BG area and population outside CBSA counties is reached by
    the fixed-budget portfolio. It is not disease-specific and does not require
    a new health-region file.
    """
    sel = [_canon(s) for s in selected_order]
    sel = [s for s in dict.fromkeys(sel) if s]
    if not sel:
        return {}

    metro_counties = _derive_cbsa_metro_counties(county_shp=county_shp, cbsa_shp=cbsa_shp)

    links = load_bg_link_window(bg_link_dir, start_date, end_date, direction=direction)  # type: ignore[arg-type]
    if links is None or links.empty or "bg_fips" not in links.columns or "wwtp_clean" not in links.columns:
        return {}

    links = links.copy()
    links["bg_fips"] = links["bg_fips"].astype(str).str.strip()
    links["county_fips"] = links["bg_fips"].str.zfill(12).str[:5]
    links["wwtp_clean"] = links["wwtp_clean"].astype(str).str.strip().str.lower()

    if weight_from not in links.columns:
        return {}
    for c in ["Volume", "Area", "Population", weight_from]:
        if c in links.columns:
            links[c] = pd.to_numeric(links[c], errors="coerce").fillna(0.0)

    bg_pop = links.groupby("bg_fips")["Population"].max() if "Population" in links.columns else pd.Series(dtype=float)
    bg_area = links.groupby("bg_fips")["Area"].max() if "Area" in links.columns else pd.Series(dtype=float)
    bg_county = links.groupby("bg_fips")["county_fips"].first()
    if bg_area.empty:
        return {}

    if metro_counties:
        nonmetro_bgs = bg_county[~bg_county.astype(str).str.zfill(5).isin(metro_counties)].index
    else:
        # If CBSA county derivation fails, return NaN rather than silently using all counties.
        return {
            "nonmetro_area_reach_frac": np.nan,
            "nonmetro_pop_reach_frac": np.nan,
            "nonmetro_bg_reach_frac": np.nan,
            "n_nonmetro_bgs_total": np.nan,
            "n_nonmetro_bgs_reached": np.nan,
        }

    if len(nonmetro_bgs) == 0:
        return {
            "nonmetro_area_reach_frac": np.nan,
            "nonmetro_pop_reach_frac": np.nan,
            "nonmetro_bg_reach_frac": np.nan,
            "n_nonmetro_bgs_total": 0.0,
            "n_nonmetro_bgs_reached": 0.0,
        }

    w_df = (
        links.groupby(["wwtp_clean", "bg_fips"], as_index=False)[weight_from]
        .sum()
        .rename(columns={weight_from: "w"})
    )

    mode = str(bg_weight_mode).strip().lower()
    if str(union_mode).lower() == "binary":
        w_df["wi"] = 1.0
    elif mode == "share":
        bg_total_w = w_df.groupby("bg_fips")["w"].sum().rename("bg_total_w")
        w_df = w_df.merge(bg_total_w, on="bg_fips", how="left")
        w_df["wi"] = np.where(w_df["bg_total_w"] > 0, w_df["w"] / w_df["bg_total_w"], 0.0)
    elif mode in ("intensity_cap", "intensity_exp"):
        pop_df = bg_pop.rename("bg_pop").reset_index()
        w_df = w_df.merge(pop_df, on="bg_fips", how="left")
        w_df["bg_pop"] = pd.to_numeric(w_df["bg_pop"], errors="coerce").fillna(0.0)
        denom = float(tau) * (w_df["bg_pop"].values + float(eps_pop))
        denom = np.where(denom > 0, denom, np.nan)
        x = w_df["w"].values / denom
        if mode == "intensity_cap":
            w_df["wi"] = np.minimum(1.0, np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0))
        else:
            w_df["wi"] = 1.0 - np.exp(-np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        w_df["wi"] = pd.to_numeric(w_df["wi"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        return {}

    wwtp_to_bgwi: Dict[str, Dict[str, float]] = {}
    for wwtp, sub in w_df.groupby("wwtp_clean"):
        wwtp_to_bgwi[str(wwtp)] = dict(zip(sub["bg_fips"].astype(str), sub["wi"].astype(float)))

    covered = pd.Series(0.0, index=bg_area.index, dtype=float)
    for wwtp in sel:
        bgwi = wwtp_to_bgwi.get(wwtp, {})
        if not bgwi:
            continue
        bgs = [b for b in bgwi.keys() if b in covered.index]
        if not bgs:
            continue
        wi = pd.Series(bgwi, dtype=float).reindex(bgs).fillna(0.0)
        umode = str(union_mode).lower()
        if umode == "binary":
            covered.loc[bgs] = np.maximum(covered.loc[bgs].values, (wi.values > 0).astype(float))
        elif umode == "prob_union":
            covered.loc[bgs] = 1.0 - (1.0 - covered.loc[bgs].values) * (1.0 - wi.values)
        else:
            covered.loc[bgs] = np.minimum(1.0, covered.loc[bgs].values + wi.values)

    nonmetro_bgs = [b for b in nonmetro_bgs if b in covered.index]
    cov_nm = covered.reindex(nonmetro_bgs).fillna(0.0).clip(0.0, 1.0)
    area_nm = bg_area.reindex(nonmetro_bgs).fillna(0.0)
    pop_nm = bg_pop.reindex(nonmetro_bgs).fillna(0.0) if not bg_pop.empty else pd.Series(0.0, index=nonmetro_bgs)

    area_total = float(area_nm.sum())
    pop_total = float(pop_nm.sum())
    area_reached = float((area_nm * cov_nm).sum())
    pop_reached = float((pop_nm * cov_nm).sum())
    bg_reached = float((cov_nm > 0).sum())
    bg_total = float(len(nonmetro_bgs))

    return {
        "nonmetro_area_reach_frac": area_reached / area_total if area_total > 0 else np.nan,
        "nonmetro_pop_reach_frac": pop_reached / pop_total if pop_total > 0 else np.nan,
        "nonmetro_bg_reach_frac": bg_reached / bg_total if bg_total > 0 else np.nan,
        "n_nonmetro_bgs_total": bg_total,
        "n_nonmetro_bgs_reached": bg_reached,
    }


# -----------------------------------------------------------------------------
# Strategy evaluation and plotting
# -----------------------------------------------------------------------------


def _safe_last(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return float(df[col].iloc[-1])


def _site_centroid_xy_table(sewersheds_gdf=None) -> pd.DataFrame:
    """
    Return WWTP centroid coordinates for site-level CSV outputs.

    Columns:
      - wwtp_clean
      - x_center, y_center: projected centroid coordinates in EPSG:3857 meters
      - lon_center, lat_center: centroid coordinates in WGS84 degrees

    The projected x/y columns are useful for plotting and distance context;
    lon/lat are easier for manual map checks.
    """
    if gpd is None or sewersheds_gdf is None:
        return pd.DataFrame(columns=["wwtp_clean", "x_center", "y_center", "lon_center", "lat_center"])
    try:
        sg = sewersheds_gdf.copy()
        if "wwtp" in sg.columns:
            sg["wwtp_clean"] = sg["wwtp"].map(_canon)
        else:
            sg["wwtp_clean"] = sg.index.map(_canon)

        sg = sg.dropna(subset=["geometry"]).copy()
        sg = sg[sg["wwtp_clean"].astype(str).str.len() > 0]
        if sg.empty:
            return pd.DataFrame(columns=["wwtp_clean", "x_center", "y_center", "lon_center", "lat_center"])

        if sg.crs is None:
            sg_wgs = sg.set_crs(epsg=4326, allow_override=True)
        else:
            sg_wgs = sg.to_crs(epsg=4326)
        sg_xy = sg_wgs.to_crs(epsg=3857)

        c_xy = sg_xy.geometry.centroid
        c_ll = sg_xy.geometry.centroid.to_crs(epsg=4326)

        out = pd.DataFrame({
            "wwtp_clean": sg_xy["wwtp_clean"].astype(str).values,
            "x_center": c_xy.x.astype(float).values,
            "y_center": c_xy.y.astype(float).values,
            "lon_center": c_ll.x.astype(float).values,
            "lat_center": c_ll.y.astype(float).values,
        })
        out = out.drop_duplicates("wwtp_clean", keep="first")
        return out
    except Exception:
        return pd.DataFrame(columns=["wwtp_clean", "x_center", "y_center", "lon_center", "lat_center"])


def _add_xy_columns(df: pd.DataFrame, xy_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach centroid coordinate columns to any site-level dataframe with wwtp_clean."""
    if df is None or df.empty or "wwtp_clean" not in df.columns:
        return df
    if xy_table is None or xy_table.empty or "wwtp_clean" not in xy_table.columns:
        for c in ["x_center", "y_center", "lon_center", "lat_center"]:
            if c not in df.columns:
                df[c] = np.nan
        return df
    base = df.copy()
    base["wwtp_clean"] = base["wwtp_clean"].map(_canon)
    xy = xy_table.copy()
    xy["wwtp_clean"] = xy["wwtp_clean"].map(_canon)
    # Avoid duplicate coordinate columns if called twice.
    base = base.drop(columns=[c for c in ["x_center", "y_center", "lon_center", "lat_center"] if c in base.columns], errors="ignore")
    return base.merge(xy, on="wwtp_clean", how="left")


def _write_strategy_order_csv(
    strategy_orders: Mapping[str, Sequence[str]],
    out_dir: str,
    label: str,
    xy_table: Optional[pd.DataFrame] = None,
) -> str:
    """Write strategy-order outputs.

    The original wide CSV is retained for backward compatibility. A new long
    CSV is also written with one row per strategy-site and centroid coordinates.
    """
    max_len = max((len(v) for v in strategy_orders.values()), default=0)
    out = pd.DataFrame({k: pd.Series(list(v)) for k, v in strategy_orders.items()})
    path = os.path.join(out_dir, f"level1_strategy_orders_{label}.csv")
    out.to_csv(path, index=False, encoding="utf-8")

    rows = []
    for strategy, order in strategy_orders.items():
        for i, site in enumerate(order, start=1):
            rows.append({
                "strategy": strategy,
                "order": i,
                "wwtp_clean": _canon(site),
            })
    long_df = pd.DataFrame(rows)
    long_df = _add_xy_columns(long_df, xy_table)
    long_path = os.path.join(out_dir, f"level1_strategy_orders_long_{label}.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8")
    print(f"[Output] strategy-order long CSV with XY saved: {long_path}")
    return path




def compute_covid_risk_benefit_stats(
    selected_order: Sequence[str],
    covid_risk_series: Optional[pd.Series],
    features: pd.DataFrame,
) -> Dict[str, float]:
    """
    Retrospective health-risk benefit.

    This uses the monthly COVID import/export risk proxy from the weekly risk
    summary (for example the mean of import_risk_COVID and export_risk_COVID).
    It is NOT wastewater viral concentration and is NOT used as a Level-1
    selection objective. It is a secondary health-relevance check applied
    uniformly to all final portfolios.
    """
    feat = _prepare_features(features)
    sel = [_canon(s) for s in selected_order if _canon(s) in feat.index]
    sel = list(dict.fromkeys(sel))

    if covid_risk_series is None or len(covid_risk_series) == 0:
        return {
            "covid_risk_signal_cum": np.nan,
            "covid_risk_signal_frac": np.nan,
            "high_covid_risk_sites_selected": np.nan,
            "high_covid_risk_sites_selected_frac_of_all_high_covid_risk": np.nan,
        }

    risk = pd.to_numeric(covid_risk_series, errors="coerce")
    risk.index = risk.index.map(_canon)
    risk = risk.reindex(feat.index).fillna(0.0).clip(lower=0.0)

    total = float(risk.sum())
    sel_total = float(risk.reindex(sel).fillna(0.0).sum())
    frac = sel_total / total if total > 0 else np.nan

    q75 = float(risk.quantile(0.75)) if risk.notna().any() else np.nan
    if not np.isnan(q75):
        high_all = set(risk[risk >= q75].index.tolist())
    else:
        high_all = set()
    high_sel = [w for w in sel if w in high_all]
    high_frac = len(high_sel) / len(high_all) if high_all else np.nan

    return {
        "covid_risk_signal_cum": sel_total,
        "covid_risk_signal_frac": frac,
        "high_covid_risk_sites_selected": float(len(high_sel)),
        "high_covid_risk_sites_selected_frac_of_all_high_covid_risk": high_frac,
    }



def _find_primary_strategy_key(
    strategy_orders: Mapping[str, Sequence[str]],
    primary_strategy: str = "Ours Level 1",
) -> str:
    """Locate the Ours/primary strategy key robustly."""
    keys = list(strategy_orders.keys())
    if primary_strategy in strategy_orders:
        return primary_strategy
    for k in keys:
        kk = str(k).lower()
        if "ours" in kk or "integrated" in kk:
            return k
    return keys[-1] if keys else primary_strategy


def _unique_clean_order(order: Sequence[str], valid_index: Optional[Iterable[str]] = None) -> List[str]:
    """Canonicalize an ordered WWTP list and remove duplicates while preserving order."""
    valid = set(_canon(x) for x in valid_index) if valid_index is not None else None
    out: List[str] = []
    seen = set()
    for x in order:
        k = _canon(x)
        if not k or k in seen:
            continue
        if valid is not None and k not in valid:
            continue
        out.append(k)
        seen.add(k)
    return out


def _write_pairwise_overlap_summary(
    strategy_orders: Mapping[str, Sequence[str]],
    out_dir: str,
    label: str,
    primary_strategy: str = "Ours Level 1",
    output_prefix: str = "level1_overlap",
    primary_order: Optional[Sequence[str]] = None,
    primary_label: Optional[str] = None,
    valid_index: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Write a simple pairwise portfolio-overlap summary.

    For Level 1:
      primary = Ours Level 1
      comparators = population-only / spatial / CBSA / etc.

    For Level 2:
      primary = Level-2 TX-RAW top-N sites
      comparators = all Level-1 strategies, including Ours Level 1.

    Outputs:
      {output_prefix}_summary_{label}.csv
      {output_prefix}_sites_{label}.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    clean_orders = {
        str(k): _unique_clean_order(v, valid_index=valid_index)
        for k, v in strategy_orders.items()
    }

    if primary_order is None:
        primary_key = _find_primary_strategy_key(clean_orders, primary_strategy=primary_strategy)
        primary = clean_orders.get(primary_key, [])
        primary_name = primary_label or str(primary_key)
    else:
        primary = _unique_clean_order(primary_order, valid_index=valid_index)
        primary_name = primary_label or primary_strategy

    pset = set(primary)
    rows = []
    site_rows = []

    for comp_name, comp_order in clean_orders.items():
        # For Level 1 overlap, skip Ours-vs-Ours. For Level 2, keep all comparators.
        if primary_order is None and str(comp_name) == str(primary_name):
            continue

        cset = set(comp_order)
        overlap = sorted(pset & cset)
        primary_only = sorted(pset - cset)
        comparator_only = sorted(cset - pset)
        union = pset | cset

        rows.append({
            "comparison": f"{primary_name} vs {comp_name}",
            "primary_strategy": primary_name,
            "comparator_strategy": comp_name,
            "n_primary": int(len(pset)),
            "n_comparator": int(len(cset)),
            "n_overlap": int(len(overlap)),
            "primary_overlap_frac": (len(overlap) / len(pset)) if pset else np.nan,
            "comparator_overlap_frac": (len(overlap) / len(cset)) if cset else np.nan,
            "jaccard": (len(overlap) / len(union)) if union else np.nan,
            "overlap_sites": "; ".join(overlap),
            "primary_only_sites": "; ".join(primary_only),
            "comparator_only_sites": "; ".join(comparator_only),
        })

        for s in sorted(union):
            site_rows.append({
                "primary_strategy": primary_name,
                "comparator_strategy": comp_name,
                "wwtp_clean": s,
                "in_primary": s in pset,
                "in_comparator": s in cset,
                "overlap": (s in pset and s in cset),
            })

    summary = pd.DataFrame(rows)
    site_df = pd.DataFrame(site_rows)
    summary_path = os.path.join(out_dir, f"{output_prefix}_summary_{label}.csv")
    sites_path = os.path.join(out_dir, f"{output_prefix}_sites_{label}.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    site_df.to_csv(sites_path, index=False, encoding="utf-8")
    print(f"[Overlap] summary saved: {summary_path}")
    print(f"[Overlap] site detail saved: {sites_path}")
    return summary


def _build_level2_txraw_top_order(
    diffusion_csv: Optional[str],
    features_index: Iterable[str],
    total_N: int,
) -> List[str]:
    """
    Build a Level-2 top-N order from TX-RAW contribution scores.
    This is used only for overlap diagnostics; it does not alter Level-1 selection.
    """
    if not diffusion_csv or not os.path.exists(str(diffusion_csv)):
        return []
    try:
        ddf = pd.read_csv(diffusion_csv)
        name_col = next((c for c in ddf.columns if str(c).lower() in ["wwtp", "site", "name", "wwtp_clean"]), None)
        score_col = next((c for c in ddf.columns if "score" in str(c).lower()), None)
        if name_col is None:
            return []

        valid = set(_canon(x) for x in features_index)
        ddf = ddf.copy()
        ddf["_wwtp_clean"] = ddf[name_col].map(_canon)
        ddf = ddf[ddf["_wwtp_clean"].isin(valid)]
        if ddf.empty:
            return []

        if score_col is not None:
            ddf["_txraw_score"] = pd.to_numeric(ddf[score_col], errors="coerce")
            ddf = ddf.dropna(subset=["_txraw_score"]).sort_values("_txraw_score", ascending=False, kind="mergesort")
        else:
            ddf["_txraw_score"] = np.arange(len(ddf), 0, -1)

        out: List[str] = []
        seen = set()
        for w in ddf["_wwtp_clean"].tolist():
            if w and w not in seen:
                out.append(w)
                seen.add(w)
            if len(out) >= int(total_N):
                break
        return out
    except Exception as e:
        print(f"[Level2-Overlap][WARN] Could not build TX-RAW top order: {e}")
        return []




def _load_txraw_scores_for_diagnostic(
    diffusion_csv: Optional[str],
    valid_index: Iterable[str],
) -> pd.DataFrame:
    """Load Level-2 TX-RAW scores as a site-level diagnostic table."""
    if not diffusion_csv or not os.path.exists(str(diffusion_csv)):
        return pd.DataFrame()
    try:
        ddf = pd.read_csv(diffusion_csv)
        name_col = next((c for c in ddf.columns if str(c).lower() in {"wwtp", "site", "name", "wwtp_clean"}), None)
        score_col = next((c for c in ddf.columns if "score" in str(c).lower()), None)
        if name_col is None or score_col is None:
            return pd.DataFrame()

        valid = set(_canon(x) for x in valid_index)
        out = ddf[[name_col, score_col]].copy()
        out["wwtp_clean"] = out[name_col].map(_canon)
        out["level2_txraw_score"] = pd.to_numeric(out[score_col], errors="coerce")
        out = out.dropna(subset=["wwtp_clean", "level2_txraw_score"])
        out = out[out["wwtp_clean"].isin(valid)]
        if out.empty:
            return pd.DataFrame()

        out = out.sort_values("level2_txraw_score", ascending=False, kind="mergesort")
        out = out.drop_duplicates("wwtp_clean")
        out["level2_txraw_rank"] = out["level2_txraw_score"].rank(ascending=False, method="min")
        return out[["wwtp_clean", "level2_txraw_score", "level2_txraw_rank"]]
    except Exception as e:
        print(f"[TXRAW-Rank][WARN] Failed to load TX-RAW scores: {e}")
        return pd.DataFrame()


def _write_level2_score_vs_rank_diagnostic(
    features: pd.DataFrame,
    sewersheds_gdf,
    diffusion_csv: Optional[str],
    out_dir: str,
    label: str,
    strategy_orders: Optional[Mapping[str, Sequence[str]]] = None,
    total_N: int = 20,
    extra_txraw_csvs: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Diagnostic: compare Level-2 TX-RAW scores with rankings from simple Level-1 methods.

    Outputs:
      level2_txraw_vs_pop_spatial_rank_<label>.csv
      level2_txraw_vs_pop_spatial_rank_<label>.png

    Only population-only and Ours L1 ranks are shown. CBSA is omitted because
    the CBSA strategy is a representative-site allocation plus refill rather than a
    continuous full ranking that is as straightforward to interpret.
    """
    os.makedirs(out_dir, exist_ok=True)
    feat = _prepare_features(features)
    tx = _load_txraw_scores_for_diagnostic(diffusion_csv, valid_index=feat.index)
    if tx.empty:
        return pd.DataFrame()

    rank_orders: Dict[str, List[str]] = {}
    rank_orders["Population-only"] = _full_population_order(feat)

    # Compare Level-2 high-contribution scores with the proposed Level-1 order.
    # For the diagnostic plot we use a COMPLETE Ours L1 rank: selected sites remain first,
    # and the remaining candidates are appended by the same fallback full-order extension
    # used elsewhere in this script. This makes the Ours panel directly comparable to the
    # population panel and allows both overlap (blue) and missed high-contribution sites
    # (orange) to be shown on the same complete x-axis ranking.
    ours_order = None
    if strategy_orders:
        for key in ["Ours L1", "Ours Level 1", "Ours"]:
            if key in strategy_orders and strategy_orders.get(key) is not None:
                ours_order = strategy_orders.get(key)
                break
    if ours_order is not None:
        rank_orders["Ours L1"] = _full_ours_level1_order(feat, ours_order)

    if not rank_orders:
        return pd.DataFrame()

    out = tx.copy()

    # Add optional Level-2 TX-RAW scores for other pathogens to the CSV only.
    # These are not used in the visualization or selection; they help compare
    # whether the disease-dynamic contribution pattern is COVID-specific.
    if extra_txraw_csvs:
        for disease_name, disease_csv in extra_txraw_csvs.items():
            if not disease_csv:
                continue
            extra = _load_txraw_scores_for_diagnostic(disease_csv, valid_index=feat.index)
            if extra is None or extra.empty:
                continue
            dslug = re.sub(r"[^0-9a-zA-Z]+", "_", str(disease_name).strip().lower()).strip("_")
            extra = extra.rename(columns={
                "level2_txraw_score": f"level2_{dslug}_txraw_score",
                "level2_txraw_rank": f"level2_{dslug}_txraw_rank",
            })
            out = out.merge(extra[["wwtp_clean", f"level2_{dslug}_txraw_score", f"level2_{dslug}_txraw_rank"]],
                            on="wwtp_clean", how="left")

    for method, order in rank_orders.items():
        full_order = _unique_clean_order(order, valid_index=feat.index)
        rank_map = {w: i + 1 for i, w in enumerate(full_order)}
        col = re.sub(r"[^0-9a-zA-Z]+", "_", str(method).strip().lower()).strip("_") + "_rank"
        out[col] = out["wwtp_clean"].map(rank_map).astype(float)

    if "pop_served" in feat.columns:
        pop_series = pd.to_numeric(feat["pop_served"], errors="coerce")
        out["population_served"] = out["wwtp_clean"].map(pop_series.to_dict())
        out["population_served_rank"] = out["population_served"].rank(ascending=False, method="min")

    out["top_level2_txraw"] = out["level2_txraw_rank"] <= total_N

    csv_path = os.path.join(out_dir, f"level2_txraw_vs_rank_{label}.csv")
    out.sort_values("level2_txraw_rank").to_csv(csv_path, index=False, encoding="utf-8")

    plot_methods = []
    if "population_only_rank" in out.columns:
        plot_methods.append(("Population-only rank", "population_only_rank"))
    if "ours_l1_rank" in out.columns:
        plot_methods.append(("Ours L1 rank", "ours_l1_rank"))

    if plot_methods:
        ncols = len(plot_methods)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, (title, rank_col) in zip(axes.ravel(), plot_methods):
            top_x = out[rank_col] <= total_N
            top_y = out["top_level2_txraw"].fillna(False).astype(bool)
            overlap = top_x & top_y
            tx_only = top_y & (~top_x)
            other = ~top_y

            # Blue + orange together represent the full Level-2 high-contribution top-N set.
            # Blue = high-contribution sites also ranked in the x-axis top-N; orange = high-contribution sites not ranked in the x-axis top-N.
            ax.scatter(
                out.loc[other, rank_col], out.loc[other, "level2_txraw_score"],
                s=30, c="#bdbdbd", alpha=0.55, edgecolors="none", zorder=1
            )
            ax.scatter(
                out.loc[overlap, rank_col], out.loc[overlap, "level2_txraw_score"],
                s=72, c="#1f78b4", alpha=0.95, edgecolors="white", linewidths=0.70, zorder=3
            )
            ax.scatter(
                out.loc[tx_only, rank_col], out.loc[tx_only, "level2_txraw_score"],
                s=78, c="#d95f02", alpha=0.95, edgecolors="white", linewidths=0.70, zorder=4
            )

            ax.axvline(total_N + 0.5, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_title(title, fontsize=13.5)
            ax.set_xlabel(f"{title} (1 = highest priority)", fontsize=15)
            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.tick_params(axis="both", labelsize=10.5)

            # Label Level-2 top-N sites not captured by the x-axis top-N method.


            corr = out[[rank_col, "level2_txraw_rank"]].corr(method="spearman").iloc[0, 1]
            overlap_n = int(overlap.sum())
            missed_n = int(tx_only.sum())
            ax.text(
                0.98, 0.97,
                f"Spearman ρ={corr:.2f}\nOverlap={overlap_n}/{total_N}\nHigh-contribution only={missed_n}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=15,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"),
            )

        axes[0].set_ylabel("Level-2 contribution score", fontsize=15)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f78b4",
                   markeredgecolor="white", markersize=9.5, label=f"High-contribution sites also ranked in top {total_N}"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d95f02",
                   markeredgecolor="white", markersize=9.5, label=f"High-contribution sites not ranked in top {total_N}"),

        ]
        # Put a shared legend below the figure, outside the panels.
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            fontsize=15,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            handletextpad=0.6,
            columnspacing=1.4,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig_path = os.path.join(out_dir, f"level2_txraw_vs_rank_{label}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[TXRAW-Rank] plot saved: {fig_path}")

    print(f"[TXRAW-Rank] CSV saved: {csv_path}")
    return out



def _write_level2_all_pathogen_txraw_scores(
    features: pd.DataFrame,
    out_dir: str,
    label: str,
    txraw_csvs: Mapping[str, str],
) -> pd.DataFrame:
    """
    Export a site-level CSV with Level-2 TX-RAW scores/ranks for all supplied pathogens.
    This is CSV-only; it does not change any figure or selection logic.
    """
    os.makedirs(out_dir, exist_ok=True)
    feat = _prepare_features(features)
    base = pd.DataFrame({"wwtp_clean": list(feat.index)})

    if "wwtp" in feat.columns:
        base["wwtp"] = feat["wwtp"].reindex(feat.index).astype(str).values
    else:
        base["wwtp"] = base["wwtp_clean"]

    if "pop_served" in feat.columns:
        base["population_served"] = pd.to_numeric(feat["pop_served"], errors="coerce").reindex(feat.index).values
        base["population_served_rank"] = pd.Series(base["population_served"]).rank(ascending=False, method="min").values

    for disease_name, csv_path in (txraw_csvs or {}).items():
        if not csv_path:
            continue
        dslug = re.sub(r"[^0-9a-zA-Z]+", "_", str(disease_name).strip().lower()).strip("_")
        scores = _load_txraw_scores_for_diagnostic(csv_path, valid_index=feat.index)
        if scores is None or scores.empty:
            continue
        scores = scores.rename(columns={
            "level2_txraw_score": f"level2_{dslug}_txraw_score",
            "level2_txraw_rank": f"level2_{dslug}_txraw_rank",
        })
        base = base.merge(scores[["wwtp_clean", f"level2_{dslug}_txraw_score", f"level2_{dslug}_txraw_rank"]],
                          on="wwtp_clean", how="left")

    out_path = os.path.join(out_dir, f"level2_txraw_scores_all_pathogens_{label}.csv")
    base.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[TXRAW-All] all-pathogen TX-RAW score CSV saved: {out_path}")
    return base



def evaluate_level1_strategy_orders(
    strategy_orders: Mapping[str, Sequence[str]],
    features: pd.DataFrame,
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    out_dir: str,
    label: str,
    direction: str = "Destination",
    union_mode: str = "cap_sum",
    weight_from: str = "Volume",
    bg_weight_mode: str = "intensity_exp",
    tau: float = 2.0,
    eps_pop: float = 100.0,
    pop_served_total_override: Optional[float] = None,
    G=None,
    sewersheds_gdf=None,
    ej_scores: Optional[pd.Series] = None,
    covid_risk_series: Optional[pd.Series] = None,
    utility_county_csv: Optional[str] = None,
    metro_counties: Optional[Iterable[str]] = None,
    small_component_max_size: int = 2,
    cbsa_shp: Optional[str] = None,
    make_plots: bool = True,
    level2_diffusion_csv: Optional[str] = None,
    level2_extra_diffusion_csvs: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Evaluate all Level-1 strategies and write CSV + figures.

    Outputs
    -------
    level1_strategy_orders_{label}.csv
    level1_cumulative_{label}_{strategy}.csv
    level1_strategy_summary_{label}.csv
    heatmap_{label}.png
    level1_cumulative_overlay_{label}.png
    spatial_map_{label}.png  (when sewersheds_gdf is provided)
    """
    os.makedirs(out_dir, exist_ok=True)
    feat = _prepare_features(features)
    pop_served = pd.to_numeric(feat.get("pop_served", pd.Series(0.0, index=feat.index)), errors="coerce").fillna(0.0)
    xy_table = _site_centroid_xy_table(sewersheds_gdf)

    # Canonicalize and trim duplicates while preserving order.
    clean_orders: Dict[str, List[str]] = {}
    for strategy, order in strategy_orders.items():
        vals = []
        for s in order:
            key = _canon(s)
            if key and key in feat.index and key not in vals:
                vals.append(key)
        clean_orders[strategy] = vals

    _write_strategy_order_csv(clean_orders, out_dir, label, xy_table=xy_table)

    # Simple portfolio-overlap diagnostics requested for reviewer response.
    # Level 1: Ours Level 1 vs each simple baseline.
    _write_pairwise_overlap_summary(
        strategy_orders=clean_orders,
        out_dir=out_dir,
        label=label,
        primary_strategy="Ours Level 1",
        output_prefix="level1_overlap",
        valid_index=feat.index,
    )

    # Level 2: top-N disease-dynamic TX-RAW sites vs each Level-1 portfolio.
    # This is diagnostic only and does not change Level-1 selection.
    if level2_diffusion_csv:
        ours_key = _find_primary_strategy_key(clean_orders, primary_strategy="Ours Level 1")
        n_level2 = len(clean_orders.get(ours_key, [])) if ours_key in clean_orders else max(len(v) for v in clean_orders.values())
        level2_order = _build_level2_txraw_top_order(
            diffusion_csv=level2_diffusion_csv,
            features_index=feat.index,
            total_N=n_level2,
        )
        if level2_order:
            _write_pairwise_overlap_summary(
                strategy_orders=clean_orders,
                out_dir=out_dir,
                label=label,
                primary_strategy=f"Level 2 TX-RAW top {n_level2}",
                primary_order=level2_order,
                primary_label=f"Level 2 TX-RAW top {n_level2}",
                output_prefix="level2_overlap",
                valid_index=feat.index,
            )

            # Level-2 score-vs-rank diagnostic for continuous simple rankings.
            # CBSA is omitted because it is an allocation/refill strategy rather than
            # a directly interpretable full rank sequence.
            extra_csvs = dict(level2_extra_diffusion_csvs or {})
            _write_level2_score_vs_rank_diagnostic(
                features=feat,
                sewersheds_gdf=sewersheds_gdf,
                diffusion_csv=level2_diffusion_csv,
                out_dir=out_dir,
                label=label,
                strategy_orders=strategy_orders,
                total_N=n_level2,
                extra_txraw_csvs=extra_csvs,
            )

            all_txraw_csvs = {"covid": level2_diffusion_csv}
            all_txraw_csvs.update(extra_csvs)
            _write_level2_all_pathogen_txraw_scores(
                features=feat,
                out_dir=out_dir,
                label=label,
                txraw_csvs=all_txraw_csvs,
            )

    # Export CBSA baseline source detail:
    # which selected sites came from official CBSA representatives versus
    # non-mobility farthest-first refill.
    try:
        if (
            "Census metro/micro-area" in clean_orders
            and sewersheds_gdf is not None
            and cbsa_shp
        ):
            cbsa_order_check, cbsa_source_detail = build_cbsa_metro_micro_order(
                features=feat,
                sewersheds_gdf=sewersheds_gdf,
                cbsa_shp=cbsa_shp,
                total_N=len(clean_orders.get("Census metro/micro-area", [])),
                return_details=True,
            )
            if cbsa_source_detail is not None and not cbsa_source_detail.empty:
                cbsa_source_detail = _add_xy_columns(cbsa_source_detail, xy_table)
                cbsa_source_detail.to_csv(
                    os.path.join(out_dir, f"cbsa_sources_{label}.csv"),
                    index=False,
                    encoding="utf-8",
                )
                print(f"[CBSA] selection-source CSV saved: {os.path.join(out_dir, f'cbsa_sources_{label}.csv')}")
    except Exception as e:
        print(f"[CBSA][WARN] Could not export CBSA selection-source CSV: {e}")

    rows = []
    cumulative_tables: Dict[str, pd.DataFrame] = {}

    for strategy, order in clean_orders.items():
        if not order:
            continue
        avg_rank = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order)
        df_cum = build_unique_cumulative_table_bg(
            avg_rank=avg_rank,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,  # type: ignore[arg-type]
            union_mode=union_mode,  # type: ignore[arg-type]
            weight_from=weight_from,  # type: ignore[arg-type]
            bg_weight_mode=bg_weight_mode,  # type: ignore[arg-type]
            tau=tau,
            eps_pop=eps_pop,
            pop_served=pop_served,
            pop_served_total_override=pop_served_total_override,
        )
        if df_cum.empty:
            # Fallback: additive only, so the comparison still writes something.
            sub = feat.reindex(order).fillna(0.0)
            df_cum = pd.DataFrame({
                "wwtp_clean": order,
                "k_sites": np.arange(1, len(order) + 1),
                "pop_served_delta": pd.to_numeric(sub.get("pop_served", 0), errors="coerce").fillna(0.0).values,
                "od_volume_total_delta": pd.to_numeric(sub.get("od_volume_total", 0), errors="coerce").fillna(0.0).values,
                "pop_reached_unique_delta": pd.to_numeric(sub.get("pop_covered_by_od", 0), errors="coerce").fillna(0.0).values,
                "area_reached_unique_delta": pd.to_numeric(sub.get("area_reached", 0), errors="coerce").fillna(0.0).values,
            })
            for c in ["pop_served", "od_volume_total", "pop_reached_unique", "area_reached_unique"]:
                dcol = f"{c}_delta"
                ccol = f"{c}_cum"
                if dcol in df_cum.columns:
                    df_cum[ccol] = df_cum[dcol].cumsum()

        df_cum = _add_xy_columns(df_cum, xy_table)
        cumulative_tables[strategy] = df_cum
        safe_strategy = strategy.lower().replace("/", "_").replace(" ", "_").replace(":", "")
        os.makedirs(out_dir, exist_ok=True)

        # Shorten per-strategy filenames to avoid Windows path-length errors.
        # The output folder can already be long; a filename like
        # level1_cumulative_2024-01_level1_N20_census_metro_micro-area.csv
        # may exceed the Windows path limit even when out_dir exists.
        short_strategy_map = {
            "population_only": "pop",
            "spatial_coverage": "spatial",
            "census_metro_micro_area": "cbsa",
            "census_metro_micro-area": "cbsa",
            "ours_level_1": "ours",
            "integrated_level_1": "ours",
        }
        safe_strategy_short = short_strategy_map.get(safe_strategy, safe_strategy[:18])
        df_cum.to_csv(
            os.path.join(out_dir, f"cum_{safe_strategy_short}.csv"),
            index=False,
            encoding="utf-8",
        )

        # Portfolio summary at final N.
        row = {
            "strategy": strategy,
            "n_selected": float(len(order)),
            "pop_served_cum": _safe_last(df_cum, "pop_served_cum"),
            "pop_served_cum_frac": _safe_last(df_cum, "pop_served_cum_frac"),
            "pop_reached_unique_cum": _safe_last(df_cum, "pop_reached_unique_cum"),
            "pop_reached_unique_cum_frac": _safe_last(df_cum, "pop_reached_unique_cum_frac"),
            "area_reached_unique_cum": _safe_last(df_cum, "area_reached_unique_cum"),
            "area_reached_unique_cum_frac": _safe_last(df_cum, "area_reached_unique_cum_frac"),
            "od_volume_total_cum": _safe_last(df_cum, "od_volume_total_cum"),
            "od_volume_total_cum_frac": _safe_last(df_cum, "od_volume_total_cum_frac"),
        }
        row.update(compute_spatial_ej_stats(
            selected_order=order,
            features=feat,
            G=G,
            ej_scores=ej_scores,
            utility_county_csv=utility_county_csv,
            metro_counties=metro_counties,
            small_component_max_size=small_component_max_size,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
        ))
        row.update(compute_mobility_county_contribution_stats(
            selected_order=order,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            union_mode=union_mode,
            weight_from=weight_from,
            bg_weight_mode=bg_weight_mode,
            tau=tau,
            eps_pop=eps_pop,
        ))
        row.update(compute_cbsa_nonmetro_reach_stats(
            selected_order=order,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            union_mode=union_mode,
            weight_from=weight_from,
            bg_weight_mode=bg_weight_mode,
            tau=tau,
            eps_pop=eps_pop,
            cbsa_shp=cbsa_shp,
            county_shp="../ZoneSelection/Input/Census/COCounty.shp",
        ))
        row.update(compute_covid_risk_benefit_stats(
            selected_order=order,
            covid_risk_series=covid_risk_series,
            features=feat,
        ))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    # Reviewer-facing balance diagnostics. These are not selection objectives;
    # they summarize whether a strategy is balanced across core mobility-reach
    # and design-representation dimensions rather than optimized for only one metric.
    core_cols = [
        "pop_served_cum_frac",
        "pop_reached_unique_cum_frac",
        "area_reached_unique_cum_frac",
        "od_volume_total_cum_frac",  # activity intensity, not unique coverage
    ]
    core_cols = [c for c in core_cols if c in summary.columns]
    if core_cols:
        summary["balance_core_mean"] = summary[core_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        summary["balance_core_min"] = summary[core_cols].apply(pd.to_numeric, errors="coerce").min(axis=1, skipna=True)

    design_cols = [
        "area_reached_unique_cum_frac",
        "od_volume_total_cum_frac",
        "mobility_counties_contributing_frac_5pct",
        "site_marginal_area_evenness",
        "component_representation_frac",
    ]
    design_cols = [c for c in design_cols if c in summary.columns and pd.to_numeric(summary[c], errors="coerce").notna().any()]
    if design_cols:
        design_vals = summary[design_cols].apply(pd.to_numeric, errors="coerce")
        summary["minimum_design_benefit_score"] = design_vals.min(axis=1, skipna=True)
        summary["mean_design_benefit_score"] = design_vals.mean(axis=1, skipna=True)
        summary["design_benefit_metric_count"] = float(len(design_cols))

    # Backward-compatible column name used by the plotting helper.
    # This excludes COVID-risk because Level 1 is a backbone design step;
    # disease-specific signals are treated as optional Level 2 / SI diagnostics.
    if "minimum_design_benefit_score" in summary.columns:
        summary["minimum_benefit_score"] = summary["minimum_design_benefit_score"]
    elif core_cols:
        summary["minimum_benefit_score"] = summary[core_cols].apply(pd.to_numeric, errors="coerce").min(axis=1, skipna=True)

    # Separate small table for the retrospective health-risk check.
    # This is kept separate from the main Level-1 design metrics.
    health_cols = [
        "strategy",
        "covid_risk_signal_cum",
        "covid_risk_signal_frac",
        "high_covid_risk_sites_selected",
        "high_covid_risk_sites_selected_frac_of_all_high_covid_risk",
    ]
    health_cols = [c for c in health_cols if c in summary.columns]
    if len(health_cols) > 1:
        os.makedirs(out_dir, exist_ok=True)
        summary[health_cols].to_csv(
            os.path.join(out_dir, f"health_covid_{label}.csv"),
            index=False,
            encoding="utf-8",
        )

    summary_path = os.path.join(out_dir, f"level1_strategy_summary_{label}.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    if make_plots:
        # Main-text fixed-budget benefit chart at N=21. Write this first so it
        # is not lost if optional SI/map products fail later.
        plot_level1_portfolio_bar_summary(
            summary,
            out_path=os.path.join(out_dir, f"level1_portfolio_benefit_bar_N21_{label}.png"),
        )

        plot_level1_strategy_heatmap(
            summary,
            out_path=os.path.join(out_dir, f"heatmap_{label}.png"),
        )
        # Main-text site-level baseline heatmap: rows are Ours Level-1 selected
        # sites; columns show how non-mobility baselines rank those same sites
        # across the full candidate pool.
        plot_level1_baseline_rank_heatmap_for_ours_selected(
            features=feat,
            strategy_orders=clean_orders,
            out_path=os.path.join(out_dir, f"rank_heatmap_{label}.png"),
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
            utility_county_csv=utility_county_csv,
            primary_strategy="Ours Level 1",
        )

        # Secondary health-oriented validation: COVID import/export risk is a
        # retrospective mobility-linked health-risk proxy, not a Level-1
        # selection criterion.
        plot_level1_health_risk_secondary(
            summary,
            out_path=os.path.join(out_dir, f"health_covid_{label}.png"),
        )

        # Supplemental full-candidate expansion curve for Ours Level 1 only.
        ours_key = None
        for _k in clean_orders.keys():
            if _k == "Ours Level 1" or "ours" in str(_k).lower() or "integrated" in str(_k).lower():
                ours_key = _k
                break
        full_orders_all = _build_full_strategy_orders_for_si(
            features=feat,
            strategy_orders=clean_orders,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
            utility_county_csv=utility_county_csv,
            primary_strategy="Ours Level 1",
        )
        full_orders_si = {}
        if "Ours Level 1" in full_orders_all:
            full_orders_si["Ours Level 1"] = full_orders_all["Ours Level 1"]
        elif ours_key is not None and ours_key in full_orders_all:
            full_orders_si["Ours Level 1"] = full_orders_all[ours_key]

        full_cumulative_tables = _build_cumulative_tables_from_orders(
            strategy_orders=full_orders_si,
            pop_served=pop_served,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            union_mode=union_mode,
            weight_from=weight_from,
            bg_weight_mode=bg_weight_mode,
            tau=tau,
            eps_pop=eps_pop,
            pop_served_total_override=pop_served_total_override,
        )
        plot_level1_cumulative_overlay(
            full_cumulative_tables,
            out_path=os.path.join(out_dir, f"cumul_from20_{label}.png"),
            x_min=20,
            vline_at=21,
            title="Ours Level 1 supplemental expansion curve",
        )
        plot_level1_marginal_benefit_curve(
            full_cumulative_tables,
            out_path=os.path.join(out_dir, f"level1_incremental_benefit_fullrank_{label}.png"),
            strategy="Ours Level 1",
            x_min=1,
            x_max=70,
            vline_at=21,
            title="Incremental benefit by additional site",
        )
        if sewersheds_gdf is not None:
            plot_level1_spatial_portfolio_map(
                strategy_orders=clean_orders,
                sewersheds_gdf=sewersheds_gdf,
                G=G,
                ej_scores=ej_scores,
                cbsa_shp=cbsa_shp,
                county_shp="../ZoneSelection/Input/Census/tl_2024_us_county/tl_2024_us_county.shp",
                out_path=os.path.join(out_dir, f"spatial_map_{label}.png"),
            )

    print(f"[Level1] Strategy comparison saved: {summary_path}")
    return summary


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def _normalize_for_heatmap(summary: pd.DataFrame, raw_col: str) -> pd.Series:
    """Return 0-1 values for a raw summary column."""
    if raw_col not in summary.columns:
        return pd.Series(np.nan, index=summary.index)
    x = pd.to_numeric(summary[raw_col], errors="coerce")
    # Fractions are already 0-1.
    if raw_col.endswith("_frac") or "_frac" in raw_col:
        return x.clip(0.0, 1.0)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=summary.index)
    return (x - lo) / (hi - lo)


def _format_annotation(row: pd.Series, col: str) -> str:
    v = row.get(col, np.nan)
    if pd.isna(v):
        return ""
    if col.endswith("_frac") or "_frac" in col:
        return f"{100 * float(v):.0f}%"
    if "mean_nearest_neighbor" in col:
        return f"{float(v):.0f} km"
    if "ej_mean" in col or "ej_pop_weighted" in col:
        return f"{float(v):.2f}"
    return f"{float(v):.0f}"



def _short_strategy_label(name: str) -> str:
    """Compact labels for site-level metric-rank + selection-order heatmaps."""
    n = str(name).strip()
    mapping = {
        "Population-only baseline": "Pop-only",
        "Population-only": "Pop-only",
        "Spatial-only baseline": "Area-only",
        "Area-only": "Area-only",
        "Metro/mobility baseline": "Mobility",
        "Mobility-activity-only": "Mobility",
        "Greedy unique-area": "Spatial",
        "Spatial coverage": "Spatial",
        "Spatial greedy": "Spatial",
        "Spatial greedy": "Spatial",
        "Census metro/micro-area": "CBSA",
        
        "Existing network": "Existing",
        "Current network": "Existing",
        "Integrated Level 1": "Ours L1",
        "Ours Level 1": "Ours L1",
    }
    return mapping.get(n, n.replace(" baseline", ""))



def _full_population_order(features: pd.DataFrame) -> List[str]:
    feat = _prepare_features(features)
    if "pop_served" not in feat.columns:
        return feat.index.tolist()
    score = pd.to_numeric(feat["pop_served"], errors="coerce").fillna(0.0)
    return score.sort_values(ascending=False, kind="mergesort").index.tolist()


def plot_level1_baseline_rank_heatmap_for_ours_selected(
    features: pd.DataFrame,
    strategy_orders: Mapping[str, Sequence[str]],
    out_path: str,
    bg_link_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    direction: str = "Destination",
    sewersheds_gdf=None,
    cbsa_shp: Optional[str] = None,
    utility_county_csv: Optional[str] = None,
    primary_strategy: str = "Ours Level 1",
) -> pd.DataFrame:
    """
    Main-text baseline heatmap.

    Rows are the final Ours Level-1 selected sites under the fixed budget
    (usually N=21). Columns show the *full candidate-pool rank* assigned to
    those same sites by the non-mobility baselines and by Ours Level 1.

    This avoids mixing the full-candidate ranking narrative with the final
    fixed-budget portfolio narrative: the figure asks, "For the sites selected
    by Ours Level 1, how would simpler non-mobility baselines rank them?"
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat = _prepare_features(features)
    n_all = max(len(feat), 1)

    # Canonicalize selected strategy orders.
    clean_orders: Dict[str, List[str]] = {}
    for strategy, order in strategy_orders.items():
        vals: List[str] = []
        for s in order:
            key = _canon(s)
            if key and key in feat.index and key not in vals:
                vals.append(key)
        clean_orders[str(strategy)] = vals

    # Locate Ours strategy robustly.
    primary_key = None
    for k in clean_orders.keys():
        if k == primary_strategy or k.lower().replace("integrated", "ours") in {"ours level 1", "ours l1"}:
            primary_key = k
            break
    if primary_key is None:
        for k in clean_orders.keys():
            if "ours" in k.lower() or "integrated" in k.lower():
                primary_key = k
                break
    if primary_key is None:
        primary_key = next(iter(clean_orders.keys()))

    rows = list(clean_orders.get(primary_key, []))
    if not rows:
        return pd.DataFrame()

    # Build full non-mobility baseline rankings across all candidate sites.
    full_orders: Dict[str, List[str]] = {}
    full_orders["Population-only"] = _full_population_order(feat)

    if sewersheds_gdf is not None:
        full_orders["Spatial coverage"] = build_spatial_geometry_greedy_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            total_N=n_all,
        )
    if sewersheds_gdf is not None and cbsa_shp:
        cbsa_full = build_cbsa_metro_micro_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
            total_N=n_all,
        )
        if cbsa_full:
            full_orders["Census metro/micro-area"] = cbsa_full

    # Ours column is the fixed-budget selected order (1..N) for the rows shown.
    full_orders["Ours Level 1"] = rows

    # Map to rank/order values.
    rank_table = pd.DataFrame(index=rows)
    for strategy, order in full_orders.items():
        omap = {site: i + 1 for i, site in enumerate(order)}
        rank_table[strategy] = [omap.get(r, np.nan) for r in rows]

    # Display names: use compact county-based aliases (e.g., "Pueblo",
    # "El Paso A", "El Paso B") instead of long WWTP names.
    name_map = feat["wwtp"].astype(str).to_dict() if "wwtp" in feat.columns else {idx: idx for idx in feat.index}
    county_map = load_dominant_county_map(_resolve_local_path(utility_county_csv))
    base_county = []
    for r in rows:
        nm = str(county_map.get(r, "")).replace(" County", "").strip()
        base_county.append(nm if nm else str(name_map.get(r, r)))
    counts = {}
    for nm in base_county:
        counts[nm] = counts.get(nm, 0) + 1
    seen_local = {}
    ylabels = []
    for nm in base_county:
        seen_local[nm] = seen_local.get(nm, 0) + 1
        if counts.get(nm, 0) > 1:
            suffix = chr(ord("A") + seen_local[nm] - 1)
            ylabels.append(f"{nm} {suffix}")
        else:
            ylabels.append(nm)

    # Companion CSV.
    out_csv = os.path.splitext(out_path)[0] + ".csv"
    csv_out = pd.DataFrame({"wwtp_clean": rows, "site_alias": ylabels, "wwtp_full_name": [name_map.get(r, r) for r in rows]})
    for strategy in rank_table.columns:
        csv_out[f"rank_{_short_strategy_label(strategy).replace(' ', '_').replace('/', '_')}"] = rank_table[strategy].values
    csv_out.to_csv(out_csv, index=False, encoding="utf-8")

    vals = rank_table.astype(float)
    # Rank 1 = high priority, rendered as stronger green; blank = light gray.
    color_vals = 1.0 - ((vals - 1.0) / max(n_all - 1.0, 1.0))
    color_vals = color_vals.clip(0.0, 1.0).values
    color_vals = np.ma.masked_invalid(color_vals)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f1f1f1")

    n_rows, n_cols = vals.shape
    fig_h = max(4.6, 0.31 * n_rows + 1.00)
    fig_w = max(4.2, 0.56 * n_cols + 1.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(color_vals, aspect="auto", vmin=0, vmax=1, cmap=cmap)

    for i in range(n_rows):
        for j, strategy in enumerate(rank_table.columns):
            v = vals.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.5, fontweight="bold")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7.4, color="#999999")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([_short_strategy_label(c) for c in rank_table.columns], rotation=18, ha="right", fontsize=8.0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ylabels, fontsize=7.2)

    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="#d9e8c8", linewidth=0.7)
    ax.axvline(n_cols - 0.5, color="#d9e8c8", linewidth=0.7)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="#e5eddc", linewidth=0.6)

    ax.set_title("Baseline ranks of Ours Level-1 selected sites", fontsize=10.8, pad=10)
    ax.set_xlabel("Cell values are full candidate-pool ranks", fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return csv_out

def plot_level1_site_rank_strategy_order_heatmap(
    features: pd.DataFrame,
    strategy_orders: Mapping[str, Sequence[str]],
    out_path: str,
    max_rows: int = 36,
    primary_strategy: str = "Integrated Level 1",
) -> pd.DataFrame:
    """
    Site-level heatmap with two clearly separated blocks:

      left block  = site-level metric ranks across all candidate WWTPs
      right block = strategy-specific selection order under the fixed budget

    This is intended for the reviewer-facing figure/SI figure that shows how
    the integrated Level-1 portfolio differs from simple greedy baselines.  A
    blank strategy-order cell means the site was not selected by that strategy.

    Outputs
    -------
    PNG/PDF figure at `out_path` and a companion CSV with the same stem:
      level1_site_metric_rank_and_strategy_orders_*.csv
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat = _prepare_features(features)

    # ---- metric rank columns shown on the left ----
    metric_specs = [
        ("pop_served", "Pop\nserved"),
        ("od_volume_total", "Commute\nvolume"),
        ("pop_covered_by_od", "Population\nreached"),
        ("area_reached", "Area\nreached"),
    ]
    # Optional heatmap-only columns if present in the feature table.
    optional_specs = [
        ("risk_score", "Mobility\nrisk"),
        ("ej_score", "EJ\nscore"),
        ("EJ_Combined", "EJ\nscore"),
        ("CombinedScore", "EJ\nscore"),
    ]
    for c, lab in optional_specs:
        if c in feat.columns and pd.to_numeric(feat[c], errors="coerce").notna().any():
            metric_specs.append((c, lab))
            break

    metric_specs = [(c, lab) for c, lab in metric_specs if c in feat.columns]
    if not metric_specs or not strategy_orders:
        return pd.DataFrame()

    # Canonicalize strategy orders and remove duplicates while preserving order.
    clean_orders: Dict[str, List[str]] = {}
    for strategy, order in strategy_orders.items():
        vals: List[str] = []
        for s in order:
            key = _canon(s)
            if key and key in feat.index and key not in vals:
                vals.append(key)
        clean_orders[str(strategy)] = vals

    # ---- row ordering: integrated selected sites first, then baseline-only sites ----
    primary_key = primary_strategy if primary_strategy in clean_orders else next(iter(clean_orders.keys()))
    rows: List[str] = list(clean_orders.get(primary_key, []))

    # Add any baseline-only selected sites by earliest order across strategies.
    candidates = set().union(*[set(v) for v in clean_orders.values()]) if clean_orders else set()
    candidates = [s for s in candidates if s not in rows]
    def _best_order(site: str) -> float:
        best = np.inf
        for order in clean_orders.values():
            if site in order:
                best = min(best, float(order.index(site) + 1))
        return best
    candidates = sorted(candidates, key=lambda x: (_best_order(x), x))
    rows = (rows + candidates)[: int(max_rows)]
    if not rows:
        return pd.DataFrame()

    # Use display names when available.
    if "wwtp" in feat.columns:
        name_map = feat["wwtp"].astype(str).to_dict()
    else:
        name_map = {idx: idx for idx in feat.index}
    ylabels = [name_map.get(r, r) for r in rows]

    # ---- metric ranks ----
    metric_rank_df = pd.DataFrame(index=feat.index)
    for col, _ in metric_specs:
        metric_rank_df[col] = _rank_desc(feat[col])

    metric_vals = metric_rank_df.reindex(rows)[[c for c, _ in metric_specs]].astype(float)
    n_all = max(float(len(feat)), 1.0)
    # Higher color intensity = better rank (rank 1).
    metric_color = 1.0 - ((metric_vals - 1.0) / max(n_all - 1.0, 1.0))
    metric_color = metric_color.clip(0.0, 1.0).values

    # ---- strategy order columns ----
    strategy_names = list(clean_orders.keys())
    order_table = pd.DataFrame(index=rows)
    for strategy, order in clean_orders.items():
        omap = {site: i + 1 for i, site in enumerate(order)}
        order_table[strategy] = [omap.get(r, np.nan) for r in rows]

    order_vals = order_table.astype(float)
    max_order = np.nanmax(order_vals.values) if np.isfinite(order_vals.values).any() else 1.0
    order_color = 1.0 - ((order_vals - 1.0) / max(max_order - 1.0, 1.0))
    order_color = order_color.clip(0.0, 1.0).values
    order_color = np.ma.masked_invalid(order_color)

    # ---- write companion CSV ----
    out_csv = os.path.splitext(out_path)[0] + ".csv"
    csv_out = pd.DataFrame({"wwtp_clean": rows, "wwtp": ylabels})
    for col, lab in metric_specs:
        csv_out[f"rank_{col}"] = metric_vals[col].values
    for strategy in strategy_names:
        csv_out[f"order_{_short_strategy_label(strategy).replace(' ', '_').replace('/', '_')}"] = order_table[strategy].values
    csv_out.to_csv(out_csv, index=False, encoding="utf-8")

    # ---- plot ----
    n_metric = len(metric_specs)
    n_order = len(strategy_names)
    fig_h = max(5.0, 0.34 * len(rows) + 1.8)
    fig_w = max(9.0, 0.72 * (n_metric + n_order) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    metric_cmap = plt.get_cmap("YlGnBu")
    order_cmap = plt.get_cmap("YlOrBr").copy()
    order_cmap.set_bad(color="#f1f1f1")

    # Left block: metric ranks.
    ax.imshow(metric_color, aspect="auto", vmin=0, vmax=1, cmap=metric_cmap,
              extent=(-0.5, n_metric - 0.5, len(rows) - 0.5, -0.5))
    # Right block: strategy selection order.
    gap = 0.8
    order_x0 = n_metric + gap
    ax.imshow(order_color, aspect="auto", vmin=0, vmax=1, cmap=order_cmap,
              extent=(order_x0 - 0.5, order_x0 + n_order - 0.5, len(rows) - 0.5, -0.5))

    # Cell annotations.
    for i in range(len(rows)):
        for j, (col, _) in enumerate(metric_specs):
            v = metric_vals.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8)
        for j, strategy in enumerate(strategy_names):
            v = order_table.iloc[i, j]
            if pd.notna(v):
                ax.text(order_x0 + j, i, f"{int(v)}", ha="center", va="center", fontsize=8, fontweight="bold")
            else:
                ax.text(order_x0 + j, i, "–", ha="center", va="center", fontsize=7, color="#999999")

    # Ticks and labels.
    x_positions = list(range(n_metric)) + [order_x0 + j for j in range(n_order)]
    x_labels = [lab for _, lab in metric_specs] + [_short_strategy_label(s) for s in strategy_names]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=7.2)

    # Grid lines.
    for x in [k - 0.5 for k in range(n_metric + 1)]:
        ax.axvline(x, color="#d0d0d0", linewidth=0.6)
    for x in [order_x0 + k - 0.5 for k in range(n_order + 1)]:
        ax.axvline(x, color="#d0d0d0", linewidth=0.6)
    for y in np.arange(-0.5, len(rows), 1):
        ax.axhline(y, color="#e5e5e5", linewidth=0.5)

    # Visual separator and block titles.
    sep_x = n_metric - 0.5 + gap / 2
    ax.axvline(sep_x, color="#333333", linewidth=1.3)
    ax.text((n_metric - 1) / 2, -1.15, "Site-level metric ranks", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1f4e79")
    ax.text(order_x0 + (n_order - 1) / 2, -1.15, "Strategy selection order", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#92400e")

    ax.set_xlim(-0.5, order_x0 + n_order - 0.5)
    ax.set_ylim(len(rows) - 0.5, -1.45)
    ax.set_title("Level-1 site ranks and fixed-budget selection order", fontsize=12, pad=16)
    ax.set_xlabel("Lower numbers indicate higher rank or earlier selection; blank means not selected", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return csv_out



def plot_level1_strategy_heatmap(summary: pd.DataFrame, out_path: str) -> None:
    """
    Strategy-level heatmap for the main Level-1 baseline comparison.

    This view emphasizes (i) size-driven coverage, (ii) mobility-linked BG reach,
    and (iii) simple statewide spatial-distribution diagnostics.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = summary.copy().reset_index(drop=True)

    columns = [
        ("pop_served_cum_frac", "Population\nserved"),
        ("od_volume_total_cum_frac", "Mobility\ntrip"),
        ("pop_reached_unique_cum_frac", "Population\nreached"),
        ("area_reached_unique_cum_frac", "Area\nreached"),
        ("n_dominant_counties_represented", "Counties\nrepr."),
        ("n_components_represented", "Network\ncomponents"),
        ("n_cbsa_represented", "CBSA\nareas"),
        ("non_metro_site_frac", "Non-metro\nshare"),
    ]
    columns = [(c, lab) for c, lab in columns if c in df.columns and df[c].notna().any()]
    if not columns:
        return

    mat = np.column_stack([_normalize_for_heatmap(df, c).values for c, _ in columns])
    labels = [lab for _, lab in columns]
    ylabels = df["strategy"].astype(str).tolist()

    fig_h = max(2.8, 0.54 * len(ylabels) + 0.90)
    fig_w = max(6.8, 0.82 * len(labels) + 1.40)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9.4)
    ax.tick_params(axis="x", rotation=0)

    for i in range(mat.shape[0]):
        for j, (raw_col, _) in enumerate(columns):
            text = _format_annotation(df.iloc[i], raw_col)
            ax.text(j, i, text, ha="center", va="center", fontsize=8.6)

    ax.set_title("Level-1 strategy comparison", fontsize=10.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized value", fontsize=8.8)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def _full_ours_level1_order(features: pd.DataFrame, selected_order: Sequence[str]) -> List[str]:
    """
    Full-candidate extension of Ours Level 1 for SI expansion curves.
    """
    feat = _prepare_features(features)
    selected = []
    seen = set()
    for s in selected_order:
        k = _canon(s)
        if k in feat.index and k not in seen:
            selected.append(k)
            seen.add(k)

    rank_cols = [c for c in ["pop_served", "od_volume_total", "pop_covered_by_od", "area_reached"] if c in feat.columns]
    if rank_cols:
        rank_mat = pd.concat([_rank_desc(feat[c]) for c in rank_cols], axis=1)
        overall_rank = rank_mat.mean(axis=1, skipna=True)
        remaining = [s for s in overall_rank.sort_values(kind="mergesort").index.tolist() if s not in seen]
    else:
        remaining = [s for s in feat.index.tolist() if s not in seen]

    return selected + remaining


def _build_full_strategy_orders_for_si(
    features: pd.DataFrame,
    strategy_orders: Mapping[str, Sequence[str]],
    bg_link_dir: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    direction: str,
    sewersheds_gdf=None,
    cbsa_shp: Optional[str] = None,
    utility_county_csv: Optional[str] = None,
    primary_strategy: str = "Ours Level 1",
) -> Dict[str, List[str]]:
    """
    Build full-candidate orders for SI expansion curves.
    Main text remains fixed-budget N=21.
    """
    feat = _prepare_features(features)
    n_all = len(feat)
    full_orders: Dict[str, List[str]] = {}

    full_orders["Population-only"] = _full_population_order(feat)

    if sewersheds_gdf is not None:
        full_orders["Spatial coverage"] = build_spatial_geometry_greedy_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            total_N=n_all,
        )

    if sewersheds_gdf is not None and cbsa_shp:
        cbsa_full = build_cbsa_metro_micro_order(
            features=feat,
            sewersheds_gdf=sewersheds_gdf,
            cbsa_shp=cbsa_shp,
            total_N=n_all,
        )
        if cbsa_full:
            full_orders["Census metro/micro-area"] = cbsa_full

    ours_key = None
    for k in strategy_orders.keys():
        if k == primary_strategy or "ours" in str(k).lower() or "integrated" in str(k).lower():
            ours_key = k
            break
    if ours_key is not None:
        full_orders["Ours Level 1"] = _full_ours_level1_order(feat, strategy_orders[ours_key])

    return full_orders


def _build_cumulative_tables_from_orders(
    strategy_orders: Mapping[str, Sequence[str]],
    pop_served: pd.Series,
    bg_link_dir: str,
    start_date: str,
    end_date: str,
    direction: str,
    union_mode: str,
    weight_from: str,
    bg_weight_mode: str,
    tau: float,
    eps_pop: float,
    pop_served_total_override: Optional[float],
) -> Dict[str, pd.DataFrame]:
    """Build cumulative BG-union tables for an arbitrary set of strategy orders."""
    out: Dict[str, pd.DataFrame] = {}
    for strategy, order in strategy_orders.items():
        if not order:
            continue
        avg_rank = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=list(order))
        df_cum = build_unique_cumulative_table_bg(
            avg_rank=avg_rank,
            bg_link_dir=bg_link_dir,
            start_date=start_date,
            end_date=end_date,
            direction=direction,  # type: ignore[arg-type]
            union_mode=union_mode,  # type: ignore[arg-type]
            weight_from=weight_from,  # type: ignore[arg-type]
            bg_weight_mode=bg_weight_mode,  # type: ignore[arg-type]
            tau=tau,
            eps_pop=eps_pop,
            pop_served=pop_served,
            pop_served_total_override=pop_served_total_override,
        )
        out[strategy] = df_cum
    return out


def plot_level1_portfolio_bar_summary(summary: pd.DataFrame, out_path: str) -> None:
    """
    Main-text fixed-budget benefit bar chart.

    Level 1 is presented as a balanced backbone comparison across core service,
    mobility-linked reach, spatial reach, and non-metro coverage. CBSA counts
    remain available in the broader heatmap/SI, but are not repeated here.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if summary is None or summary.empty:
        print("[Level1][bar] skipped: empty summary")
        return

    desired = ["Population-only", "Spatial coverage", "Census metro/micro-area", "Ours Level 1"]
    alias = {
        "Spatial greedy": "Spatial coverage",
        "Greedy unique-area": "Spatial coverage",
        "Integrated Level 1": "Ours Level 1",
        "Ours L1": "Ours Level 1",
    }

    df = summary.copy()
    df["strategy"] = df["strategy"].astype(str).replace(alias)
    df = df[df["strategy"].isin(desired)].copy()
    if df.empty:
        print("[Level1][bar] skipped: no matching strategies:", sorted(summary["strategy"].astype(str).unique()))
        return

    order = [s for s in desired if s in set(df["strategy"])]
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy").reset_index(drop=True)

    def _score_fraction(col: str):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce")
        if not s.notna().any():
            return None
        return s.clip(0.0, 1.0) * 100.0

    metric_series = []
    for col, lab in [
        ("pop_served_cum_frac", "Population\nserved"),
        ("od_volume_total_cum_frac", "Mobility\ntrip"),
        ("pop_reached_unique_cum_frac", "Population\nreached"),
        ("area_reached_unique_cum_frac", "Area\nreached"),
    ]:
        s = _score_fraction(col)
        if s is not None:
            metric_series.append((col, lab, s))

    s_nm = _score_fraction("nonmetro_area_reach_frac")
    nm_label = "Non-metro\ncoverage"
    if s_nm is None:
        s_nm = _score_fraction("non_metro_site_frac")
    if s_nm is not None:
        metric_series.append(("nonmetro_or_share", nm_label, s_nm))

    if not metric_series:
        print("[Level1][bar] skipped: no plottable metric columns")
        return

    x = np.arange(len(metric_series))
    n = len(df)
    width = min(0.105, 0.62 / max(n, 1))

    # Fixed, low-saturation strategy encoding based on the original
    # blue/orange/green/red palette. Sparse hatch patterns provide a redundant
    # cue for grayscale printing and readers with color-vision deficiencies.
    strategy_styles = {
        "Population-only":         {"color": "#5B8DB8", "hatch": ""},
        "Spatial coverage":        {"color": "#D9A441", "hatch": "/"},
        "Census metro/micro-area": {"color": "#76A66A", "hatch": "\\"},
        "Ours Level 1":            {"color": "#C77C8D", "hatch": "x"},
    }

    fig, ax = plt.subplots(figsize=(4.9, 2.95))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [float(s.iloc[i]) if pd.notna(s.iloc[i]) else np.nan for _, _, s in metric_series]
        strategy = str(row["strategy"])
        style = strategy_styles.get(strategy, {"color": "#999999", "hatch": ""})
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            vals,
            width=width,
            label=_short_strategy_label(strategy),
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="#303030",
            linewidth=0.55,
        )

    ax.set_ylabel("Score (0–100)", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in metric_series], fontsize=8.1, linespacing=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(n, 5), frameon=False, fontsize=7.8)
    ax.set_title("Level-1 comparison", fontsize=9.4)
    fig.tight_layout(pad=0.55)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Level1][bar] saved: {out_path}")


def plot_level1_health_risk_secondary(summary: pd.DataFrame, out_path: str) -> None:
    """
    Secondary health-relevance check.

    Shows retrospective mobility-linked COVID import/export risk retained.
    This is intentionally separate from the main Level-1 design bar chart:
    Level 1 is a general backbone design step, not a COVID-specific optimizer.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if summary is None or summary.empty or "covid_risk_signal_frac" not in summary.columns:
        return

    desired = ["Population-only", "Spatial coverage", "Census metro/micro-area", "Ours Level 1"]
    df = summary.copy()
    df["strategy"] = df["strategy"].astype(str)
    df = df[df["strategy"].isin(desired)].copy()
    if df.empty:
        return

    vals = pd.to_numeric(df["covid_risk_signal_frac"], errors="coerce")
    if not vals.notna().any():
        return

    order = [s for s in desired if s in set(df["strategy"])]
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy")

    x = np.arange(len(df))
    y = pd.to_numeric(df["covid_risk_signal_frac"], errors="coerce").values * 100.0

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    ax.bar(x, y, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_strategy_label(str(s)) for s in df["strategy"].astype(str)], rotation=20, ha="right")
    ax.set_ylabel("Retained risk proxy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Secondary health-risk check", fontsize=12)
    ax.text(
        0.5, -0.30,
        "Retrospective COVID import/export risk proxy; not used for Level-1 selection.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="0.25",
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def plot_level1_cumulative_overlay(
    cumulative_tables: Mapping[str, pd.DataFrame],
    out_path: str,
    x_min: int = 0,
    vline_at: Optional[int] = None,
    title: str = "Cumulative Level-1 portfolio performance by strategy",
) -> None:
    """Overlay cumulative fraction curves for the main Level-1 metrics."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metric_specs = [
        ("pop_served_cum_frac", "Population served"),
        ("pop_reached_unique_cum_frac", "BG population via mobility"),
        ("area_reached_unique_cum_frac", "BG area via mobility"),
        ("od_volume_total_cum_frac", "Mobility activity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metric_specs):
        for strategy, df in cumulative_tables.items():
            if df is None or df.empty or col not in df.columns:
                continue
            ax.plot(df["k_sites"], df[col], marker="o", markersize=2.5, linewidth=1.5, label=strategy)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of sites")
        ax.set_ylabel("Cumulative fraction")
        max_k_all = 0
        for _df in cumulative_tables.values():
            if _df is not None and (not _df.empty) and "k_sites" in _df.columns:
                max_k_all = max(max_k_all, int(pd.to_numeric(_df["k_sites"], errors="coerce").max()))
        if max_k_all > 0:
            tick_step = 10 if max_k_all > 30 else 5
            tick_end = int(np.ceil(max_k_all / float(tick_step)) * tick_step)
            start_tick = int(np.floor(float(x_min) / tick_step) * tick_step)
            ax.set_xticks(np.arange(start_tick, tick_end + 1, tick_step))
            ax.set_xlim(max(0, x_min), tick_end)
        if vline_at is not None:
            ax.axvline(vline_at, color="0.35", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)




def _nice_scale_length_m(width_m: float) -> float:
    """Return a manuscript-friendly scalebar length in meters."""
    if not np.isfinite(width_m) or width_m <= 0:
        return 100000.0
    target = width_m / 5.0
    candidates_km = np.array([10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750, 1000], dtype=float)
    candidates_m = candidates_km * 1000.0
    return float(candidates_m[np.argmin(np.abs(candidates_m - target))])


def _add_north_arrow_and_scalebar(ax) -> None:
    """Add a simple north arrow and Web-Mercator scalebar to a Matplotlib map axis."""
    try:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        width = x1 - x0
        height = y1 - y0
        if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
            return

        # North arrow in axes coordinates.
        ax.annotate(
            "N",
            xy=(0.94, 0.92),
            xytext=(0.94, 0.80),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"),
            zorder=100,
        )

        # Scale bar in map units. The maps are projected to EPSG:3857 above,
        # so the label is approximate but appropriate for visual reference.
        scale_m = _nice_scale_length_m(width)
        bar_x0 = x0 + 0.08 * width
        bar_y0 = y0 + 0.07 * height
        ax.plot([bar_x0, bar_x0 + scale_m], [bar_y0, bar_y0], color="black", lw=2.0, zorder=100)
        tick = 0.012 * height
        ax.plot([bar_x0, bar_x0], [bar_y0 - tick, bar_y0 + tick], color="black", lw=1.2, zorder=100)
        ax.plot([bar_x0 + scale_m, bar_x0 + scale_m], [bar_y0 - tick, bar_y0 + tick], color="black", lw=1.2, zorder=100)
        ax.text(
            bar_x0 + scale_m / 2.0,
            bar_y0 + 0.025 * height,
            f"{int(round(scale_m / 1000.0))} km",
            ha="center",
            va="bottom",
            fontsize=8,
            zorder=100,
        )
    except Exception:
        return




def plot_level1_marginal_benefit_curve(
    cumulative_tables,
    out_path: str,
    strategy: str = "Ours Level 1",
    x_min: int = 1,
    x_max=None,
    vline_at=21,
    title: str = "Incremental benefit by additional site",
) -> None:
    """
    Supplemental marginal-benefit figure for the ordered Level-1 expansion.

    Values are percentage-point contributions to the full ranked-site total
    for each metric, so metrics with different units can be compared on the
    same axis. This is not a baseline-comparison figure.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not cumulative_tables:
        print("[Level1][marginal] skipped: no cumulative tables")
        return

    key = strategy if strategy in cumulative_tables else None
    if key is None:
        for k in cumulative_tables.keys():
            if "ours" in str(k).lower() or "integrated" in str(k).lower():
                key = k
                break
    if key is None:
        key = next(iter(cumulative_tables.keys()))

    df = cumulative_tables.get(key)
    if df is None or df.empty or "k_sites" not in df.columns:
        print("[Level1][marginal] skipped: empty table")
        return

    df = df.copy()
    df["k_sites"] = pd.to_numeric(df["k_sites"], errors="coerce")
    df = df[df["k_sites"].notna()].copy()
    if x_max is not None:
        df = df[df["k_sites"] <= int(x_max)].copy()
    if x_min is not None:
        df = df[df["k_sites"] >= int(x_min)].copy()
    if df.empty:
        print("[Level1][marginal] skipped: no rows in requested x range")
        return

    metric_specs = [
        ("pop_served_delta", "pop_served_cum", "Population Served"),
        ("od_volume_total_delta", "od_volume_total_cum", "Mobility Trip"),
        ("pop_reached_unique_delta", "pop_reached_unique_cum", "Population Reached"),
        ("area_reached_unique_delta", "area_reached_unique_cum", "Area Reached"),
    ]

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    plotted = False
    for delta_col, cum_col, label in metric_specs:
        if delta_col not in df.columns or cum_col not in df.columns:
            continue
        denom_series = pd.to_numeric(cumulative_tables[key][cum_col], errors="coerce").dropna()
        denom = float(denom_series.iloc[-1]) if len(denom_series) else np.nan
        if not np.isfinite(denom) or denom <= 0:
            continue
        y = pd.to_numeric(df[delta_col], errors="coerce").fillna(0.0) / denom * 100.0
        ax.plot(
            df["k_sites"].values,
            y.values,
            marker="o",
            markersize=2.6,
            linewidth=1.4,
            label=label,
        )
        plotted = True

    if not plotted:
        print("[Level1][marginal] skipped: no plottable marginal columns")
        plt.close(fig)
        return

    if vline_at is not None:
        ax.axvline(float(vline_at), linestyle="--", linewidth=1.0, alpha=0.55)
        ymax = ax.get_ylim()[1]
        ax.text(float(vline_at) + 0.4, ymax * 0.92, f"N={int(vline_at)}", fontsize=7.6, va="top")

    ax.set_xlabel("Additional site in ranked order", fontsize=8.4)
    ax.set_ylabel("Incremental contribution\n(% of full-ranked total)", fontsize=8.4)
    ax.set_title(title, fontsize=9.2)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7.8)
    ax.legend(loc="upper right", frameon=True, fontsize=7.4)
    fig.tight_layout(pad=0.55)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Level1][marginal] saved: {out_path}")



def _add_simple_north_arrow(ax, x: float = 0.94, y: float = 0.91, size: float = 0.055) -> None:
    """Add a small unobtrusive north arrow in axes coordinates."""
    try:
        ax.annotate(
            "N",
            xy=(x, y),
            xytext=(x, y - size),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", linewidth=0.9, color="0.20"),
            color="0.20",
            zorder=20,
        )
    except Exception:
        pass


def _add_simple_scalebar_km(ax, length_km: int = 100, x: float = 0.055, y: float = 0.055) -> None:
    """Add a simple scalebar for Web Mercator meters."""
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xmax - xmin
        dy = ymax - ymin
        length_m = float(length_km) * 1000.0
        x0 = xmin + x * dx
        y0 = ymin + y * dy
        ax.plot([x0, x0 + length_m], [y0, y0], color="0.15", linewidth=1.8, solid_capstyle="butt", zorder=20)
        tick_h = 0.012 * dy
        ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="0.15", linewidth=1.2, zorder=20)
        ax.plot([x0 + length_m, x0 + length_m], [y0 - tick_h, y0 + tick_h], color="0.15", linewidth=1.2, zorder=20)
        ax.text(
            x0 + length_m / 2.0,
            y0 + 0.018 * dy,
            f"{int(length_km)} km",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.8),
            zorder=21,
        )
    except Exception:
        pass




def _add_shared_north_arrow(fig, x: float = 0.955, y: float = 0.935, size: float = 0.030) -> None:
    """Add one small shared north arrow in figure coordinates."""
    try:
        import matplotlib.patches as mpatches
        arrow = mpatches.FancyArrowPatch(
            (x, y - size),
            (x, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color="0.18",
            zorder=30,
        )
        fig.add_artist(arrow)
        fig.text(
            x,
            y + 0.008,
            "N",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="0.18",
            zorder=31,
        )
    except Exception:
        pass


def _add_shared_scalebar_km(fig, ax, length_km: int = 100) -> None:
    """Add one shared scalebar using the first map axis as coordinate reference."""
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = xmax - xmin
        dy = ymax - ymin
        length_m = float(length_km) * 1000.0

        x0_data = xmin + 0.055 * dx
        y0_data = ymin + 0.050 * dy
        p0 = fig.transFigure.inverted().transform(ax.transData.transform((x0_data, y0_data)))
        p1 = fig.transFigure.inverted().transform(ax.transData.transform((x0_data + length_m, y0_data)))
        tick = 0.010

        line = plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=fig.transFigure,
                          color="0.15", linewidth=2.0, solid_capstyle="butt", zorder=30)
        fig.add_artist(line)
        fig.add_artist(plt.Line2D([p0[0], p0[0]], [p0[1]-tick/2, p0[1]+tick/2],
                                  transform=fig.transFigure, color="0.15", linewidth=1.2, zorder=30))
        fig.add_artist(plt.Line2D([p1[0], p1[0]], [p1[1]-tick/2, p1[1]+tick/2],
                                  transform=fig.transFigure, color="0.15", linewidth=1.2, zorder=30))
        fig.text((p0[0] + p1[0]) / 2.0, p0[1] + 0.010, f"{int(length_km)} km",
                 ha="center", va="bottom", fontsize=11.5, color="0.15",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
                 zorder=31)
    except Exception:
        pass



def plot_level1_spatial_portfolio_map(
    strategy_orders: Mapping[str, Sequence[str]],
    sewersheds_gdf,
    G=None,
    ej_scores: Optional[pd.Series] = None,
    cbsa_shp: Optional[str] = None,
    county_shp: Optional[str] = None,
    out_path: str = "level1_spatial_portfolio_map.png",
) -> None:
    """
    Clean 4-panel spatial comparison map for SI.

    Shows county boundaries (preferred) or a light basemap fallback, plus
    all candidate WWTP points and larger selected WWTP points.
    """
    if gpd is None or sewersheds_gdf is None:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sg = sewersheds_gdf.copy()
    sg_plot = sg if sg.crs is None else sg.to_crs(epsg=3857)
    sg_plot["wwtp_clean"] = sg_plot["wwtp"].map(_canon) if "wwtp" in sg_plot.columns else sg_plot.index.map(_canon)

    county_plot = None
    county_path = _resolve_local_path(county_shp)
    if county_path is None or not os.path.exists(str(county_path)):
        # Fallback to the original project county boundary path used in earlier maps.
        county_path = _resolve_local_path(r"../ZoneSelection/Input/Census/COCounty.shp")
    if county_path and os.path.exists(county_path):
        try:
            county_tmp = gpd.read_file(county_path)
            if county_tmp.crs is not None and sg_plot.crs is not None:
                county_plot = county_tmp.to_crs(sg_plot.crs)
            else:
                county_plot = county_tmp
        except Exception:
            county_plot = None

    if county_plot is None:
        county_cols = [c for c in sg_plot.columns if str(c).lower() in {"county", "county_name", "cnty_name", "county_nam"}]
        if county_cols:
            try:
                cc = county_cols[0]
                county_plot = sg_plot[[cc, "geometry"]].dropna().dissolve(by=cc).reset_index()
            except Exception:
                county_plot = None

    xmin, ymin, xmax, ymax = sg_plot.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = dx * 0.055 if dx > 0 else 10000
    pad_y = dy * 0.055 if dy > 0 else 10000
    map_xlim = (xmin - pad_x, xmax + pad_x)
    map_ylim = (ymin - pad_y, ymax + pad_y)

    if county_plot is not None and not county_plot.empty:
        try:
            bbox_poly = gpd.GeoSeries.from_bbox((map_xlim[0], map_ylim[0], map_xlim[1], map_ylim[1])).set_crs(sg_plot.crs)
            county_plot = county_plot[county_plot.intersects(bbox_poly.iloc[0])].copy()
        except Exception:
            pass

    # Optional very light basemap fallback only if county boundaries are unavailable.
    add_basemap = False
    try:
        import contextily as ctx  # type: ignore
        add_basemap = county_plot is None or county_plot.empty
    except Exception:
        ctx = None
        add_basemap = False

    cent = sg_plot.geometry.centroid
    all_x = cent.x.values
    all_y = cent.y.values

    n = len(strategy_orders)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 5.0 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for ax, (strategy, order) in zip(axes, strategy_orders.items()):
        ax.set_facecolor("white")

        if county_plot is not None and not county_plot.empty:
            county_plot.boundary.plot(ax=ax, color="#bdbdbd", linewidth=0.72, alpha=0.96, zorder=2)
        elif add_basemap and ctx is not None:
            try:
                ax.set_xlim(*map_xlim)
                ax.set_ylim(*map_ylim)
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.42, attribution=False)
            except Exception:
                pass

        # All candidate WWTPs: very weak background points.
        ax.scatter(
            all_x, all_y,
            s=8,
            color="#9d9d9d",
            alpha=0.18,
            edgecolors="none",
            label="All candidate WWTPs",
            zorder=3,
        )

        # Selected WWTPs: larger and more legible.
        sel = set(_canon(s) for s in order)
        mask_sel = sg_plot["wwtp_clean"].isin(sel)
        if mask_sel.any():
            c_sel = sg_plot.loc[mask_sel].geometry.centroid
            ax.scatter(
                c_sel.x, c_sel.y,
                s=64,
                color="#1f78b4",
                edgecolors="white",
                linewidths=0.75,
                alpha=0.99,
                label="Selected WWTPs",
                zorder=4,
            )

        ax.set_xlim(*map_xlim)
        ax.set_ylim(*map_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_on()

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.80)
            spine.set_edgecolor("#6f6f6f")

        ax.text(
            0.03, 0.97,
            _short_strategy_label(strategy),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.8),
            zorder=10,
        )

    for ax in axes[n:]:
        ax.set_axis_off()

    if len(axes) > 0:
        _add_shared_north_arrow(fig)
        _add_shared_scalebar_km(fig, axes[0], length_km=100)

    # Clean figure-level legend. Use proxy artists so county boundary is labeled
    # without requiring repeated legend entries from every subplot.
    legend_handles = [
        Line2D([0], [0], color="#bdbdbd", lw=1.0, label="County boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9d9d9d",
               markeredgecolor="none", markersize=6.2, alpha=0.35, label="All WWTPs"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f78b4",
               markeredgecolor="white", markeredgewidth=0.7, markersize=9.0, label="Selected sites"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.030),
        ncol=3,
        fontsize=13,
        frameon=False,
        handletextpad=0.65,
        columnspacing=1.45,
    )
    fig.tight_layout(rect=[0, 0.060, 1, 1])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
