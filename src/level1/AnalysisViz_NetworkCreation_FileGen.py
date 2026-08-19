"""Export Level 1 network nodes, edges, sentinel tables, and contribution sites.

This module contains output helpers only. It does not perform ranking or site
selection; those decisions are made by the Level 1 workflow.
"""
import os
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
from typing import Optional

def _ensure_dir_for_file(filepath: str):
    outdir = os.path.dirname(filepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)


def export_network_edges_shp(
    G: nx.Graph,
    sewersheds: gpd.GeoDataFrame,
    outpath: str,
    include_distance_km: bool = True,
    include_component_id: bool = True,
):
    """
    Export network edges as LINE features (shp or gpkg).

    Geometry: LineString between node centroids.
    CRS: inherited from sewersheds.crs.

    Tip: Use .gpkg to avoid shapefile limitations (field name <=10 chars, etc.).
    """
    _ensure_dir_for_file(outpath)

    # component id lookup
    comp_id = {}
    if include_component_id:
        comps = list(nx.connected_components(G))
        for i, comp in enumerate(comps, start=1):
            for n in comp:
                comp_id[n] = i

    rows = []
    for u, v in G.edges:
        p1 = G.nodes[u].get("centroid", None)
        p2 = G.nodes[v].get("centroid", None)
        if (p1 is None) or (p2 is None):
            continue

        # node labels (optional)
        w1 = str(G.nodes[u].get("wwtp", u))
        w2 = str(G.nodes[v].get("wwtp", v))

        geom = LineString([(p1.x, p1.y), (p2.x, p2.y)])

        rec = {
            # keep these short for shapefile compatibility
            "u": int(u) if isinstance(u, (int, np.integer)) else str(u),
            "v": int(v) if isinstance(v, (int, np.integer)) else str(v),
            "wwtp_u": w1[:80],
            "wwtp_v": w2[:80],
        }

        if include_distance_km:
            # IMPORTANT:
            # distance is meaningful only if centroids are in a projected CRS in meters (e.g., EPSG:3857)
            rec["dist_km"] = float(p1.distance(p2) / 1000.0)

        if include_component_id:
            rec["comp_id"] = int(comp_id.get(u, -1))

        rows.append({**rec, "geometry": geom})

    edges_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=sewersheds.crs)

    if outpath.lower().endswith(".gpkg"):
        edges_gdf.to_file(outpath, layer="edges", driver="GPKG")
    else:
        edges_gdf.to_file(outpath, driver="ESRI Shapefile")

    print(f"[Export] edges saved -> {outpath} (n={len(edges_gdf)})")


def export_network_nodes_shp(
    G: nx.Graph,
    sewersheds: gpd.GeoDataFrame,
    outpath: str,
):
    """
    Export network nodes as POINT features (shp or gpkg).
    Geometry: node centroid.
    """
    _ensure_dir_for_file(outpath)

    rows = []
    for n in G.nodes:
        p = G.nodes[n].get("centroid", None)
        if p is None:
            continue
        rows.append(
            {
                "node": int(n) if isinstance(n, (int, np.integer)) else str(n),
                "wwtp": str(G.nodes[n].get("wwtp", n))[:80],
                "geometry": p,
            }
        )

    nodes_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=sewersheds.crs)

    if outpath.lower().endswith(".gpkg"):
        nodes_gdf.to_file(outpath, layer="nodes", driver="GPKG")
    else:
        nodes_gdf.to_file(outpath, driver="ESRI Shapefile")

    print(f"[Export] nodes saved -> {outpath} (n={len(nodes_gdf)})")


def export_sentinel_points_table(
    sewersheds_gdf,
    selected_wwtps ,
    official_sites,
    avg_rank,
    decision_log,
    diffusion_score_lower,
    out_csv,
    out_xlsx,
):
    """
    Export WWTP points + attributes for ArcGIS (CSV/Excel).

    Categories follow the selection-conditional HTML output semantics:
      - transmission_reinforced: selected==1 AND (official==1 OR post_diffusion != "none")
      - recommended: selected==1 AND official==0 AND post_diffusion=="none"
      - existing: selected==0 AND official==1
      - other: everything else

    Required columns in sewersheds_gdf:
      - geometry (polygons)
      - wwtp (string site name)

    Notes:
      - Exports centroid lat/lon in EPSG:4326
      - avg_rank can be a Series indexed by WWTP
      - diffusion_score_lower is optional; just stored as txraw_sc if present
      - decision_log is optional; used for post_diffusion/component metadata
    """
    _ensure_dir_for_file(out_csv)
    if out_xlsx:
        _ensure_dir_for_file(out_xlsx)

    def canon(x: str) -> str:
        return str(x).strip().lower()

    # Prepare centroids in WGS84
    g = sewersheds_gdf.to_crs(epsg=4326).copy()
    g["centroid"] = g.geometry.centroid
    g["lat"] = g["centroid"].y
    g["lon"] = g["centroid"].x

    # Normalize sets
    selected_set = {canon(s) for s in (selected_wwtps or [])}
    official_set = {canon(s) for s in (official_sites or set())}

    # avg_rank lookup (lower-cased join)
    avg_rank_lower = {}
    if avg_rank is not None and len(avg_rank) > 0:
        for k, v in avg_rank.items():
            try:
                avg_rank_lower[canon(k)] = float(v) if pd.notna(v) else np.nan
            except Exception:
                avg_rank_lower[canon(k)] = np.nan

    # Diffusion-score lookup using normalized lowercase identifiers
    diffusion_score_lower = diffusion_score_lower or {}

    rows = []
    for _, row in g.iterrows():
        w = (row.get("wwtp") or "").strip()
        if not w:
            continue
        if "historic" in w.lower():
            continue

        wl = canon(w)

        selected = int(wl in selected_set)
        official = int(wl in official_set)

        # decision log: try exact key, then case-insensitive fallback
        d = (decision_log or {}).get(w, {}) or {}
        if not d and decision_log:
            for kk in decision_log.keys():
                if canon(kk) == wl:
                    d = decision_log.get(kk, {}) or {}
                    break

        post_diff = d.get("post_diffusion", "none")  # expected: "none" or something like "diffusion"/"promoted"/etc.

        txraw_sc = diffusion_score_lower.get(wl, np.nan)
        try:
            txraw_sc = float(txraw_sc) if pd.notna(txraw_sc) else np.nan
        except Exception:
            txraw_sc = np.nan

        # --- category logic (selection-conditional) ---
        if selected == 1 and (official == 1 or str(post_diff).lower() != "none"):
            category = "transmission_reinforced"
            symbol = "green"
        elif selected == 1 and official == 0 and str(post_diff).lower() == "none":
            category = "recommended"
            symbol = "red"
        elif selected == 0 and official == 1:
            category = "existing"
            symbol = "blue"
        else:
            category = "other"
            symbol = "gray"

        rec = {
            "wwtp": w,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),

            # core flags
            "selected": selected,     # in top-K sentinel set
            "official": official,     # in existing sentinel list
            "category": category,     # existing/recommended/transmission_reinforced/other
            "symbol": symbol,         # for easy ArcGIS symbology

            # ranking + transmission attributes
            "avg_rank": avg_rank_lower.get(wl, np.nan),
            "txraw_sc": txraw_sc,     # optional; may be NaN if not available

            # traceability (optional but useful)
            "post_diff": post_diff,
            "comp_id": d.get("component_id", np.nan),
            "comp_sz": d.get("component_size", np.nan),
        }

        # Keep status if present (nice for filtering)
        if "act_sts" in g.columns:
            rec["act_sts"] = row.get("act_sts")

        rows.append(rec)

    df = pd.DataFrame(rows)

    # Helpful sorting (selected first, then reinforced, then rank)
    cat_order = {
        "transmission_reinforced": 0,
        "recommended": 1,
        "existing": 2,
        "other": 9,
    }
    df["cat_ord"] = df["category"].map(cat_order).fillna(9).astype(int)

    df = df.sort_values(
        by=["selected", "cat_ord", "avg_rank"],
        ascending=[False, True, True],
        na_position="last",
    ).drop(columns=["cat_ord"])

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Export] Sentinel points table CSV -> {out_csv}  (n={len(df)})")

    if out_xlsx:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="sentinel_points", index=False)
        print(f"[Export] Sentinel points table Excel -> {out_xlsx}")


def export_top_transmission_sites(
    diffusion_score_lower: dict,
    out_csv: str,
    top_k: int = 20,
    min_score: Optional[float] = None,
):
    """
    Export top transmission (TX) sites based on diffusion / TX-RAW scores.

    Parameters
    ----------
    diffusion_score_lower : dict
        Mapping {wwtp_lowercase: transmission_score}
    out_csv : str
        Output CSV path
    top_k : int
        Number of top TX sites to export
    min_score : float, optional
        Minimum TX score threshold (applied before ranking)

    Output columns
    --------------
    wwtp       : site name (lowercase key)
    tx_score   : transmission score
    tx_rank    : rank (1 = highest transmission)
    """
    _ensure_dir_for_file(out_csv)

    rows = []
    for w, sc in diffusion_score_lower.items():
        try:
            sc = float(sc)
        except Exception:
            continue
        if np.isnan(sc):
            continue
        if min_score is not None and sc < min_score:
            continue

        rows.append({
            "wwtp": w,
            "tx_score": sc,
        })

    if not rows:
        print("[Export] No valid TX scores found — CSV not written")
        return

    df = pd.DataFrame(rows)

    df = df.sort_values("tx_score", ascending=False).reset_index(drop=True)
    df["tx_rank"] = df.index + 1

    if top_k is not None:
        df = df.head(top_k)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Export] Top TX sites CSV -> {out_csv}  (n={len(df)})")
