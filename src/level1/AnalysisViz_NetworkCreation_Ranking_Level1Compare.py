"""Run the reported Level 1 ranking, subnetwork selection, and comparisons.

Inputs are authorized WWTP, sewershed, population, mobility, and clinical-risk
tables. Outputs include the fixed-capacity sentinel set, network diagnostics,
baseline comparisons, and cumulative-benefit inputs.
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import os
import folium
import numpy as np
from typing import List, Dict, Optional, Tuple
import glob
import re

from AnalysisViz_CumulativeCoverageV1 import (
    save_cumulative_outputs,
    save_bg_unique_outputs,

)

from AnalysisViz_Level1BaselineComparison import (
    make_level1_baseline_orders,
    evaluate_level1_strategy_orders,
)

from AnalysisViz_NetworkCreation_FileGen import export_network_edges_shp, export_network_nodes_shp, export_sentinel_points_table, export_top_transmission_sites
from AnalysisViz_NetworkRanking_DiffusionFilePicker import glob_best_txraw_csv

def _pick_best_txraw_csv_optional(search_dirs):
    """
    Robust optional picker for FLU/RSV TX-RAW files.

    The COVID picker path uses glob_best_txraw_csv("run_COVID"). Some FLU/RSV
    runs may have a slightly different naming pattern or grid file layout, so
    this function first tries the same picker and then falls back to a recursive
    file search inside run_FLU / run_RSV.
    """
    for d in search_dirs:
        if not d:
            continue

        d = str(d)
        if not os.path.exists(d):
            continue

        # 1) Try the same official picker used for COVID.
        try:
            p = glob_best_txraw_csv(d)
            if p and os.path.exists(p):
                return p
        except Exception as e:
            print(f"[Picker] Optional TX-RAW picker failed for {d}; using fallback search. Reason: {e}")

        # 2) If grid_results.csv exists, use its best tag to prefer the matching TX-RAW file.
        try:
            grid_path = os.path.join(d, "grid_results.csv")
            if os.path.exists(grid_path):
                grid = pd.read_csv(grid_path)
                best_row = None
                if "rmse" in grid.columns:
                    best_row = grid.sort_values("rmse", ascending=True, kind="mergesort").iloc[0]
                elif "mae" in grid.columns:
                    best_row = grid.sort_values("mae", ascending=True, kind="mergesort").iloc[0]
                elif len(grid) > 0:
                    best_row = grid.iloc[0]

                if best_row is not None and "tag" in grid.columns:
                    tag = str(best_row.get("tag", "")).strip()
                    if tag:
                        tagged_candidates = []
                        for pat in [
                            os.path.join(d, f"**/*{tag}*RAW*TX*.csv"),
                            os.path.join(d, f"**/*RAW*TX*{tag}*.csv"),
                            os.path.join(d, f"**/*{tag}*_table.csv"),
                        ]:
                            tagged_candidates.extend(glob.glob(pat, recursive=True))
                        tagged_candidates = [p for p in tagged_candidates if os.path.isfile(p)]
                        if tagged_candidates:
                            tagged_candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                            return tagged_candidates[0]
        except Exception as e:
            print(f"[Picker] Optional TX-RAW grid fallback failed for {d}: {e}")

        # 3) General recursive fallback. Prefer sentinel RAW TX tables, then any RAW/TX csv.
        candidates = []
        for pat in [
            os.path.join(d, "**/sentinel_RAW_TX*_table.csv"),
            os.path.join(d, "**/*sentinel*RAW*TX*_table.csv"),
            os.path.join(d, "**/*RAW*TX*_table.csv"),
            os.path.join(d, "**/*RAW*TX*.csv"),
            os.path.join(d, "**/*TX*RAW*.csv"),
            os.path.join(d, "**/*tx*raw*.csv"),
        ]:
            candidates.extend(glob.glob(pat, recursive=True))

        candidates = sorted(set([p for p in candidates if os.path.isfile(p)]))
        if candidates:
            # Prefer table outputs, then newest modified file.
            def _candidate_key(p):
                base = os.path.basename(p).lower()
                table_bonus = 1 if "_table" in base or "table" in base else 0
                sentinel_bonus = 1 if "sentinel" in base else 0
                return (sentinel_bonus, table_bonus, os.path.getmtime(p))
            candidates.sort(key=_candidate_key, reverse=True)
            return candidates[0]

    return ""



def _find_basic_ours_order_csv(run_root: str, month_label: str, suffix: str, total_N: int) -> str:
    """
    Find an Ours selected-order CSV produced by the basic ranking script.

    Expected preferred file from v76+ basic script:
      outputs/.../networkmaps/<month><suffix>/ours_selected_order_basic_<month><suffix>_N20.csv

    Fallbacks are intentionally broad so older outputs can also be used.
    """
    candidates = []

    # Most likely location if the basic ranking script was run first.
    networkmaps_dir = os.path.join(run_root, "networkmaps", f"{month_label}{suffix}")
    cumulative_dir = os.path.join(run_root, "cumulative_coverage", f"{month_label}{suffix}")
    baseline_dir = os.path.join(run_root, "level1_baseline_comparison", f"{month_label}_level1")

    search_dirs = [networkmaps_dir, cumulative_dir, baseline_dir, run_root]
    patterns = [
        f"ours_selected_order_basic_{month_label}{suffix}_N{int(total_N)}.csv",
        f"ours_selected_order_QA_{month_label}{suffix}_N{int(total_N)}.csv",
        f"*ours*selected*order*{month_label}*N{int(total_N)}*.csv",
        f"*selected_order*{month_label}*N{int(total_N)}*.csv",
        f"level1_strategy_orders_long_{month_label}_level1_N{int(total_N)}.csv",
    ]

    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(d, pat)))
            candidates.extend(glob.glob(os.path.join(d, "**", pat), recursive=True))

    candidates = sorted(set([c for c in candidates if os.path.isfile(c)]))
    if not candidates:
        return ""

    # Prefer explicit basic outputs over fallback strategy-order files.
    def _score(p):
        b = os.path.basename(p).lower()
        return (
            5 if "ours_selected_order_basic" in b else 0,
            4 if "ours_selected_order_qa" in b else 0,
            2 if "selected_order" in b else 0,
            1 if "strategy_orders_long" in b else 0,
            os.path.getmtime(p),
        )

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _load_basic_ours_order_csv(path: str, total_N: int = 20) -> list:
    """
    Load Ours order from the basic ranking script CSV.

    Supports:
    - explicit basic/QA format with columns: rank, wwtp, wwtp_clean
    - strategy long format with columns: strategy, order/rank, wwtp_clean
    """
    if not path or not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    cols_lower = {c.lower(): c for c in df.columns}

    # If this is the long strategy-order file, filter Ours.
    if "strategy" in cols_lower:
        sc = cols_lower["strategy"]
        mask = df[sc].astype(str).str.lower().str.contains("ours", na=False)
        df = df.loc[mask].copy()

    # Sort by available order/rank column.
    for c in ["rank", "order", "site_order", "selection_order", "k", "position"]:
        if c in cols_lower and cols_lower[c] in df.columns:
            df["_order_tmp"] = pd.to_numeric(df[cols_lower[c]], errors="coerce")
            df = df.sort_values("_order_tmp", kind="mergesort")
            break

    # Pick the cleanest site-name column available.
    site_col = None
    for c in ["wwtp_clean", "site_clean", "wwtp_lc", "site", "wwtp", "plant", "name"]:
        if c in cols_lower:
            site_col = cols_lower[c]
            break

    if site_col is None:
        return []

    order = []
    seen = set()
    for x in df[site_col].tolist():
        k = str(x).strip().lower()
        if not k or k == "nan" or k in seen:
            continue
        order.append(k)
        seen.add(k)
        if len(order) >= int(total_N):
            break
    return order


def _export_basic_ours_order_for_reuse(
    selected_ordered,
    run_root: str,
    month_label: str,
    suffix: str,
    total_N: int,
):
    """
    Export exact selected_ordered from the basic/ranking workflow so the baseline
    script can reuse it later instead of recomputing Ours.
    """
    out_dir = os.path.join(run_root, "networkmaps", f"{month_label}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ours_selected_order_basic_{month_label}{suffix}_N{int(total_N)}.csv")
    pd.DataFrame({
        "rank": list(range(1, len(selected_ordered) + 1)),
        "wwtp": [str(x) for x in selected_ordered],
        "wwtp_clean": [str(x).strip().lower() for x in selected_ordered],
    }).to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OursReuse] Basic Ours selected order exported: {out_path}")
    return out_path


state_population = 5937082
site_population_Total = 3643190

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set Times New Roman as the default font family
plt.rcParams["font.family"] =  "Arial"

# Ensure weight and style are set to normal
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.style"] = "normal"

def _build_comp_size_lookup(G: nx.Graph):
    """Map exact WWTP name -> connected-component size."""
    name_to_node = {}
    for n in G.nodes:
        nm = (G.nodes[n].get("wwtp") or str(n)).strip()
        if nm:
            name_to_node[nm] = n

    comps = list(nx.connected_components(G))
    node_to_size = {}
    for comp in comps:
        sz = len(comp)
        for n in comp:
            node_to_size[n] = sz

    return {nm: node_to_size.get(node, 0) for nm, node in name_to_node.items()}


def build_expanded_selected_order(
    G: nx.Graph,
    ordered_all: list,          # global ordering list
    baseline_21: list,          # Fixed-size reference list (length=total_N).
    selected_raw_set: set,      # output from select_sentinels_simple BEFORE trimming
    expand_cap: int = 70,
    backforth_pattern=("S", "M", "M"),  # 1 singleton then 2 multi
):
    """
    Returns:
      expanded_list: baseline_21 + saturation_extras + backforth extras (up to expand_cap)
      phase_label: dict site -> baseline|saturation|backforth
    """
    phase = {}
    out = []
    picked = set()

    # --- Phase 0: baseline (first total_N)
    for w in baseline_21:
        if w not in picked:
            out.append(w); picked.add(w)
            phase[w] = "baseline"

    if len(out) >= expand_cap:
        return out[:expand_cap], phase

    # --- Phase 1: saturation extras (still from "current config" result)
    saturation_extras = [w for w in ordered_all if (w in selected_raw_set) and (w not in picked)]
    for w in saturation_extras:
        if len(out) >= expand_cap:
            break
        out.append(w); picked.add(w)
        phase[w] = "saturation"

    if len(out) >= expand_cap:
        return out[:expand_cap], phase

    # --- Phase 2: back-and-forth pools (ignore constraints)
    name_to_csize = _build_comp_size_lookup(G)
    def csize(w): return name_to_csize.get(w, 0)

    singleton_pool = [w for w in ordered_all if (w not in picked) and (csize(w) == 1)]
    multi_pool     = [w for w in ordered_all if (w not in picked) and (csize(w) >= 2)]
    unknown_pool   = [w for w in ordered_all if (w not in picked) and (csize(w) == 0)]

    iS = iM = iU = 0
    pat = list(backforth_pattern)
    pi = 0

    while len(out) < expand_cap and (iS < len(singleton_pool) or iM < len(multi_pool) or iU < len(unknown_pool)):
        mode = pat[pi % len(pat)]
        pi += 1

        if mode == "S":
            if iS < len(singleton_pool):
                w = singleton_pool[iS]; iS += 1
            elif iU < len(unknown_pool):
                w = unknown_pool[iU]; iU += 1
            elif iM < len(multi_pool):
                w = multi_pool[iM]; iM += 1
            else:
                break
        else:  # "M"
            if iM < len(multi_pool):
                w = multi_pool[iM]; iM += 1
            elif iS < len(singleton_pool):
                w = singleton_pool[iS]; iS += 1
            elif iU < len(unknown_pool):
                w = unknown_pool[iU]; iU += 1
            else:
                break

        out.append(w); picked.add(w)
        phase[w] = "backforth"

    return out[:expand_cap], phase


# =============================================================
# Helpers: time windows & file utils
# =============================================================
def export_rank_lists(rank_table: pd.DataFrame, out_dir: str, label: str, k_list=(10, 20, 30, 40, 50)):
    """
    rank_table: must contain 'wwtp' and 'final_rank'
    Writes:
      - ranked_master_{label}.csv: full list in rank order
      - ranked_topK_{label}.csv  : columns top10/top20/... with names
    """
    os.makedirs(out_dir, exist_ok=True)

    master = (rank_table[["wwtp", "final_rank"]]
              .dropna(subset=["wwtp", "final_rank"])
              .sort_values("final_rank", ascending=True, kind="mergesort")
              .reset_index(drop=True))
    master.to_csv(os.path.join(out_dir, f"ranked_master_{label}.csv"), index=False, encoding="utf-8")

    names = master["wwtp"].astype(str).tolist()
    topk = {f"top{k}": pd.Series(names[:min(k, len(names))]) for k in k_list}
    pd.DataFrame(topk).to_csv(os.path.join(out_dir, f"ranked_topK_{label}.csv"), index=False, encoding="utf-8")

def iter_month_windows(start_date: str, end_date: str):
    """Yield (month_start_iso, month_end_iso, label) for each calendar month in [start_date, end_date]."""
    s = pd.to_datetime(start_date).to_period("M")
    e = pd.to_datetime(end_date).to_period("M")
    for p in pd.period_range(s, e, freq="M"):
        m_start = pd.Timestamp(p.start_time).date().isoformat()
        m_end   = pd.Timestamp(p.end_time).date().isoformat()
        yield m_start, m_end, str(p)  # label like '2024-10'

def _ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _norm_name(s: str) -> str:
    return str(s).replace("_", " ").strip().title()

# =============================================================
# Data loading & base network construction
# =============================================================

def load_and_prepare_data(sewershed_path, zip_path, commute_path):
    sewersheds = gpd.read_file(sewershed_path).to_crs(epsg=3857)
    assert 'wwtp' in sewersheds.columns, "Missing 'wwtp' column in sewershed shapefile."

    zip_gdf = gpd.read_file(zip_path).to_crs(epsg=3857)
    zip_gdf = zip_gdf.explode(index_parts=False).reset_index(drop=True)

    commute_df = pd.read_csv(commute_path)
    commute_df["ZIP"] = commute_df["NAME"].str.extract(r'(\d{5})')
    zip_gdf["ZIP"] = zip_gdf["GEOID10"].astype(str).str.zfill(5)

    zip_gdf = zip_gdf.merge(commute_df, on="ZIP", how="left")
    valid_zips = zip_gdf[zip_gdf["commute_distance_miles"].notna()]

    sewer_zip = gpd.sjoin(sewersheds, valid_zips[["ZIP", "commute_distance_miles", "geometry"]], how="left", predicate="intersects")
    sewer_zip_grouped = sewer_zip.groupby(sewer_zip.index).agg({"commute_distance_miles": "mean"})
    sewersheds = sewersheds.join(sewer_zip_grouped)

    sewersheds["buffer_radius_meters"] = sewersheds["commute_distance_miles"] * 1609.34
    sewersheds["geometry_buffer"] = sewersheds.geometry.buffer(sewersheds["buffer_radius_meters"])

    return sewersheds

def build_sewershed_graph(sewersheds):
    G = nx.Graph()
    for idx, row in sewersheds.iterrows():
        G.add_node(idx, wwtp=row['wwtp'], centroid=row.geometry.centroid, full_geom=row.geometry)

    for i, row_i in sewersheds.iterrows():
        for j, row_j in sewersheds.iterrows():
            if j <= i:
                continue
            if row_i.geometry_buffer.intersects(row_j.geometry_buffer):
                G.add_edge(i, j)

    return G


def plot_static_network(
    G,
    sewersheds,
    outpath="outputs/images/colorado_sewershed_network.png",
    roads=None,
    county_boundary=None
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx

    _ensure_dir_for_file(outpath)

    # --- Load if file paths ---
    if isinstance(sewersheds, str):
        sewersheds = gpd.read_file(sewersheds)

    if isinstance(roads, str):
        roads = gpd.read_file(roads)

    if isinstance(county_boundary, str):
        county_boundary = gpd.read_file(county_boundary)

    # --- Reproject all to same CRS ---
    sewersheds = sewersheds.to_crs(epsg=3857)

    if roads is not None:
        roads = roads.to_crs(epsg=3857)

    if county_boundary is not None:
        county_boundary = county_boundary.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- County boundary ---
    if county_boundary is not None:
        county_boundary.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.8,
            alpha=0.6,
            zorder=1
        )

    # --- Sewersheds ---
    sewersheds.plot(
        ax=ax,
        facecolor="lightblue",
        edgecolor="black",
        alpha=0.6,
        zorder=2
    )

    # --- Roads ---
    if roads is not None:
        roads.plot(
            ax=ax,
            color="blue",
            linewidth=0.6,
            alpha=0.4,
            zorder=3
        )

    # --- Recompute centroids from current plotted sewersheds ---
    centroids = sewersheds.geometry.centroid

    # --- Network edges ---
    for u, v in G.edges:
        try:
            p1 = centroids.loc[u]
            p2 = centroids.loc[v]
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                color="gray",
                linewidth=0.5,
                alpha=0.7,
                zorder=4
            )
        except Exception:
            continue

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron
    )

    ax.set_title("Colorado Sewershed Network")
    ax.set_axis_off()

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def export_interactive_map(G, sewersheds, outpath="outputs/maps/colorado_sewershed_network.html"):
    _ensure_dir_for_file(outpath)
    sewersheds_wgs84 = sewersheds.to_crs(epsg=4326)
    centroids_wgs84 = sewersheds_wgs84.geometry.centroid
    html_map = folium.Map(location=[centroids_wgs84.y.mean(), centroids_wgs84.x.mean()],
                          zoom_start=7, tiles="CartoDB Positron")

    for u, v in G.edges:
        p1 = sewersheds_wgs84.loc[u].geometry.centroid
        p2 = sewersheds_wgs84.loc[v].geometry.centroid
        folium.PolyLine([(p1.y, p1.x), (p2.y, p2.x)], color="#4a4a4a", weight=1, opacity=0.6).add_to(html_map)

    for idx, row in sewersheds_wgs84.iterrows():
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=6,
            color="blue",
            weight=1,
            fill=True,
            fill_opacity=0.6,
            popup=f"WWTP: {row['wwtp']}"
        ).add_to(html_map)
        folium.GeoJson(row.geometry, name=row['wwtp'], style_function=lambda x: {'fillOpacity': 0.1, 'color': 'black', 'weight': 0.5}).add_to(html_map)

    folium.LayerControl(collapsed=False).add_to(html_map)
    html_map.save(outpath)

def clean_graph_for_graphml(G):
    """Strip geometry and unsupported attributes for GraphML export. Keep only 'wwtp' on nodes."""
    for node in G.nodes:
        attrs = G.nodes[node]
        keys_to_delete = [k for k in attrs if k not in {"wwtp"}]
        for k in keys_to_delete:
            del attrs[k]
    for _, _, attrs in G.edges(data=True):
        attrs.clear()

# =============================================================
# OD coverage enrichment (in/out volume, pop, area, counties)
# =============================================================

def enrich_nodes_with_od_coverage(G, weekly_od_dir, wwtp_shapefile, start_date=None, end_date=None):
    """
    Enrich each WWTP node in G with:
    - pop_served, shape_area (from shapefile)
    - OD inflow/outflow totals and diversity (volume, area, population, counties)
    - Time-filtered using start_date and end_date (inclusive)
    """
    from collections import defaultdict

    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    wwtp_gdf = gpd.read_file(wwtp_shapefile)
    wwtp_gdf['wwtp'] = wwtp_gdf['wwtp'].astype(str).str.strip().str.lower()
    wwtp_meta = wwtp_gdf.set_index('wwtp')[
        [c for c in ['Shape_Area', 'Shape_Are', 'pop_served'] if c in wwtp_gdf.columns]
    ].copy()
    if 'Shape_Area' not in wwtp_meta.columns and 'Shape_Are' in wwtp_meta.columns:
        wwtp_meta.rename(columns={'Shape_Are': 'Shape_Area'}, inplace=True)

    od_agg = defaultdict(lambda: {
        "volume_in": 0.0,
        "volume_out": 0.0,
        "area_in": 0.0,
        "area_out": 0.0,
        "pop_from_in": 0.0,
        "pop_to_out": 0.0,
        "counties_in": set(),
        "counties_out": set()
    })

    for fname in os.listdir(weekly_od_dir):
        if not fname.endswith(".csv"):
            continue
        date_str = fname.replace("weekly_d_", "").replace("weekly_o_", "").replace(".csv", "")
        try:
            file_date = pd.to_datetime(date_str)
        except Exception:
            continue
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        fpath = os.path.join(weekly_od_dir, fname)
        df = pd.read_csv(fpath)

        if "Sewershed-O" in df.columns:
            df['wwtp'] = df["Sewershed-O"].astype(str).str.strip().str.lower()
            df['county'] = df["County"].astype(str).str.zfill(5)
            direction = "out"
        elif "Sewershed-D" in df.columns:
            df['wwtp'] = df["Sewershed-D"].astype(str).str.strip().str.lower()
            df['county'] = df["County"].astype(str).str.zfill(5)
            direction = "in"
        else:
            continue

        df = df[[c for c in ["wwtp", "county", "Volume", "Area", "Population"] if c in df.columns]].dropna()

        for _, row in df.iterrows():
            w = row["wwtp"]
            c = row["county"]
            vol = float(row["Volume"]) if "Volume" in df.columns else 0.0
            area = float(row["Area"]) if "Area" in df.columns else 0.0
            pop = float(row["Population"]) if "Population" in df.columns else 0.0
            if direction == "in":
                od_agg[w]["volume_in"] += vol
                od_agg[w]["area_in"] += area
                od_agg[w]["pop_from_in"] += pop
                od_agg[w]["counties_in"].add(c)
            else:
                od_agg[w]["volume_out"] += vol
                od_agg[w]["area_out"] += area
                od_agg[w]["pop_to_out"] += pop
                od_agg[w]["counties_out"].add(c)

    for node in G.nodes:
        name = str(G.nodes[node].get("wwtp", "")).strip().lower()
        if name in wwtp_meta.index:
            meta = wwtp_meta.loc[name]
            if "pop_served" in meta.index:
                G.nodes[node]["pop_served"] = meta["pop_served"]
            if "Shape_Area" in meta.index:
                G.nodes[node]["shape_area"] = meta["Shape_Area"]
        if name in od_agg:
            data = od_agg[name]
            vol_in, vol_out = data["volume_in"], data["volume_out"]
            G.nodes[node]["od_volume_in"] = vol_in
            G.nodes[node]["od_volume_out"] = vol_out
            G.nodes[node]["od_volume_total"] = vol_in + vol_out
            G.nodes[node]["area_from_od_in_counties"] = data["area_in"]
            G.nodes[node]["area_to_od_out_counties"] = data["area_out"]
            G.nodes[node]["pop_from_od_in_counties"] = data["pop_from_in"]
            G.nodes[node]["pop_to_od_out_counties"] = data["pop_to_out"]
            G.nodes[node]["pop_covered_by_od"] = data["pop_from_in"] + data["pop_to_out"]
            G.nodes[node]["counties_inflow_from"] = sorted(data["counties_in"])
            G.nodes[node]["counties_outflow_to"] = sorted(data["counties_out"])

# =============================================================
# Ranking utilities (base + optional risk/viral signals)
# =============================================================

BASE_METRICS = [
    "pop_served",          # Population served (from shapefile)
    "od_volume_total",     # Total OD volume (in+out)
    "pop_covered_by_od",   # Population reached via OD
    "area_reached"         # Area reached via OD
]

def build_feature_table_from_graph(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for node in G.nodes:
        attrs = G.nodes[node]
        wwtp = (attrs.get("wwtp") or str(node)).strip()
        if "historic" in wwtp.lower():
            continue
        vol_in = attrs.get("od_volume_in", 0.0) or 0.0
        vol_out = attrs.get("od_volume_out", 0.0) or 0.0
        vol_total = attrs.get("od_volume_total")
        if vol_total is None:
            vol_total = vol_in + vol_out

        pop_served = attrs.get("pop_served", np.nan)
        pop_in = attrs.get("pop_from_od_in_counties", 0.0) or 0.0
        pop_out = attrs.get("pop_to_od_out_counties", 0.0) or 0.0
        pop_cov = attrs.get("pop_covered_by_od")
        if pop_cov is None:
            pop_cov = pop_in + pop_out

        area_in = attrs.get("area_from_od_in_counties", 0.0) or 0.0
        area_out = attrs.get("area_to_od_out_counties", 0.0) or 0.0
        area_total = area_in + area_out

        rows.append({
            "wwtp": wwtp,
            "pop_served": pop_served,
            "od_volume_total": vol_total,
            "pop_covered_by_od": pop_cov,
            "area_reached": area_total,
        })
    df = pd.DataFrame(rows)
    df["wwtp_clean"] = df["wwtp"].astype(str).str.strip().str.lower()
    return df.set_index("wwtp_clean").sort_index()


# =============================================================
# Level-1 baseline comparison helper
# =============================================================

def compute_integrated_level1_order(
        G: nx.Graph,
        base_features: pd.DataFrame,
        total_N: int = 20,
        selection_fraction: float = 0.25,
        singleton_top_pct: float = 0.30,
        singleton_drop_km: float = 100.0,
        singleton_bottom_pct: float = 0.20,
        singleton_reserve_k: int = 0,
        isolation_bonus_km: float = 150.0,
        risk_weight: float = 0.5,
        coverage_weight: float = 0.5,
) -> list:
    """
    Reproduce the integrated Level-1 selection order without disease/TX-RAW signals.

    This is used for baseline comparison. It intentionally uses only the Level-1
    backbone metrics:
      - pop_served
      - od_volume_total
      - pop_covered_by_od
      - area_reached

    It then applies the same subnetwork/singleton logic used by the main selector.
    Returned names are canonical lowercase wwtp_clean values so they match BG-Link
    and cumulative-coverage functions.
    """
    if base_features is None or base_features.empty:
        return []

    candidate_fields = [
        c for c in ["pop_served", "od_volume_total", "pop_covered_by_od", "area_reached"]
        if c in base_features.columns
    ]
    if not candidate_fields:
        return []

    # select_sentinels_simple expects original WWTP labels that match G.nodes[n]['wwtp'].
    feat_orig = base_features.copy()
    if "wwtp" in feat_orig.columns:
        feat_orig.index = feat_orig["wwtp"].astype(str).str.strip()
    else:
        feat_orig.index = feat_orig.index.astype(str).str.strip()

    rank_df = feat_orig[candidate_fields].rank(ascending=False, method="min")
    avg_rank = rank_df.mean(axis=1, skipna=True)

    (
        selected_set,
        dropped_singletons,
        honored_diff,
        drop_reasons,
        keep_reasons_singletons,
        reserved_singletons,
        decision_log,
    ) = select_sentinels_simple(
        G,
        rank_df=rank_df,
        features=feat_orig,
        selection_fraction=selection_fraction,
        counties_in_attr="counties_inflow_from",
        counties_out_attr="counties_outflow_to",
        singleton_top_pct=singleton_top_pct,
        singleton_drop_km=singleton_drop_km,
        diffusion_csv_path="",  # Level 1 only: no disease/TX-RAW signal
        large_component_min_nodes=2,
        singleton_bottom_pct=singleton_bottom_pct,
        singleton_reserve_k=singleton_reserve_k,
        singleton_reserve_frac=0.0,
        isolation_bonus_km=isolation_bonus_km,
        risk_weight=risk_weight,
        coverage_weight=coverage_weight,
        saturate_components=True,
        saturation_gain_min=1,
    )

    ordered_all = avg_rank.sort_values(kind="mergesort").index.tolist()
    total_N = int(total_N)

    # Final fixed-budget order must match the main ranking script exactly.
    # Earlier versions of this baseline helper only reserved hard singletons here,
    # which could make "Ours Level 1" differ from AnalysisViz_NetworkCreation_Ranking.py.
    reserved_set = set(reserved_singletons)
    raw_ordered = [s for s in ordered_all if s in selected_set]

    hard_singletons = {
        w for w, info in decision_log.items()
        if (w in selected_set)
        and (str(info.get("decision", "")).lower() == "keep")
        and bool(info.get("is_singleton", False))
    }

    # Sites selected to satisfy the multi-site component quota.
    # Exclude within-component saturation additions because those are expansion candidates,
    # not minimum quota requirements.
    quota_core = {
        w for w, info in decision_log.items()
        if (w in selected_set)
        and (str(info.get("decision", "")).lower() == "keep")
        and (not bool(info.get("is_singleton", False)))
        and (str(info.get("method", "")) != "saturation_component")
    }

    reserved_singleton_ordered = [s for s in ordered_all if s in hard_singletons and s in reserved_set]
    other_singleton_ordered = [s for s in ordered_all if s in hard_singletons and s not in reserved_set]
    quota_core_ordered = [s for s in ordered_all if s in quota_core and s not in hard_singletons]

    minimum_ordered = []
    for pool in (reserved_singleton_ordered, other_singleton_ordered, quota_core_ordered):
        for s in pool:
            if s not in minimum_ordered:
                minimum_ordered.append(s)

    if len(minimum_ordered) > total_N:
        selected_ordered = minimum_ordered[:total_N]
    else:
        remaining_ordered = [s for s in raw_ordered if s not in set(minimum_ordered)]
        selected_ordered = minimum_ordered + remaining_ordered[: max(0, total_N - len(minimum_ordered))]

    if len(selected_ordered) < total_N:
        selected_set_now = set(selected_ordered)
        fillers = [s for s in ordered_all if s not in selected_set_now]
        selected_ordered += fillers[: (total_N - len(selected_ordered))]

    return [str(s).strip().lower() for s in selected_ordered]


def _find_latest_raw_summary_csv(search_dir="outputs/csv", prefix="weekly_top10_summary_multi") -> Optional[str]:
    pattern = os.path.join(search_dir, f"{prefix}_*.csv")
    candidates = [p for p in glob.glob(pattern) if "_rank_" not in os.path.basename(p)]
    if not candidates:
        return None
    rgx = re.compile(r"(\d{8})_(\d{8})")
    ranged = []
    for p in candidates:
        m = rgx.search(os.path.basename(p))
        if m:
            ranged.append((p, m.group(1), m.group(2)))
        else:
            ranged.append((p, "", ""))
    ranged.sort(key=lambda x: (x[2], x[1], os.path.getmtime(x[0])), reverse=True)
    return ranged[0][0]

def load_monthly_risk_signals(summary_csv: str, month_label: str, include_wval: bool = False) -> pd.DataFrame:
    """
    From the weekly summary CSV (created by Analysis_ODDiffusionRisk.save_weekly_risk_summary),
    compute monthly averages per WWTP for COVID-only risk columns:
      - 'import_risk_COVID'
      - 'export_risk_COVID'
      - 'speed_norm_COVID'      (early-detection metric)
      - (optional) 'WVAL_COVID'
    Returns a DataFrame indexed by wwtp_clean with these monthly means.
    """
    # Default: restrict to COVID risks only
    use_cols_prefixes = ["import_risk_COVID", "export_risk_COVID"]
    if include_wval:
        use_cols_prefixes.append("WVAL_COVID")

    df = pd.read_csv(summary_csv, parse_dates=["week"]) \
        .assign(wwtp_clean=lambda d: d["wwtp"].astype(str).str.strip().str.lower())
    df["ym"] = df["week"].dt.to_period("M").astype(str)
    month_df = df[df["ym"] == month_label]
    if month_df.empty:
        return pd.DataFrame(index=[])

    # Collect exact matching columns (prefixes are full names here)
    metric_cols = [c for c in month_df.columns if any(c.startswith(p) for p in use_cols_prefixes)]
    if not metric_cols:
        return pd.DataFrame(index=[])

    monthly = (month_df.groupby("wwtp_clean", as_index=True)[metric_cols]
               .mean(numeric_only=True))
    return monthly


# =============================================================
# Environmental-justice ranking utilities
# =============================================================

def load_ej_scores(
        ej_csv: str,
        name_col_candidates: Tuple[str, ...] = ("wwtp", "wwtp_id", "name"),
        score_col: str = "CombinedScore",
        higher_is_worse: bool = True,
) -> pd.Series:
    """Load EJ scores as a Series indexed by canonical wwtp_clean.

    Expected EJ CSV: at least a WWTP/site name column and a numeric score column.
    By convention, EJ CombinedScore is often higher = worse burden; set
    higher_is_worse=False if your score is already higher=better.
    """
    ej = pd.read_csv(ej_csv)
    name_col = next((c for c in ej.columns if c.lower() in name_col_candidates), None)
    if name_col is None or score_col not in ej.columns:
        raise KeyError(
            f"EJ CSV must contain one of {name_col_candidates} and '{score_col}'. Found columns: {list(ej.columns)}"
        )

    s = pd.to_numeric(ej[score_col], errors="coerce")
    idx = ej[name_col].astype(str).str.strip().str.lower()
    out = pd.Series(s.values, index=idx)
    out = out[~out.index.duplicated(keep="first")]

    # If higher_is_worse, we keep the raw score and later rank descending (worst first)
    # so that rank=1 means highest EJ burden.
    return out


def build_ej_only_rank_table(
        base_features: pd.DataFrame,
        ej_scores: pd.Series,
        higher_is_worse: bool = True,
) -> pd.DataFrame:
    """Build a monthly rank table where final_rank is EJ-only.

    Output columns:
      - wwtp
      - final_rank   (lower = better priority; rank=1 is highest EJ burden if higher_is_worse)
      - rank_ej
      - ej_score
    """
    mask = ~base_features.index.str.contains("historic", case=False, na=False)
    bf = base_features.loc[mask].copy()

    # base_features is indexed by wwtp_clean
    ej_aligned = ej_scores.reindex(bf.index)

    # rank rule: if higher score = worse burden, prioritize higher scores first
    rank_ej = ej_aligned.rank(ascending=not higher_is_worse, method="min")

    out = pd.DataFrame(index=bf.index)
    out["ej_score"] = ej_aligned
    out["rank_ej"] = rank_ej
    out["final_rank"] = rank_ej

    # Restore original-case label for reporting/maps
    name_lookup = bf.reset_index().set_index("wwtp_clean")["wwtp"]
    out.insert(0, "wwtp", name_lookup.reindex(out.index))

    out = out.sort_values("final_rank", ascending=True)

    # Keep the same column order style as the standard rank tables
    return out[["wwtp", "final_rank", "rank_ej", "ej_score"]]


def export_monthly_ej_rank_sheet(
        writer: pd.ExcelWriter,
        month_label: str,
        base_features: pd.DataFrame,
        ej_scores: pd.Series,
        higher_is_worse: bool = True
) -> pd.DataFrame:
    rank_table = build_ej_only_rank_table(base_features, ej_scores, higher_is_worse=higher_is_worse)
    rank_table.reset_index(drop=True).to_excel(writer, sheet_name=month_label, index=False)
    return rank_table
# =============================================================
# Enhanced heatmap + interactive map with TX-RAW overlay
# =============================================================


from typing import Iterable, Set, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import networkx as nx
from math import inf

def select_sentinels_simple(
        G: nx.Graph,
        rank_df: pd.DataFrame,              # rows = WWTP names; columns = category ranks (lower = better)
        features: pd.DataFrame,             # rows = WWTP names; columns = raw metrics for medians (CO-only guard)
        selection_fraction: float = 0.25,
        counties_in_attr: str = "counties_inflow_from",
        counties_out_attr: str = "counties_outflow_to",
        singleton_top_pct: float = 0.30,    # "top X%" protection
        singleton_drop_km: float = 100.0,   # nearest-other distance rule
        diffusion_csv_path: str ="",
        large_component_min_nodes: int = 2,  # minimum size of a large subnetwork
        singleton_bottom_pct: float = 0.20,  # lower-ranked singleton fraction excluded from diffusion
        # --- reserve & scoring knobs for singletons ---
        singleton_reserve_k: int = 0,  # e.g., keep top-K singletons by importance
        singleton_reserve_frac: float = 0.0,  # alternative: fraction of total_N (pre-convert to K upstream)
        isolation_bonus_km: float = 150.0,  # isolation bonus threshold
        risk_weight: float = 0.5,  # importance weight for risk-like info (if present)
        coverage_weight: float = 0.5,  # importance weight for population coverage

        saturate_components: bool = False,
        saturation_gain_min: int = 1,
) -> Tuple[
    Set[str],  # selected_set
    List[str],  # dropped_singletons_list
    Set[str],  # honored_diffusion_set
    Dict[str, str],  # drop_reasons {singleton -> reason}
    Dict[str, str],  # keep_reasons_singletons {singleton -> reason}
    Set[str],  # reserved_singletons (protected from global trim)
    Dict[str, Dict[str, object]]  # decision_log per site
]:
    """
    Selection with:
      - CO-only guards using base features medians
      - diffusion priority (TX-RAW)
      - singleton prune/keep rules (+ optional top-K reserved singletons)
      - coverage-first fill within multi-node components
    Returns selected_set, dropped_singletons, honored_diffusion, drop_reasons, keep_reasons_singletons,
            reserved_singletons, decision_log.

    decision_log includes entries for kept/dropped/trimmed AND for not-chosen sites in multi-node components.
    """

    def canon(s):  # canonical (case-insensitive) key
        return str(s).strip().lower()

    # --- helpers: county coverage & geometry ---
    def county_set_for_wwtp(name: str) -> Set[str]:
        for n in G.nodes:
            nm = (G.nodes[n].get("wwtp") or str(n)).strip()
            if nm == name:
                cin = set(G.nodes[n].get(counties_in_attr, []) or [])
                cout = set(G.nodes[n].get(counties_out_attr, []) or [])
                return {str(x).zfill(5) for x in cin} | {str(x).zfill(5) for x in cout}
        return set()

    name_to_geom: Dict[str, object] = {}
    for n in G.nodes:
        nm = (G.nodes[n].get("wwtp") or str(n)).strip()
        if nm:
            name_to_geom[nm] = G.nodes[n].get("centroid")

    def nearest_other_km(name: str) -> float:
        p = name_to_geom.get(name)
        if p is None:
            return float("inf")
        best = float("inf")
        for other, q in name_to_geom.items():
            if other == name or q is None:
                continue
            d = p.distance(q)
            if d < best:
                best = d
        return best / 1000.0 if best != float("inf") else float("inf")

    # --- rank thresholds: top X% and bottom Y% ---
    import math
    col_top_thresh: Dict[str, float] = {}
    col_bottom_thresh: Dict[str, float] = {}
    for c in rank_df.columns:
        s = pd.to_numeric(rank_df[c], errors="coerce").dropna()
        if s.empty:
            col_top_thresh[c] = -float("inf")
            col_bottom_thresh[c] = -float("inf")
        else:
            col_top_thresh[c] = math.ceil(singleton_top_pct * len(s))
            col_bottom_thresh[c] = math.floor((1.0 - singleton_bottom_pct) * len(s))

    def is_top_any_category(site: str) -> bool:
        if site not in rank_df.index:
            return False
        row = rank_df.loc[site]
        for c in rank_df.columns:
            val = row.get(c)
            if pd.notna(val) and float(val) <= col_top_thresh[c]:
                return True
        return False

    def is_bottom_all_categories(site: str) -> bool:
        if site not in rank_df.index:
            return False
        row = rank_df.loc[site]
        for c in rank_df.columns:
            val = row.get(c)
            if pd.isna(val):
                continue
            if float(val) < col_bottom_thresh[c]:
                return False
        return True

    # --- DIFFUSION CSV (flexible columns; optional 'score') --- (use canonical, case-insensitive)
    diffusion_sites_lower: Set[str] = set()
    diffusion_score_lower: Dict[str, float] = {}
    honored_diffusion: Set[str] = set()

    if diffusion_csv_path:
        try:
            ddf = pd.read_csv(diffusion_csv_path)
            site_col = next((c for c in ddf.columns if c.lower() in ["wwtp", "site", "name"]), None)
            score_col = next((c for c in ddf.columns if "score" in c.lower()), None)
            if site_col is not None:
                if score_col is not None:
                    ddf = ddf.sort_values(score_col, ascending=False)
                    scores = pd.to_numeric(ddf[score_col], errors="coerce")
                else:
                    scores = pd.Series([1.0] * len(ddf), index=ddf.index)
                for nm_raw, s in zip(ddf[site_col].astype(str), scores.tolist()):
                    nm = canon(nm_raw)
                    diffusion_sites_lower.add(nm)
                    if pd.notna(s):
                        diffusion_score_lower[nm] = float(s)
                # If no scores populated, backfill with −avg_rank (lower rank -> higher priority)
                if (not diffusion_score_lower) and (not rank_df.empty):
                    avg_rank_for_diff = rank_df.mean(axis=1)
                    for nm in diffusion_sites_lower:
                        # avg_rank_for_diff index is original case; search best match by case-insensitive:
                        # Try exact, then case-insensitive match
                        if nm in map(canon, avg_rank_for_diff.index):
                            # build a mapping once
                            idx_map = {canon(k): k for k in avg_rank_for_diff.index}
                            s = -float(avg_rank_for_diff.loc[idx_map[nm]])
                            diffusion_score_lower[nm] = s
        except Exception as e:
            print(f"[WARN] Could not load diffusion CSV '{diffusion_csv_path}': {e}")

    # --- global rank + medians for low-impact guard ---
    avg_rank = rank_df.mean(axis=1)
    co_metrics = ["pop_served", "od_volume_total", "pop_covered_by_od", "area_reached"]
    med = features[co_metrics].median(skipna=True)

    # ---------- Build a global importance score for singletons ----------
    risk_like_cols = [c for c in rank_df.columns if ("risk" in c.lower()) or ("transmission" in c.lower())]

    # nearest km for all names
    all_names = {(G.nodes[n].get("wwtp") or str(n)).strip(): n for n in G.nodes}
    nearest_km_all = {}
    for nm, node_id in all_names.items():
        p = G.nodes[node_id].get("centroid")
        if p is None:
            nearest_km_all[nm] = float("inf")
            continue
        best = float("inf")
        for nm2, node2 in all_names.items():
            if nm2 == nm:
                continue
            q = G.nodes[node2].get("centroid")
            if q is None:
                continue
            d = p.distance(q)
            if d < best:
                best = d
        nearest_km_all[nm] = best / 1000.0 if best < float("inf") else float("inf")

    # assemble features aligned to rank_df index
    feat = pd.DataFrame(index=rank_df.index)
    feat["final_rank"] = avg_rank
    feat["pop_served"] = features.reindex(rank_df.index).get("pop_served")
    feat["pop_cov"] = features.reindex(rank_df.index).get("pop_covered_by_od")
    if risk_like_cols:
        r = rank_df[risk_like_cols].mean(axis=1)
        feat["risk_score"] = 1.0 / r.replace(0, np.nan)  # smaller rank -> larger score
    else:
        feat["risk_score"] = 0.0

    def _mm(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.notna().sum() < 2:
            return pd.Series(0.0, index=x.index)
        lo, hi = x.min(), x.max()
        if hi <= lo:
            return pd.Series(0.0, index=x.index)
        return (x - lo) / (hi - lo)

    mm_rank_inv = 1.0 - _mm(feat["final_rank"])  # smaller rank -> larger score
    mm_pop = _mm(feat["pop_served"]) * 0.5 + _mm(feat["pop_cov"]) * 0.5
    mm_risk = _mm(feat["risk_score"])
    iso_bonus = pd.Series(
        {nm: (1.0 if nearest_km_all.get(nm, 0.0) >= isolation_bonus_km else 0.0) for nm in feat.index})

    singleton_importance = (mm_rank_inv
                            + coverage_weight * mm_pop
                            + risk_weight * mm_risk
                            + 0.5 * iso_bonus).fillna(0.0)

    # --- Component IDs & sizes ---
    comps_all = list(nx.connected_components(G))
    node_to_compid: Dict[object, int] = {}
    for cid, comp in enumerate(comps_all, start=1):
        for n in comp:
            node_to_compid[n] = cid

    def comp_id_of_name(name: str) -> Tuple[int, int]:
        for n in G.nodes:
            nm = (G.nodes[n].get("wwtp") or str(n)).strip()
            if nm == name:
                cid = node_to_compid.get(n, -1)
                size = len(next((c for c in comps_all if n in c), []))
                return cid, size
        return -1, 0

    # --- per-site decision log ---
    decision_log: Dict[str, Dict[str, object]] = {}

    # discover singletons & reserve top-K by importance
    singleton_names: List[str] = []
    for comp in comps_all:
        names_here = []
        for n in comp:
            nm = (G.nodes[n].get("wwtp") or str(n)).strip()
            if nm in rank_df.index and "historic" not in nm.lower():
                names_here.append(nm)
        if len(names_here) == 1:
            singleton_names.extend(names_here)

    reserved_singletons: Set[str] = set()
    reserve_count = int(singleton_reserve_k) if singleton_reserve_k else 0
    if reserve_count > 0 and singleton_names:
        s_imp = singleton_importance.reindex(singleton_names).dropna()
        top_keep = s_imp.sort_values(ascending=False).head(reserve_count).index.tolist()
        reserved_singletons = set(top_keep)

    # ---------- main selection by component ----------
    selected: Set[str] = set()
    dropped_singletons: List[str] = []
    drop_reasons: Dict[str, str] = {}
    keep_reasons_singletons: Dict[str, str] = {}

    for comp in comps_all:
        names = []
        for n in comp:
            nm = (G.nodes[n].get("wwtp") or str(n)).strip()
            if nm in rank_df.index and "historic" not in nm.lower():
                names.append(nm)
        if not names:
            continue

        if len(names) == 1:
            # --- SINGLETON COMPONENT ---
            w = names[0]
            cid, csize = comp_id_of_name(w)

            raw = features.loc[w] if w in features.index else None
            low_impact = False
            if raw is not None:
                low_impact = all(
                    pd.notna(raw.get(k)) and float(raw.get(k)) < float(med.get(k, np.nan))
                    for k in co_metrics
                )

            weak_singleton = is_bottom_all_categories(w)
            not_top_any = not is_top_any_category(w)
            near_dist_km = nearest_other_km(w)
            near_other = near_dist_km <= float(singleton_drop_km)
            is_diff = (canon(w) in diffusion_sites_lower)

            # Reserved? keep outright
            if w in reserved_singletons:
                selected.add(w)
                keep_reasons_singletons[w] = "reserved_singleton_high_importance"
                decision_log[w] = {
                    "decision": "keep",
                    "reason": "reserved_singleton_high_importance",
                    "component_id": cid, "component_size": csize,
                    "is_singleton": True, "method": "reserved",
                    "nearest_km": near_dist_km, "diffusion_flag": is_diff,
                    "low_impact_all4<median": low_impact,
                    "top_any_category": not not_top_any,
                    "bottom_all_categories": weak_singleton
                }
                continue

            # Drop if rules match AND either not diffusion or diffusion but weak
            if (low_impact or (not_top_any and near_other)) and (not is_diff or weak_singleton):
                reason_bits = []
                if low_impact:
                    reason_bits.append("low_impact_all4<median")
                if not_top_any and near_other:
                    reason_bits.append(
                        f"not_top_any & nearest≤{int(singleton_drop_km)}km (actual≈{near_dist_km:.1f}km)")
                if is_diff and weak_singleton:
                    reason_bits.append("diffusion_flagged_but_bottom_all_categories")

                reason = " | ".join(reason_bits) if reason_bits else "rule_matched"
                drop_reasons[w] = reason
                dropped_singletons.append(w)
                decision_log[w] = {
                    "decision": "drop",
                    "reason": reason,
                    "component_id": cid, "component_size": csize,
                    "is_singleton": True, "method": "pruning_rule",
                    "nearest_km": near_dist_km, "diffusion_flag": is_diff,
                    "low_impact_all4<median": low_impact,
                    "top_any_category": not not_top_any,
                    "bottom_all_categories": weak_singleton
                }
                continue

            # Keep (passes rules or diffusion helps)
            selected.add(w)
            keep_bits = []
            method = "singleton_pass"
            if is_diff and not weak_singleton:
                honored_diffusion.add(w)
                keep_bits.append("kept_by_diffusion")
                method = "diffusion_singleton"
            if not low_impact:
                keep_bits.append("not_low_impact")
            if is_top_any_category(w):
                keep_bits.append(f"top_any≤{int(round(singleton_top_pct * 100))}%")
            if not near_other:
                keep_bits.append(f"nearest>{int(singleton_drop_km)}km (actual≈{near_dist_km:.1f}km)")

            keep_reasons_singletons[w] = " | ".join(keep_bits) if keep_bits else "kept_by_quota"
            decision_log[w] = {
                "decision": "keep",
                "reason": keep_reasons_singletons[w],
                "component_id": cid, "component_size": csize,
                "is_singleton": True, "method": method,
                "nearest_km": near_dist_km, "diffusion_flag": is_diff,
                "low_impact_all4<median": low_impact
            }
            continue

        # --- MULTI-NODE COMPONENT ---
        sub = rank_df.mean(axis=1).loc[rank_df.index.intersection(names)].copy()
        if sub.empty:
            continue

        n_select = max(1, int(np.ceil(selection_fraction * len(sub))))

        chosen: List[str] = []
        covered: Set[str] = set()
        site_cov = {w: county_set_for_wwtp(w) for w in sub.index}
        cid = next((cid for n in comp for cid in [node_to_compid[n]]), -1)

        # Preselect diffusion sites (priority)
        comp_diff = [w for w in names if (canon(w) in diffusion_sites_lower) and (w in sub.index)]
        if len(names) >= int(large_component_min_nodes) and comp_diff:
            comp_diff_sorted = sorted(
                comp_diff,
                key=lambda w: (
                    -(diffusion_score_lower.get(canon(w), 0.0)),
                    float(sub.loc[w])
                )
            )
            for w in comp_diff_sorted:
                if len(chosen) >= n_select:
                    break
                if w not in chosen:
                    chosen.append(w)
                    honored_diffusion.add(w)
                    gain = len(site_cov[w])  # first additions have full gain
                    decision_log[w] = {
                        "decision": "keep",
                        "reason": "diffusion_priority",
                        "component_id": cid,
                        "component_size": len(comp),
                        "is_singleton": False,
                        "method": "diffusion_priority",
                        "coverage_gain": int(gain),
                        "avg_rank": float(sub.loc[w]),
                        "diffusion_score": float(diffusion_score_lower.get(canon(w), 0.0))
                    }
                    covered |= site_cov[w]

        # Fill remaining slots via coverage-first, then better avg rank
        remaining = [w for w in sub.index if w not in chosen]
        while remaining and len(chosen) < n_select:
            best = None
            best_key = None
            best_gain = 0
            for w in remaining:
                gain = len(site_cov[w] - covered)
                key = (-gain, float(sub.loc[w]))
                if best is None or key < best_key:
                    best, best_key = w, key
                    best_gain = gain
            chosen.append(best)
            decision_log[best] = {
                "decision": "keep",
                "reason": "coverage_first_then_rank",
                "component_id": cid,
                "component_size": len(comp),
                "is_singleton": False,
                "method": "coverage_gain",
                "coverage_gain": int(best_gain),
                "avg_rank": float(sub.loc[best]),
                "diffusion_flag": canon(best) in diffusion_sites_lower
            }
            covered |= site_cov[best]
            remaining.remove(best)

            # ---------------------------------------------------
            # Saturate within the component after satisfying its initial quota.
            # Only adds a site if it contributes >= saturation_gain_min new counties
            # ---------------------------------------------------
            if saturate_components:
                remaining = [w for w in sub.index if w not in chosen]
                while remaining:
                    best = None
                    best_key = None
                    best_gain = 0

                    for w in remaining:
                        gain = len(site_cov[w] - covered)
                        if gain < int(saturation_gain_min):
                            continue
                        key = (-gain, float(sub.loc[w]))  # same scoring
                        if best is None or key < best_key:
                            best, best_key = w, key
                            best_gain = gain

                    if best is None:
                        break

                    chosen.append(best)
                    decision_log[best] = {
                        "decision": "keep",
                        "reason": f"saturation_add_gain>={int(saturation_gain_min)}",
                        "component_id": cid,
                        "component_size": len(comp),
                        "is_singleton": False,
                        "method": "saturation_component",
                        "coverage_gain": int(best_gain),
                        "avg_rank": float(sub.loc[best]),
                        "diffusion_flag": canon(best) in diffusion_sites_lower
                    }
                    covered |= site_cov[best]
                    remaining.remove(best)

        # mark selected from this component
        selected.update(chosen)

        # Record nonselected sites for component-level selection diagnostics.
        # use the final covered set to compute residual coverage gain if this site had been added
        for w in sub.index:
            if w in chosen:
                continue
            potential_gain = len(site_cov[w] - covered)
            decision_log.setdefault(w, {
                "decision": "not_chosen",
                "reason": "not_selected_in_component",
                "component_id": cid,
                "component_size": len(comp),
                "is_singleton": False,
                "method": "component_quota_or_lower_gain",
                "coverage_gain_if_added": int(potential_gain),
                "avg_rank": float(sub.loc[w]),
                "diffusion_flag": canon(w) in diffusion_sites_lower
            })

    reserved_singletons_set = set(reserved_singletons)
    to_remove = {s for s in keep_reasons_singletons.keys() if s not in reserved_singletons_set}

    if to_remove:
        print(f"[Rule A] Removing {len(to_remove)} non-reserved singleton(s): "
              f"{', '.join(list(to_remove)[:6])}{'…' if len(to_remove) > 6 else ''}")
        selected -= to_remove

    return selected, dropped_singletons, honored_diffusion, drop_reasons, keep_reasons_singletons, reserved_singletons, decision_log


def visualize_rank_with_subnetwork_coverage(
        G,
        save_path="outputs/sentinel_rank_heatmap.png",
        sewersheds_gdf=None,
        output_html_map="outputs/maps/sentinel_ranked_map.html",
        county_alias_map=None,
        # ----- selection / ranking controls -----
        selection_fraction=0.25,
        singleton_top_pct=0.30,
        singleton_drop_km=100.0,
        singleton_bottom_pct=0.20,
        total_N=20,

        # ----- singleton reserve & scoring knobs -----
        singleton_reserve_k: int = 0,
        singleton_reserve_frac: float = 0.0,
        isolation_bonus_km: float = 150.0,
        risk_weight: float = 0.5,
        coverage_weight: float = 0.5,

        # ----- DISPLAY-ONLY series (do NOT affect ranking) -----
        risk_series: Optional[pd.Series] = None,  # e.g., combined COVID risk

        # ----- OPTIONAL: override selection/ordering with an external final-rank series -----
        # If provided, this ONLY changes the greedy ordering used for selection + heatmap row sorting.
        # The heatmap columns (pop/OD/pop/area + optional risk/EJ/TX-RAW) are kept the same.
        final_rank_override: Optional[pd.Series] = None,

        # ----- diffusion / annotation -----
        diffusion_csv: Optional[str] = None,

        # ----- misc / rendering -----
        ej_csv: Optional[str] = None,
        # ----- ranking strategy -----
        ranking_mode: str = "base",  # "base" | "risk" | "ej"
        ej_scores: Optional[pd.Series] = None,  # preloaded EJ CombinedScore by wwtp_clean
        ej_higher_is_worse: bool = True,
        render_heatmap: bool = True,
        month_label: Optional[str] = None,
        sets_out_dir: Optional[str] = None,
        heatmap_suffix="",
        heatmap_title_suffix="",
        n_official_extras: int = 6,
        write_summaries: bool = True,
        write_singleton_files: bool = False,
        two_phase_mode: bool = True,
        bg_link_dir: Optional[str] = None,
        bg_start_date: Optional[str] = None,
        bg_end_date: Optional[str] = None,
        write_bg_unique_for_selected: bool = False,
        # Optional HTML overlays
        primary_roads_shp: Optional[str] = None,
        airports_shp: Optional[str] = None,
        airport_name_field: Optional[str] = None,
):
    """
    Ranks WWTPs and selects sentinels. In two_phase_mode (default=True):
      - Phase A: selection is done WITHOUT diffusion (clean baseline).
      - Phase B: diffusion CSV (if provided) is used ONLY to tag sites:
          * 'reinforced_*'  (selected ∩ diffusion)
          * 'normal_*'         (diffusion − selected)
      Visualization layer "Diffusion impact (reinforced / removed)" uses:
          green star   = reinforced (selected ∩ diffusion)
          blue check   = selected-only (baseline selected but not diffusion)
          dark grey -  = in diffusion but not selected (cut)
    """
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    if county_alias_map is None:
        county_alias_map = {}
    def canon(s):
        return str(s).strip().lower()

    # ---- Build features directly from graph node attributes ----
    rows = []
    for node in G.nodes:
        attrs = G.nodes[node]
        wwtp = (attrs.get("wwtp") or str(node)).strip()
        if "historic" in wwtp.lower():
            continue

        vol_in = attrs.get("od_volume_in", 0.0) or 0.0
        vol_out = attrs.get("od_volume_out", 0.0) or 0.0
        vol_total = attrs.get("od_volume_total")
        if vol_total is None:
            vol_total = vol_in + vol_out

        pop_served = attrs.get("pop_served", np.nan)
        pop_in = attrs.get("pop_from_od_in_counties", 0.0) or 0.0
        pop_out = attrs.get("pop_to_od_out_counties", 0.0) or 0.0
        pop_cov = attrs.get("pop_covered_by_od")
        if pop_cov is None:
            pop_cov = pop_in + pop_out

        area_in = attrs.get("area_from_od_in_counties", 0.0) or 0.0
        area_out = attrs.get("area_to_od_out_counties", 0.0) or 0.0
        area_total = area_in + area_out

        rows.append({
            "wwtp": wwtp,
            "pop_served": pop_served,
            "od_volume_total": vol_total,
            "pop_covered_by_od": pop_cov,
            "area_reached": area_total,
        })

    features = pd.DataFrame(rows).set_index("wwtp").dropna(how="all")
    if features.empty:
        print("No valid features available for ranking.")
        return

    candidate_fields = ["pop_served", "od_volume_total", "pop_covered_by_od", "area_reached"]

    scaler = MinMaxScaler()
    features_norm = pd.DataFrame(
        scaler.fit_transform(features[candidate_fields].fillna(0)),
        columns=candidate_fields,
        index=features.index,
    )

    # Base ranks (lower rank = better, i.e., larger raw value is better)
    rank_df = features[candidate_fields].rank(ascending=False, method="min")
    pretty_names = {
        "pop_served": "Population \nServed",
        "od_volume_total": "Commute \nVolume",
        "pop_covered_by_od": "Population \nReached",
        "area_reached": "Area\n Reached",
    }
    rank_df.columns = [pretty_names.get(c, c) for c in rank_df.columns]

    # -----------------------------------------------------------------
    # Build TWO rank tables:
    #   - rank_df_display: what we show in the heatmap (may include risk)
    #   - rank_df_sel: what we use for selection (depends on ranking_mode)
    #   IMPORTANT:
    #     * TX-RAW and EJ must NOT affect ranking unless explicitly requested.
    # -----------------------------------------------------------------
    rank_df_base = rank_df.copy()
    rank_df_display = rank_df_base.copy()
    rank_df_sel = rank_df_base.copy()

    # ---- Risk rank (can affect selection if ranking_mode == "risk") ----
    if risk_series is not None and not risk_series.dropna().empty:
        aligned = risk_series.reindex(rank_df_base.index)
        risk_rank = aligned.rank(ascending=False, method="min")  # higher risk => better (rank=1)
        rank_df_display["Mobility\nRisk"] = risk_rank

        if str(ranking_mode).lower() == "risk":
            # Risk participates in SELECTION
            rank_df_sel["Mobility\nRisk"] = risk_rank



    # ---- EJ-only selection mode ----
    ej_rank = None
    if str(ranking_mode).lower() == "ej":
        # Prefer preloaded ej_scores (wwtp_clean -> CombinedScore); fallback to reading ej_csv
        if ej_scores is None or len(getattr(ej_scores, "index", [])) == 0:
            ej_scores_local = load_ej_scores(ej_csv) if ej_csv else pd.Series(dtype=float)
        else:
            ej_scores_local = pd.to_numeric(ej_scores, errors="coerce")
        ej_scores_local = ej_scores_local.reindex(
            pd.Index(rank_df_base.index).astype(str).str.strip().str.lower()
        )
        # Map ej_scores_local (lowercase index) back to the original-case WWTP index
        idx_map = {str(k).strip().lower(): k for k in rank_df_base.index}
        ej_aligned = pd.Series(index=rank_df_base.index, dtype=float)
        for k_l, v in ej_scores_local.dropna().items():
            if k_l in idx_map:
                ej_aligned.loc[idx_map[k_l]] = float(v)

        # higher_is_worse=True => higher burden gets rank=1 (best priority)
        asc = not bool(ej_higher_is_worse)
        ej_rank = ej_aligned.rank(ascending=asc, method="min")

        # Selection uses ONLY EJ ranks
        rank_df_sel = pd.DataFrame({"EJ": ej_rank})
        # Heatmap can still display EJ column; we add it later (existing logic)

    # NOTE: In two-phase, we DO NOT add a "Transmission" (diffusion) column into the ranking table.
    #       TX-RAW can be displayed (heatmap/map) but must not change selection order.

    # Composite rank (lower = better) used for ordering & "official extras" picking
    if ej_rank is not None and ej_rank.notna().any():
        avg_rank = ej_rank.copy()
    else:
        avg_rank = rank_df_sel.mean(axis=1)
    # --- compute reserve_k (either absolute or from frac of total_N) ---
    reserve_k = int(singleton_reserve_k or round(singleton_reserve_frac * total_N))

    # =========================================================
    # Phase A: Selection call WITHOUT diffusion participation
    # =========================================================

    sel_diffusion_path = None if two_phase_mode else diffusion_csv  # two-phase -> ignore diffusion in selection

    # =========================================================
    # EJ strategy (PURE): NO subnetwork enforcement, NO singleton rules,
    # NO coverage constraints, NO diffusion participation.
    # Selection is simply top-N by EJ rank.
    # =========================================================
    if str(ranking_mode).lower() == "ej":
        ordered = avg_rank.sort_values(kind="mergesort").index.tolist()
        selected_wwtps = set(ordered[: int(total_N)])
        dropped_singletons = set()
        honored_diff = set()
        drop_reasons = {}
        keep_reasons_singletons = {}
        reserved_singletons = set()
        decision_log = {}  # pure EJ: keep dict for downstream tagging/exports
    else:

        sel_diffusion_path = None if two_phase_mode else diffusion_csv  # two-phase -> ignore diffusion in selection

        (selected_wwtps,
         dropped_singletons,
         honored_diff,  # will be empty in two-phase because we ignore diffusion in selection
         drop_reasons,
         keep_reasons_singletons,
         reserved_singletons,
         decision_log) = select_sentinels_simple(
            G,
            rank_df=rank_df_sel,
            features=features,
            selection_fraction=selection_fraction,
            counties_in_attr="counties_inflow_from",
            counties_out_attr="counties_outflow_to",
            singleton_top_pct=singleton_top_pct,
            singleton_drop_km=singleton_drop_km,
            diffusion_csv_path=sel_diffusion_path or "",  # <- empty in two-phase
            large_component_min_nodes=2,
            singleton_bottom_pct=singleton_bottom_pct,
            singleton_reserve_k=reserve_k,
            singleton_reserve_frac=0.0,  # already converted to K
            isolation_bonus_km=isolation_bonus_km,
            risk_weight=risk_weight,
            coverage_weight=coverage_weight,
            saturate_components=True,
            saturation_gain_min=1  # minimum gain of one additional county
        )

        selected_raw_set = set(selected_wwtps)
        total_N = int(total_N)
        ordered_all = avg_rank.sort_values(kind="mergesort").index.tolist()

        # ------------------------------------------------------------------
        # Final fixed-budget trimming with hard singleton reserve and
        # minimum subnetwork-quota preservation.
        #
        # Interpretation:
        #   - Singleton subnetworks that passed the singleton screen are treated
        #     as hard spatial reserves.
        #   - Multi-site subnetwork quota picks are treated as minimum
        #     representation, not as final proportional allocation.
        #   - Saturation / expansion sites remain optional and are used only
        #     after the minimum representation set is kept.
        #
        # This avoids the previous behavior where raw_ordered[:total_N] could
        # accidentally remove a protected singleton or a required quota pick
        # when total_N is reduced from 21 to 20.
        # ------------------------------------------------------------------
        raw_ordered = [s for s in ordered_all if s in selected_wwtps]
        reserved_set = set(reserved_singletons)

        # Sites selected by the pruning/singleton rules.
        hard_singletons = {
            w for w, info in decision_log.items()
            if (w in selected_wwtps)
            and (str(info.get("decision", "")).lower() == "keep")
            and bool(info.get("is_singleton", False))
        }

        # Sites selected to satisfy the multi-site component quota.
        # Exclude within-component saturation additions because those are
        # useful expansion candidates but not minimum quota requirements.
        quota_core = {
            w for w, info in decision_log.items()
            if (w in selected_wwtps)
            and (str(info.get("decision", "")).lower() == "keep")
            and (not bool(info.get("is_singleton", False)))
            and (str(info.get("method", "")) != "saturation_component")
        }

        reserved_singleton_ordered = [s for s in ordered_all if s in hard_singletons and s in reserved_set]
        other_singleton_ordered = [s for s in ordered_all if s in hard_singletons and s not in reserved_set]
        quota_core_ordered = [s for s in ordered_all if s in quota_core and s not in hard_singletons]

        minimum_ordered = []
        for pool in (reserved_singleton_ordered, other_singleton_ordered, quota_core_ordered):
            for s in pool:
                if s not in minimum_ordered:
                    minimum_ordered.append(s)

        if len(minimum_ordered) > total_N:
            print(
                f"[MinQuota][WARN] hard singleton + quota-minimum set "
                f"({len(minimum_ordered)}) exceeds total_N={total_N}. "
                "Keeping highest-priority minimum sites by the hard-priority order."
            )
            selected_ordered = minimum_ordered[:total_N]
        else:
            remaining_ordered = [s for s in raw_ordered if s not in set(minimum_ordered)]
            selected_ordered = minimum_ordered + remaining_ordered[: max(0, total_N - len(minimum_ordered))]

        # Fill if still too few. This should be rare and only occurs if the
        # selected set has fewer than total_N sites.
        if len(selected_ordered) < total_N:
            selected_set = set(selected_ordered)
            fillers = [s for s in ordered_all if s not in selected_set]
            selected_ordered += fillers[: (total_N - len(selected_ordered))]

        print(
            f"[MinQuota] final N={total_N}: "
            f"hard_singletons={len(set(selected_ordered) & hard_singletons)}/{len(hard_singletons)}, "
            f"quota_core={len(set(selected_ordered) & quota_core)}/{len(quota_core)}, "
            f"reserved_singletons={len(set(selected_ordered) & reserved_set)}/{len(reserved_set)}"
        )

        # ------------------------------------------------------------------
        # Display order for cumulative/expansion figures.
        #
        # The membership of the fixed-budget portfolio is selected_ordered and
        # remains unchanged. However, selected_ordered begins with hard-reserve
        # singletons and quota-minimum sites because that is the internal
        # constraint-enforcement sequence. For cumulative benefit visualization,
        # that internal order can make the early part of the curve look
        # artificially weak or jumpy.
        #
        # Therefore, for display only, we sort the selected N sites back by the
        # overall Level-1 rank, then append all remaining sites by the same
        # full ranked order. This does not change which N sites are selected.
        # ------------------------------------------------------------------
        selected_set_for_display = set(selected_ordered)
        selected_display_order = [s for s in ordered_all if s in selected_set_for_display]
        remaining_display_order = [s for s in ordered_all if s not in selected_set_for_display]
        ordered_all_display = selected_display_order + remaining_display_order

        print(
            f"[DisplayOrder] selected membership unchanged: N={len(selected_ordered)}; "
            f"cumulative figures use overall-rank display order."
        )



        # Expand the ranked order to support full cumulative-benefit curves.
        expanded_list, phase_label = build_expanded_selected_order(
            G=G,
            ordered_all=ordered_all_display,
            baseline_21=selected_display_order,  # display order only; membership unchanged
            selected_raw_set=selected_raw_set,  # saturation extras come from here
            expand_cap = len(ordered_all_display),  # change if needed
            backforth_pattern=("S", "M", "M"),  #  rule
        )
        print("expanded_list len:", len(expanded_list))
        print("unique len:", len(set(expanded_list)))
        dups = pd.Series(expanded_list).duplicated()
        print("duplicate count:", int(dups.sum()))
        if dups.any():
            print("dups:", pd.Series(expanded_list)[dups].tolist())

        print("[EXPAND] counts:",
              "baseline=", sum(v == "baseline" for v in phase_label.values()),
              "saturation=", sum(v == "saturation" for v in phase_label.values()),
              "backforth=", sum(v == "backforth" for v in phase_label.values()))

        # Save the expansion order used by manuscript diagnostics.
        try:
            df_expand = pd.DataFrame({
                "order": range(1, len(expanded_list) + 1),
                "wwtp": expanded_list,
                "phase": [phase_label.get(w, "") for w in expanded_list],
                "avg_rank": [float(avg_rank.get(w, np.nan)) for w in expanded_list],
            })
            df_expand.to_csv(os.path.join(os.path.dirname(save_path), f"{month_label}_expansion_order.csv"),
                             index=False, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not write expansion_order.csv: {e}")

        selected_wwtps = set(selected_ordered)
        # ---------------------------------------------------
        # BG-unique cumulative coverage for the same selected_ordered list
        # ---------------------------------------------------
        if write_bg_unique_for_selected:
            if (bg_link_dir is None) or (bg_start_date is None) or (bg_end_date is None):
                print("[BG_SELECTED] Skipped: bg_link_dir/start/end not provided.")
            else:
                # Canonicalize selected list to match BG-Link keys (usually lowercase)
                # Use display order for the selected-portfolio cumulative curve.
                # Membership is identical to selected_ordered.
                sel_clean = [str(s).strip().lower() for s in selected_display_order]

                # Build an avg_rank that forces EXACT order = selected_ordered
                # (0,1,2,... means the function will process in this order)
                avg_rank_sel = pd.Series(range(len(sel_clean)), index=sel_clean, dtype=float)

                # pop_served Series must be indexed by wwtp_clean (lowercase)
                pop_served_series = (
                    features["pop_served"]
                    .copy()
                )
                pop_served_series.index = pop_served_series.index.astype(str).str.strip().str.lower()

                save_bg_unique_outputs(
                    avg_rank=avg_rank_sel,
                    bg_link_dir=bg_link_dir,
                    start_date=bg_start_date,
                    end_date=bg_end_date,
                    out_dir=os.path.dirname(save_path),
                    label=f"{month_label}_BG_SELECTED_N{len(sel_clean)}",
                    direction="Destination",
                    union_mode="cap_sum",
                    weight_from="Volume",

                    # Use the same exponential-intensity settings as the main workflow.
                    bg_weight_mode="intensity_exp",
                    tau=2,
                    eps_pop=100.0,

                    pop_served=pop_served_series,
                    pop_served_total_override=state_population

                )

                # BG-unique coverage for the full expansion list (diagnostic curve)
                full_clean = [str(s).strip().lower() for s in expanded_list]
                avg_rank_full = pd.Series(range(len(full_clean)), index=full_clean, dtype=float)

                save_bg_unique_outputs(
                    avg_rank=avg_rank_full,
                    bg_link_dir=bg_link_dir,
                    start_date=bg_start_date,
                    end_date=bg_end_date,
                    out_dir=os.path.dirname(save_path),
                    label=f"{month_label}_BG_EXPAND_N{len(full_clean)}",  # distinct label; won't overwrite trimmed
                    direction="Destination",
                    union_mode="cap_sum",
                    weight_from="Volume",
                    bg_weight_mode="intensity_exp",
                    tau=2,
                    eps_pop=100.0,
                    pop_served=pop_served_series,
                    pop_served_total_override=state_population
                )
                print("[BG_EXPAND] Wrote BG-unique cumulative for expanded list.")

                print("[BG_SELECTED] Wrote BG-unique cumulative for selected sites.")

    # --- singleton explanations (baseline) ---
    if write_summaries:
        try:
            rows = []
            all_singletons = set(drop_reasons.keys()) | set(keep_reasons_singletons.keys())
            for w in sorted(all_singletons):
                is_kept = w in selected_wwtps
                rows.append({
                    "wwtp": w,
                    "decision": "kept" if is_kept else "dropped",
                    "reason": keep_reasons_singletons.get(w) if is_kept else drop_reasons.get(w),
                    "reserved": w in reserved_singletons,
                    "avg_rank": float(avg_rank.get(w, np.nan))
                })
            if rows:
                folder = os.path.dirname(save_path)
                os.makedirs(folder, exist_ok=True)
                df_single = pd.DataFrame(rows)
                df_single.to_csv(os.path.join(folder, "singleton_decisions.csv"), index=False, encoding="utf-8")
                with open(os.path.join(folder, "singleton_decisions.txt"), "w", encoding="utf-8") as f:
                    f.write("# Singleton Decisions (Baseline selection; no diffusion)\n\n## Dropped\n")
                    for w in sorted(drop_reasons):
                        f.write(f"- {w}: {drop_reasons[w]}\n")
                    f.write("\n## Kept\n")
                    for w in sorted(keep_reasons_singletons):
                        flag = " [RESERVED]" if w in reserved_singletons else ""
                        f.write(f"- {w}{flag}: {keep_reasons_singletons[w]}\n")
                print(f"[Explain] Singleton decisions written to {folder}")
        except Exception as e:
            print(f"[Warn] Could not write singleton explanations: {e}")

    # =========================================================
    # Phase B: Post-selection diffusion tags (annotation only)
    #        - subnetwork-aware (singleton vs component)
    # =========================================================
    # Build component lookup for names
    name_to_node = {}
    for n in G.nodes:
        nm = (G.nodes[n].get("wwtp") or str(n)).strip()
        if nm:
            name_to_node[canon(nm)] = n

    comps = list(nx.connected_components(G))
    node_to_compid = {}
    for cid, comp in enumerate(comps, start=1):
        for n in comp:
            node_to_compid[n] = cid
    comp_size = {cid: len(comp) for cid, comp in enumerate(comps, start=1)}

    def comp_info(name_lower):
        n = name_to_node.get(name_lower)
        if n is None:
            return -1, 0
        cid = node_to_compid.get(n, -1)
        return cid, comp_size.get(cid, 0)

    selected_lower = {canon(s) for s in selected_wwtps}

    # Diffusion set from CSV (names only)
    diffusion_lower = set()
    diffusion_score_lower = {}  # retain scores for map popups and labels when available

    TOP_DIFFUSION_N = 20  # number of leading diffusion scores retained

    # ---- Add TX-RAW (diffusion) rank to heatmap if a diffusion CSV is provided ----
    # This is DISPLAY-ONLY. Selection still ignores diffusion in two-phase mode.
    if diffusion_csv and os.path.exists(diffusion_csv):
        try:
            ddf = pd.read_csv(diffusion_csv)
            name_col = next((c for c in ddf.columns if c.lower() in ["wwtp", "site", "name"]), None)
            score_col = next((c for c in ddf.columns if "score" in c.lower()), None)
            if name_col is not None:
                # canon → original-case index mapping for heatmap alignment
                idx_map = {str(k).strip().lower(): k for k in rank_df_display.index}

                # Heatmap diffusion-score column
                tx_score = pd.Series(index=rank_df_display.index, dtype=float)
                if score_col is not None:
                    for nm_raw, s in zip(ddf[name_col].astype(str), pd.to_numeric(ddf[score_col], errors="coerce")):
                        key = str(nm_raw).strip().lower()
                        if key in idx_map and pd.notna(s):
                            tx_score.loc[idx_map[key]] = float(s)
                else:
                    # Presence-only: present=1, absent=NaN
                    for nm_raw in ddf[name_col].dropna().astype(str):
                        key = str(nm_raw).strip().lower()
                        if key in idx_map:
                            tx_score.loc[idx_map[key]] = 1.0

                if tx_score.notna().any():
                    tx_rank = tx_score.rank(ascending=False, method="min")
                    rank_df_display["Transmission"] = tx_rank

                # Build normalized diffusion-score lookup
                # Use scores if available; otherwise take first TOP_DIFFUSION_N names
                ddf_names = ddf[name_col].dropna().astype(str)

                if score_col is not None:
                    ddf_scores = pd.to_numeric(ddf[score_col], errors="coerce")
                    ddf_use = (
                        pd.DataFrame({"name": ddf_names, "score": ddf_scores})
                        .dropna(subset=["name"])
                        .sort_values("score", ascending=False)
                    )
                    top_rows = ddf_use.head(TOP_DIFFUSION_N).itertuples(index=False)
                    for nm_raw, s in top_rows:
                        nm_can = canon(nm_raw) if 'canon' in globals() else str(nm_raw).strip().lower()
                        diffusion_lower.add(nm_can)
                        if pd.notna(s):
                            diffusion_score_lower[nm_can] = float(s)
                else:
                    # No score: honor file order, unique, top N
                    for nm_raw in ddf_names.drop_duplicates().head(TOP_DIFFUSION_N):
                        nm_can = canon(nm_raw) if 'canon' in globals() else str(nm_raw).strip().lower()
                        diffusion_lower.add(nm_can)
                        diffusion_score_lower.setdefault(nm_can, 1.0)

        except Exception as e:
            print(f"[WARN] Could not build TX-RAW rank column / diffusion set: {e}")

    reinforced_lower = selected_lower & diffusion_lower  # selected & diffusion
    normal_lower = selected_lower - diffusion_lower

    # Component has selected?
    comp_has_selected = {}
    for nm_l in selected_lower:
        cid, _ = comp_info(nm_l)
        if cid != -1:
            comp_has_selected[cid] = True

    # Helper to recover original case (for CSVs/maps)
    def _orig_case(idx_like):
        for ix in rank_df_display.index:
            if canon(ix) == idx_like:
                return ix
        return idx_like

    # Tag 'reinforced_*' into decision_log
    for nm_l in reinforced_lower:
        cid, size = comp_info(nm_l)
        tag = "reinforced_singleton" if size == 1 else "reinforced_component"
        nm = _orig_case(nm_l)
        decision_log.setdefault(nm, {}).update({
            "post_diffusion": tag,
            "component_id": cid,
            "component_size": size
        })

    # Tag 'cut_*' into decision_log
    for nm_l in normal_lower:
        cid, size = comp_info(nm_l)
        if size == 1:
            tag = "normal_singleton"
        else:
            tag = "normal_component_redundant" if comp_has_selected.get(cid, False) else "normal_component_uncovered"
        nm = _orig_case(nm_l)
        d = decision_log.setdefault(nm, {})
        # Don't overwrite an existing concrete decision like 'keep' — just annotate
        if "decision" not in d:
            d["decision"] = "not_selected"
        d.update({"post_diffusion": tag, "component_id": cid, "component_size": size})

    # Small console summary
    if diffusion_lower:
        print("\n[Phase-B Diffusion Annotation]")
        print(f"  Reinforced (selected ∩ diffusion): {len(reinforced_lower)}")
        print(f"  Normal (selected - diffusion  ):       {len(normal_lower)}\n")

    # ---- Heatmap (selected + a few official-but-not-selected) ----
    if render_heatmap and selected_wwtps:
        # official sites to label in heatmap (⚑)
        official_sites = set()
        if sewersheds_gdf is not None and "act_sts" in sewersheds_gdf.columns:
            sewersheds_wgs84 = sewersheds_gdf.to_crs(epsg=4326)
            sentinel_mask = (
                    sewersheds_wgs84["act_sts"].astype(str).str.strip().str.lower()
                    == "sentinel surveillance site"
            )
            official_sites = {
                (row.get("wwtp") or "").strip()
                for _, row in sewersheds_wgs84[sentinel_mask].iterrows()
                if (row.get("wwtp") or "").strip()
                   and "historic" not in (row.get("wwtp") or "").lower()
            }

        selected_set = set(selected_wwtps)
        off_only = list(official_sites - selected_set)
        # choose a few by best avg rank
        off_only_sorted = sorted(off_only, key=lambda w: float(avg_rank.get(w, np.inf)))[:n_official_extras]

        rows_to_plot = [r for r in list(selected_set) + off_only_sorted if r in rank_df_display.index]
        final_df = rank_df_display.loc[rows_to_plot].copy()
        final_df = final_df.loc[avg_rank.loc[final_df.index].sort_values().index]

        # ---- ADD EJ Combined as a heatmap-only column (does not affect selection) ----
        if ej_csv:
            try:
                ej = pd.read_csv(ej_csv)
                # Locate a name column matching the normalized WWTP labels.
                name_col = next((c for c in ej.columns if c.lower() in ["wwtp", "wwtp_id", "name"]), None)
                if (name_col is not None) and ("CombinedScore" in ej.columns):
                    # Canonicalize on lowercase for safe alignment with heatmap rows
                    idx_map = {str(k).strip().lower(): k for k in final_df.index}
                    ej_key = ej[name_col].astype(str).str.strip().str.lower()
                    ej_vals = pd.to_numeric(ej["CombinedScore"], errors="coerce")

                    ej_series_lower = pd.Series(ej_vals.values, index=ej_key)
                    # Reindex to heatmap rows (by their lowercase keys), then restore original casing
                    aligned = ej_series_lower.reindex(idx_map.keys())
                    aligned.index = [idx_map[i] for i in aligned.index]

                    # Convert EJ CombinedScore (0–1 or 0–100, higher=worse) to a "rank" like other columns (lower=better)
                    ej_rank = aligned.rank(ascending=False, method="min")

                    # Insert as the last column with a pretty header
                    final_df["EJ"] = ej_rank
                else:
                    print("[EJ] Could not find name column (wwtp/wwtp_id/name) and CombinedScore in EJ CSV.")
            except Exception as e:
                print(f"[EJ] Failed to append EJ Combined to heatmap: {e}")

        # -------------------------------------------------------------------------------

        def _row_label(name: str) -> str:
            disp = county_alias_map.get(str(name).strip().lower(), str(name).strip())
            tags = []
            if name in selected_set:
                tags.append("▲")
            if name in official_sites:
                tags.append("■")
            tag = (" " + "".join(tags)) if tags else ""
            return f"{disp}{tag}"

        ylabels = [_row_label(nm) for nm in final_df.index]

        heatmap_title = f"Sentinel Site Ranking"
        filename_tag = f"_heatmap_twophase{heatmap_suffix}.png"

        extra_note = ""
        if off_only_sorted:
            extra_note = f"\n(includes {len(off_only_sorted)} official not selected)"

        # heatmap_title += f"  •  sf={int(round(selection_fraction * 100))}%, N={total_N}{extra_note}"

        save_dir = os.path.dirname(save_path)
        base = os.path.splitext(os.path.basename(save_path))[0]
        heatmap_path = os.path.join(save_dir, base + filename_tag)

        fig_h = max(5.6, 0.62 * len(final_df) + 0.55)
        fig, ax = plt.subplots(figsize=(10.4, fig_h))
        hm = sns.heatmap(
            final_df,
            cmap="viridis_r",
            annot=True,
            fmt=".0f",
            linewidths=0.3,
            linecolor="gray",
            annot_kws={"size": 18},
            cbar=True,
            cbar_kws={"label": "Rank scale", "shrink": 0.82, "pad": 0.02},
            ax=ax,
        )
        ax.set_yticklabels(ylabels, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=18)
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("Rank scale", fontsize=13)
        note = (
            "Numbers are within-column ranks (1 = highest priority / strongest contribution). "
            "Lower ranks are better. ▲ selected site; ■ existing official sentinel site."
        )
        fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=10.0, color="dimgray")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Rank heatmap saved to {heatmap_path} (official-only shown with ⚑; selected with ★)")

    # ---- Map layers ----
    if sewersheds_gdf is not None:
        try:
            sewersheds_wgs84
        except NameError:
            sewersheds_wgs84 = sewersheds_gdf.to_crs(epsg=4326)
        centroids = sewersheds_wgs84.geometry.centroid
        m = folium.Map(location=[centroids.y.mean(), centroids.x.mean()], zoom_start=7, tiles="CartoDB Positron")

        # edges overlay (optional)
        edges_fg = folium.FeatureGroup(name="Network Edges", show=False)
        for u, v in G.edges:
            try:
                p1 = sewersheds_wgs84.loc[u].geometry.centroid
                p2 = sewersheds_wgs84.loc[v].geometry.centroid
                folium.PolyLine([(p1.y, p1.x), (p2.y, p2.x)], color="#777777", weight=1, opacity=0.5).add_to(edges_fg)
            except Exception:
                continue
        edges_fg.add_to(m)

        # Canonical sets for maps
        selected_lower = {canon(s) for s in selected_wwtps}
        try:
            official_sites
        except NameError:
            official_sites = set()
        official_lower = {canon(s) for s in official_sites} if official_sites else set()

        # --- Diffusion impact layer (reinforced / removed) ---
        diffusion_lower_keys = set(diffusion_lower)
        if diffusion_csv and diffusion_lower_keys:
            diff_fg = folium.FeatureGroup(name="Transmission reinforced", show=True)

            def is_reinforced(name):
                return canon(name) in reinforced_lower  # selected ∩ diffusion

            def is_normal(name):
                return canon(name) in normal_lower  # diffusion \ selected

            for _, row in sewersheds_wgs84.iterrows():
                name = (row.get("wwtp") or "").strip()
                if not name or ("historic" in name.lower()):
                    continue
                c = row.geometry.centroid

                if is_reinforced(name):
                    folium.Marker(
                        location=[c.y, c.x],
                        icon=folium.Icon(color="green", icon="star", prefix="fa"),
                        popup=f"Selected & Diffusion (reinforced): {name}"
                    ).add_to(diff_fg)
                elif is_normal(name):
                    folium.Marker(
                        location=[c.y, c.x],
                        icon=folium.Icon(color="orange", icon="trash", prefix="fa"),
                        popup=f"Removal candidate (selected but NOT high transmission): {name}"
                    ).add_to(diff_fg)

            diff_fg.add_to(m)

        # Selected-versus-official layer
        sel_fg = folium.FeatureGroup(name="Selected vs Official", show=True)
        overlaps_s = selected_lower & official_lower
        only_sel = selected_lower - official_lower
        only_off_sel = official_lower - selected_lower

        for _, row in sewersheds_wgs84.iterrows():
            name = (row.get("wwtp") or "").strip()
            if not name or ("historic" in name.lower()):
                continue
            nm_l = canon(name)
            c = row.geometry.centroid
            if nm_l in overlaps_s:
                folium.Marker(
                    location=[c.y, c.x],
                    icon=folium.Icon(color="green", icon="star", prefix="fa"),
                    popup=f"Overlap (Selected & Official): {name}"
                ).add_to(sel_fg)
            elif nm_l in only_sel:
                folium.Marker(
                    location=[c.y, c.x],
                    icon=folium.Icon(color="red", icon="star", prefix="fa"),
                    popup=f"Selected only: {name}"
                ).add_to(sel_fg)
            elif nm_l in only_off_sel:
                folium.Marker(
                    location=[c.y, c.x],
                    icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
                    popup=f"Official only: {name}"
                ).add_to(sel_fg)
        sel_fg.add_to(m)
        # --- Optional overlay: Primary Roads (line) ---
        if primary_roads_shp:
            try:
                roads_gdf = gpd.read_file(primary_roads_shp)
                if roads_gdf.crs is None:
                    print("[Warn] Primary roads shapefile has no CRS; skipping roads layer.")
                elif not roads_gdf.empty:
                    roads_wgs84 = roads_gdf.to_crs(epsg=4326)

                    roads_fg = folium.FeatureGroup(name="Primary Roads", show=False)

                    folium.GeoJson(
                        roads_wgs84,
                        name="Primary Roads",
                        style_function=lambda feat: {
                            "color": "#1f4e79",  # dark blue roads
                            "weight": 2,
                            "opacity": 0.85
                        }
                    ).add_to(roads_fg)

                    roads_fg.add_to(m)
                    print(f"[Map] Added primary roads layer: {primary_roads_shp}")
            except Exception as e:
                print(f"[Warn] Could not add primary roads layer: {e}")

        # --- Optional overlay: Airports (point) ---
        if airports_shp:
            try:
                airports_gdf = gpd.read_file(airports_shp)
                if airports_gdf.crs is None:
                    print("[Warn] Airports shapefile has no CRS; skipping airport layer.")
                elif not airports_gdf.empty:
                    airports_wgs84 = airports_gdf.to_crs(epsg=4326)

                    airports_fg = folium.FeatureGroup(name="Airports", show=False)

                    for _, row in airports_wgs84.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue

                        if geom.geom_type == "Point":
                            x, y = geom.x, geom.y
                        else:
                            # fallback in case geometry is multipoint or something unexpected
                            c = geom.centroid
                            x, y = c.x, c.y

                        label = "Airport"
                        if airport_name_field and airport_name_field in airports_wgs84.columns:
                            val = row.get(airport_name_field)
                            if val is not None:
                                label = str(val)


                        folium.CircleMarker(
                            location=[y, x],
                            radius=10,  #
                            color="purple",
                            fill=False,  #
                            fill_opacity=0.8,
                            weight=3,  # line thickness
                            popup=label,
                        ).add_to(airports_fg)

                        # Optional: add a tiny plane label
                        folium.map.Marker(
                            [y, x],
                            icon=folium.DivIcon(
                                html='<div style="font-size:10px;">✈️</div>'
                            )
                        ).add_to(airports_fg)

                    airports_fg.add_to(m)
                    print(f"[Map] Added airports layer: {airports_shp}")
            except Exception as e:
                print(f"[Warn] Could not add airports layer: {e}")
        folium.LayerControl(collapsed=False).add_to(m)
        os.makedirs(os.path.dirname(output_html_map), exist_ok=True)
        m.save(output_html_map)
        print(f"Sentinel map saved to {output_html_map}")
        # --- Export top transmission (TX) sites ---
        top_tx_csv = os.path.join(
            os.path.dirname(output_html_map),
            "top_transmission_sites.csv"
        )

        export_top_transmission_sites(
            diffusion_score_lower=diffusion_score_lower,
            out_csv=top_tx_csv,
            top_k=20,
        )

        # Export an ArcGIS-compatible point table using the same map inputs
        points_csv = os.path.join(os.path.dirname(output_html_map), "sentinel_points_for_arcgis.csv")
        points_xlsx = os.path.join(os.path.dirname(output_html_map), "sentinel_points_for_arcgis.xlsx")

        export_sentinel_points_table(
            sewersheds_gdf=sewersheds_gdf,
            selected_wwtps=selected_wwtps,
            official_sites=official_sites if "official_sites" in locals() else set(),
            avg_rank=avg_rank,
            decision_log=decision_log,
            diffusion_score_lower=diffusion_score_lower if "diffusion_score_lower" in locals() else {},
            out_csv=points_csv,
            out_xlsx=points_xlsx,
        )

    # --- Subnetwork decision log (CSV) with post_diffusion column ---
    try:
        if decision_log:
            # ensure post_diffusion column always present
            for k, v in decision_log.items():
                if isinstance(v, dict) and "post_diffusion" not in v:
                    v["post_diffusion"] = "none"
            df_dec = (pd.DataFrame.from_dict(decision_log, orient="index")
                      .reset_index()
                      .rename(columns={"index": "wwtp"}))
            dec_csv = os.path.join(os.path.dirname(save_path), "subnetwork_decisions.csv")
            df_dec.to_csv(dec_csv, index=False, encoding="utf-8")
            print(f"[Explain] Subnetwork decisions written to {dec_csv}")
    except Exception as e:
        print(f"[Warn] Could not write subnetwork decisions: {e}")

    # --- Optional site-category summary (heavy) ---
    # --- Site-category summary (always write when write_summaries=True) ---
    if write_summaries:
        try:
            # official_sites may not exist depending on branch; ensure defined
            try:
                official_sites
            except NameError:
                official_sites = set()

            summary_rows = []
            selected_set = set(selected_wwtps)
            official_set = set(official_sites) if official_sites else set()

            # diffusion_lower exists earlier; ensure defined
            try:
                diffusion_lower
            except NameError:
                diffusion_lower = set()

            all_sites = selected_set | official_set | {s for s in rank_df_display.index}
            canon = lambda s: str(s).strip().lower()
            for s in sorted(all_sites):
                s_l = canon(s)
                summary_rows.append({
                    "wwtp": s,
                    "Sentinel_Selected": s in selected_set,
                    "Diffusion_TXRAW": s_l in diffusion_lower,
                    "Official": s in official_set,
                    "Official_not_selected": (s in official_set) and (s not in selected_set),
                    "post_diffusion": decision_log.get(s, {}).get("post_diffusion", "none")
                })

            df_summary = pd.DataFrame(summary_rows)
            out_csv = os.path.join(os.path.dirname(save_path), "site_category_summary.csv")
            out_txt = os.path.join(os.path.dirname(save_path), "site_category_summary.txt")
            df_summary.to_csv(out_csv, index=False, encoding="utf-8")

            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("# Summary of Site Categories (Two-phase: baseline selection; diffusion annotated)\n")
                f.write(f"Total unique sites listed: {len(df_summary)}\n\n")
                f.write("Category counts:\n")
                f.write(f"  Sentinel Selected: {df_summary['Sentinel_Selected'].sum()}\n")
                f.write(f"  Diffusion/TX-RAW: {(df_summary['Diffusion_TXRAW']).sum()}\n")
                f.write(f"  Official Sentinel: {df_summary['Official'].sum()}\n")
                f.write(f"  Official but NOT selected: {df_summary['Official_not_selected'].sum()}\n\n")
                # Breakdown of post-diffusion tags
                pf = df_summary["post_diffusion"].value_counts()
                f.write("Post-diffusion tags:\n")
                for tag, cnt in pf.items():
                    f.write(f"  {tag}: {cnt}\n")
        except Exception as e:
            print(f"[WARN] Could not write site category summary: {e}")


# =============================================================
# Export monthly rankings for all sites
# =============================================================

def rank_columns(frame: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Rank each numeric column; lower rank = better (descending scores by default)."""
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        out[f"rank_{col}"] = frame[col].rank(ascending=ascending, method="min")
    return out


def build_month_rank_table(
        base_features: pd.DataFrame,
        risk_features: Optional[pd.DataFrame] = None,
        final_rank_strategy: str = "mean"
) -> pd.DataFrame:
    """Combine base feature ranks and (optional) risk/viral ranks into one table."""
    mask = ~base_features.index.str.contains("historic", case=False, na=False)
    base_features = base_features.loc[mask]

    base_rank = rank_columns(base_features[[c for c in BASE_METRICS if c in base_features.columns]].fillna(0))

    if risk_features is not None and not risk_features.empty:
        risk_features = risk_features.loc[risk_features.index.intersection(base_features.index)]
        risk_cols = [
            c for c in risk_features.columns
            if c.startswith("import_risk_")
               or c.startswith("export_risk_")
               or c.startswith("WVAL_")
        ]
        if risk_cols:
            risk_combined = risk_features[risk_cols].mean(axis=1, skipna=True).to_frame("risk_combined")
            risk_rank = rank_columns(risk_combined.fillna(0))
            risk_rank.columns = ["rank_risk_combined"]
            rank_df = base_rank.join(risk_rank, how="outer").fillna(np.nan)
        else:
            rank_df = base_rank.copy()
    else:
        rank_df = base_rank.copy()

    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    if final_rank_strategy == "median":
        final = rank_df[rank_cols].median(axis=1, skipna=True)
    elif final_rank_strategy == "min":
        final = rank_df[rank_cols].min(axis=1, skipna=True)
    else:
        final = rank_df[rank_cols].mean(axis=1, skipna=True)

    out = rank_df.copy()
    out.insert(0, "final_rank", final)

    name_lookup = base_features.reset_index().set_index("wwtp_clean")["wwtp"]
    out.insert(0, "wwtp", name_lookup.reindex(out.index))

    out = out.sort_values("final_rank", ascending=True)

    cols = list(out.columns)
    base_rank_cols = [c for c in cols if c.startswith("rank_") and any(c.endswith(x) for x in BASE_METRICS)]
    risk_rank_cols = [c for c in cols if c == "rank_risk_combined"]
    ordered = ["wwtp", "final_rank"] + base_rank_cols + risk_rank_cols
    return out.loc[:, ordered]


def export_monthly_rank_sheet(
        writer: pd.ExcelWriter,
        month_label: str,
        base_features: pd.DataFrame,
        summary_csv: Optional[str] = None,
        include_risk: bool = True,
        include_wval: bool = False,
        final_rank_strategy: str = "mean",
):
    risk_month = None
    if include_risk and summary_csv and os.path.exists(summary_csv):
        risk_month = load_monthly_risk_signals(summary_csv, month_label, include_wval=include_wval)

    rank_table = build_month_rank_table(base_features, risk_month, final_rank_strategy)
    rank_table.to_excel(writer, sheet_name=month_label, index=False)

    # Return the rank table for reuse by the monthly workflow
    return rank_table


def summarize_common_top_sites(
        excel_path: str,
        out_dir: str,
        variant_label: str,
        top_k: int = 10,
        months_limit: Optional[int] = 12,
        top_n_display: int = 25
):
    _ensure_dir_for_file(os.path.join(out_dir, "_placeholder.txt"))
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names

    try:
        sheets_sorted = sorted(sheets, key=lambda x: pd.Period(x, freq="M"))
    except Exception:
        sheets_sorted = sheets

    target_sheets = sheets_sorted[-months_limit:] if months_limit else sheets_sorted

    counts = {}
    for sh in target_sheets:
        df = pd.read_excel(xls, sh)
        if "wwtp" not in df.columns:
            continue
        top_sites = df["wwtp"].head(top_k).dropna().astype(str)
        for w in top_sites:
            counts[w] = counts.get(w, 0) + 1

    if not counts:
        print(f"[Summary] No top-{top_k} sites found in {variant_label}.")
        return

    s = pd.Series(counts, name="months_in_topK").sort_values(ascending=False)
    csv_path = os.path.join(out_dir, f"common_top_sites_counts_top{top_k}_{variant_label}.csv")
    s.to_csv(csv_path, header=True, index=True)

    plt.figure(figsize=(10, max(6, 0.35 * len(s.head(top_n_display)))))
    s.head(top_n_display).sort_values().plot(kind="barh")
    plt.xlabel("Number of months in top-{}".format(top_k))
    if target_sheets:
        first_m, last_m = target_sheets[0], target_sheets[-1]
        plt.title(f"Common Top Sites ({variant_label}) — {first_m} to {last_m}")
    else:
        plt.title(f"Common Top Sites ({variant_label})")

    out_png = os.path.join(out_dir, f"common_top_sites_top{top_k}_{variant_label}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Summary] Saved: {csv_path} and {out_png}")


# =============================================================
# Main monthly pipeline (now emits an extra TX-RAW HTML page)
# =============================================================
def build_county_alias_map_from_table(csv_path, sewersheds_gdf=None):
    tbl = pd.read_csv(csv_path)

    # normalize
    tbl["Utility_clean"] = tbl["Utility"].astype(str).str.strip().str.lower()
    tbl["county_name"] = tbl["county_name"].astype(str).str.strip().str.title()

    # choose dominant county per utility
    tbl_dom = (
        tbl.sort_values(["Utility_clean", "PctOfUtility"], ascending=[True, False])
           .drop_duplicates("Utility_clean", keep="first")
           .copy()
    )

    # if sewersheds provided, restrict to utilities that actually exist there
    if sewersheds_gdf is not None and "wwtp" in sewersheds_gdf.columns:
        valid = set(sewersheds_gdf["wwtp"].astype(str).str.strip().str.lower())
        tbl_dom = tbl_dom[tbl_dom["Utility_clean"].isin(valid)].copy()

    # Build the utility-to-county lookup used during aggregation.
    util_to_county = dict(zip(tbl_dom["Utility_clean"], tbl_dom["county_name"]))

    # reverse: county -> utilities
    county_to_utils = {}
    for util, county in util_to_county.items():
        county_to_utils.setdefault(county, []).append(util)

    # final alias map
    alias_map = {}
    for county, utils in county_to_utils.items():
        utils_sorted = sorted(utils)
        if len(utils_sorted) == 1:
            alias_map[utils_sorted[0]] = county
        else:
            for i, util in enumerate(utils_sorted):
                suffix = chr(65 + i)   # A, B, C...
                alias_map[util] = f"{county} {suffix}"

    return alias_map

def _extract_existing_sentinel_order_for_level1(
    sewersheds_gdf,
    features_index,
    total_N: int = 20,
):
    """
    Extract the existing/current sentinel WWTP portfolio from the sewershed
    attribute table for baseline comparison.

    Expected attribute:
      act_sts == "sentinel surveillance site"

    Historic sites are excluded if the WWTP name or act_sts string contains
    "historic". The output is canonicalized to match base_features.index.
    If more than total_N active sentinel sites are found, the first total_N
    in the shapefile order are used and a warning is printed. This keeps the
    comparison as an existing operational portfolio rather than optimizing it
    by the Level-1 score.
    """
    if sewersheds_gdf is None or "wwtp" not in sewersheds_gdf.columns or "act_sts" not in sewersheds_gdf.columns:
        print("[Existing] Skipped: sewersheds_gdf has no wwtp/act_sts columns.")
        return []

    feat_set = {str(x).strip().lower() for x in list(features_index)}
    out = []
    seen = set()

    for _, row in sewersheds_gdf.iterrows():
        name = str(row.get("wwtp", "")).strip()
        status = str(row.get("act_sts", "")).strip().lower()
        if not name:
            continue
        lname = name.lower()

        if status != "sentinel surveillance site":
            continue
        if "historic" in status or "historic" in lname:
            continue

        key = lname
        if key in feat_set and key not in seen:
            out.append(key)
            seen.add(key)

    total_N = int(total_N)
    if len(out) > total_N:
        print(f"[Existing] Found {len(out)} active sentinel sites; using first {total_N} for N={total_N} comparison.")
        out = out[:total_N]
    else:
        print(f"[Existing] Found {len(out)} active sentinel sites for baseline comparison.")

    return out




# =============================================================
# Level 2 membership diagnostic:
# population-only vs proposed Level 1 vs Level 2 contribution
# =============================================================

def _find_strategy_key(strategy_orders, keyword):
    """Find a strategy key containing keyword, case-insensitive."""
    keyword = str(keyword).lower()
    for k in strategy_orders.keys():
        if keyword in str(k).lower():
            return k
    return None


def _safe_spearman(x, y):
    """Return Spearman correlation, or NaN if not enough paired data."""
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return np.nan
    return float(tmp.corr(method="spearman").iloc[0, 1])


def _load_level2_tx_scores(level2_csv):
    """
    Load Level 2 TX/contribution scores from a CSV.

    Expected site-name column:
        wwtp, site, or name

    Expected score column:
        any column containing 'score'

    If no score column is found, file order is used as a fallback score.
    """
    if not level2_csv or not os.path.exists(level2_csv):
        print(f"[Level2 diagnostic] Level 2 CSV not found: {level2_csv}")
        return pd.Series(dtype=float)

    df = pd.read_csv(level2_csv)

    name_col = next(
        (c for c in df.columns if c.lower() in ["wwtp", "site", "name"]),
        None,
    )

    score_col = next(
        (c for c in df.columns if "score" in c.lower()),
        None,
    )

    if name_col is None:
        raise KeyError(
            f"No site-name column found in {level2_csv}. "
            f"Expected one of: wwtp, site, name. "
            f"Columns found: {list(df.columns)}"
        )

    if score_col is None:
        print(
            f"[Level2 diagnostic][WARN] No score column found in {level2_csv}; "
            "using file order as fallback score."
        )
        df["_score_fallback"] = np.arange(len(df), 0, -1)
        score_col = "_score_fallback"

    out = pd.Series(
        pd.to_numeric(df[score_col], errors="coerce").values,
        index=df[name_col].astype(str).str.strip().str.lower(),
        name="level2_score",
    )

    out = out[~out.index.duplicated(keep="first")]
    return out.dropna()


def plot_level2_membership_diagnostic(
    features,
    ours_rank,
    strategy_orders,
    level2_csv,
    out_dir,
    label,
    total_N=20,
):
    """
    SI diagnostic comparing:
      1. Population-only selected sites
      2. Proposed Level 1 selected sites
      3. Top Level 2 contribution sites

    Outputs:
      - level2_membership_diagnostic_{label}.csv
      - level2_membership_vs_population_{label}.png
      - level2_membership_vs_ours_rank_{label}.png

    Interpretation:
      - Marker color = selection group:
          Both selected,
          Proposed L1 only,
          Pop only,
          Neither selected.
      - Marker size = population served.
      - In the population plot, black edge marks top-N Level 2 contribution sites.
      - In the rank plot, only top-N Level 2 contribution sites are shown.
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Features
    # -------------------------
    feat = features.copy()
    feat.index = feat.index.astype(str).str.strip().str.lower()

    if "wwtp" in feat.columns:
        name_lookup = feat["wwtp"].astype(str)
    else:
        name_lookup = pd.Series(feat.index, index=feat.index)

    if "pop_served" in feat.columns:
        pop_served = pd.to_numeric(feat["pop_served"], errors="coerce")
    else:
        pop_served = pd.Series(np.nan, index=feat.index, dtype=float)

    pop_rank = pop_served.rank(ascending=False, method="min")

    # -------------------------
    # Proposed Level 1 rank
    # -------------------------
    ours_rank = ours_rank.copy()
    ours_rank.index = ours_rank.index.astype(str).str.strip().str.lower()
    ours_rank = pd.to_numeric(ours_rank, errors="coerce")

    # -------------------------
    # Strategy selected sets
    # -------------------------
    pop_key = _find_strategy_key(strategy_orders, "population")
    ours_key = _find_strategy_key(strategy_orders, "ours")

    if pop_key is None:
        raise KeyError(
            f"Could not find population-only strategy in keys: {list(strategy_orders.keys())}"
        )

    if ours_key is None:
        raise KeyError(
            f"Could not find Ours Level 1 strategy in keys: {list(strategy_orders.keys())}"
        )

    pop_selected = {
        str(x).strip().lower()
        for x in strategy_orders[pop_key][:int(total_N)]
        if str(x).strip()
    }

    ours_selected = {
        str(x).strip().lower()
        for x in strategy_orders[ours_key][:int(total_N)]
        if str(x).strip()
    }

    # -------------------------
    # Level 2 contribution score
    # -------------------------
    level2_score = _load_level2_tx_scores(level2_csv)

    if level2_score.empty:
        print("[Level2 diagnostic] Skipped because Level 2 score table is empty.")
        return

    # -------------------------
    # Build diagnostic table
    # -------------------------
    idx = feat.index.union(level2_score.index).union(ours_rank.index)

    diag = pd.DataFrame(index=idx)
    diag["wwtp"] = name_lookup.reindex(idx)
    diag["wwtp"] = diag["wwtp"].where(diag["wwtp"].notna(), diag.index.to_series())

    diag["pop_served"] = pop_served.reindex(idx)
    diag["pop_rank"] = pop_rank.reindex(idx)
    diag["ours_level1_rank"] = ours_rank.reindex(idx)
    diag["level2_score"] = level2_score.reindex(idx)

    # Keep only sites with Level 2 scores
    diag = diag.dropna(subset=["level2_score"]).copy()

    if diag.empty:
        print("[Level2 diagnostic] Skipped because no sites match Level 2 scores.")
        return

    # Top Level 2 contribution sites
    top_l2 = set(
        diag.sort_values("level2_score", ascending=False)
            .head(int(total_N))
            .index
            .tolist()
    )

    diag["pop_selected_topN"] = diag.index.isin(pop_selected)
    diag["ours_selected_topN"] = diag.index.isin(ours_selected)
    diag["top_level2_contribution"] = diag.index.isin(top_l2)

    def _membership_group(row):
        if row["pop_selected_topN"] and row["ours_selected_topN"]:
            return "Both selected"
        if row["ours_selected_topN"] and not row["pop_selected_topN"]:
            return "Proposed L1 only"
        if row["pop_selected_topN"] and not row["ours_selected_topN"]:
            return "Pop only"
        return "Neither selected"

    diag["selection_group"] = diag.apply(_membership_group, axis=1)

    diag["high_l2_missed_by_population"] = (
        diag["top_level2_contribution"] & (~diag["pop_selected_topN"])
    )

    diag["high_l2_missed_by_ours"] = (
        diag["top_level2_contribution"] & (~diag["ours_selected_topN"])
    )

    diag["high_l2_missed_by_both"] = (
        diag["top_level2_contribution"]
        & (~diag["pop_selected_topN"])
        & (~diag["ours_selected_topN"])
    )

    diag["ours_only_high_l2"] = (
        diag["ours_selected_topN"]
        & (~diag["pop_selected_topN"])
        & diag["top_level2_contribution"]
    )

    diag["population_only_high_l2"] = (
        diag["pop_selected_topN"]
        & (~diag["ours_selected_topN"])
        & diag["top_level2_contribution"]
    )

    diag["pop_percentile"] = diag["pop_served"].rank(pct=True)

    # -------------------------
    # Summary statistics
    # -------------------------
    n_top_l2 = int(diag["top_level2_contribution"].sum())
    n_pop_capture = int((diag["top_level2_contribution"] & diag["pop_selected_topN"]).sum())
    n_ours_capture = int((diag["top_level2_contribution"] & diag["ours_selected_topN"]).sum())
    n_missed_both = int(diag["high_l2_missed_by_both"].sum())
    n_ours_only_high_l2 = int(diag["ours_only_high_l2"].sum())
    n_pop_only_high_l2 = int(diag["population_only_high_l2"].sum())

    rho_pop = _safe_spearman(diag["pop_rank"], diag["level2_score"])
    rho_ours = _safe_spearman(diag["ours_level1_rank"], diag["level2_score"])

    # -------------------------
    # Save diagnostic CSV
    # -------------------------
    out_csv = os.path.join(out_dir, f"level2_membership_diagnostic_{label}.csv")
    diag.sort_values("level2_score", ascending=False).to_csv(
        out_csv,
        index=False,
        encoding="utf-8",
    )
    print(f"[Level2 diagnostic] CSV saved: {out_csv}")

    # -------------------------
    # Plot style
    # -------------------------
    group_colors = {
        "Both selected": "#2ca02c",
        "Proposed L1 only": "#d62728",
        "Pop only": "#1f77b4",
        "Neither selected": "#bdbdbd",
    }

    group_order = [
        "Both selected",
        "Proposed L1 only",
        "Pop only",
        "Neither selected",
    ]

    pop_for_size = diag["pop_served"].fillna(0).clip(lower=0)

    if pop_for_size.max() > 0:
        point_sizes = 35 + 260 * np.sqrt(pop_for_size / pop_for_size.max())
    else:
        point_sizes = pd.Series(60, index=diag.index)

    l2_threshold = diag.loc[list(top_l2), "level2_score"].min()

    # =========================================================
    # Plot 1: Level 2 contribution vs population served
    # =========================================================
    plot_df = diag.copy()
    plot_df["pop_served_plot"] = plot_df["pop_served"]
    plot_df.loc[plot_df["pop_served_plot"] <= 0, "pop_served_plot"] = np.nan
    plot_df = plot_df.dropna(subset=["pop_served_plot", "level2_score"])

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))

        for group in group_order:
            sub = plot_df[plot_df["selection_group"] == group]
            if sub.empty:
                continue

            edge_colors = np.where(sub["top_level2_contribution"], "black", "none")
            line_widths = np.where(sub["top_level2_contribution"], 0.9, 0.0)

            ax.scatter(
                sub["pop_served_plot"],
                sub["level2_score"],
                s=point_sizes.reindex(sub.index),
                color=group_colors.get(group, "#999999"),
                label=group,
                alpha=0.78,
                edgecolors=edge_colors,
                linewidths=line_widths,
            )

        ax.set_xscale("log")

        ax.axhline(
            l2_threshold,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
            label=f"Top {int(total_N)} Level 2 threshold",
        )

        ax.set_xlabel("Population served")
        ax.set_ylabel("Annual-average Level 2 contribution score")
        ax.set_title("Level 2 contribution vs population served")
        ax.grid(True, alpha=0.25)

        ax.legend(frameon=True, fontsize=8, loc="best")
        fig.tight_layout()

        out_png = os.path.join(out_dir, f"level2_membership_vs_population_{label}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[Level2 diagnostic] Plot saved: {out_png}")

    else:
        print("[Level2 diagnostic] Population plot skipped: no positive population values.")

    # =========================================================
    # Plot 2: Top Level 2 contribution sites vs proposed Level 1 rank
    # =========================================================
    # SI/manuscript version: only plot top Level 2 sites.
    plot_df = diag[
        diag["top_level2_contribution"]
    ].dropna(subset=["ours_level1_rank", "level2_score"]).copy()

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))

        for group in group_order:
            sub = plot_df[plot_df["selection_group"] == group]
            if sub.empty:
                continue

            ax.scatter(
                sub["ours_level1_rank"],
                sub["level2_score"],
                s=point_sizes.reindex(sub.index),
                color=group_colors.get(group, "#999999"),
                label=group,
                alpha=0.82,
                edgecolors="black",
                linewidths=0.8,
            )

        ax.axvline(
            int(total_N),
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label=f"Raw top {int(total_N)} Level 1 rank threshold",
        )

        ax.set_title("Top Level 2 contribution WWTPs by Level 1 rank and selection status")
        ax.set_xlabel("Raw proposed Level 1 rank (lower = higher priority)")
        ax.set_ylabel("Annual-average Level 2 contribution score")
        ax.grid(True, alpha=0.25)

        ax.legend(frameon=True, fontsize=8, loc="best")
        fig.tight_layout()

        out_png = os.path.join(out_dir, f"level2_membership_vs_ours_rank_{label}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[Level2 diagnostic] Plot saved: {out_png}")

    else:
        print("[Level2 diagnostic] Rank plot skipped: no Level 1 rank values.")

    # -------------------------
    # Console summary
    # -------------------------
    print("\n[Level2 diagnostic summary]")
    print(f"  Top Level 2 sites: {n_top_l2}")
    print(f"  Captured by population-only top-{int(total_N)}: {n_pop_capture}/{n_top_l2}")
    print(f"  Captured by Proposed Level 1 top-{int(total_N)}: {n_ours_capture}/{n_top_l2}")
    print(f"  Missed by both: {n_missed_both}")
    print(f"  Proposed-L1-only and top Level 2: {n_ours_only_high_l2}")
    print(f"  Pop-only and top Level 2: {n_pop_only_high_l2}")
    print(f"  Spearman rho, population rank vs Level 2: {rho_pop:.3f}")
    print(f"  Spearman rho, Proposed Level 1 rank vs Level 2: {rho_ours:.3f}\n")

def main(
        selection_fraction: float = 0.25,
        total_N: int = 20,
        singleton_top_pct: float = 0.30,
        singleton_drop_km: float = 95.0,
        singleton_bottom_pct: float = 0.20,
        singleton_reserve_k: int = 0,  # conservative default (off)
        singleton_reserve_frac: float = 0.0,  # only used if reserve_k==0
        isolation_bonus_km: float = 150.0,
        risk_weight: float = 0.5,
        coverage_weight: float = 0.5,
):
    # --- tag and run root for this parameter combo ---
    reserve_k = singleton_reserve_k or int(round(singleton_reserve_frac * total_N))
    param_tag = (
        f"sf{int(round(selection_fraction * 100)):02d}"
        f"_N{int(total_N)}"
        f"_top{int(round(singleton_top_pct * 100)):02d}"
        f"_km{int(round(singleton_drop_km))}"
        f"_bot{int(round(singleton_bottom_pct * 100)):02d}"
        f"_rsv{reserve_k:02d}"
    )

    run_root = os.path.join("outputs", "param_sweeps", param_tag)
    os.makedirs(run_root, exist_ok=True)
    # Analysis inputs and optional paths.
    sewershed_path = "../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp"
    zip_path = "../ZoneSelection/Input/Census/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp"
    commute_path = "../ZoneSelection/Input/Commute/US/zip_code_commute_distance_adjusted.csv"

    # Global analysis window (inclusive). We'll split this by calendar month.
    analysis_start = "2024-01-01"
    analysis_end = "2024-12-31"

    # Viral/risk ranking options
    INCLUDE_RISK_SIGNALS = True  # toggle to ignore import/export risk signals
    FINAL_RANK_STRATEGY = "mean"  # 'mean' | 'median' | 'min'
    INCLUDE_WVAL = False  # WVAL off by default

    # Where to find weekly summary with import/export risks
    SUMMARY_CSV_OVERRIDE = "outputs/csv/weekly_top10_summary_multi_rank_all_20230109_20241230.csv"
    if SUMMARY_CSV_OVERRIDE and os.path.exists(SUMMARY_CSV_OVERRIDE):
        summary_csv = SUMMARY_CSV_OVERRIDE
    else:
        summary_csv = _find_latest_raw_summary_csv(search_dir="outputs/csv", prefix="weekly_top10_summary_multi")

    # Optional Level 2 contribution CSV selected using validation metrics.
    RAW_TX_CSV = "run_COVID/test.csv"
    if not os.path.exists(RAW_TX_CSV):
        RAW_TX_CSV = glob_best_txraw_csv("run_COVID")

    # Optional: best Level-2 TX-RAW files for other pathogens.
    # These are exported to CSV only and do not change visualization or selection.
    FLU_TX_CSV = _pick_best_txraw_csv_optional([
        "run_FLU", "run_Flu", "run_flu", "run_INFLUENZA", "run_Influenza", "run_influenza"
    ])
    RSV_TX_CSV = _pick_best_txraw_csv_optional([
        "run_RSV", "run_rsv"
    ])
    print(f"[Picker] Best COVID TX-RAW CSV: {RAW_TX_CSV}")
    print(f"[Picker] Best FLU TX-RAW CSV: {FLU_TX_CSV if FLU_TX_CSV else 'not found'}")
    print(f"[Picker] Best RSV TX-RAW CSV: {RSV_TX_CSV if RSV_TX_CSV else 'not found'}")
    if not FLU_TX_CSV:
        print("[Picker][WARN] FLU not found. Checked run_FLU / run_Flu / run_flu / run_INFLUENZA variants.")
    if not RSV_TX_CSV:
        print("[Picker][WARN] RSV not found. Checked run_RSV / run_rsv.")



    # Optional EJ CSV (CombinedScore). Used ONLY for EJ-only ranking mode and for display in heatmaps.
    EJ_CSV = r"../ZoneSelection/Outfile/EJ/wwtp_combined_rank_ej.csv"  # adjust if needed
    ej_scores = load_ej_scores(EJ_CSV)
    # --- Build base data once ---
    sewersheds = load_and_prepare_data(sewershed_path, zip_path, commute_path)
    base_G = build_sewershed_graph(sewersheds)
    # --- Export static network to GIS line/point layers ---
    export_network_edges_shp(
        base_G,
        sewersheds,
        outpath=os.path.join(run_root, "shp", "co_sewershed_edges.shp"),
    )

    export_network_nodes_shp(
        base_G,
        sewersheds,
        outpath=os.path.join(run_root, "shp", "co_sewershed_nodes.shp"),
    )

    historic_nodes = [n for n in base_G.nodes if "historic" in str(base_G.nodes[n].get("wwtp", "")).lower()]
    base_G.remove_nodes_from(historic_nodes)

    # Optional: static overview
    # plot_static_network(base_G, sewersheds, outpath=os.path.join(run_root, "images", "colorado_sewershed_network.png"))
    roads_shp = r"..\ZoneSelection\Input\Transportation\Road_CO.shp"
    county_boundary_shp = r"..\ZoneSelection\Input\Census\COCounty.shp"
    plot_static_network(
        base_G,
        sewersheds,
        outpath=os.path.join(run_root, "images", "colorado_sewershed_network.png"),
        roads=roads_shp,
        county_boundary=None,
    )
    export_interactive_map(base_G, sewersheds,
                           outpath=os.path.join(run_root, "maps", "colorado_sewershed_network.html"))

    start_tag = pd.to_datetime(analysis_start).strftime("%Y%m")
    end_tag = pd.to_datetime(analysis_end).strftime("%Y%m")

    # --- EJ file (static across months) ---
    EJ_CSV = r"../ZoneSelection/Outfile/EJ/wwtp_combined_rank_ej.csv"
    ej_scores = None
    if EJ_CSV and os.path.exists(EJ_CSV):
        try:
            ej_scores = load_ej_scores(EJ_CSV)
        except Exception as e:
            print(f"[EJ] Could not load EJ scores from {EJ_CSV}: {e}")

    for variant in ("base", "risk"):
        include_risk_now = (variant == "risk") and INCLUDE_RISK_SIGNALS
        include_ej_now = (variant == "ej")
        suffix = "_risk" if include_risk_now else ("_ej" if include_ej_now else "_base")

        # Excel for this run+variant
        excel_out_dir = os.path.join(run_root, "rank_excel")
        os.makedirs(excel_out_dir, exist_ok=True)
        excel_out = os.path.join(
            excel_out_dir,
            f"monthly_wwtp_ranks_{start_tag}_{end_tag}{suffix}.xlsx"
        )

        with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
            for m_start, m_end, m_label in iter_month_windows(analysis_start, analysis_end):
                G = base_G.copy()
                enrich_nodes_with_od_coverage(
                    G,
                    wwtp_shapefile=sewershed_path,
                    weekly_od_dir="../ZoneSelection/Outfile/ODData/Weekly",
                    start_date=m_start,
                    end_date=m_end
                )
                nx.set_node_attributes(G, nx.degree_centrality(G), "degree_centrality")

                # 1) Build base features for this month
                base_features = build_feature_table_from_graph(G)

                # 2) Export monthly rank sheet AND capture the rank table
                #    - base/risk variants: existing rank table builder
                #    - ej variant: EJ-only ranking (final_rank = rank_ej)
                if include_ej_now:
                    if ej_scores is None:
                        raise FileNotFoundError(
                            f"EJ-only variant requested, but EJ_CSV not found/loaded: {EJ_CSV}"
                        )
                    rank_table = build_ej_only_rank_table(
                        base_features=base_features,
                        ej_scores=ej_scores,
                        higher_is_worse=True,
                    )
                    # Write a readable sheet (no wwtp_clean index)
                    rank_table.reset_index().rename(columns={"index": "wwtp_clean"}).to_excel(
                        writer, sheet_name=m_label, index=False
                    )
                else:
                    rank_table = export_monthly_rank_sheet(
                        writer,
                        month_label=m_label,
                        base_features=base_features,
                        summary_csv=summary_csv,
                        include_risk=include_risk_now,
                        include_wval=INCLUDE_WVAL,
                        final_rank_strategy=FINAL_RANK_STRATEGY,
                    )

                # 3) Run cumulative coverage for this month (same month, same variant)
                #    - features index = wwtp_clean
                #    - rank_table index = wwtp_clean, column 'final_rank'
                # -----------------------------------------------------------
                # Cumulative coverage plots (pop, OD pop, commute volume, area, risk)
                # -----------------------------------------------------------
                cumulative_out_dir = os.path.join(
                    run_root, "cumulative_coverage", f"{m_label}{suffix}"
                )
                os.makedirs(cumulative_out_dir, exist_ok=True)

                # Start with coverage-related columns; index = wwtp_clean
                metric_cols = [
                    "pop_served",
                    "pop_covered_by_od",
                    "od_volume_total",  # commute volume
                    "area_reached",
                ]
                metric_cols = [c for c in metric_cols if c in base_features.columns]
                cum_features = base_features[metric_cols].copy()

                # --- Optional: add a per-site risk_score for cumulative plots (risk variant only) ---
                if summary_csv:
                    risk_month_for_cum = load_monthly_risk_signals(
                        summary_csv,
                        m_label,
                        include_wval=INCLUDE_WVAL,
                    )
                    if risk_month_for_cum is not None and not risk_month_for_cum.empty:
                        covid_cols_cum = [
                            c
                            for c in risk_month_for_cum.columns
                            if c.startswith("import_risk_")
                               or c.startswith("export_risk_")
                               or c.startswith("WVAL_")
                        ]
                        if covid_cols_cum:
                            risk_combined_cum = risk_month_for_cum[covid_cols_cum].mean(
                                axis=1, skipna=True
                            )
                            # align to wwtp_clean index used in base_features
                            cum_features["risk_score"] = risk_combined_cum.reindex(
                                cum_features.index
                            )

                # final_rank series aligned on the same index (wwtp_clean)
                avg_rank = rank_table["final_rank"]
                # This writes:
                #   - CSV of cumulative deltas + totals (including risk_score if present)
                #   - Absolute cumulative plot (pop/OD/commute on left axis, area on right axis)
                #   - Fraction plot (0–1) for all metrics, including risk_score
                #   - Marginal-gain bar plot
                save_cumulative_outputs(
                    features=cum_features,
                    avg_rank=avg_rank,
                    out_dir=cumulative_out_dir,
                    label=f"{m_label}{suffix}",
                )

                # ============================================================
                # BG-level unique cumulative coverage
                # ============================================================

                save_bg_unique_outputs(
                    avg_rank=avg_rank,
                    bg_link_dir="../ZoneSelection/Outfile/ODData/BG_Link_Weekly",
                    start_date=m_start,
                    end_date=m_end,
                    out_dir=cumulative_out_dir,
                    label=f"{m_label}{suffix}_BG",
                    direction="Destination",
                    union_mode="cap_sum",
                    weight_from="Volume",

                    # Exponential saturation of BG-level trip intensity
                    bg_weight_mode="intensity_exp",
                    tau=2,  # controls how fast curves rise
                    eps_pop=100.0,  # stabilizer for small BGs
                    # --------------------------------

                    pop_served=base_features["pop_served"],
                    pop_served_total_override=state_population
                )

                # ============================================================
                # Level 1: baseline vs integrated backbone comparison
                # ============================================================
                # This comparison is only for the Level-1/base variant.
                # It is disease-agnostic and answers the reviewer-facing question:
                # do simple baselines perform as well as the integrated backbone?
                if variant == "base":
                    level1_out_dir = os.path.join(
                        run_root, "level1_baseline_comparison", f"{m_label}_level1"
                    )
                    os.makedirs(level1_out_dir, exist_ok=True)

                    baseline_orders = make_level1_baseline_orders(
                        features=base_features,
                        total_N=total_N,
                        include_ej_only=False,
                        ej_scores=ej_scores,
                        bg_link_dir="../ZoneSelection/Outfile/ODData/BG_Link_Weekly",
                        start_date=m_start,
                        end_date=m_end,
                        direction="Destination",
                        sewersheds_gdf=sewersheds,
                        cbsa_shp="../ZoneSelection/Input/Census/tl_2024_us_cbsa/tl_2024_us_cbsa.shp",
                        include_cbsa_baseline=True,
                        include_legacy_mobility_area_baselines=False,
                    )

                    # Reuse the exact Ours result from the basic ranking script if available.
                    # The base variant is evaluated first so the baseline comparison can
                    # read that output instead of recomputing Ours.
                    basic_ours_csv = _find_basic_ours_order_csv(
                        run_root=run_root,
                        month_label=m_label,
                        suffix=suffix,
                        total_N=total_N,
                    )
                    integrated_order = _load_basic_ours_order_csv(basic_ours_csv, total_N=total_N)

                    if integrated_order:
                        print(f"[OursReuse] Using Ours Level 1 from basic ranking CSV: {basic_ours_csv}")
                    else:
                        # Fallback only: use selected_ordered from this run if no basic CSV is found.
                        print("[OursReuse][WARN] Basic Ours selected-order CSV not found; falling back to this run's selected_ordered.")
                        _name_to_clean = {}
                        if base_features is not None and not base_features.empty:
                            for _idx, _row in base_features.iterrows():
                                _clean_idx = str(_idx).strip().lower()
                                _name_to_clean[_clean_idx] = _clean_idx
                                if "wwtp" in base_features.columns:
                                    _name_to_clean[str(_row.get("wwtp", "")).strip().lower()] = _clean_idx
                        integrated_order = []
                        for _s in selected_ordered:
                            _key = str(_s).strip().lower()
                            _clean = _name_to_clean.get(_key, _key)
                            if _clean not in integrated_order:
                                integrated_order.append(_clean)

                    if len(integrated_order) != int(total_N):
                        print(
                            f"[OursReuse][WARN] Ours Level 1 order length={len(integrated_order)} "
                            f"but total_N={int(total_N)}."
                        )

                    strategy_orders = dict(baseline_orders)
                    # Main reviewer-facing comparison excludes the existing operational portfolio.
                    # Existing sites can be evaluated separately if needed, but are not included
                    # as a baseline bar or preserved in the recommended Level-1 set.
                    strategy_orders["Ours Level 1"] = integrated_order

                    # Monthly COVID risk signal for evaluation-only benefit metric.
                    # This does not affect non-mobility baselines; it is used
                    # only to evaluate how much COVID risk signal each final
                    # N=21 portfolio captures.
                    covid_risk_series_for_benefit = None
                    if summary_csv:
                        risk_month_for_benefit = load_monthly_risk_signals(
                            summary_csv,
                            m_label,
                            include_wval=INCLUDE_WVAL,
                        )
                        if risk_month_for_benefit is not None and not risk_month_for_benefit.empty:
                            covid_cols_benefit = [
                                c for c in risk_month_for_benefit.columns
                                if c.startswith("import_risk_COVID") or c.startswith("export_risk_COVID")
                            ]
                            if covid_cols_benefit:
                                covid_risk_series_for_benefit = risk_month_for_benefit[covid_cols_benefit].mean(axis=1, skipna=True)

                    evaluate_level1_strategy_orders(
                        strategy_orders=strategy_orders,
                        features=base_features,
                        bg_link_dir="../ZoneSelection/Outfile/ODData/BG_Link_Weekly",
                        start_date=m_start,
                        end_date=m_end,
                        out_dir=level1_out_dir,
                        label=f"{m_label}_level1_N{total_N}",
                        direction="Destination",
                        union_mode="cap_sum",
                        weight_from="Volume",
                        bg_weight_mode="intensity_exp",
                        tau=2,
                        eps_pop=100.0,
                        pop_served_total_override=state_population,
                        G=G,
                        sewersheds_gdf=sewersheds,
                        ej_scores=ej_scores,
                        covid_risk_series=covid_risk_series_for_benefit,
                        utility_county_csv="../ZoneSelection/Input/WWTP_CO/utility_county_adj.csv",
                        cbsa_shp="../ZoneSelection/Input/Census/tl_2024_us_cbsa/tl_2024_us_cbsa.shp",
                        small_component_max_size=2,
                        make_plots=True,
                        level2_diffusion_csv=RAW_TX_CSV,
                        level2_extra_diffusion_csvs={
                            "flu": FLU_TX_CSV,
                            "rsv": RSV_TX_CSV,
                        },
                    )


                    # ------------------------------------------------------------
                    # Additional Level 2 membership diagnostic for SI:
                    # population-only vs proposed Level 1 vs Level 2 contribution.
                    # This is a diagnostic figure only; it does not change selection.
                    # ------------------------------------------------------------
                    plot_level2_membership_diagnostic(
                        features=base_features,
                        ours_rank=rank_table["final_rank"],
                        strategy_orders=strategy_orders,
                        level2_csv=RAW_TX_CSV,
                        out_dir=level1_out_dir,
                        label=f"{m_label}_level1_N{total_N}",
                        total_N=total_N,
                    )


                # 4) Generate the monthly map and rank heatmap
                out_dir = os.path.join(run_root, "networkmaps", f"{m_label}{suffix}")
                os.makedirs(out_dir, exist_ok=True)

                heat_path = os.path.join(out_dir, f"sentinel_rank_heatmap_{m_label}{suffix}.png")
                map_path_normal = os.path.join(out_dir, f"sentinel_ranked_map_{m_label}{suffix}.html")
                map_path_txraw = os.path.join(out_dir, f"sentinel_ranked_map_{m_label}{suffix}_txraw.html")

                # Prepare risk series for the heatmap (risk variant only)
                risk_series_for_plot = None
                if summary_csv:
                    risk_month = load_monthly_risk_signals(summary_csv, m_label, include_wval=INCLUDE_WVAL)
                    if risk_month is not None and not risk_month.empty:
                        covid_cols = [c for c in risk_month.columns if
                                      c.startswith("import_risk_C")
                                      or c.startswith("export_risk_C")]

                        if covid_cols:
                            risk_combined = risk_month[covid_cols].mean(axis=1, skipna=True)
                            name_map = base_features.reset_index().set_index("wwtp_clean")["wwtp"]
                            risk_series_for_plot = risk_combined.reindex(name_map.index).rename(index=name_map)

                # For heatmap-only EJ column (and EJ-only selection override), reuse the same EJ_CSV

                # 1) PRIMARY call: render heatmap (enhanced) + TX-RAW overlay map
                # Optional: override the selection/row ordering with EJ-only final ranks
                final_rank_override = None
                if include_ej_now:
                    # rank_table index = wwtp_clean, but visualize() uses original-case WWTP names
                    name_map = base_features.reset_index().set_index("wwtp_clean")["wwtp"]
                    final_rank_override = avg_rank.rename(index=name_map)

                # -----------------------------
                # Heatmap DISPLAY controls (variant-specific)
                # -----------------------------
                # -----------------------------
                # Heatmap DISPLAY controls (variant-specific)
                # -----------------------------
                if variant == "base":
                    # Level 1 backbone: disease-agnostic.
                    # Do NOT show/use COVID risk or TX-RAW in the base selection heatmap.
                    heat_risk_series = None
                    heat_tx_csv = None
                    rank_mode = "base"
                elif variant == "risk":
                    # Level 2 refinement: disease/transmission-aware overlay.
                    heat_risk_series = risk_series_for_plot
                    heat_tx_csv = RAW_TX_CSV
                    rank_mode = "risk"
                else:
                    heat_risk_series = None
                    heat_tx_csv = None
                    rank_mode = "base"
                county_alias_map = build_county_alias_map_from_table(
                    "../ZoneSelection/Input/WWTP_CO/utility_county_adj.csv",
                    sewersheds_gdf=sewersheds
                )
                visualize_rank_with_subnetwork_coverage(
                    G,
                    save_path=heat_path,
                    sewersheds_gdf=sewersheds,
                    output_html_map=map_path_normal,  # base map (no TX overlay)
                    county_alias_map=county_alias_map,
                    risk_series=heat_risk_series,  # <- variant-specific
                    diffusion_csv=heat_tx_csv,  # <- variant-specific (None removes Transmission col)
                    ej_csv=None,  # <- variant-specific (None removes EJ col)
                    ranking_mode=rank_mode,

                    ej_scores=ej_scores,
                    ej_higher_is_worse=True,

                    selection_fraction=selection_fraction,
                    render_heatmap=True,
                    month_label=f"{m_label}{suffix}",

                    singleton_top_pct=singleton_top_pct,
                    singleton_drop_km=singleton_drop_km,
                    singleton_bottom_pct=singleton_bottom_pct,
                    total_N=total_N,

                    singleton_reserve_k=reserve_k,
                    isolation_bonus_km=isolation_bonus_km,
                    risk_weight=risk_weight,
                    coverage_weight=coverage_weight,
                    # Align BG-unique coverage with the selected portfolio
                    bg_link_dir="../ZoneSelection/Outfile/ODData/BG_Link_Weekly",
                    bg_start_date=m_start,
                    bg_end_date=m_end,
                    write_bg_unique_for_selected=True,
                    primary_roads_shp=r"..\ZoneSelection\Input\Transportation\Road_CO.shp",
                    airports_shp=r"..\ZoneSelection\Input\Transportation\Airport_CO_5.shp",
                    airport_name_field="ARPT_NAME"  # field containing airport names
                )

        # Summarize frequent top-K sites across months for this variant
        summary_dir = os.path.join(run_root, f"summary{suffix}")
        os.makedirs(summary_dir, exist_ok=True)
        summarize_common_top_sites(
            excel_path=excel_out,
            out_dir=summary_dir,
            variant_label=variant,
            top_k=15,
            months_limit=12,
            top_n_display=15
        )

        print(f"Monthly ranking maps complete for {variant}. Excel saved to: {excel_out}")

        # Assemble manuscript-ready Level-1 panels immediately after the base
        # Level-1 monthly figures have been generated. This avoids a separate
        # manual panel-assembly BAT step.
        if variant == "base":
            assemble_level1_panel_outputs(
                level1_root_dir=os.path.join(run_root, "level1_baseline_comparison"),
                month_a="2024-06",
                month_b="2024-12",
                make_bar_1x2=True,
                make_heatmap_1x2=True,
                make_bar_4x3=True,
            )


def make_avg_rank_from_month(rank_table: pd.DataFrame) -> pd.Series:
    """
    Given a monthly rank table (output of build_month_rank_table), return a
    Series 'avg_rank' indexed by wwtp name, where lower = better.

    We just use final_rank; if you later want multi-month averaging, you can
    extend this.
    """
    if "wwtp" not in rank_table.columns or "final_rank" not in rank_table.columns:
        raise KeyError("rank_table must include 'wwtp' and 'final_rank' columns.")
    avg_rank = (
        rank_table[["wwtp", "final_rank"]]
        .assign(
            wwtp_clean=lambda d: d["wwtp"].astype(str).str.strip().str.lower()
        )
        .set_index("wwtp_clean")["final_rank"]
    )
    return avg_rank


def run_cumulative_coverage_for_month(
        G,
        rank_table: pd.DataFrame,
        out_dir: str,
        label: str,
):
    """
    Connect the network ranking to cumulative coverage:

      1) Build features (pop_served, pop_covered_by_od, area_reached) from graph G
      2) Build avg_rank from the monthly rank table
      3) Call save_cumulative_outputs(features, avg_rank, out_dir, label)

    The cumulative curves will then reflect the benefit of adding sites in the
    order given by final_rank (which already includes early detection).
    """
    # 1. Build the feature table from the graph.
    base_features = build_feature_table_from_graph(G)

    # keep only the columns that CumulativeCoverage expects
    features = base_features[["pop_served", "pop_covered_by_od", "area_reached"]].copy()
    features.index = features.index.astype(str).str.shetrip().str.lower()

    # 2. Build avg_rank for this month
    avg_rank = make_avg_rank_from_month(rank_table)
    avg_rank.index = avg_rank.index.astype(str).str.strip().str.lower()

    # 3. Align and call save_cumulative_outputs
    common = features.index.intersection(avg_rank.index)
    if common.empty:
        raise ValueError("No overlap between features index and avg_rank index.")

    features_use = features.loc[common]
    avg_rank_use = avg_rank.loc[common]

    os.makedirs(out_dir, exist_ok=True)
    save_cumulative_outputs(
        features=features_use,
        avg_rank=avg_rank_use,
        out_dir=out_dir,
        label=label,
    )


# -----------------------------------------------------------------------------
# Level-1 manuscript panel assembly
# -----------------------------------------------------------------------------

def _level1_panel_load_font(size=28, bold=False):
    """Load a broadly available font for manuscript panels."""
    try:
        from PIL import ImageFont
    except Exception:
        return None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _level1_panel_fit_image(img, box_w: int, box_h: int):
    """Resize an image to fit inside a panel box."""
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size)


def _level1_month_folder(root_dir: str, month_str: str):
    """Find a monthly Level-1 output folder such as 2024-06_level1."""
    from pathlib import Path
    root = Path(root_dir)
    cand = root / f"{month_str}_level1"
    if cand.exists():
        return cand
    hits = sorted(root.glob(f"{month_str}*_level1"))
    if hits:
        return hits[0]
    return None


def _level1_find_panel_image(month_folder, kind: str):
    """Find a monthly bar chart or baseline heatmap image."""
    if month_folder is None:
        return None
    patterns = {
        "bar": ["level1_portfolio_benefit_bar_N21*.png"],
        "heatmap": ["level1_baseline_rank_heatmap_ours_selected*.png"],
        "health": ["level1_health_risk_secondary_covid_import_export*.png"],
    }.get(kind, [])
    for pat in patterns:
        hits = sorted(month_folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _level1_month_folders_sorted(root_dir: str):
    """Return monthly Level-1 folders sorted by year-month."""
    from pathlib import Path
    root = Path(root_dir)
    folders = [p for p in root.glob("*_level1") if p.is_dir()]
    def _key(p):
        stem = p.name.split("_level1")[0]
        try:
            y, m = stem.split("-")[:2]
            return (int(y), int(m))
        except Exception:
            return (9999, 9999)
    return sorted(folders, key=_key)


def _level1_draw_panel(canvas, img_path, x: int, y: int, box_w: int, box_h: int, title: str, letter: str = None):
    """Draw a single image panel on a PIL canvas."""
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(canvas)
    title_font = _level1_panel_load_font(24, bold=True)
    label = f"{letter}. {title}" if letter else title
    draw.text((x, y), label, fill=(0, 0, 0), font=title_font)
    top_pad = 40
    body = Image.open(img_path)
    body = _level1_panel_fit_image(body, box_w, box_h - top_pad)
    ix = x + (box_w - body.width) // 2
    iy = y + top_pad
    canvas.paste(body, (ix, iy))


def _level1_save_canvas(canvas, out_base):
    """Save a PIL canvas as PNG and PDF."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_base.with_suffix(".png")
    out_pdf = out_base.with_suffix(".pdf")
    canvas.save(out_png)
    try:
        canvas.save(out_pdf, "PDF", resolution=300.0)
    except Exception as e:
        print(f"[Level1 panels] PDF save skipped for {out_pdf}: {e}")
    print(f"[Level1 panels] Saved: {out_png}")
    print(f"[Level1 panels] Saved: {out_pdf}")


def assemble_level1_panel_outputs(
    level1_root_dir: str,
    month_a: str = "2024-06",
    month_b: str = "2024-12",
    make_bar_1x2: bool = True,
    make_heatmap_1x2: bool = True,
    make_bar_4x3: bool = True,
):
    """
    Assemble manuscript-ready Level-1 panels after monthly Level-1 figures exist.

    Outputs:
      - level1_bar_panel_1x2_<month_a>_and_<month_b>.png/pdf
      - level1_heatmap_panel_1x2_<month_a>_and_<month_b>.png/pdf
      - level1_bar_panel_4x3_all_months.png/pdf

    Monthly single figures are left unchanged.
    """
    try:
        from PIL import Image
        from pathlib import Path
    except Exception as e:
        print(f"[Level1 panels] PIL/Pillow unavailable; panel assembly skipped: {e}")
        return

    root = Path(level1_root_dir)
    if not root.exists():
        print(f"[Level1 panels] Root not found; skipped: {root}")
        return

    def _make_1x2(kind: str, out_name: str, box_h: int):
        f_a = _level1_month_folder(str(root), month_a)
        f_b = _level1_month_folder(str(root), month_b)
        img_a = _level1_find_panel_image(f_a, kind)
        img_b = _level1_find_panel_image(f_b, kind)
        if img_a is None or img_b is None:
            print(f"[Level1 panels] Missing {kind} image for {month_a} or {month_b}; skipped {out_name}.")
            return
        margin, gap = 22, 16
        box_w = 780
        W = margin * 2 + box_w * 2 + gap
        H = margin * 2 + box_h
        canvas = Image.new("RGB", (W, H), "white")
        title_a = f"{month_a} benefit bar" if kind == "bar" else f"{month_a} baseline heatmap"
        title_b = f"{month_b} benefit bar" if kind == "bar" else f"{month_b} baseline heatmap"
        _level1_draw_panel(canvas, img_a, margin, margin, box_w, box_h, title_a, letter="A")
        _level1_draw_panel(canvas, img_b, margin + box_w + gap, margin, box_w, box_h, title_b, letter="B")
        _level1_save_canvas(canvas, root / out_name)

    if make_bar_1x2:
        _make_1x2("bar", f"level1_bar_panel_1x2_{month_a}_and_{month_b}", box_h=600)
    if make_heatmap_1x2:
        _make_1x2("heatmap", f"level1_heatmap_panel_1x2_{month_a}_and_{month_b}", box_h=760)

    if make_bar_4x3:
        folders = _level1_month_folders_sorted(str(root))
        images = []
        for mf in folders[:12]:
            img = _level1_find_panel_image(mf, "bar")
            if img is not None:
                images.append((mf.name.split("_level1")[0], img))
        if not images:
            print(f"[Level1 panels] No monthly bar charts found under {root}; skipped 4x3 panel.")
            return

        ncols, nrows = 3, 4
        margin, gap_x, gap_y = 18, 12, 16
        box_w, box_h = 500, 380
        W = margin * 2 + ncols * box_w + (ncols - 1) * gap_x
        H = margin * 2 + nrows * box_h + (nrows - 1) * gap_y
        canvas = Image.new("RGB", (W, H), "white")
        for idx, (month_label, img_path) in enumerate(images[:12]):
            r, c = divmod(idx, ncols)
            x = margin + c * (box_w + gap_x)
            y = margin + r * (box_h + gap_y)
            letter = chr(ord("A") + idx)
            _level1_draw_panel(canvas, img_path, x, y, box_w, box_h, month_label, letter=letter)
        _level1_save_canvas(canvas, root / "level1_bar_panel_4x3_all_months")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel selection runner")

    parser.add_argument("--selection_fraction", type=float, default=0.25)
    parser.add_argument("--total_N", type=int, default=20)
    parser.add_argument("--singleton_top_pct", type=float, default=0.30)
    parser.add_argument("--singleton_drop_km", type=float, default=100.0)
    parser.add_argument("--singleton_bottom_pct", type=float, default=0.20)
    parser.add_argument("--singleton_reserve_k", type=int, default=6)
    parser.add_argument("--singleton_reserve_frac", type=float, default=0.0)
    parser.add_argument("--isolation_bonus_km", type=float, default=150.0)
    parser.add_argument("--risk_weight", type=float, default=0.5)
    parser.add_argument("--coverage_weight", type=float, default=0.5)

    args = parser.parse_args()

    print("\nRunning with parameters:")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")

    main(
        selection_fraction=args.selection_fraction,
        total_N=args.total_N,
        singleton_top_pct=args.singleton_top_pct,
        singleton_drop_km=args.singleton_drop_km,
        singleton_bottom_pct=args.singleton_bottom_pct,
        singleton_reserve_k=args.singleton_reserve_k,
        singleton_reserve_frac=args.singleton_reserve_frac,
        isolation_bonus_km=args.isolation_bonus_km,
        risk_weight=args.risk_weight,
        coverage_weight=args.coverage_weight,
    )

    print("\nFinished successfully.\n")
