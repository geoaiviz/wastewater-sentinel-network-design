"""Visualization utilities for Level 2 validation and contribution outputs.

All paths are supplied by the caller. These functions create diagnostics and
maps only; they do not train the model or select sentinel sites.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.colors as mcolors
from typing import List, Optional, Tuple


# Polygon and centroid loading helpers.
def load_polygons(shp_path):
    """
    Returns GeoDataFrame of polygons in WGS84.
    """
    gdf = gpd.read_file(shp_path).copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def load_county_centroids(polys_path, name_field=None):
    """
    Returns a GeoDataFrame with columns:
      - 'county' (lowercased, stripped)
      - geometry (Point centroid guaranteed to lie inside the polygon)
    """
    gdf = gpd.read_file(polys_path)

    # pick a name field if not provided
    if name_field is None:
        for cand in ("LABEL", "COUNTY" ):
            if cand in gdf.columns:
                name_field = cand
                break
    if name_field is None:
        raise ValueError("Could not infer county name field. Pass name_field or rename a column to 'CountyName'.")

    # ensure projected CRS for good centroids, fall back to current
    try:
        gdf_proj = gdf.to_crs(3857)  # Web Mercator projection used for display
    except Exception:
        gdf_proj = gdf

    # robust 'inside' centroids
    pts_proj = gdf_proj.representative_point()  # always inside polygon
    pts = pts_proj.to_crs(gdf.crs) if gdf_proj.crs and gdf.crs and gdf_proj.crs != gdf.crs else pts_proj

    out = gpd.GeoDataFrame({
        "county": gdf[name_field].astype(str).str.strip().str.lower(),
        "geometry": pts
    }, geometry="geometry", crs=gdf.crs)
    return out, gdf_proj


def plot_wwtp_spokes(
    risk_wc,
    w_names,
    c_names,
    wwtp_gdf,
    county_centroids_gdf=None,     # optional: if None, derive from polygons_gdf
    out_dir=".",
    tag="",
    K=10,
    M=5,
    backdrop="boundary",            # "none" | "boundary"
    polygons_gdf=None,              # optional for backdrop or county centroids
    # Labeling options
    annotate_sites=True,
    annotate_counties=True,
    site_fontsize=8,
    county_fontsize=8,
    county_alpha=0.85,
    label_top_m_only=True,          # label only counties used by top-M spokes
    min_weight=0.0,                 # skip if weight ≤ this
    require_finite=True,
    strict_match=False,
    return_diagnostics=False,
    debug_print=True,
):
    """
    Draw 'spokes' from top-K WWTPs to their top-M counties by risk.
    Also labels WWTP site names and county names (configurable).
    """
    import os, re
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Helper: normalize names (matching)
    # --------------------------------------------------------
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        s = s.replace("&", "and")
        s = re.sub(r"\s+", " ", s)
        if not strict_match:
            s = re.sub(r"\bcounty\b$", "", s).strip()
        return s

    # --------------------------------------------------------
    # Helper: convert polygons → inside points
    # --------------------------------------------------------
    def _to_inside_points(gdf):
        G = gdf.copy()
        if G.crs is None:
            G = G.set_crs(epsg=4326, allow_override=True)
        crs_orig = G.crs
        try:
            Gp = G.to_crs(3857)
        except Exception:
            Gp = G

        def _pointify(geom):
            if geom is None or getattr(geom, "is_empty", False):
                return None
            gt = getattr(geom, "geom_type", "")
            if gt == "Point":
                return geom
            try:
                return geom.representative_point()
            except Exception:
                try:
                    return geom.centroid
                except Exception:
                    return None

        pts_proj = Gp.geometry.apply(_pointify)
        pts = pts_proj.to_crs(crs_orig) if hasattr(Gp, "crs") and Gp.crs != crs_orig else pts_proj
        G = G.copy()
        G.set_geometry(pts, inplace=True)
        return G

    # --------------------------------------------------------
    # Normalize inputs & keep ORIGINAL names for labels
    # --------------------------------------------------------
    w_names_norm = [_norm(w) for w in w_names]
    c_names_norm = [_norm(c) for c in c_names]

    WW = wwtp_gdf.copy()
    if "wwtp" not in WW.columns:
        for cand in ("WWTP", "name", "Name", "utility", "Utility", "site", "SiteName"):
            if cand in WW.columns:
                WW["wwtp"] = WW[cand].astype(str)
                break
    if "wwtp" not in WW.columns:
        raise KeyError("wwtp_gdf must have a 'wwtp' column (or a recognizable name field).")

    # keep original site name for labeling
    WW["wwtp_label"] = WW["wwtp"].astype(str)
    WW["name_l"] = WW["wwtp_label"].map(_norm)
    WW = _to_inside_points(WW)

    # County centroids: provided or derive from polygons
    if county_centroids_gdf is not None:
        CC = county_centroids_gdf.copy()
        if "county" not in CC.columns:
            raise KeyError("county_centroids_gdf must have a 'county' column.")
        CC["county_label"] = CC["county"].astype(str)
        CC["county_l"] = CC["county_label"].map(_norm)
        if not all(getattr(g, "geom_type", "") == "Point" for g in CC.geometry):
            CC = _to_inside_points(CC)
    elif polygons_gdf is not None:
        Gp = polygons_gdf.copy()
        name_field = None
        for cand in ("LABEL", "COUNTY", "NAME", "Name", "CountyName"):
            if cand in Gp.columns:
                name_field = cand; break
        if name_field is None:
            raise ValueError("polygons_gdf lacks a recognizable county name field.")
        Gp = _to_inside_points(Gp)
        CC = gpd.GeoDataFrame({
            "county_label": Gp[name_field].astype(str),
            "geometry": Gp.geometry
        }, geometry="geometry", crs=Gp.crs)
        CC["county_l"] = CC["county_label"].map(_norm)
    else:
        raise ValueError("Provide county_centroids_gdf or polygons_gdf.")

    # --------------------------------------------------------
    # CRS alignment (prefer WWTP CRS)
    # --------------------------------------------------------
    if WW.crs is None:
        WW = WW.set_crs(epsg=4326, allow_override=True)
    if CC.crs is None:
        CC = CC.set_crs(WW.crs or "EPSG:4326", allow_override=True)
    try:
        if WW.crs != CC.crs:
            CC = CC.to_crs(WW.crs)
    except Exception as e:
        if debug_print:
            print("[plot_wwtp_spokes] WARNING: CRS mismatch could not be fixed:", e)

    # Backdrop boundary
    if backdrop == "boundary" and polygons_gdf is not None:
        P = polygons_gdf.copy()
        if P.crs is None:
            P = P.set_crs(WW.crs or "EPSG:4326", allow_override=True)
        if P.crs != WW.crs:
            P = P.to_crs(WW.crs)
    else:
        P = None

    # --------------------------------------------------------
    # XY lookups + name lookups (normalized → original label)
    # --------------------------------------------------------
    def _ptxy(g):
        try:
            if g is None or getattr(g, "is_empty", False):
                return None
            return (float(g.x), float(g.y))
        except Exception:
            return None

    WW["_xy"] = WW.geometry.apply(_ptxy)
    CC["_xy"] = CC.geometry.apply(_ptxy)
    w_xy = dict(zip(WW["name_l"], WW["_xy"]))
    c_xy = dict(zip(CC["county_l"], CC["_xy"]))
    w_lab = dict(zip(WW["name_l"], WW["wwtp_label"]))
    c_lab = dict(zip(CC["county_l"], CC["county_label"]))

    # --------------------------------------------------------
    # Risk filtering & ranking
    # --------------------------------------------------------
    R = np.asarray(risk_wc, dtype=float)
    if require_finite:
        R = np.where(np.isfinite(R), R, 0.0)
    w_scores = R.sum(axis=1)
    topw_idx = np.argsort(-w_scores)[: int(K)]

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 9))

    if P is not None:
        try:
            P.boundary.plot(ax=ax, linewidth=0.5, alpha=0.3)
        except Exception:
            pass

    try:
        WW.plot(ax=ax, markersize=12, alpha=0.25, color="grey")
    except Exception:
        pass

    total_segments = 0
    dropped_by_weight = 0
    w_miss = c_miss = 0
    used_counties = set()

    for wi in topw_idx:
        wname_n = w_names_norm[wi]
        wxy = w_xy.get(wname_n)
        if wxy is None:
            w_miss += 1
            continue

        # draw top-M spokes from this WWTP
        row = R[wi, :]
        topj = np.argsort(-row)[: int(M)]
        segments = []
        for j in topj:
            v = row[j]
            if not np.isfinite(v) or v <= min_weight:
                dropped_by_weight += 1
                continue
            cname_n = c_names_norm[j]
            cxy_j = c_xy.get(cname_n)
            if cxy_j is None:
                c_miss += 1
                continue
            segments.append([wxy, cxy_j])
            used_counties.add(cname_n)

        if segments:
            lc = LineCollection(segments, linewidths=1.2, alpha=0.55)
            ax.add_collection(lc)
            total_segments += len(segments)

        # site marker & label
        ax.scatter([wxy[0]], [wxy[1]], s=60, zorder=5)
        if annotate_sites:
            ax.text(
                wxy[0], wxy[1],
                w_lab.get(wname_n, wname_n),
                fontsize=site_fontsize, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

    # county labels (only those used by the spokes unless label_top_m_only=False)
    if annotate_counties:
        if label_top_m_only:
            to_label = [cn for cn in used_counties]
        else:
            to_label = list(c_xy.keys())

        for cname_n in to_label:
            cxy_pt = c_xy.get(cname_n)
            if cxy_pt is None:
                continue
            ax.scatter([cxy_pt[0]], [cxy_pt[1]], s=20, alpha=0.6)
            ax.text(
                cxy_pt[0], cxy_pt[1],
                c_lab.get(cname_n, cname_n),
                fontsize=county_fontsize, ha="center", va="center",
                alpha=county_alpha,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)
            )

    ax.set_title(f"Top {K} WWTPs → top {M} counties by risk {tag}".strip())
    ax.set_axis_off()
    fig.tight_layout()

    out_png = os.path.join(out_dir, f"wwtp_spokes_{K}x{M}{'_'+tag if tag else ''}.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    diag = {
        "segments_drawn": int(total_segments),
        "wwtp_points_missing_xy": int(WW["_xy"].isna().sum()),
        "county_points_missing_xy": int(CC["_xy"].isna().sum()),
        "wwtp_names_not_matched": int(w_miss),
        "county_names_not_matched": int(c_miss),
        "edges_dropped_by_weight": int(dropped_by_weight),
        "min_weight": float(min_weight),
        "out_png": out_png,
        "counties_labeled": len(used_counties) if annotate_counties else 0,
    }
    if debug_print:
        print("[plot_wwtp_spokes] diagnostics:", diag)

    return (out_png, diag) if return_diagnostics else out_png


def plot_risk_signal_correlation_heatmap(csv_path, out_dir, title="", vmax=1.0, tag=""):
    """
    Load a correlation CSV (produced by pipe.save_risk_signal_correlations)
    and plot a heatmap.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    df_num = df.apply(pd.to_numeric, errors="coerce")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df_num.values,
        cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=df_num.columns, yticklabels=df_num.index,
        cbar_kws={"label": "Correlation"}
    )
    plt.title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    outp = os.path.join(out_dir, f"heatmap_{tag}.png")
    plt.savefig(outp, dpi=160)
    plt.close()
    print(f"[plot_risk_signal_correlation_heatmap] → {outp}")
    return outp


def plot_risk_signal_top_pairs(csv_path, out_dir, title="", k=25, tag=""):
    """
    Flatten correlation matrix and plot top-|corr| pairs.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    df_num = df.apply(pd.to_numeric, errors="coerce")

    vals = df_num.values
    rows = df_num.index.to_numpy()
    cols = df_num.columns.to_numpy()

    rr, cc = np.meshgrid(np.arange(vals.shape[0]), np.arange(vals.shape[1]), indexing="ij")
    flat = pd.DataFrame({
        "row": rows[rr.ravel()],
        "col": cols[cc.ravel()],
        "corr": vals.ravel()
    }).dropna(subset=["corr"])

    top = flat.reindex(flat["corr"].abs().sort_values(ascending=False).index).head(k)

    plt.figure(figsize=(10, max(6, int(0.35 * len(top)))))
    ylabels = [f"{r} → {c}" for r, c in zip(top["row"], top["col"])]
    plt.barh(range(len(top)), top["corr"].to_numpy())
    plt.yticks(range(len(top)), ylabels, fontsize=7)
    plt.xlabel("Correlation")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    outp = os.path.join(out_dir, f"top_pairs_{tag}.png")
    plt.savefig(outp, dpi=160)
    plt.close()
    print(f"[plot_risk_signal_top_pairs] → {outp}")
    return outp


def load_labeled_matrix(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path, index_col=0)
    names = list(df.index)
    df = df.reindex(index=names, columns=names)
    return df.values.astype(float), names


def _safe_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]


def _merge_with_lonlat(df_top: pd.DataFrame, wwtp_gdf: gpd.GeoDataFrame, on: str = "node") -> pd.DataFrame:
    keep = _safe_cols(wwtp_gdf, ["node", "wwtp", "lon", "lat", "pop_served"])
    m = df_top.merge(wwtp_gdf[keep], on=on, how="left", suffixes=("", "_gdf"))

    for c in ["wwtp", "lon", "lat", "pop_served"]:
        cg = f"{c}_gdf"
        if cg in m.columns and c in m.columns:
            m[c] = m[c].where(m[c].notna(), m[cg])
        elif cg in m.columns and c not in m.columns:
            m[c] = m[cg]
    m = m.drop(columns=[c for c in m.columns if c.endswith("_gdf")], errors="ignore")
    return m

import numpy as np
import pandas as pd


from sklearn.cluster import SpectralClustering


# ------------------------- helpers -------------------------

def _prep_affinity(A, top_q=0.85):
    """Symmetrize, zero diag, sparsify by global quantile, rescale to [0,1]."""
    A = np.array(A, dtype=np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    if top_q is not None:
        pos = A[A > 0]
        thr = np.quantile(pos, top_q) if pos.size else 0.0
        A = np.where(A >= thr, A, 0.0)
    m = A.max()
    if m > 0:
        A = A / m
    return A


def _auto_k_by_eigengap(A, k_max=8, k_min=2):
    """Choose k via eigengap on normalized Laplacian with fallbacks."""
    n = A.shape[0]
    if n <= 2:
        return max(2, n)
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    Dm12 = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dm12 @ A @ Dm12
    w = np.linalg.eigvalsh(L)  # ascending
    if np.ptp(w) < 1e-5:
        return max(k_min, 2)
    k_max = min(k_max, n - 1)
    gaps = np.diff(w[:k_max + 1])
    # skip first trivial gap; ensure at least 2 clusters
    k_hat = int(np.argmax(gaps[1:]) + 1)
    k_hat = max(k_min, min(k_hat, k_max))
    return k_hat


def _spectral_labels(A, force_k=None, k_max=8, k_min=2, top_q=0.85, random_state=0):
    """Compute cluster labels from an affinity A."""
    A_aff = _prep_affinity(A, top_q=top_q)
    if (A_aff > 0).sum() == 0:
        # empty after sparsification → trivial split
        n = A_aff.shape[0]
        if force_k and force_k >= 2:
            k = int(force_k)
        else:
            k = max(k_min, 2) if n >= 2 else 1
        return np.zeros(n, dtype=int) if k == 1 else np.r_[0, np.ones(n - 1, dtype=int)]
    k = int(force_k) if (force_k is not None and force_k >= 2) else _auto_k_by_eigengap(
        A_aff, k_max=k_max, k_min=k_min
    )
    clus = SpectralClustering(
        n_clusters=k, affinity="precomputed",
        assign_labels="kmeans", random_state=random_state
    )
    return clus.fit_predict(A_aff)


def _find_default_matrix_path(pred_true_csv, tag):
    """Best-effort to locate a transmission matrix saved by the training script."""
    base_dir = os.path.dirname(pred_true_csv)
    # Check the standard filenames produced by the modeling workflow.
    cands = [
        os.path.join(base_dir, f"transmission_risk_matrix_overall{tag}.csv"),
        os.path.join(base_dir, f"transmission_predicted{tag}.csv"),
        os.path.join(base_dir, f"transmission_TRUE_proxy_corr{tag}.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _select_wwtp_submatrix(df, node_prefix_wwtp="W_"):
    """Return WWTP-only matrix and ordered WWTP node names from a square DataFrame."""
    cols = [c for c in df.columns if str(c).startswith(node_prefix_wwtp)]
    rows = [r for r in df.index if str(r).startswith(node_prefix_wwtp)]
    df = df.loc[rows, cols]
    nodes = list(df.index)
    return df.values.astype(np.float32), nodes


def _similarity_from_pred_csv(pred_true_csv, node_prefix_wwtp="W_"):
    """
    Build WWTP similarity by correlating predicted time series:
    uses columns 'pred_W_xxx' in the pred_vs_true CSV.
    """
    df = pd.read_csv(pred_true_csv, index_col=0)
    pred_cols = [c for c in df.columns if c.startswith("pred_{}".format(node_prefix_wwtp))]
    if not pred_cols:
        raise ValueError("No columns like 'pred_W_...' found in {}".format(pred_true_csv))
    # map 'pred_W_xxx' -> 'W_xxx'
    colmap = {c: c.replace("pred_", "") for c in pred_cols}
    P = df[pred_cols].rename(columns=colmap)
    # correlation as similarity; ensure finite values
    P = P.replace([np.inf, -np.inf], np.nan).astype(float)
    S = P.corr(method="pearson", min_periods=3).fillna(0.0)
    return S.values.astype(np.float32), list(S.columns)


def load_wwtp_points(shp_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path).copy()
    if "wwtp" not in gdf.columns:
        for cand in ["WWTP", "name", "Name", "Utility", "utility"]:
            if cand in gdf.columns:
                gdf["wwtp"] = gdf[cand].astype(str)
                break
    if "wwtp" not in gdf.columns:
        raise KeyError("Shapefile must contain a 'wwtp' column (or map it).")

    gdf["wwtp"] = gdf["wwtp"].astype(str).str.strip().str.lower()
    gdf["node"] = gdf["wwtp"].apply(lambda x: f"W_{x}")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    cent = gdf.geometry.centroid
    gdf["lon"] = cent.x.astype(float)
    gdf["lat"] = cent.y.astype(float)

    if "pop_served" in gdf.columns:
        gdf["pop_served"] = pd.to_numeric(gdf["pop_served"], errors="coerce")
    return gdf


# ---------------------------
# VIZ: heatmap of risk / corr
# ---------------------------

def plot_risk_heatmap(
        mat: np.ndarray,
        names: List[str],
        title: str,
        output_dir: str,
        skip_self: bool = False,
        highlight_self: bool = False,
        figsize: Tuple[float, float] = (24, 20)
):
    H = mat.copy()
    if skip_self:
        np.fill_diagonal(H, 0.0)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        H,
        xticklabels=names,
        yticklabels=names,
        cmap=mcolors.LinearSegmentedColormap.from_list("green_red", ["green", "yellow", "red"]),
        cbar_kws={"label": "Normalized Influence"},
        annot=False,
        linewidths=0.2,
        linecolor="white",
    )
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    if highlight_self:
        for i in range(len(names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, linewidth=1.2, linestyle="--"))
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "-").lower()
    os.makedirs(output_dir, exist_ok=True)
    outp = os.path.join(output_dir, f"{safe}.png")
    plt.savefig(outp, dpi=180)
    plt.close()
    print(f"[plot_risk_heatmap] → {outp}")


# ---------------------------
# Ranking + saving top sites
# ---------------------------

def rank_sentinel_sites(
        risk_matrix: np.ndarray,
        names: List[str],
        k: Optional[int] = 10,
        focus: str = "wwtp",
        metric: str = "outbound",
) -> pd.DataFrame:
    types = np.array(["county" if n.startswith("C_") else "wwtp" for n in names])

    M = risk_matrix.copy()
    np.fill_diagonal(M, 0.0)

    outbound = M.sum(axis=1)
    inbound = M.sum(axis=0)
    if metric == "outbound":
        score = outbound
    elif metric == "inbound":
        score = inbound
    else:
        score = outbound + inbound

    df = pd.DataFrame({"node": names, "type": types, "score": score})
    if focus in {"wwtp", "county"}:
        df = df[df["type"] == focus].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    if k is not None and k > 0:
        df = df.head(k).copy()
    return df


def save_top_sentinel_sites(
        df_top: pd.DataFrame,
        wwtp_gdf: gpd.GeoDataFrame,
        out_dir: str,
        tag: str = "_viz",
) -> pd.DataFrame:
    """
    Save a ranked table of top WWTP sentinel sites w
    ith lon/lat (and pop_served if available).

    Accepts df_top in any of these shapes:
      1) ['wwtp','score']                (your case)
      2) ['wwtp','risk_score']           (auto-renamed to 'score')
      3) ['node','score']                (node like 'W_<normalized name>')
      4) ['type','node','score']         (will filter type == 'wwtp')

    Writes: <out_dir>/top_sentinel_sites{tag}.csv
    Returns: merged DataFrame with columns like ['rank','wwtp','node','score','lon','lat',...]
    """
    import os
    import numpy as np

    def _norm(s):
        return str(s).strip().lower()

    # ----------------------------
    # Prep wwtp_gdf to ensure 'wwtp','node','lon','lat' exist
    # ----------------------------
    G = wwtp_gdf.copy()

    # find/construct 'wwtp' if missing
    if "wwtp" not in G.columns:
        for cand in ["WWTP", "wwtp_name", "name", "Name", "Utility", "utility", "site", "SiteName"]:
            if cand in G.columns:
                G["wwtp"] = G[cand].astype(str)
                break
    if "wwtp" not in G.columns:
        raise KeyError("wwtp_gdf must have 'wwtp' or a recognizable name field (e.g., 'name', 'Utility').")

    if "node" not in G.columns:
        G["node"] = "W_" + G["wwtp"].map(_norm)

    # ensure lon/lat exist (derive from geometry if needed)
    if ("lon" not in G.columns) or ("lat" not in G.columns):
        if hasattr(G, "geometry") and G.geometry is not None:
            # assume geometry is set; make robust centroids and convert to WGS84
            try:
                g_proj = G
                if g_proj.crs is None:
                    g_proj = g_proj.set_crs(epsg=4326, allow_override=True)
                pts_proj = g_proj.representative_point() if hasattr(g_proj, "representative_point") else g_proj.centroid
                pts = pts_proj.to_crs(epsg=4326) if g_proj.crs and g_proj.crs.to_epsg() != 4326 else pts_proj
                G["lon"] = pts.x.astype(float)
                G["lat"] = pts.y.astype(float)
            except Exception as e:
                raise KeyError(f"Could not compute lon/lat from wwtp_gdf geometry: {e}")
        else:
            raise KeyError("wwtp_gdf must have 'lon'/'lat' columns or a valid 'geometry' to compute them.")

    # ----------------------------
    # Harmonize df_top to ['node','wwtp','score']
    # ----------------------------
    T = df_top.copy()

    # optional filter by type
    if "type" in T.columns:
        T = T[T["type"].astype(str).str.lower() == "wwtp"].copy()

    # accept 'risk_score' as 'score'
    if "score" not in T.columns:
        if "risk_score" in T.columns:
            T = T.rename(columns={"risk_score": "score"})
        else:
            raise KeyError("df_top must include either 'score' or 'risk_score'.")

    # ensure node/wwtp
    if "node" in T.columns:
        T["node"] = T["node"].astype(str)
        if "wwtp" not in T.columns:
            T["wwtp"] = T["node"].str.replace(r"^W_", "", regex=True)
    elif "wwtp" in T.columns:
        T["wwtp"] = T["wwtp"].astype(str)
        T["node"] = "W_" + T["wwtp"].map(_norm)
    else:
        raise KeyError("df_top must include 'node' or 'wwtp'.")

    # ----------------------------
    # Merge coords & finalize
    # ----------------------------
    M = _merge_with_lonlat(T, G, on="node")  # keeps ['wwtp','lon','lat','pop_served'] if present
    M = M.sort_values("score", ascending=False).reset_index(drop=True)
    M.insert(0, "rank", np.arange(1, len(M) + 1))

    # tidy column order
    preferred = ["rank", "wwtp", "node", "score", "lon", "lat", "pop_served"]
    cols = [c for c in preferred if c in M.columns] + [c for c in M.columns if c not in preferred]
    M = M[cols]

    # warn if any missing coords
    if M[["lon", "lat"]].isna().any(axis=1).sum() > 0:
        miss = int(M[["lon", "lat"]].isna().any(axis=1).sum())
        print(f"[save_top_sentinel_sites] Warning: {miss} row(s) missing lon/lat after merge.")

    # write CSV
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"top_sentinel_sites{tag}.csv")
    M.to_csv(path, index=False)
    print(f"[save_top_sentinel_sites] saved {len(M)} rows → {path}")
    return M



# ---------------------------
# Point map of top WWTPs
# ---------------------------

def _sizes_from_scores(scores: np.ndarray, base: float = 80, span: float = 220) -> np.ndarray:
    if scores.size == 0 or np.all(np.isnan(scores)) or (np.nanmax(scores) == np.nanmin(scores)):
        return np.full_like(scores, base, dtype=float)
    lo, hi = np.nanmin(scores), np.nanmax(scores)
    return base + span * (scores - lo) / (hi - lo + 1e-9)


def plot_top_sites_map(
        df_top: pd.DataFrame,
        wwtp_gdf: gpd.GeoDataFrame,
        title: str,
        out_dir: str,
        tag: str = "_viz",
        annotate: bool = True,
        county_polys_gdf: gpd.GeoDataFrame = None,
        sewershed_gdf: gpd.GeoDataFrame = None,
        use_basemap: bool = True
):
    """
    Plot top WWTP sites based on transmission/self scores with county boundaries,
    optional sewershed polygons, and an optional light basemap.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import contextily as cx

    def _norm(s):
        return str(s).strip().lower()

    # --- Ensure WWTP CRS = GCS_North_American_1983 (EPSG:4269) ---
    G = wwtp_gdf.copy()
    if G.crs is None:
        G = G.set_crs(epsg=4269, allow_override=True)
    elif G.crs.to_epsg() != 4269:
        G = G.to_crs(epsg=4269)

    # --- Harmonize identifiers ---
    if "wwtp" not in G.columns:
        for cand in ["WWTP", "wwtp_name", "name", "Name", "Utility", "utility", "site", "SiteName"]:
            if cand in G.columns:
                G["wwtp"] = G[cand].astype(str)
                break
    if "wwtp" not in G.columns:
        raise KeyError("wwtp_gdf must have a recognizable 'wwtp' column.")
    G["wwtp_norm"] = G["wwtp"].map(_norm)
    if "node" not in G.columns:
        G["node"] = "W_" + G["wwtp_norm"]

    # --- Compute lon/lat if missing ---
    if ("lon" not in G.columns) or ("lat" not in G.columns):
        cent = G.geometry.representative_point()
        G["lon"] = cent.x.astype(float)
        G["lat"] = cent.y.astype(float)

    # --- Filter and merge top sites ---
    T = df_top.copy()
    if "type" in T.columns:
        T = T[T["type"].astype(str).str.lower() == "wwtp"].copy()
    if "node" in T.columns:
        if "wwtp" not in T.columns:
            T["wwtp"] = T["node"].str.replace("^W_", "", regex=True)
    elif "wwtp" in T.columns:
        T["node"] = "W_" + T["wwtp"].map(_norm)
    else:
        raise KeyError("df_top must include either 'node' or 'wwtp'.")
    if "score" not in T.columns:
        raise KeyError("df_top must include a 'score' column.")

    M = T.merge(G[["node", "wwtp", "lon", "lat"]], on="node", how="left").dropna(subset=["lon", "lat"]).copy()
    if M.empty:
        print("[plot_top_sites_map] No top WWTP rows or missing coordinates.")
        return

    # --- Size scaling ---
    def _sizes_from_scores(scores, base=80, span=220):
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0 or np.nanmax(scores) == np.nanmin(scores):
            return np.full_like(scores, base, dtype=float)
        lo, hi = np.nanmin(scores), np.nanmax(scores)
        return base + span * (scores - lo) / (hi - lo + 1e-9)

    sizes = _sizes_from_scores(M["score"])

    # --- Begin plotting ---
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    # 1. County boundaries (grey outline)
    if county_polys_gdf is not None:
        try:
            if county_polys_gdf.crs is None:
                county_polys_gdf = county_polys_gdf.set_crs(epsg=4269, allow_override=True)
            elif county_polys_gdf.crs.to_epsg() != 4269:
                county_polys_gdf = county_polys_gdf.to_crs(epsg=4269)
            county_polys_gdf.boundary.plot(ax=ax, linewidth=0.7, edgecolor="grey", alpha=0.65, zorder=1)
        except Exception as e:
            print("[plot_top_sites_map] county boundary overlay skipped:", e)

    # 2. Optional sewershed polygons
    if sewershed_gdf is not None:
        try:
            if sewershed_gdf.crs is None:
                sewershed_gdf = sewershed_gdf.set_crs(epsg=4269, allow_override=True)
            elif sewershed_gdf.crs.to_epsg() != 4269:
                sewershed_gdf = sewershed_gdf.to_crs(epsg=4269)
            sewershed_gdf.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.35, zorder=2)
        except Exception as e:
            print("[plot_top_sites_map] sewershed overlay skipped:", e)

    # 3. All WWTPs (faint background)
    ax.scatter(G["lon"], G["lat"], s=10, alpha=0.25, color="#6baed6", label="All WWTPs", zorder=3)

    # 4. Top WWTPs (scaled circles)
    ax.scatter(M["lon"], M["lat"], s=sizes, alpha=0.92, color="#f58634",
               edgecolor="none", label="Top WWTPs (size ∝ score)", zorder=4)

    # 5. Add light basemap
    if use_basemap:
        try:
            cx.add_basemap(ax, crs="EPSG:4269",
                           source=cx.providers.OpenStreetMap.Mapnik,
                           alpha=0.25, zorder=0)

        # cx.add_basemap(ax, crs="EPSG:4269",
        #                source=cx.providers.CartoDB.Positron,  # light grey, modern
        #                alpha=0.3, zorder=0)

        except Exception as e:
            print("[plot_top_sites_map] basemap skipped:", e)

    # 6. Labels (offset, no 'W_')
    if annotate:
        for _, r in M.iterrows():
            # pick the available name field safely
            name = None
            for col in ["wwtp_y", "wwtp_x", "wwtp", "node"]:
                if col in M.columns:
                    name = str(r[col])
                    break
            if name is None:
                continue

            name = name.replace("W_", "").strip()
            ax.text(float(r["lon"]) + 0.05, float(r["lat"]) + 0.05, name,
                    fontsize=8, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    zorder=5)

    # 7. Title and axes
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)  # remove grid lines

    # 8. Legend
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, alpha=0.4,
               label="All WWTPs", color="#6baed6"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, alpha=0.9,
               label="Top WWTPs (size ∝ score)", color="#f58634"),
    ]
    if county_polys_gdf is not None:
        legend_elements.append(Patch(facecolor="none", edgecolor="grey", linewidth=0.7, label="County boundary"))
    if sewershed_gdf is not None:
        legend_elements.append(Patch(facecolor="none", edgecolor="black", linewidth=0.6, label="WWTP sewershed"))
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    # 9. Save
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"top_sentinel_map{tag}.png".replace(" ", "_")
    outp = os.path.join(out_dir, fname)
    plt.savefig(outp, dpi=250)
    plt.close()
    print(f"[plot_top_sites_map] → {outp}")


def plot_top_sites_map_html(
        df_top: pd.DataFrame,
        wwtp_gdf: gpd.GeoDataFrame,
        out_dir: str,
        tag: str = "_viz",
        county_polys_gdf: gpd.GeoDataFrame = None,
        sewershed_gdf: gpd.GeoDataFrame = None,
        title: str = "Top WWTP Sentinels (interactive)",
        top_color: str = "#f58634",
        all_color: str = "#6baed6",
        outline_color: str = "black",
        min_radius: int = 5,
        max_radius: int = 16,
        label_fontsize: str = "11px",
        label_offset: float = 0.02        # degrees
):
    """
    Interactive Folium map with permanent labels and adjustable sizes.
    """
    import os, numpy as np, folium
    os.makedirs(out_dir, exist_ok=True)

    def _norm(s): return str(s).strip().lower()

    # --- normalize & prep ---
    G = wwtp_gdf.copy()
    if G.crs is None: G = G.set_crs(epsg=4269)
    elif G.crs.to_epsg() != 4269: G = G.to_crs(epsg=4269)
    if "wwtp" not in G.columns:
        for cand in ["WWTP","wwtp_name","name","Name","Utility","utility","site","SiteName"]:
            if cand in G.columns: G["wwtp"] = G[cand].astype(str); break
    G["wwtp_norm"] = G["wwtp"].map(_norm)
    if "node" not in G.columns: G["node"] = "W_" + G["wwtp_norm"]
    if ("lon" not in G.columns) or ("lat" not in G.columns):
        cent = G.geometry.representative_point(); G["lon"] = cent.x; G["lat"] = cent.y

    # top sites table
    T = df_top.copy()
    if "type" in T.columns: T = T[T["type"].str.lower()=="wwtp"]
    if "node" not in T.columns:
        if "wwtp" in T.columns: T["node"] = "W_" + T["wwtp"].map(_norm)
        else: raise KeyError("df_top must have node or wwtp")
    if "score" not in T.columns: raise KeyError("df_top must have score")

    Gm = G[["node","wwtp"]].rename(columns={"wwtp":"wwtp_name"})
    Gc = G[["node","lon","lat"]]
    M = T.merge(Gm,on="node",how="left").merge(Gc,on="node",how="left").dropna(subset=["lon","lat"])
    if M.empty: print("No top WWTP rows."); return
    G_4326 = G.to_crs(epsg=4326)
    cent_4326 = G_4326.geometry.representative_point()
    G_4326["lon"],G_4326["lat"]=cent_4326.x,cent_4326.y
    M = M.drop(columns=["lon","lat"],errors="ignore").merge(G_4326[["node","lon","lat"]],on="node",how="left")

    # --- map base ---
    latc, lonc = M["lat"].mean(), M["lon"].mean()
    m = folium.Map(location=[latc, lonc], zoom_start=7, tiles="cartodbpositron")

    # optional boundaries
    if county_polys_gdf is not None and not county_polys_gdf.empty:
        folium.GeoJson(county_polys_gdf.to_crs(epsg=4326).__geo_interface__,
                       style_function=lambda x:{"fillOpacity":0,"color":"grey","weight":1},
                       name="County boundaries").add_to(m)
    if sewershed_gdf is not None and not sewershed_gdf.empty:
        folium.GeoJson(sewershed_gdf.to_crs(epsg=4326).__geo_interface__,
                       style_function=lambda x:{"fillOpacity":0,"color":"black","weight":1},
                       name="WWTP sewersheds").add_to(m)

    # --- all WWTPs (blue + outline) ---
    all_layer = folium.FeatureGroup(name="All WWTPs", show=True)
    for _, r in G_4326.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=3,
                            color=outline_color, weight=0.7,
                            fill=True, fill_color=all_color, fill_opacity=0.8).add_to(all_layer)
    all_layer.add_to(m)

    # --- size scaling ---
    scores = np.asarray(M["score"], float)
    lo, hi = np.nanpercentile(scores, [5,95]) if len(scores) else (0,1)
    if hi <= lo: hi = lo + 1e-9
    def s2r(v):
        v = float(np.clip(v, lo, hi))
        return min_radius + (max_radius - min_radius)*(v-lo)/(hi-lo)

    # --- top sites (orange + outline + labels) ---
    top_layer = folium.FeatureGroup(name="Top WWTPs (size ∝ score)", show=True)
    for _, r in M.iterrows():
        name = str(r.get("wwtp_name", r.get("node",""))).replace("W_","").strip()
        lat, lon = float(r["lat"]), float(r["lon"])
        folium.CircleMarker(
            [lat, lon],
            radius=float(s2r(r["score"])),
            color=outline_color, weight=0.9,
            fill=True, fill_color=top_color, fill_opacity=0.9
        ).add_to(top_layer)
        # label slightly offset to the east/north
        folium.map.Marker(
            [lat + label_offset, lon + label_offset],
            icon=folium.DivIcon(
                html=f'<div style="font-size:{label_fontsize};color:#000000;white-space:nowrap;">{name}</div>'
            )
        ).add_to(top_layer)
    top_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html=os.path.join(out_dir,f"top_sentinel_map{tag}.html")
    m.save(out_html)
    print(f"[plot_top_sites_map_html] → {out_html}")
    return out_html


# ---------------------------
# Network visualization
# ---------------------------

def plot_network_map(
        risk_matrix: np.ndarray,
        names: List[str],
        wwtp_gdf: gpd.GeoDataFrame,
        df_rank: Optional[pd.DataFrame] = None,
        max_edges: int = 200,
        edge_quantile: float = 0.97,
        title: str = "Predicted Transmission Network (WWTPs)",
        out_dir: str = ".",
        tag: str = "_viz",
):
    types = np.array(["county" if n.startswith("C_") else "wwtp" for n in names])
    idx_w = np.where(types == "wwtp")[0]
    if idx_w.size == 0:
        print("[plot_network_map] No WWTP nodes to draw.")
        return

    loc = dict(zip(wwtp_gdf["node"], zip(wwtp_gdf["lon"], wwtp_gdf["lat"])))

    node_size = {n: 60.0 for n in names}
    if df_rank is not None and not df_rank.empty:
        m = df_rank.set_index("node")["score"].to_dict()
        vals = np.array(list(m.values()), dtype=float)
        lo, hi = (np.nanmin(vals), np.nanmax(vals)) if vals.size else (0.0, 1.0)
        for n in names:
            v = m.get(n, lo)
            node_size[n] = 80 + 220 * ((v - lo) / (hi - lo + 1e-9)) if hi > lo else 80.0

    M = risk_matrix.copy()
    np.fill_diagonal(M, 0.0)
    keep = np.zeros_like(M, dtype=bool)
    keep[np.ix_(idx_w, idx_w)] = True
    W = np.where(keep, M, 0.0)

    flat = W[W > 0].ravel()
    if flat.size == 0:
        print("[plot_network_map] No positive edges.")
        return
    thr = np.quantile(flat, edge_quantile)
    src, dst = np.where(W >= thr)
    if len(src) > max_edges:
        weights = W[src, dst]
        order = np.argsort(-weights)[:max_edges]
        src, dst = src[order], dst[order]

    edges = []
    for i, j in zip(src, dst):
        u, v = names[i], names[j]
        if u.startswith("W_") and v.startswith("W_") and (u in loc) and (v in loc):
            edges.append((u, v, W[i, j]))

    plt.figure(figsize=(11, 9))
    plt.scatter(wwtp_gdf["lon"], wwtp_gdf["lat"], s=8, alpha=0.25, label="WWTPs")
    for n, (x, y) in loc.items():
        plt.scatter([x], [y], s=node_size.get(n, 60.0), alpha=0.9)
    if edges:
        ew = np.array([w for (_, _, w) in edges], dtype=float)
        ewl = 0.5 + 3.5 * (ew - ew.min()) / (ew.max() - ew.min() + 1e-9) if ew.size else np.full(len(edges), 1.5)
        for k, (u, v, w) in enumerate(edges):
            x1, y1 = loc[u];
            x2, y2 = loc[v]
            plt.plot([x1, x2], [y1, y2], linewidth=ewl[k], alpha=0.5)

    plt.title(title)
    plt.xlabel("Longitude");
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.2);
    plt.legend()
    fname = f"network_map{tag}.png".replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, fname)
    plt.tight_layout();
    plt.savefig(outp, dpi=180);
    plt.close()
    print(f"[plot_network_map] → {outp}")


# ---------------------------
# Pred vs True plotting (grouped only)
# ---------------------------

def plot_predictions_grouped_by_range_from_csv(
        pred_true_csv: str,
        out_dir: str,
        names: Optional[List[str]] = None,
        title_prefix: str = "Pred_vs_True_grouped",
        q_span: int = 2,
        q_center: int = 3,
        cols: int = 4,
        smooth_win: int = 3,
        smooth_center: bool = True,
        show_raw_background: bool = True,
):
    """
    Create grouped figures comparing model predictions to the *trend* of the true series
    instead of the raw spiky true values.

    What changes vs previous version:
    - We build y_true_trend = rolling-mean-smoothed(true) with window = smooth_win.
    - We plot:
        blue  = y_true_trend   (the target the model is actually learning)
        orange= y_pred         (model output)
      Optionally:
        grey  = raw true (spiky) just for visual context.

    Inputs
    ------
    pred_true_csv : CSV produced by training/inference. Must have columns like:
        true_<node>, pred_<node>
    out_dir       : where to write PNGs + manifest CSV
    names         : optional explicit node order. If None, inferred from pred_ cols.
    q_span        : how coarsely to bin by span (y_max - y_min)
    q_center      : how coarsely to bin by center ((y_max + y_min)/2)
    cols          : subplots per row
    smooth_win    : rolling window size (in timesteps) for computing trend(true)
    smooth_center : if True, use centered rolling avg
    show_raw_background : if True, also draw the raw true series in faint grey

    Outputs
    -------
    - {title_prefix}_groupXX_nN.png
    - {title_prefix}_groups_manifest.csv
    """

    import math
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(pred_true_csv, index_col=0)

    # infer node names if not provided
    if names is None:
        names = sorted([c[len("pred_"):] for c in df.columns if c.startswith("pred_")])

    # helper to build smoothed trend from raw true
    def _smooth_trend(y_raw: np.ndarray) -> np.ndarray:
        s = pd.Series(y_raw, dtype=float)
        y_trend = (
            s.rolling(window=smooth_win,
                      center=smooth_center,
                      min_periods=1)
             .mean()
             .to_numpy()
        )
        return y_trend

    # --- compute stats per node for grouping (using BOTH pred and trend(true)) ---
    stats = []
    for n in names:
        true_col = f"true_{n}"
        pred_col = f"pred_{n}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        y_true_raw = df[true_col].astype(float).values
        y_pred     = df[pred_col].astype(float).values
        y_true_tr  = _smooth_trend(y_true_raw)

        # use both pred and smoothed true to estimate plotting range
        y_all = np.concatenate([y_true_tr, y_pred])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            continue

        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        span = y_max - y_min
        center = 0.5 * (y_max + y_min)
        stats.append({
            "node": n,
            "y_min": y_min,
            "y_max": y_max,
            "span": span,
            "center": center,
        })

    if not stats:
        print("[plot_predictions_grouped_by_range_from_csv] No valid series to plot.")
        return

    stat_df = pd.DataFrame(stats)

    # helper to generate quantile bins but be robust to duplicates
    def quantile_bin(series: pd.Series, q: int):
        if q <= 1 or np.nanmin(series.values) == np.nanmax(series.values):
            # everything same -> single bin
            return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
        try:
            return pd.qcut(series, q=q, labels=False, duplicates="drop")
        except Exception:
            # fallback: use rank percentile -> cut
            ranks = series.rank(method="average", pct=True)  # 0..1
            return pd.cut(ranks, bins=q, labels=False, include_lowest=True)

    # create hierarchical grouping: first by span_bin (variance), then by center_bin (level)
    stat_df["span_bin"] = quantile_bin(stat_df["span"], q_span)

    grouped = []
    for _, sub in stat_df.groupby("span_bin"):
        sub = sub.copy()
        sub["center_bin"] = quantile_bin(sub["center"], q_center)
        grouped.append(sub)
    stat_df = pd.concat(grouped, ignore_index=True)

    # We'll generate a figure per unique (span_bin, center_bin)
    T = len(df)
    group_keys = sorted(
        stat_df[["span_bin", "center_bin"]].drop_duplicates().itertuples(index=False, name=None)
    )

    manifest = []

    for gi, (sb, cb) in enumerate(group_keys, start=1):
        # which nodes in this subgroup?
        members = (
            stat_df[(stat_df["span_bin"] == sb) & (stat_df["center_bin"] == cb)]
            .sort_values("center")["node"]
            .tolist()
        )
        if not members:
            continue

        # y-limit based on trend + pred only (not raw spike)
        vals_for_ylim = []
        for n in members:
            y_true_raw = df[f"true_{n}"].astype(float).values
            y_pred     = df[f"pred_{n}"].astype(float).values
            y_true_tr  = _smooth_trend(y_true_raw)

            vals_for_ylim.append(y_true_tr)
            vals_for_ylim.append(y_pred)

        vals_for_ylim = np.concatenate(vals_for_ylim)
        vals_for_ylim = vals_for_ylim[np.isfinite(vals_for_ylim)]
        if vals_for_ylim.size == 0:
            y_min, y_max = (0.0, 1.0)
        else:
            y_min, y_max = float(np.min(vals_for_ylim)), float(np.max(vals_for_ylim))

        # layout
        rows = int(math.ceil(len(members) / cols))
        plt.figure(figsize=(5.6 * cols, 3.0 * rows))

        for idx, n in enumerate(members):
            ax = plt.subplot(rows, cols, idx + 1)
            x = np.arange(T)

            y_true_raw = df[f"true_{n}"].astype(float).values
            y_true_tr  = _smooth_trend(y_true_raw)
            y_pred     = df[f"pred_{n}"].astype(float).values

            # optional faint raw background for spike context
            if show_raw_background:
                ax.plot(
                    x, y_true_raw,
                    color="0.8", linewidth=0.7,
                    label="raw(true)" if idx == 0 else None,
                )

            # smoothed 'trend(true)' in blue
            ax.plot(
                x, y_true_tr,
                color="tab:blue", linewidth=1.6,
                label="trend(true)" if idx == 0 else None,
            )

            # model pred in orange dashed
            ax.plot(
                x, y_pred,
                color="tab:orange", linewidth=1.3, linestyle="--",
                label="pred" if idx == 0 else None,
            )

            ax.set_title(n, fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=8)

        # legend: only once per figure, using first axes' labels
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, loc="upper right", fontsize=8)

        plt.tight_layout()

        # save figure
        fname = f"{title_prefix}_group{gi:02d}_n{len(members)}.png"
        outp = os.path.join(out_dir, fname.replace(" ", "_"))
        plt.savefig(outp, dpi=170)
        plt.close()
        print(f"[plot_predictions_grouped_by_range_from_csv] → {outp}")

        # manifest row
        sb_span = stat_df.loc[stat_df["span_bin"] == sb, "span"]
        cb_cent = stat_df.loc[(stat_df["span_bin"] == sb) & (stat_df["center_bin"] == cb), "center"]
        span_lo, span_hi = (
            (float(np.nanmin(sb_span)), float(np.nanmax(sb_span))) if sb_span.size else (0.0, 0.0)
        )
        cent_lo, cent_hi = (
            (float(np.nanmin(cb_cent)), float(np.nanmax(cb_cent))) if cb_cent.size else (0.0, 0.0)
        )
        manifest.append({
            "group_id": gi,
            "n_members": len(members),
            "members": ",".join(members),
            "span_min": span_lo,
            "span_max": span_hi,
            "center_min": cent_lo,
            "center_max": cent_hi,
            "y_min": float(y_min),
            "y_max": float(y_max),
        })

    # write manifest CSV
    if manifest:
        man_df = pd.DataFrame(manifest)
        man_path = os.path.join(out_dir, f"{title_prefix}_groups_manifest.csv")
        man_df.to_csv(man_path, index=False)
        print(f"[plot_predictions_grouped_by_range_from_csv] → manifest {man_path}")


def _eigengap_k(L, kmin=2, kmax=10):
    """Choose k via eigengap on normalized Laplacian."""
    # Use smallest k where gap is maximized among first kmax-1 gaps
    evals, _ = np.linalg.eigh(L)
    evals = np.sort(evals)
    gaps = np.diff(evals[:max(kmax, 3)])
    if len(gaps) == 0:
        return kmin
    idx = np.argmax(gaps[kmin - 1:kmax - 1]) + (kmin - 1)
    k = max(kmin, min(kmax, idx + 1))
    return int(k)


def _spectral_cluster_from_corr(corr_abs, k=None, kmin=2, kmax=10):
    """Spectral clustering on similarity (abs corr). Returns labels 0..k-1."""
    S = corr_abs.copy()
    np.fill_diagonal(S, 1.0)
    # Build normalized Laplacian
    d = S.sum(axis=1) + 1e-8
    Dinvhalf = np.diag(1.0 / np.sqrt(d))
    L = np.eye(S.shape[0]) - Dinvhalf @ S @ Dinvhalf
    if k is None:
        k = _eigengap_k(L, kmin=kmin, kmax=kmax)
    # Use k smallest eigenvectors of L
    evals, evecs = np.linalg.eigh(L)
    X = evecs[:, :k]  # [N,k]
    # k-means (tiny, deterministic)
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    C = X[idx]  # init centers
    for _ in range(30):
        # assign
        dist = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # [N,k]
        lab = np.argmin(dist, axis=1)
        # update
        for j in range(k):
            pts = X[lab == j]
            if len(pts) > 0:
                C[j] = pts.mean(axis=0)
    return lab


def build_proxy_corr_from_predtrue_csv(pred_true_csv):
    """Load pred_vs_true CSV, build abs Pearson corr of TRUE series (proxy similarity)."""
    df = pd.read_csv(pred_true_csv, index_col=0)
    nodes = sorted({c.replace("true_", "") for c in df.columns if c.startswith("true_")})
    Y = pd.DataFrame({n: df[f"true_{n}"] for n in nodes})
    # Optional mask support if present
    mask_cols = [f"mask_{n}" for n in nodes]
    if all(c in df.columns for c in mask_cols):
        M = pd.DataFrame({n: df[f"mask_{n}"] > 0 for n in nodes})
        Y = Y.where(M, np.nan)
    corr = Y.corr(method="pearson", min_periods=2).abs().fillna(0.0).values
    return corr, nodes


def proxy_group_colors_on_map(
        pred_true_csv,
        wwtp_gdf,
        out_dir,
        tag="",
        k=None,
        matrix_csv=None,
        node_prefix_wwtp="W_",
        k_max=8,
        k_min=2,
        top_q=0.85,
        random_state=0,
        name_field_candidates=("wwtp_name", "wwtp", "name", "WWTP_NAME", "WWTP", "site", "SiteName"),
):
    """
    Cluster WWTPs by network affinity and color them on a map.

    Parameters
    ----------
    pred_true_csv : str
        Path to 'pred_vs_true_*.csv' produced by the training script.
        Used for fallback similarity if no transmission matrix is found.
    wwtp_gdf : GeoDataFrame
        Points of WWTPs. Must contain a name field matching your node naming
        (e.g., 'W_<lowercased name>'), or at least a raw name we can normalize.
    out_dir : str
        Directory to save outputs (PNG and CSV).
    tag : str
        Suffix used in your pipeline (e.g., '_covid_combined').
    k : int or None
        If None, auto-select; otherwise force this number of clusters (>=2).
    matrix_csv : str or None
        Optional direct path to a square transmission matrix CSV (rows/cols = nodes).
        If None, we try to locate one next to pred_true_csv using tag.
    node_prefix_wwtp : str
        Prefix of WWTP nodes in your matrices (default 'W_').
    k_max, k_min, top_q, random_state : spectral clustering knobs.
    name_field_candidates : tuple[str]
        Column candidates in wwtp_gdf for names.

    Outputs
    -------
    - PNG figure in out_dir: 'wwtp_clusters{tag}.png'
    - CSV in out_dir: 'wwtp_clusters{tag}.csv' with ['wwtp_node','cluster']
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load or construct WWTP affinity matrix
    if matrix_csv is None:
        matrix_csv = _find_default_matrix_path(pred_true_csv, tag)

    if matrix_csv and os.path.exists(matrix_csv):
        M = pd.read_csv(matrix_csv, index_col=0)
        try:
            A, nodes = _select_wwtp_submatrix(M, node_prefix_wwtp=node_prefix_wwtp)
        except Exception as e:
            raise RuntimeError(f"Failed to extract WWTP submatrix from {matrix_csv}: {e}")
    else:
        # Fallback: build similarity from predicted time series in pred_true_csv
        A, nodes = _similarity_from_pred_csv(pred_true_csv, node_prefix_wwtp=node_prefix_wwtp)

    # 2) Cluster labels
    labels = _spectral_labels(A, force_k=k, k_max=k_max, k_min=k_min, top_q=top_q, random_state=random_state)
    if len(np.unique(labels)) == 1:
        print("[proxy_group_colors_on_map] Auto-k collapsed to 1 cluster. "
              "Consider passing k explicitly (e.g., k=4) or raising top_q to 0.90–0.95.")

    # 3) Prepare a lookup from node name -> cluster id
    df_clusters = pd.DataFrame({"wwtp_node": nodes, "cluster": labels})

    # 4) Attach clusters to wwtp_gdf by matching names
    #    node names are like 'W_<canonical-lower-name>'; we strip the 'W_' and match.
    node_core = {n: n[len(node_prefix_wwtp):] if n.startswith(node_prefix_wwtp) else n for n in nodes}

    # find name column in gdf
    name_col = None
    for cand in name_field_candidates:
        if cand in wwtp_gdf.columns:
            name_col = cand
            break
    if name_col is None:
        raise ValueError(f"None of the name columns {name_field_candidates} found in wwtp_gdf.")

    # normalize: lower & strip
    gdf = wwtp_gdf.copy()
    gdf["_name_norm"] = gdf[name_col].astype(str).str.strip().str.lower()

    # build mapping: normalized gdf name -> 'W_<normalized>'
    node_norm = {n: core for n, core in node_core.items()}
    # prepare cluster map keyed by core name
    cluster_by_core = {node_core[n]: int(lbl) for n, lbl in zip(nodes, labels)}

    # attach clusters
    gdf["_core"] = gdf["_name_norm"]
    gdf["_cluster"] = gdf["_core"].map(cluster_by_core)

    # keep only matched rows
    gdf_matched = gdf[~gdf["_cluster"].isna()].copy()
    if gdf_matched.empty:
        raise RuntimeError("No WWTPs in gdf matched the cluster nodes. "
                           "Check name normalization or provided name column.")

    gdf_matched["_cluster"] = gdf_matched["_cluster"].astype(int)

    # 5) Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    # Optional background layer for county or basemap context

    # color by cluster
    clusters = sorted(gdf_matched["_cluster"].unique())
    # for c in clusters:
    #     gdf_matched[gdf_matched["_cluster"] == c].plot(ax=ax, markersize=36, label=f"Cluster {c}")

    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab10", len(clusters))
    for i, c in enumerate(clusters):
        gdf_matched[gdf_matched["_cluster"] == c].plot(
            ax=ax,
            markersize=36,
            color=cmap(i),
            label=f"Cluster {c}"
        )

    ax.set_title(f"WWTP Clusters {tag}".strip(), fontsize=14)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True)

    png_path = os.path.join(out_dir, f"wwtp_clusters{tag}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    # 6) Save CSV
    csv_path = os.path.join(out_dir, f"wwtp_clusters{tag}.csv")
    df_clusters.to_csv(csv_path, index=False)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {csv_path}")
    return df_clusters, gdf_matched


import numpy as np


def diagnostic_verify_mobility_regulation_viz(
        out_dir,
        tag="",
        # Inputs (CSV paths). Provide at least: flow_csv and pred_matrix_csv.
        flow_csv=None,  # REQUIRED: adjacency/flow matrix CSV (square; rows/cols = node_names)
        pred_matrix_csv=None,  # required predicted-transmission or regulated-risk matrix
        unreg_matrix_csv=None,  # OPTIONAL: unregulated sensitivity (to compare)
        clin_proxy_csv=None,  # OPTIONAL: transmission_TRUE_proxy_corr{tag}.csv
        clinxmob_proxy_csv=None,  # OPTIONAL: transmission_TRUE_proxy_corr_x_mobility{tag}.csv
        # options
        top_k=50,
        dpi=160,
        title_prefix="Verification",
):
    """
    Stand-alone verification (viz-layer) for mobility-regulated transmission.

    Expects CSVs with square matrices (rows/cols are node names).
    Saves into out_dir:
      - verify_corr_flow_vs_risk_reg{tag}.png
      - verify_scatter_flow_vs_risk_reg{tag}.png
      - verify_corr_flow_vs_risk_unreg{tag}.png              (if unreg provided)
      - verify_scatter_flow_vs_risk_unreg{tag}.png           (if unreg provided)
      - verify_top_edges_by_flow{tag}.csv
      - verify_top_edges_by_risk_reg{tag}.csv
      - verify_top_edges_by_risk_unreg{tag}.csv              (if unreg provided)
      - verify_proxy_diff_heatmap{tag}.png                   (if both proxies provided)
      - verify_summary_stats{tag}.txt
    """
    os.makedirs(out_dir, exist_ok=True)

    def _load_sq(csv_path, label):
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"{label} not found: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        if list(df.index) != list(df.columns):
            raise ValueError(f"{label} must be square with matching row/col labels.")
        return df

    def _sym_flow(A):
        F = 0.5 * (A + A.T)
        np.fill_diagonal(F, 0.0)
        m = F.max()
        return F / (m + 1e-12) if m > 0 else F

    def _flatten_offdiag(M):
        n = M.shape[0]
        triu = np.triu_indices(n, k=1)
        return M[triu]

    def _scatter(x, y, title, outpng):
        plt.figure(figsize=(6.2, 5))
        plt.scatter(x, y, s=8, alpha=0.4)
        plt.xlabel("Flow weight (sym, [0,1])")
        plt.ylabel("Risk weight")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpng, dpi=dpi)
        plt.close()

    def _corrplt(x, y, title, outpng):
        plt.figure(figsize=(6.2, 5))
        hb = plt.hexbin(x, y, gridsize=35, mincnt=1)
        plt.colorbar(hb, label="count")
        plt.xlabel("Flow weight (sym, [0,1])")
        plt.ylabel("Risk weight")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpng, dpi=dpi)
        plt.close()

    # --- load matrices ---
    df_flow = _load_sq(flow_csv, "flow_csv")
    df_pred = _load_sq(pred_matrix_csv, "pred_matrix_csv")

    # align to common node ordering
    common = [n for n in df_pred.index if n in df_flow.index]
    if not common:
        raise ValueError("No overlapping node names between flow_csv and pred_matrix_csv.")
    df_flow = df_flow.loc[common, common]
    df_pred = df_pred.loc[common, common]

    nodes = list(common)
    A = df_flow.values.astype(np.float32)
    flow_sym = _sym_flow(A)

    risk_reg = df_pred.values.astype(np.float32)

    # Optional unregulated
    risk_unreg = None
    if unreg_matrix_csv and os.path.exists(unreg_matrix_csv):
        df_unreg = _load_sq(unreg_matrix_csv, "unreg_matrix_csv").loc[common, common]
        risk_unreg = df_unreg.values.astype(np.float32)

    # --- correlations (offdiag) ---
    x = _flatten_offdiag(flow_sym)
    y_reg = _flatten_offdiag(risk_reg)
    r_reg = float(np.corrcoef(x, y_reg)[0, 1]) if x.size and y_reg.size else np.nan

    png_corr_reg = os.path.join(out_dir, f"verify_corr_flow_vs_risk_reg{tag}.png")
    _corrplt(x, y_reg, f"{title_prefix}: Flow vs Regulated Risk (r={r_reg:.3f})", png_corr_reg)

    png_scatter_reg = os.path.join(out_dir, f"verify_scatter_flow_vs_risk_reg{tag}.png")
    _scatter(x, y_reg, f"{title_prefix}: Flow vs Regulated Risk (r={r_reg:.3f})", png_scatter_reg)

    r_unreg = np.nan
    png_corr_unreg = png_scatter_unreg = None
    if risk_unreg is not None:
        y_unreg = _flatten_offdiag(risk_unreg)
        r_unreg = float(np.corrcoef(x, y_unreg)[0, 1]) if x.size and y_unreg.size else np.nan
        png_corr_unreg = os.path.join(out_dir, f"verify_corr_flow_vs_risk_unreg{tag}.png")
        _corrplt(x, y_unreg, f"{title_prefix}: Flow vs Unregulated Risk (r={r_unreg:.3f})", png_corr_unreg)
        png_scatter_unreg = os.path.join(out_dir, f"verify_scatter_flow_vs_risk_unreg{tag}.png")
        _scatter(x, y_unreg, f"{title_prefix}: Flow vs Unregulated Risk (r={r_unreg:.3f})", png_scatter_unreg)

    # --- top-edges tables ---
    def _edges_df(M, label):
        src = np.repeat(nodes, len(nodes))
        dst = np.tile(nodes, len(nodes))
        df = pd.DataFrame({"src": src, "dst": dst, label: M.flatten(), "flow": flow_sym.flatten()})
        df = df[df["src"] != df["dst"]]
        return df

    df_reg = _edges_df(risk_reg, "risk_reg")
    df_reg.sort_values("risk_reg", ascending=False).head(top_k).to_csv(
        os.path.join(out_dir, f"verify_top_edges_by_risk_reg{tag}.csv"), index=False
    )
    df_reg.sort_values("flow", ascending=False).head(top_k).to_csv(
        os.path.join(out_dir, f"verify_top_edges_by_flow{tag}.csv"), index=False
    )
    if risk_unreg is not None:
        df_unreg = _edges_df(risk_unreg, "risk_unreg")
        df_unreg.sort_values("risk_unreg", ascending=False).head(top_k).to_csv(
            os.path.join(out_dir, f"verify_top_edges_by_risk_unreg{tag}.csv"), index=False
        )

    # --- proxies comparison (if both provided) ---
    proxy_diff_png = None
    corr_proxy = np.nan
    if clin_proxy_csv and clinxmob_proxy_csv and os.path.exists(clin_proxy_csv) and os.path.exists(clinxmob_proxy_csv):
        df_pc = _load_sq(clin_proxy_csv, "clin_proxy_csv")
        df_pm = _load_sq(clinxmob_proxy_csv, "clinxmob_proxy_csv")
        # align to common node set
        common2 = [n for n in nodes if n in df_pc.index and n in df_pm.index]
        if common2:
            Pc = df_pc.loc[common2, common2].values.astype(np.float32)
            Pm = df_pm.loc[common2, common2].values.astype(np.float32)
            corr_proxy = float(np.corrcoef(_flatten_offdiag(Pc), _flatten_offdiag(Pm))[0, 1])
            proxy_diff_png = os.path.join(out_dir, f"verify_proxy_diff_heatmap{tag}.png")
            plt.figure(figsize=(6.2, 5))
            plt.imshow(Pm - Pc, cmap="bwr")
            plt.colorbar(label="Mobility-weighted – Clinical-only")
            plt.title(f"{title_prefix}: Proxy (Clin×Mob – Clin)")
            plt.tight_layout()
            plt.savefig(proxy_diff_png, dpi=dpi)
            plt.close()

    # --- summary ---
    def _nz_frac(M):
        nz = (M > 0).sum()
        tot = M.size
        return float(nz) / float(tot) if tot else np.nan

    frac_reg = _nz_frac(risk_reg)
    frac_unreg = _nz_frac(risk_unreg) if risk_unreg is not None else np.nan

    summary_txt = os.path.join(out_dir, f"verify_summary_stats{tag}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Mobility-Regulated Transmission: Verification (viz) ===\n")
        f.write(f"Nodes: {len(nodes)}\n")
        f.write(f"Corr(flow, risk_reg offdiag): {r_reg:.4f}\n")
        if risk_unreg is not None:
            f.write(f"Corr(flow, risk_unreg offdiag): {r_unreg:.4f}\n")
        f.write(f"Non-zero fraction (risk_reg): {frac_reg:.4f}\n")
        if risk_unreg is not None:
            f.write(f"Non-zero fraction (risk_unreg): {frac_unreg:.4f}\n")
        if proxy_diff_png:
            f.write(f"Corr(proxy_clin, proxy_clin×mob offdiag): {corr_proxy:.4f}\n")
            f.write(f"Proxy diff heatmap: {os.path.basename(proxy_diff_png)}\n")
        f.write("Scatter & hexbin plots saved.\n")
        f.write("Top-edges CSVs saved.\n")

    print(f"[Saved] {summary_txt}")
    return {
        "corr_flow_vs_risk_reg": r_reg,
        "corr_flow_vs_risk_unreg": r_unreg,
        "nz_frac_reg": frac_reg,
        "nz_frac_unreg": frac_unreg,
        "summary_path": summary_txt,
    }


def plot_wwtp_only_heatmap(
        risk_matrix: np.ndarray,
        names: List[str],
        out_dir: str,
        tag: str = "",
        title: str = "Transmission (WWTP ↔ WWTP, self excluded)",
        figsize: Tuple[float, float] = (12, 10)
):
    """Heatmap for WWTP×WWTP submatrix with diagonal zeroed (no self)."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # pick WWTP rows/cols
    idx = [i for i, n in enumerate(names) if n.startswith("W_")]
    if not idx:
        print("[plot_wwtp_only_heatmap] No WWTP nodes found.")
        return
    M = risk_matrix[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, 0.0)  # exclude self

    ww_names = [names[i] for i in idx]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        M, xticklabels=ww_names, yticklabels=ww_names,
        cmap="viridis", cbar_kws={"label": "Normalized Influence"},
        annot=False, linewidths=0.2, linecolor="white"
    )
    plt.title(title)
    plt.xticks(rotation=30);
    plt.yticks(rotation=0)
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"wwtp_only_heatmap{tag}.png")
    plt.tight_layout();
    plt.savefig(outp, dpi=180);
    plt.close()
    print(f"[plot_wwtp_only_heatmap] → {outp}")


# --- add to Model_DCRNN_Viz.py ---

import numpy as np

def flow_reference_from_adj(
        adj_static: np.ndarray,
        node_names,
        output_dir: str,
        symmetrize: str = "mean",  # {"mean","sum","max"}
        flow_gamma: float = 1.0,  # >1 highlights high-flow links
        flow_min_q: float =0.2,  # e.g., 0.20 to prune weakest edges globally
        normalize: bool = True,  # scale to [0,1] for plotting
        make_degree_hist: bool = True,
        tag: str = "",
) -> np.ndarray:
    """
    Build a reference flow matrix from an adjacency, save CSV + heatmap, and (optionally) degree histograms.
    Returns the processed flow matrix (after symmetrization/pruning/gamma/normalize).
    """
    A = np.array(adj_static, dtype=np.float32)

    # --- symmetrize ---
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)

    # --- global pruning by quantile ---
    if flow_min_q is not None:
        pos = F[F > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)

    # --- power transform & normalize for display ---
    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))
    if normalize:
        m = F.max()
        if m > 0:
            F = F / m

    # Save matrix CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"flow_reference_matrix_{symmetrize}{tag}.csv")
    pd.DataFrame(F, index=node_names, columns=node_names).to_csv(csv_path)

    # Heatmap using existing helper in this module
    try:
        # Reuse the heatmap utility defined in this module.
        plot_risk_heatmap(
            F, node_names,
            title=f"Flow_reference_{symmetrize}{tag}",
            output_dir=output_dir,
            skip_self=False, highlight_self=True,
        )
    except Exception as e:
        print("[flow_reference_from_adj] heatmap skipped:", e)

    # Degree histograms (nonzero counts)
    if make_degree_hist:
        indeg = (F > 0).sum(axis=0)
        outdeg = (F > 0).sum(axis=1)
        plt.figure(figsize=(7, 4))
        plt.hist(outdeg, bins=20, alpha=0.6, label="Out-degree")
        plt.hist(indeg, bins=20, alpha=0.6, label="In-degree")
        plt.xlabel("Non-zero neighbors");
        plt.ylabel("Nodes")
        plt.title(f"Flow degree histogram ({symmetrize})")
        plt.legend()
        png = os.path.join(output_dir, f"flow_reference_degree_hist_{symmetrize}{tag}.png")
        plt.tight_layout();
        plt.savefig(png, dpi=160);
        plt.close()

    return F


def flow_vs_matrix_scatter(
        M: np.ndarray,
        adj_static: np.ndarray,
        output_dir: str,
        node_names,
        title: str = "Flow vs Matrix",
        symmetrize: str = "mean",
        flow_gamma: float = 1.0,
        flow_min_q: float  = None,
        tag: str = "",
):
    """
    Scatter / hexbin comparing symmetrized flow weights vs provided matrix weights (e.g., risk).
    Saves PNGs: verify_corr_flow_vs_risk_reg{tag}.png and verify_scatter_flow_vs_risk_reg{tag}.png
    """
    # Build the symmetrized flow reference (without writing extra CSV/plots)
    import numpy as np
    A = np.array(adj_static, dtype=np.float32)
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)

    if flow_min_q is not None:
        pos = F[F > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)

    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))
    m = F.max()
    if m > 0:
        F = F / m

    # flatten off-diagonals
    n = F.shape[0]
    triu = np.triu_indices(n, k=1)
    x = F[triu]
    y = M[triu]

    os.makedirs(output_dir, exist_ok=True)

    # Hexbin correlation plot
    plt.figure(figsize=(6.2, 5))
    hb = plt.hexbin(x, y, gridsize=35, mincnt=1)
    plt.colorbar(hb, label="count")
    plt.xlabel("Flow weight (sym, [0,1])")
    plt.ylabel("Matrix weight")
    plt.title(title)
    plt.tight_layout()
    png_corr = os.path.join(output_dir, f"verify_corr_flow_vs_matrix{tag}.png")
    plt.savefig(png_corr, dpi=160)
    plt.close()

    # Scatter plot
    plt.figure(figsize=(6.2, 5))
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.xlabel("Flow weight (sym, [0,1])")
    plt.ylabel("Matrix weight")
    plt.title(title)
    plt.tight_layout()
    png_scat = os.path.join(output_dir, f"verify_scatter_flow_vs_matrix{tag}.png")
    plt.savefig(png_scat, dpi=160)
    plt.close()

    # quick console corr
    try:
        import numpy as np
        r = float(np.corrcoef(x, y)[0, 1]) if x.size and y.size else np.nan
        print(f"[flow_vs_matrix_scatter] r={r:.3f}  →  {png_corr} , {png_scat}")
    except Exception:
        pass


import numpy as np


def viz_flow_risk_correlation_panel(
    M, adj_static, node_names, output_dir,
    title="Flow ↔ Risk",
    symmetrize="sum",
    flow_gamma=1.0,
    flow_min_q=None,
    top_k=20,
    tag=""
):
    """
    Scatter off-diagonal flow vs. off-diagonal transmission risk with a NaN-safe r.
    Saves PNG + TXT; returns r (float).
    """
    import os, numpy as np, matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    # flow prep
    A = np.array(adj_static, dtype=np.float64)
    if symmetrize == "sum":   A = A + A.T
    elif symmetrize == "mean": A = 0.5 * (A + A.T)
    elif symmetrize == "max": A = np.maximum(A, A.T)
    if flow_gamma and flow_gamma != 1.0:
        A = np.power(np.clip(A, 0, None), float(flow_gamma))
    if flow_min_q is not None:
        pos = A[A > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        A = np.where(A >= thr, A, 0.0)

    # off-diagonal vectors
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    x = A[mask].ravel()
    y = np.array(M, dtype=np.float64)[mask].ravel()

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # NaN-safe Pearson
    sx = float(np.nanstd(x)); sy = float(np.nanstd(y))
    if x.size < 3 or y.size < 3 or sx < 1e-12 or sy < 1e-12:
        r = 0.0
    else:
        xz = (x - x.mean()) / sx
        yz = (y - y.mean()) / sy
        r = float(np.corrcoef(xz, yz)[0, 1])

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    ax.scatter(x, y, s=6, alpha=0.5)
    ax.set_xlabel("Flow (off-diagonal)")
    ax.set_ylabel("Transmission risk (off-diagonal)")
    ax.set_title(f"{title}\n r = {r:.3f}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"panel_flow_vs_risk{tag}.png")
    txt_path = os.path.join(output_dir, f"summary_flow_vs_risk{tag}.txt")
    fig.savefig(png_path); plt.close(fig)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"title: {title}\n")
        f.write(f"r: {r:.6f}\n")
        f.write(f"n_points: {x.size}\n")
        f.write(f"x_std: {sx:.6e} | y_std: {sy:.6e}\n")

    print(f"[viz_flow_risk_correlation_panel] → {png_path} (r = {r:.3f})")
    print(f"[viz_flow_risk_correlation_panel] → {txt_path}")
    return r



# === Add below to Model_DCRNN_Viz.py =========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the existing heatmap helper when available; otherwise define a basic fallback.
def _simple_heatmap(mat, names, title, outdir, skip_self=False):
    try:
        import seaborn as sns, matplotlib.colors as mcolors
        H = mat.copy()
        if skip_self:
            np.fill_diagonal(H, 0.0)
        plt.figure(figsize=(22, 18))
        ax = sns.heatmap(
            H, xticklabels=names, yticklabels=names,
            cmap=mcolors.LinearSegmentedColormap.from_list("green_red", ["green", "yellow", "red"]),
            annot=False, linewidths=0.2, linecolor="white", cbar_kws={"label": "Weight"}
        )
        plt.title(title); plt.xticks(rotation=90); plt.yticks(rotation=0)
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, f"{title.replace(' ','_').lower()}.png")
        plt.tight_layout(); plt.savefig(outp, dpi=170); plt.close()
        print(f"[heatmap] → {outp}")
    except Exception as e:
        print("[heatmap] skipped:", e)

def flow_reference_from_adj(
        adj_static: np.ndarray,
        node_names,
        output_dir: str,
        symmetrize: str = "mean",  # {"mean","sum","max"}
        flow_gamma: float = 1.0,   # >1 emphasizes strong links
        flow_min_q: float = None,  # prune weakest edges by global quantile (e.g., 0.20)
        normalize: bool = True,
        make_degree_hist: bool = True,
        tag: str = "",
) -> np.ndarray:
    """Build a symmetrized/pruned/power-transformed flow reference. Save CSV + heatmap (+ degree hist)."""
    A = np.array(adj_static, dtype=np.float32)
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)

    if flow_min_q is not None:
        pos = F[F > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)

    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))
    if normalize:
        m = F.max()
        if m > 0:
            F = F / m

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"flow_reference_matrix_{symmetrize}{tag}.csv")
    pd.DataFrame(F, index=node_names, columns=node_names).to_csv(csv_path)

    # heatmap
    try:
        _simple_heatmap(F, node_names, f"Flow_reference_{symmetrize}{tag}", output_dir, skip_self=False)
    except Exception:
        pass

    # degree histogram
    if make_degree_hist:
        indeg = (F > 0).sum(axis=0)
        outdeg = (F > 0).sum(axis=1)
        plt.figure(figsize=(7, 4))
        plt.hist(outdeg, bins=20, alpha=0.6, label="Out-degree")
        plt.hist(indeg, bins=20, alpha=0.6, label="In-degree")
        plt.xlabel("Non-zero neighbors"); plt.ylabel("Nodes")
        plt.title(f"Flow degree histogram ({symmetrize})")
        plt.legend()
        png = os.path.join(output_dir, f"flow_reference_degree_hist_{symmetrize}{tag}.png")
        plt.tight_layout(); plt.savefig(png, dpi=160); plt.close()
    return F

def viz_flow_risk_correlation_panel(
    M: np.ndarray,
    adj_static: np.ndarray,
    node_names,
    output_dir: str,
    title: str = "Flow vs Risk: correlation panel",
    symmetrize: str = "mean",      # {"mean","sum","max"}
    flow_gamma: float = 1.0,       # >1 emphasizes strong links
    flow_min_q: float = None,      # e.g., 0.20 to prune weakest edges
    tag: str = "",
    top_k: int = 50,
    dpi: int = 170,
):
    """
    Build symmetrized/normalized flow from adj, compare to M (risk), and save a 1×2 panel + top-edge CSVs.
    Returns a tiny dict with r, nz_frac, paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    node_names = list(node_names)

    # symmetrized reference flow
    A = np.array(adj_static, dtype=np.float32)
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)
    if flow_min_q is not None:
        pos = F[F > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)
    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))
    m = F.max()
    if m > 0:
        F = F / m

    # align shapes and flatten off-diagonals
    M = np.array(M, dtype=np.float32)
    if M.shape != F.shape:
        raise ValueError(f"[viz_flow_risk_correlation_panel] shape mismatch: risk {M.shape} vs flow {F.shape}")
    n = M.shape[0]
    triu = np.triu_indices(n, k=1)
    x = F[triu]; y = M[triu]

    # corr
    r = float(np.corrcoef(x, y)[0, 1]) if x.size and y.size else np.nan

    # 1×2 panel
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), dpi=dpi)
    hb = axes[0].hexbin(x, y, gridsize=35, mincnt=1)
    cb = fig.colorbar(hb, ax=axes[0]); cb.set_label("count")
    axes[0].set_xlabel("Flow weight (sym, [0,1])"); axes[0].set_ylabel("Risk weight")
    axes[0].set_title(f"Hexbin (r = {r:.3f})")
    axes[1].scatter(x, y, s=8, alpha=0.35)
    axes[1].set_xlabel("Flow weight (sym, [0,1])"); axes[1].set_ylabel("Risk weight"); axes[1].set_title("Scatter")
    fig.suptitle(title); fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    png = os.path.join(output_dir, f"panel_flow_vs_risk{tag}.png")
    fig.savefig(png, dpi=dpi); plt.close(fig)
    print(f"[viz_flow_risk_correlation_panel] → {png} (r = {r:.3f})")

    # top-edges CSVs
    src = np.repeat(node_names, n); dst = np.tile(node_names, n)
    df_edges = pd.DataFrame({"src": src, "dst": dst, "risk": M.flatten(), "flow": F.flatten()})
    df_edges = df_edges[df_edges["src"] != df_edges["dst"]]
    path_top_risk = os.path.join(output_dir, f"top_edges_by_risk{tag}.csv")
    df_edges.sort_values("risk", ascending=False).head(top_k).to_csv(path_top_risk, index=False)
    path_top_flow = os.path.join(output_dir, f"top_edges_by_flow{tag}.csv")
    df_edges.sort_values("flow", ascending=False).head(top_k).to_csv(path_top_flow, index=False)

    # summary
    nz_frac = float((M > 0).sum()) / float(M.size) if M.size else np.nan
    summary_txt = os.path.join(output_dir, f"summary_flow_vs_risk{tag}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Flow vs Risk: Correlation Panel ===\n")
        f.write(f"Nodes: {n}\n")
        f.write(f"Corr(offdiag): {r:.4f}\n")
        f.write(f"Non-zero fraction (risk): {nz_frac:.4f}\n")
        f.write(f"Panel PNG: {os.path.basename(png)}\n")
        f.write(f"Top edges (risk): {os.path.basename(path_top_risk)}\n")
        f.write(f"Top edges (flow): {os.path.basename(path_top_flow)}\n")
    print(f"[viz_flow_risk_correlation_panel] → {summary_txt}")
    return {"r": r, "nz_frac": nz_frac, "panel_png": png,
            "top_edges_by_risk": path_top_risk, "top_edges_by_flow": path_top_flow,
            "summary_path": summary_txt}

def flow_vs_matrix_scatter(
        M: np.ndarray,
        adj_static: np.ndarray,
        output_dir: str,
        node_names,
        title: str = "Flow vs Matrix",
        symmetrize: str = "mean",
        flow_gamma: float = 1.0,
        flow_min_q: float = None,
        tag: str = "",
):
    """Hexbin + scatter comparing symmetrized flow vs arbitrary matrix M. Saves two PNGs; prints r."""
    A = np.array(adj_static, dtype=np.float32)
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)
    if flow_min_q is not None:
        pos = F[F > 0]; thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)
    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))
    m = F.max()
    if m > 0:
        F = F / m

    n = F.shape[0]; triu = np.triu_indices(n, k=1)
    x = F[triu]; y = np.array(M, dtype=np.float32)[triu]

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6.2, 5))
    hb = plt.hexbin(x, y, gridsize=35, mincnt=1)
    plt.colorbar(hb, label="count")
    plt.xlabel("Flow weight (sym, [0,1])"); plt.ylabel("Matrix weight"); plt.title(title)
    plt.tight_layout(); corr_png = os.path.join(output_dir, f"verify_corr_flow_vs_matrix{tag}.png")
    plt.savefig(corr_png, dpi=160); plt.close()

    plt.figure(figsize=(6.2, 5))
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.xlabel("Flow weight (sym, [0,1])"); plt.ylabel("Matrix weight"); plt.title(title)
    plt.tight_layout(); scat_png = os.path.join(output_dir, f"verify_scatter_flow_vs_matrix{tag}.png")
    plt.savefig(scat_png, dpi=160); plt.close()

    try:
        r = float(np.corrcoef(x, y)[0, 1]) if x.size and y.size else np.nan
        print(f"[flow_vs_matrix_scatter] r={r:.3f}  →  {corr_png} , {scat_png}")
    except Exception:
        pass

def diagnostic_verify_mobility_regulation_viz(
        out_dir,
        tag="",
        flow_csv=None,           # square; rows/cols = node_names
        pred_matrix_csv=None,    # regulated matrix CSV
        unreg_matrix_csv=None,   # optional unregulated CSV
        clin_proxy_csv=None,     # optional clinical-only proxy CSV
        clinxmob_proxy_csv=None, # optional clin×mob proxy CSV
        top_k=50,
        dpi=160,
        title_prefix="Verification",
):
    """
    Stand-alone verification: compares symmetrized flow vs regulated and (optional) unregulated matrices.
    Saves scatter/hexbin PNGs, top-edges CSVs, optional proxy-diff heatmap, and a TXT summary.
    Returns a small dict with key metrics and summary path.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _load_sq(path, lbl):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{lbl} not found: {path}")
        df = pd.read_csv(path, index_col=0)
        if list(df.index) != list(df.columns):
            raise ValueError(f"{lbl} must be square with matching row/col labels.")
        return df

    def _sym(A):
        F = 0.5 * (A + A.T); np.fill_diagonal(F, 0.0)
        m = F.max();  return F/(m + 1e-12) if m > 0 else F

    def _off(M):
        n = M.shape[0]; triu = np.triu_indices(n, k=1); return M[triu]

    def _scatter(x, y, title, outpng):
        plt.figure(figsize=(6.2, 5))
        plt.scatter(x, y, s=8, alpha=0.4)
        plt.xlabel("Flow weight (sym, [0,1])"); plt.ylabel("Risk weight"); plt.title(title)
        plt.tight_layout(); plt.savefig(outpng, dpi=dpi); plt.close()

    def _hex(x, y, title, outpng):
        plt.figure(figsize=(6.2, 5))
        hb = plt.hexbin(x, y, gridsize=35, mincnt=1); plt.colorbar(hb, label="count")
        plt.xlabel("Flow weight (sym, [0,1])"); plt.ylabel("Risk weight"); plt.title(title)
        plt.tight_layout(); plt.savefig(outpng, dpi=dpi); plt.close()

    df_flow = _load_sq(flow_csv, "flow_csv")
    df_reg  = _load_sq(pred_matrix_csv, "pred_matrix_csv")

    common = [n for n in df_reg.index if n in df_flow.index]
    if not common:
        raise ValueError("No overlapping node names between flow_csv and pred_matrix_csv.")
    df_flow = df_flow.loc[common, common]
    df_reg  = df_reg.loc[common, common]
    nodes = list(common)

    flow = _sym(df_flow.values.astype(np.float32))
    risk_reg = df_reg.values.astype(np.float32)

    risk_unreg = None
    if unreg_matrix_csv and os.path.exists(unreg_matrix_csv):
        df_un = _load_sq(unreg_matrix_csv, "unreg_matrix_csv").loc[common, common]
        risk_unreg = df_un.values.astype(np.float32)

    x = _off(flow); y_reg = _off(risk_reg)
    r_reg = float(np.corrcoef(x, y_reg)[0, 1]) if x.size and y_reg.size else np.nan
    png_corr_reg = os.path.join(out_dir, f"verify_corr_flow_vs_risk_reg{tag}.png")
    _hex(x, y_reg, f"{title_prefix}: Flow vs Regulated Risk (r={r_reg:.3f})", png_corr_reg)
    png_scatter_reg = os.path.join(out_dir, f"verify_scatter_flow_vs_risk_reg{tag}.png")
    _scatter(x, y_reg, f"{title_prefix}: Flow vs Regulated Risk (r={r_reg:.3f})", png_scatter_reg)

    r_unreg = np.nan; png_corr_unreg = png_scatter_unreg = None
    if risk_unreg is not None:
        y_unreg = _off(risk_unreg)
        r_unreg = float(np.corrcoef(x, y_unreg)[0, 1]) if x.size and y_unreg.size else np.nan
        png_corr_unreg = os.path.join(out_dir, f"verify_corr_flow_vs_risk_unreg{tag}.png")
        _hex(x, y_unreg, f"{title_prefix}: Flow vs Unregulated Risk (r={r_unreg:.3f})", png_corr_unreg)
        png_scatter_unreg = os.path.join(out_dir, f"verify_scatter_flow_vs_risk_unreg{tag}.png")
        _scatter(x, y_unreg, f"{title_prefix}: Flow vs Unregulated Risk (r={r_unreg:.3f})", png_scatter_unreg)

    # top-edges CSVs
    def _edges_df(M, label):
        src = np.repeat(nodes, len(nodes)); dst = np.tile(nodes, len(nodes))
        df = pd.DataFrame({"src": src, "dst": dst, label: M.flatten(), "flow": flow.flatten()})
        return df[df["src"] != df["dst"]]

    df_reg_edges = _edges_df(risk_reg, "risk_reg")
    df_reg_edges.sort_values("risk_reg", ascending=False).head(top_k).to_csv(
        os.path.join(out_dir, f"verify_top_edges_by_risk_reg{tag}.csv"), index=False
    )
    df_reg_edges.sort_values("flow", ascending=False).head(top_k).to_csv(
        os.path.join(out_dir, f"verify_top_edges_by_flow{tag}.csv"), index=False
    )
    if risk_unreg is not None:
        df_un_edges = _edges_df(risk_unreg, "risk_unreg")
        df_un_edges.sort_values("risk_unreg", ascending=False).head(top_k).to_csv(
            os.path.join(out_dir, f"verify_top_edges_by_risk_unreg{tag}.csv"), index=False
        )

    # optional proxy diff heatmap
    proxy_diff_png = None; corr_proxy = np.nan
    if clin_proxy_csv and clinxmob_proxy_csv and os.path.exists(clin_proxy_csv) and os.path.exists(clinxmob_proxy_csv):
        df_pc = _load_sq(clin_proxy_csv, "clin_proxy_csv").loc[common, common]
        df_pm = _load_sq(clinxmob_proxy_csv, "clinxmob_proxy_csv").loc[common, common]
        Pc = df_pc.values.astype(np.float32); Pm = df_pm.values.astype(np.float32)
        off_pc = _off(Pc); off_pm = _off(Pm)
        corr_proxy = float(np.corrcoef(off_pc, off_pm)[0, 1]) if off_pc.size else np.nan
        proxy_diff_png = os.path.join(out_dir, f"verify_proxy_diff_heatmap{tag}.png")
        plt.figure(figsize=(6.2, 5))
        plt.imshow(Pm - Pc, cmap="bwr"); plt.colorbar(label="Mobility-weighted – Clinical-only")
        plt.title(f"{title_prefix}: Proxy (Clin×Mob – Clin)"); plt.tight_layout()
        plt.savefig(proxy_diff_png, dpi=dpi); plt.close()

    # summary
    def _nz_frac(M):
        return float((M > 0).sum()) / float(M.size) if (M is not None and M.size) else np.nan
    frac_reg = _nz_frac(risk_reg); frac_unreg = _nz_frac(risk_unreg)

    summary_txt = os.path.join(out_dir, f"verify_summary_stats{tag}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Mobility-Regulated Transmission: Verification (viz) ===\n")
        f.write(f"Nodes: {len(nodes)}\n")
        f.write(f"Corr(flow, risk_reg offdiag): {r_reg:.4f}\n")
        if risk_unreg is not None:
            f.write(f"Corr(flow, risk_unreg offdiag): {r_unreg:.4f}\n")
        f.write(f"Non-zero fraction (risk_reg): {frac_reg:.4f}\n")
        if risk_unreg is not None:
            f.write(f"Non-zero fraction (risk_unreg): {frac_unreg:.4f}\n")
        if proxy_diff_png:
            f.write(f"Corr(proxy_clin, proxy_clin×mob offdiag): {corr_proxy:.4f}\n")
            f.write(f"Proxy diff heatmap: {os.path.basename(proxy_diff_png)}\n")
        f.write("Scatter & hexbin plots saved. Top-edges CSVs saved.\n")
    print(f"[Saved] {summary_txt}")

    return {
        "corr_flow_vs_risk_reg": r_reg,
        "corr_flow_vs_risk_unreg": r_unreg,
        "nz_frac_reg": frac_reg,
        "nz_frac_unreg": frac_unreg,
        "summary_path": summary_txt,
    }

def plot_wwtp_only_heatmap(
        risk_matrix: np.ndarray,
        names,
        out_dir: str,
        tag: str = "",
        title: str = "Transmission (WWTP ↔ WWTP, self excluded)",
        figsize=(12, 10)
):
    """Heatmap for WWTP×WWTP submatrix with diagonal zeroed (no self)."""
    import seaborn as sns
    import numpy as np
    idx = [i for i, n in enumerate(names) if str(n).startswith("W_")]
    if not idx:
        print("[plot_wwtp_only_heatmap] No WWTP nodes found."); return
    M = np.array(risk_matrix, dtype=np.float32)[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, 0.0)
    ww_names = [names[i] for i in idx]
    plt.figure(figsize=figsize)
    ax = sns.heatmap(M, xticklabels=ww_names, yticklabels=ww_names,
                     cmap="viridis", annot=False, linewidths=0.2, linecolor="white",
                     cbar_kws={"label": "Normalized Influence"})
    plt.title(title); plt.xticks(rotation=90); plt.yticks(rotation=0)
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"wwtp_only_heatmap{tag}.png")
    plt.tight_layout(); plt.savefig(outp, dpi=180); plt.close()
    print(f"[plot_wwtp_only_heatmap] → {outp}")
# Flow-versus-risk correlation visualization

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flow_vs_risk_corr_panel(
    M: np.ndarray,
    adj_static: np.ndarray,
    node_names,
    output_dir: str,
    title: str = "Flow ↔ Risk: correlation panel",
    symmetrize: str = "mean",      # {"mean","sum","max"}
    flow_gamma: float = 1.0,       # >1 emphasizes strong links
    flow_min_q: float = None,      # e.g., 0.20 to prune weakest edges globally
    tag: str = "",
    top_k: int = 50,
    dpi: int = 170,
):
    """
    Build a symmetrized/normalized reference flow from `adj_static`, compare it to `M` (risk),
    and save a 1×2 panel (hexbin + scatter) alongside top-edge CSVs.

    Outputs in `output_dir`:
      - panel_flow_vs_risk{tag}.png
      - top_edges_by_risk{tag}.csv
      - top_edges_by_flow{tag}.csv
      - summary_flow_vs_risk{tag}.txt

    Returns:
      {
        "r": float,            # Pearson corr on off-diagonals (flow vs M)
        "nz_frac": float,      # fraction of non-zeros in M
        "panel_png": str,      # path to PNG
        "top_edges_by_risk": str,
        "top_edges_by_flow": str,
        "summary_path": str
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    node_names = list(node_names)

    # --- symmetrize + prune + power-transform + normalize flow reference ---
    A = np.array(adj_static, dtype=np.float32)
    if symmetrize == "sum":
        F = A + A.T
    elif symmetrize == "max":
        F = np.maximum(A, A.T)
    else:  # "mean"
        F = 0.5 * (A + A.T)
    np.fill_diagonal(F, 0.0)

    if flow_min_q is not None:
        pos = F[F > 0]
        thr = np.quantile(pos, float(flow_min_q)) if pos.size else 0.0
        F = np.where(F >= thr, F, 0.0)

    if flow_gamma != 1.0:
        F = np.power(F, float(flow_gamma))

    m = F.max()
    if m > 0:
        F = F / m

    # --- align/validate and flatten off-diagonals ---
    M = np.array(M, dtype=np.float32)
    if M.shape != F.shape:
        raise ValueError(f"[flow_vs_risk_corr_panel] Shape mismatch: risk {M.shape} vs flow {F.shape}")
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    x = F[iu]
    y = M[iu]

    # --- correlation ---
    r = float(np.corrcoef(x, y)[0, 1]) if x.size and y.size else np.nan

    # --- 1×2 panel ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), dpi=dpi)

    hb = axes[0].hexbin(x, y, gridsize=35, mincnt=1)
    cb = fig.colorbar(hb, ax=axes[0]); cb.set_label("count")
    axes[0].set_xlabel("Flow weight (sym, [0,1])")
    axes[0].set_ylabel("Risk weight")
    axes[0].set_title(f"Hexbin (r = {r:.3f})")

    axes[1].scatter(x, y, s=8, alpha=0.35)
    axes[1].set_xlabel("Flow weight (sym, [0,1])")
    axes[1].set_ylabel("Risk weight")
    axes[1].set_title("Scatter")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    png = os.path.join(output_dir, f"panel_flow_vs_risk{tag}.png")
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"[flow_vs_risk_corr_panel] → {png} (r = {r:.3f})")

    # --- top-edges CSVs ---
    src = np.repeat(node_names, n)
    dst = np.tile(node_names, n)
    df_edges = pd.DataFrame({"src": src, "dst": dst, "risk": M.flatten(), "flow": F.flatten()})
    df_edges = df_edges[df_edges["src"] != df_edges["dst"]].copy()

    path_top_risk = os.path.join(output_dir, f"top_edges_by_risk{tag}.csv")
    df_edges.sort_values("risk", ascending=False).head(top_k).to_csv(path_top_risk, index=False)
    path_top_flow = os.path.join(output_dir, f"top_edges_by_flow{tag}.csv")
    df_edges.sort_values("flow", ascending=False).head(top_k).to_csv(path_top_flow, index=False)

    # --- summary TXT ---
    nz_frac = float((M > 0).sum()) / float(M.size) if M.size else np.nan
    summary_txt = os.path.join(output_dir, f"summary_flow_vs_risk{tag}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Flow vs Risk: Correlation Panel ===\n")
        f.write(f"Nodes: {n}\n")
        f.write(f"Corr(offdiag): {r:.4f}\n")
        f.write(f"Non-zero fraction (risk): {nz_frac:.4f}\n")
        f.write(f"Panel PNG: {os.path.basename(png)}\n")
        f.write(f"Top edges (risk): {os.path.basename(path_top_risk)}\n")
        f.write(f"Top edges (flow): {os.path.basename(path_top_flow)}\n")
    print(f"[flow_vs_risk_corr_panel] → {summary_txt}")

    return {
        "r": r,
        "nz_frac": nz_frac,
        "panel_png": png,
        "top_edges_by_risk": path_top_risk,
        "top_edges_by_flow": path_top_flow,
        "summary_path": summary_txt,
    }

#
# # === quick standalone plotting run ===
# import geopandas as gpd, pandas as pd
# import Model_DCRNN_Viz as viz
#
# # Load previously generated results.
# # sc00_bfu_tp00_a03_pc1_vn1_nm1_arn1_sl4_hd128_l1_do0_l2100_cos0_ucm_seed42
# df_tx   = pd.read_csv("run_COVID/sentinel_RAW_TX_sc00_bfu_tp00_a03_pc1_vn1_nm1_arn1_sl4_hd128_l1_do0_l2100_cos0_UCM_seed42_table.csv")     # adjust path
# df_tx_map = df_tx.rename(columns={"score": "score"})
# wwtp_gdf    = load_wwtp_points("../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp")
#
# # --- optional boundary and sewershed shapefiles ---
# county_polys_gdf = gpd.read_file("../ZoneSelection/Input/Census/COCounty.shp")
# # sewershed_gdf    = gpd.read_file("../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp")
#
# # --- re-plot without rerunning model ---
# viz.plot_top_sites_map(
#     df_top=df_tx_map,
#     wwtp_gdf=wwtp_gdf,
#     title="Top WWTP Sentinels (RAW TX)",
#     out_dir="outputs/",
#     tag="_RAW_TX",
#     annotate=True,
#     county_polys_gdf=county_polys_gdf,
#     sewershed_gdf=None,       # skip if not aligned
#     use_basemap=True          # set False for a plain background
# )
#
#
# viz.plot_top_sites_map_html(
#     df_top=df_tx_map,
#     wwtp_gdf=wwtp_gdf,
#     out_dir="run_COVID",
#     tag="_RAW_TX",
#     county_polys_gdf=county_polys_gdf,
# )
