"""
AnalysisViz_CumulativeCoverage.py

Helpers to:
  - build a cumulative coverage table (sorted by final rank)
  - export CSV
  - generate cumulative and marginal-gain plots

Assumes:
  features: DataFrame indexed by WWTP name, columns may include:
      'pop_served', 'pop_covered_by_od', 'od_volume_total', 'area_reached',
      optional: 'risk_score'  (or other scalar risk metric per site)
  avg_rank: Series indexed by same WWTP names, lower = better rank
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Set Times New Roman as the default font family
plt.rcParams["font.family"] = "Arial"


# Ensure weight and style are set to normal
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.style"] = "normal"

# Which metrics we treat as "benefits"
# Only metrics that actually exist in features will be used.
CUM_METRICS: Dict[str, str] = {
    "pop_served": "Population Served",
    "pop_covered_by_od": "OD Population Reached",
    "od_volume_total": "Commute Volume (Trips)",
    "area_reached": "Area Reached",
    # optional: site-level risk score (e.g., combined import/export/speed_norm)
    "risk_score": "Risk Score",
}

COLOR_MAP = {
    "pop_served": "#1f77b4",        # blue
    "pop_reached": "#ff7f0e",       # orange
    "od_volume": "#9467bd",         # purple
    "area_reached": "#2ca02c",      # green
}


def build_cumulative_table(features: pd.DataFrame,
                           avg_rank: pd.Series) -> pd.DataFrame:
    """
    Sort sites by final rank and compute:
      - per-site contribution (delta)
      - cumulative sum
      - cumulative fraction of total
    Returns a tidy table with row per site in greedy order.
    """
    # Align indices (defensive)
    # Keep the full ranking list/order (so cumulative CSV list == ranking list)
    rank_idx = avg_rank.index.astype(str).str.strip().str.lower()
    feat = features.copy()
    feat.index = feat.index.astype(str).str.strip().str.lower()

    df = feat.reindex(rank_idx).copy()  # <- preserves rank order universe
    df = df.fillna(0.0)  # <- missing features become 0 contribution
    df["final_rank"] = avg_rank.reindex(rank_idx).values

    df = df.sort_values("final_rank", kind="mergesort")  # stable sort
    df = df.reset_index().rename(columns={"index": "wwtp"})

    # k_sites: 1, 2, 3, ... as we add sites
    df["k_sites"] = np.arange(1, len(df) + 1)

    # ensure we have numeric versions of any cumulative metrics we will use
    # Use a local copy so we don't mutate the global metric dict across calls
    metrics = {k: v for k, v in CUM_METRICS.items() if k in df.columns}

    # ensure numeric
    for col in metrics.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # compute delta, cumulative and fractional cumulative for each metric present
    for col in metrics.keys():
        vals = df[col].astype(float).values
        cum = np.cumsum(vals)
        total = cum[-1] if len(cum) > 0 else 0.0

        df[f"{col}_delta"] = vals
        df[f"{col}_cum"] = cum
        df[f"{col}_cum_frac"] = cum / total if total > 0 else np.nan

    return df


def plot_cumulative_curves(cum_df: pd.DataFrame,
                           out_path: str,
                           title_suffix: str = "") -> None:
    """
    Line plot: cumulative benefits vs number of sites included (absolute values).

    X axis: site count (1, 2, 3, ...) – can also be read as "new site added in order".

    Left Y axis:
        - Population Served (cumulative)
        - OD Population Reached (cumulative)
        - Commute Volume (cumulative)

    Right Y axis:
        - Area Reached (cumulative)

    Risk is better interpreted in fraction space, so it is not plotted here;
    see plot_cumulative_fractions().
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = cum_df["k_sites"].values

    fig, ax_left = plt.subplots(figsize=(9, 6))

    # ---- Left axis: population & volume metrics ----
    lines_left = []
    labels_left = []

    if "pop_served_cum" in cum_df.columns:
        l, = ax_left.plot(
            x,
            cum_df["pop_served_cum"].values,
            marker="o",
            label=CUM_METRICS.get("pop_served", "Population Served"),
        )
        lines_left.append(l)
        labels_left.append(l.get_label())

    if "pop_covered_by_od_cum" in cum_df.columns:
        l, = ax_left.plot(
            x,
            cum_df["pop_covered_by_od_cum"].values,
            marker="o",
            label=CUM_METRICS.get("pop_covered_by_od", "OD Population Reached"),
        )
        lines_left.append(l)
        labels_left.append(l.get_label())

    if "od_volume_total_cum" in cum_df.columns:
        l, = ax_left.plot(
            x,
            cum_df["od_volume_total_cum"].values,
            marker="o",
            label=CUM_METRICS.get("od_volume_total", "Commute Volume"),
        )
        lines_left.append(l)
        labels_left.append(l.get_label())

    ax_left.set_xlabel("Number of Top-Ranked Sites Included)")
    ax_left.set_ylabel("Population / Commute Volume (cumulative)")
    title = "Cumulative Coverage Benefits (Absolute)"
    # if title_suffix:
    #     title += f" — {title_suffix}"
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.3)

    # ---- Right axis: area metric ----
    lines_right = []
    labels_right = []
    if "area_reached_cum" in cum_df.columns:
        ax_right = ax_left.twinx()
        l_area, = ax_right.plot(
            x,
            cum_df["area_reached_cum"].values,
            marker="o",
            color="green",
            label=CUM_METRICS.get("area_reached", "Area Reached"),
        )
        ax_right.set_ylabel("Area Reached (cumulative)")
        lines_right.append(l_area)
        labels_right.append(l_area.get_label())
    else:
        ax_right = None

    # ---- Combine legends ----
    all_lines = lines_left + lines_right
    all_labels = labels_left + labels_right
    if all_lines:
        ax_left.legend(all_lines, all_labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Cumulative] Absolute-value plot saved: {out_path}")


def plot_cumulative_fractions(cum_df: pd.DataFrame,
                              out_path: str,
                              title_suffix: str = "") -> None:
    """
    Line plot: cumulative FRACTIONS (0–1) for all metrics, including risk.

    This shows, for example, what fraction of:
      - total population,
      - total OD population,
      - total commute volume,
      - total area,
      - total risk_score

    is captured as we add the top-k ranked sites.
    All lines share a common 0–1 scale, so shapes are directly comparable.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = cum_df["k_sites"].values

    plt.figure(figsize=(9, 6))

    for col, pretty in CUM_METRICS.items():
        frac_col = f"{col}_cum_frac"
        if frac_col not in cum_df.columns:
            continue
        y = cum_df[frac_col].values
        # Risk often benefits from a dashed line to visually distinguish it
        if col == "risk_score":
            plt.plot(
                x,
                y,
                marker="o",
                linestyle="--",
                label=f"{pretty} (fraction)",
            )
        else:
            plt.plot(
                x,
                y,
                marker="o",
                label=f"{pretty} (fraction)",
            )

    plt.xlabel("Number of Top-Ranked Sites Included")
    plt.ylabel("Cumulative Fraction of Total (0–1)")
    title = "Cumulative Coverage Fractions"
    # if title_suffix:
    #     title += f" — {title_suffix}"
    plt.title(title)

    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Cumulative] Fraction plot saved: {out_path}")


def plot_marginal_gains(cum_df: pd.DataFrame,
                        out_path: str,
                        title_suffix: str = "") -> None:
    """
    Bar plot: marginal (delta) gain per additional site
    for a subset of key metrics.

    X axis: site count / new site added (also labeled with WWTP name if not too many).
    Y axis: delta value contributed by that site.

    We keep this focused on three interpretable metrics:
      - Δ Population Served
      - Δ OD Population
      - Δ Area Reached
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = cum_df["k_sites"].values

    plt.figure(figsize=(10, 6))
    width = 0.25

    if "pop_served_delta" in cum_df.columns:
        plt.bar(
            x - width,
            cum_df["pop_served_delta"].values,
            width=width,
            label="Δ Population Served",
        )

    if "pop_covered_by_od_delta" in cum_df.columns:
        plt.bar(
            x,
            cum_df["pop_covered_by_od_delta"].values,
            width=width,
            label="Δ OD Population",
        )

    if "area_reached_delta" in cum_df.columns:
        plt.bar(
            x + width,
            cum_df["area_reached_delta"].values,
            width=width,
            label="Δ Area Reached",
        )

    plt.xlabel("New Site Added")
    plt.ylabel("Incremental Gain per Site")
    title = "Marginal Gains by Additional Site"
    # if title_suffix:
    #     title += f" — {title_suffix}"
    plt.title(title)

    # Label x ticks with site names if manageable
    if len(cum_df) <= 30:
        plt.xticks(
            ticks=x,
            labels=cum_df["wwtp"].astype(str).tolist(),
            rotation=60,
            ha="right",
        )
    else:
        plt.xticks(ticks=x)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Cumulative] Marginal gains plot saved: {out_path}")


def save_cumulative_outputs(features: pd.DataFrame,
                            avg_rank: pd.Series,
                            out_dir: str,
                            label: str = "all") -> None:
    """
    Convenience wrapper:
      1. Build cumulative table
      2. Save CSV
      3. Save cumulative curves plot (absolute values with dual y-axis)
      4. Save cumulative FRACTION plot (0–1, including risk if present)
      5. Save marginal-gain plot
    """
    os.makedirs(out_dir, exist_ok=True)

    cum_df = build_cumulative_table(features, avg_rank)

    # 1) CSV
    csv_path = os.path.join(out_dir, f"cumulative_benefits_{label}.csv")
    cum_df.to_csv(csv_path, index=False)
    print(f"[Cumulative] CSV saved: {csv_path}")

    # 2) cumulative curves (absolute)
    png_cum = os.path.join(out_dir, f"cumulative_benefits_{label}.png")
    plot_cumulative_curves(cum_df, png_cum, title_suffix=label)

    # 3) cumulative fractions (0–1, includes risk if provided)
    png_frac = os.path.join(out_dir, f"cumulative_benefits_fraction_{label}.png")
    plot_cumulative_fractions(cum_df, png_frac, title_suffix=label)

    # 4) marginal gains
    png_delta = os.path.join(out_dir, f"marginal_gains_{label}.png")
    plot_marginal_gains(cum_df, png_delta, title_suffix=label)

# ============================================================
# V2: BG-level unique cumulative coverage using BG_Link outputs
# ============================================================

import glob
import re
from typing import Literal, Optional


def _list_weekly_bg_link_files(bg_link_dir: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None):
    """Return sorted list of weekly_bg_wwtp_YYYY-MM-DD.csv paths within [start_date, end_date]."""
    if not os.path.isdir(bg_link_dir):
        return []
    files = glob.glob(os.path.join(bg_link_dir, "weekly_bg_wwtp_*.csv"))
    if not files:
        return []

    def _date_from_path(p: str):
        m = re.search(r"weekly_bg_wwtp_(\d{4}-\d{2}-\d{2})\.csv$", os.path.basename(p))
        return pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT

    sdt = pd.to_datetime(start_date) if start_date else None
    edt = pd.to_datetime(end_date) if end_date else None

    keep = []
    for p in files:
        d = _date_from_path(p)
        if pd.isna(d):
            continue
        if sdt is not None and d < sdt:
            continue
        if edt is not None and d > edt:
            continue
        keep.append((d, p))

    keep.sort(key=lambda x: x[0])
    return [p for _, p in keep]


def load_bg_link_window(bg_link_dir: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        direction: Literal["Destination", "Origin"] = "Destination") -> pd.DataFrame:
    """
    Load BG_Link weekly files in the time window.

    Expected columns (from Process_ODData.py):
      - week, direction, bg_fips, wwtp_clean, Volume, Area, Population
    """
    files = _list_weekly_bg_link_files(bg_link_dir, start_date, end_date)
    if not files:
        return pd.DataFrame()

    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p, dtype={"bg_fips": str, "wwtp_clean": str})
        except Exception:
            continue

        if "direction" in df.columns:
            df = df[df["direction"].astype(str).str.strip().str.lower() == direction.lower()]

        # Keep only columns we use
        keep_cols = [c for c in ["week", "bg_fips", "wwtp_clean", "Volume", "Area", "Population"] if c in df.columns]
        df = df[keep_cols].copy()
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out["bg_fips"] = out["bg_fips"].astype(str).str.strip()
    out["wwtp_clean"] = out["wwtp_clean"].astype(str).str.strip().str.lower()
    for c in ["Volume", "Area", "Population"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_unique_cumulative_table_bg(
    avg_rank: pd.Series,
    bg_link_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    direction: Literal["Destination", "Origin"] = "Destination",
    # For BG-level "no double count" metrics (pop/area reached)
    union_mode: Literal["cap_sum", "binary", "prob_union"] = "cap_sum",
    weight_from: Literal["Volume", "Area", "Population"] = "Volume",
    min_weight: float = 0.0,
    # Definition of BG-to-site weight used by the BG-union operator
    #   - "share"         : original V1 (within-BG share of OD weight) -> answers "share of BG trips captured"
    #   - "intensity_cap" : trips-per-capita with hard cap at 1
    #   - "intensity_exp" : trips-per-capita with exponential saturation (recommended)
    bg_weight_mode: Literal["share", "intensity_cap", "intensity_exp"] = "share",
    tau: float = 1.0,
    eps_pop: float = 100.0,
    # Optional: resident population served (sewershed population), additive because sewersheds do not overlap
    pop_served: Optional[pd.Series] = None,
        pop_served_total_override: Optional[float] = None,

) -> pd.DataFrame:
    """
    Portfolio cumulative table in ranked order (avg_rank), combining:

    1) Additive (no-overlap) site metric:
       - pop_served: resident population served by each WWTP sewershed.
         Since WWTP sewersheds are disjoint, this is summed directly across sites.

    2) BG-level "no double count" metrics (portfolio / joint coverage):
       - pop_reached_unique: BG population reached with BG-level union.
       - area_reached_unique: BG area reached with BG-level union.

       We maintain a BG coverage state covered(bg) in [0,1] and update it as sites are added:
         binary:     covered = max(covered, wi>0)
         cap_sum:    covered = min(1, covered + wi)
         prob_union: covered = 1 - (1-covered)*(1-wi)

       The key design choice is how to define wi for a (site, BG) pair:

         A) bg_weight_mode="share"  (ORIGINAL V1)
            wi = w(site,bg) / sum_over_sites w(site,bg)
            where w is aggregated over the window from `weight_from` (default Volume).
            Interpretation: for each BG, wi is the share of that BG’s total OD weight going to the site.
            This answers: "What percentage of each BG’s trips are captured by selected sites?"

         B) bg_weight_mode="intensity_cap" (trips-per-capita, hard cap)
            wi = min(1, w(site,bg) / (tau * (Pop_bg + eps_pop)) )

         C) bg_weight_mode="intensity_exp" (trips-per-capita, exponential saturation; recommended)
            wi = 1 - exp( - w(site,bg) / (tau * (Pop_bg + eps_pop)) )

            Interpretation: wi reflects absolute mobility intensity relative to BG population,
            so small numbers of trips do NOT imply full BG coverage.

    3) OD volume totals (site-based, no BG union):
       - od_volume_total: sum of OD Volume linked to each WWTP in the window.
         This is treated as a site-total (WWTP-specific) quantity and summed directly across sites.

    Notes:
      - BG totals for Population/Area are computed as max over rows for each BG in the window.
      - Fractions for unique pop/area are normalized to the OD-connected BG universe
        present in the BG_Link table, not the full state.
    """
    links = load_bg_link_window(bg_link_dir, start_date, end_date, direction=direction)
    if links.empty:
        return pd.DataFrame()

    # Optional filter to suppress tiny/noisy rows (applied to the union weight column)
    if weight_from in links.columns and float(min_weight) > 0:
        links = links[links[weight_from] > float(min_weight)].copy()
        if links.empty:
            return pd.DataFrame()

    # BG masses (one value per BG) for unique accounting
    bg_pop = links.groupby("bg_fips")["Population"].max() if "Population" in links.columns else pd.Series(dtype=float)
    bg_area = links.groupby("bg_fips")["Area"].max() if "Area" in links.columns else pd.Series(dtype=float)
    # ---------------- QA: Compare BG_Link universe vs full BG universe ----------------
    try:
        # bg_info exists in Process_ODData script, not here,
        # so we compare only against what appears in BG_Link files.

        n_bg_link = len(bg_pop)
        total_bg_link_pop = float(bg_pop.sum())

        print("\n[QA] BG_Link Universe Check")
        print(f"[QA] Number of BGs in BG_Link window: {n_bg_link}")
        print(f"[QA] Total BG population in BG_Link window: {total_bg_link_pop:,.0f}")

    except Exception as e:
        print("[QA] BG universe check failed:", e)

    # WWTP order from ranking (lower rank = better)
    order = (
        avg_rank.dropna()
        .sort_values()
        .index.astype(str).str.strip().str.lower()
        .tolist()
    )

    # ----- (A) Site-based OD Volume totals (no BG union) -----
    if "Volume" in links.columns:
        vol_by_wwtp = links.groupby("wwtp_clean")["Volume"].sum()
    else:
        vol_by_wwtp = pd.Series(dtype=float)

    # ----- (B) Build WWTP->BG union weights wi -----
    if weight_from not in links.columns:
        raise ValueError(f"weight_from='{weight_from}' not found in BG link data columns: {list(links.columns)}")

    # Aggregate over weeks: sum weights per (wwtp,bg)
    w_df = (
        links.groupby(["wwtp_clean", "bg_fips"], as_index=False)[weight_from]
        .sum()
        .rename(columns={weight_from: "w"})
    )

    # Compute wi depending on union_mode and bg_weight_mode
    if union_mode == "binary":
        w_df["wi"] = 1.0
    else:
        mode = str(bg_weight_mode).strip().lower()
        if mode == "share":
            # Original V1: within-BG share of OD weight
            bg_total_w = w_df.groupby("bg_fips")["w"].sum().rename("bg_total_w")
            w_df = w_df.merge(bg_total_w, on="bg_fips", how="left")
            w_df["wi"] = np.where(w_df["bg_total_w"] > 0, w_df["w"] / w_df["bg_total_w"], 0.0)

        elif mode in ("intensity_cap", "intensity_exp"):
            if bg_pop.empty:
                raise ValueError("bg_weight_mode requires BG population ('Population' column) in BG_Link data, but it was not found.")
            if tau <= 0:
                raise ValueError("tau must be > 0 for intensity-based BG weights.")
            # Attach BG population to each (wwtp,bg) row
            pop_df = bg_pop.rename("bg_pop").reset_index().rename(columns={"bg_fips": "bg_fips"})
            w_df = w_df.merge(pop_df, on="bg_fips", how="left")
            w_df["bg_pop"] = pd.to_numeric(w_df["bg_pop"], errors="coerce").fillna(0.0)

            denom = tau * (w_df["bg_pop"].values + float(eps_pop))
            denom = np.where(denom > 0, denom, np.nan)
            x = w_df["w"].values / denom

            if mode == "intensity_cap":
                w_df["wi"] = np.minimum(1.0, np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0))
            else:
                # exponential saturation
                w_df["wi"] = 1.0 - np.exp(-np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

            # ensure in [0,1]
            w_df["wi"] = pd.to_numeric(w_df["wi"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

        else:
            raise ValueError(f"Unknown bg_weight_mode='{bg_weight_mode}' (use 'share', 'intensity_cap', 'intensity_exp').")

    # Map: wwtp -> {bg: wi}
    wwtp_to_bgwi = {}
    for wwtp, sub in w_df.groupby("wwtp_clean"):
        wwtp_to_bgwi[wwtp] = dict(zip(sub["bg_fips"].astype(str), sub["wi"].astype(float)))

    # ----- (C) pop_served (optional, additive) -----
    pop_served_series = None
    if pop_served is not None:
        pop_served_series = pop_served.copy()
        pop_served_series.index = pop_served_series.index.astype(str).str.strip().str.lower()
        pop_served_series = pd.to_numeric(pop_served_series, errors="coerce").fillna(0.0)

    # Coverage state over BGs for unique metrics
    covered = pd.Series(0.0, index=bg_pop.index.union(bg_area.index), dtype=float)

    pop_unique_cum = 0.0
    area_unique_cum = 0.0
    pop_served_cum = 0.0
    volume_total_cum = 0.0

    rows = []

    for k, wwtp in enumerate(order, start=1):
        # ---- pop_served delta (additive) ----
        pop_served_delta = float(pop_served_series.get(wwtp, 0.0)) if pop_served_series is not None else 0.0
        pop_served_cum += pop_served_delta

        # ---- volume_total delta (additive site-total) ----
        volume_total_delta = float(vol_by_wwtp.get(wwtp, 0.0)) if not vol_by_wwtp.empty else 0.0
        volume_total_cum += volume_total_delta

        # ---- BG-union update for unique pop/area reached ----
        bgwi = wwtp_to_bgwi.get(wwtp, {})
        if bgwi:
            covered_prev = covered.copy()

            bgs = list(bgwi.keys())
            wi = pd.Series(bgwi, dtype=float).reindex(bgs).fillna(0.0)

            if union_mode == "binary":
                covered.loc[bgs] = np.maximum(covered.loc[bgs].values, (wi.values > 0).astype(float))
            elif union_mode == "cap_sum":
                covered.loc[bgs] = np.minimum(1.0, covered.loc[bgs].values + wi.values)
            else:  # prob_union
                covered.loc[bgs] = 1.0 - (1.0 - covered.loc[bgs].values) * (1.0 - wi.values)

            delta_cov = (covered - covered_prev).clip(lower=0.0)

            pop_unique_delta = float((bg_pop.reindex(delta_cov.index).fillna(0.0) * delta_cov).sum()) if not bg_pop.empty else 0.0
            area_unique_delta = float((bg_area.reindex(delta_cov.index).fillna(0.0) * delta_cov).sum()) if not bg_area.empty else 0.0

            pop_unique_cum += pop_unique_delta
            area_unique_cum += area_unique_delta
        else:
            pop_unique_delta = 0.0
            area_unique_delta = 0.0

        rows.append({
            "wwtp_clean": wwtp,
            "k_sites": k,

            # Additive, no-overlap metric
            "pop_served_delta": pop_served_delta,
            "pop_served_cum": pop_served_cum,

            # BG-union metrics (no double count across sites)
            "pop_reached_unique_delta": pop_unique_delta,
            "pop_reached_unique_cum": pop_unique_cum,
            "area_reached_unique_delta": area_unique_delta,
            "area_reached_unique_cum": area_unique_cum,

            # Site-based OD volume totals (no BG union)
            "od_volume_total_delta": volume_total_delta,
            "od_volume_total_cum": volume_total_cum,
        })

    out = pd.DataFrame(rows)

    # ---- Fractions ----
    # ---- Fractions ----
    if pop_served_series is not None:

        if pop_served_total_override is not None:
            pop_served_total = float(pop_served_total_override)
        else:
            # default behavior (denominator = whatever is in `order`)
            pop_served_total = float(pop_served_series.reindex(order).fillna(0.0).sum())

        out["pop_served_cum_frac"] = (
            out["pop_served_cum"] / pop_served_total
            if pop_served_total > 0 else np.nan
        )
    else:
        out["pop_served_cum_frac"] = np.nan


    pop_total_bg = float(bg_pop.sum()) if not bg_pop.empty else np.nan
    area_total_bg = float(bg_area.sum()) if not bg_area.empty else np.nan
    vol_total_all = float(vol_by_wwtp.sum()) if not vol_by_wwtp.empty else np.nan

    out["pop_reached_unique_cum_frac"] = out["pop_reached_unique_cum"] / pop_total_bg if pop_total_bg and pop_total_bg > 0 else np.nan
    out["area_reached_unique_cum_frac"] = out["area_reached_unique_cum"] / area_total_bg if area_total_bg and area_total_bg > 0 else np.nan
    out["od_volume_total_cum_frac"] = out["od_volume_total_cum"] / vol_total_all if vol_total_all and vol_total_all > 0 else np.nan

    # Metadata
    out["union_mode"] = union_mode
    out["weight_from"] = weight_from
    out["bg_weight_mode"] = bg_weight_mode
    out["tau"] = tau
    out["eps_pop"] = eps_pop
    out["start_date"] = start_date
    out["end_date"] = end_date
    out["direction"] = direction

    return out


def _plot_bg_unique_cumulative(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    """
    Absolute cumulative plot for V2 portfolio metrics:
      - pop_served_cum (additive)
      - pop_reached_unique_cum (BG union)
      - od_volume_total_cum (additive site-total)
      - area_reached_unique_cum (BG union) on right axis
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = df["k_sites"].values

    fig, ax_left = plt.subplots(figsize=(9, 6))

    lines_left, labels_left = [], []

    if "pop_served_cum" in df.columns and df["pop_served_cum"].notna().any():
        l, = ax_left.plot(
            x,
            df["pop_served_cum"].values,
            marker="o",
            color=COLOR_MAP["pop_served"],
            label="Population Served (cumulative)",
        )
        lines_left.append(l); labels_left.append(l.get_label())

    if "pop_reached_unique_cum" in df.columns:
        l, = ax_left.plot(
            x,
            df["pop_reached_unique_cum"].values,
            marker="o",
            color=COLOR_MAP["pop_reached"],
            label="Population Reached (unique, BG union)",
        )
        lines_left.append(l); labels_left.append(l.get_label())

    if "od_volume_total_cum" in df.columns:
        l, = ax_left.plot(
            x,
            df["od_volume_total_cum"].values,
            marker="o",
            color=COLOR_MAP["od_volume"],
            label="OD Volume Total (cumulative)",
        )
        lines_left.append(l); labels_left.append(l.get_label())

    ax_left.set_xlabel("Number of Top-Ranked Sites Included")
    ax_left.set_ylabel("Population / OD Volume (cumulative)")
    ax_left.grid(True, alpha=0.3)

    ax_right = ax_left.twinx()
    lines_right, labels_right = [], []
    if "area_reached_unique_cum" in df.columns:
        l2, = ax_right.plot(
            x,
            df["area_reached_unique_cum"].values,
            marker="o",
            color=COLOR_MAP["area_reached"],
            label="Area Reached (unique, BG union)",
        )
        lines_right.append(l2); labels_right.append(l2.get_label())
        ax_right.set_ylabel("Area Reached (cumulative)")

    title = "Cumulative Benefits (V2: Additive + BG-Union)"
    if title_suffix:
        title += f" — {title_suffix}"
    ax_left.set_title(title)

    all_lines = lines_left + lines_right
    all_labels = labels_left + labels_right
    if all_lines:
        ax_left.legend(all_lines, all_labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_bg_unique_fractions(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    """Fraction plot (0–1). All included lines are expected to reach 1.0 when all sites are used."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = df["k_sites"].values

    plt.figure(figsize=(7, 4))

    plt.plot(x, df["pop_served_cum_frac"],linewidth=1.5,marker ='o', markersize=2,
             color=COLOR_MAP["pop_served"],
             label="Population Served")

    plt.plot(x, df["pop_reached_unique_cum_frac"],linewidth=1.5,marker ='o', markersize=2,
             color=COLOR_MAP["pop_reached"],
             label="Population Reached")

    plt.plot(x, df["area_reached_unique_cum_frac"], linewidth=1.5,marker ='o',markersize=2,
             color=COLOR_MAP["area_reached"],
             label="Area Reached")

    plt.plot(x, df["od_volume_total_cum_frac"],linewidth=1.5,marker ='o', markersize=2,
             color=COLOR_MAP["od_volume"],
             label="Mobility Trip")

    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Number of Top-Ranked Sites Included",  fontsize=12)
    plt.ylabel("Cumulative Fraction of Total (0–1)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    title = "Cumulative Fractions (V2)"
    if title_suffix:
        title += f" — {title_suffix}"
    # plt.title(title)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_bg_unique_marginal(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    """Marginal gains (delta) per added site for the V2 metrics."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = df["k_sites"].values

    plt.figure(figsize=(11, 6))
    width = 0.2

    # Order bars left-to-right per k
    if "pop_served_delta" in df.columns and df["pop_served_delta"].notna().any():
        plt.bar(x - 1.5 * width, df["pop_served_delta"].values, width=width, label="Δ Pop Served")

    if "pop_reached_unique_delta" in df.columns:
        plt.bar(x - 0.5 * width, df["pop_reached_unique_delta"].values, width=width, label="Δ Pop Reached (unique)")

    if "area_reached_unique_delta" in df.columns:
        plt.bar(x + 0.5 * width, df["area_reached_unique_delta"].values, width=width, label="Δ Area Reached (unique)")

    if "od_volume_total_delta" in df.columns:
        plt.bar(x + 1.5 * width, df["od_volume_total_delta"].values, width=width, label="Δ OD Volume Total")

    plt.xlabel("New Site Added")
    plt.ylabel("Incremental Gain per Site")
    title = "Marginal Gains (V2)"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title)

    if len(df) <= 30:
        plt.xticks(ticks=x, labels=df["wwtp_clean"].astype(str).tolist(), rotation=60, ha="right")
    else:
        plt.xticks(ticks=x)

    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def print_portfolio_summary_at_k(df: pd.DataFrame,
                                 k_values=(20, 40, 60)) -> None:
    """
    Print cumulative totals and fractions at selected portfolio sizes,
    and also print overall totals for reference.
    """

    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY (SELECTED K + TOTALS)")
    print("=" * 80)

    # ----- Overall totals (denominators) -----
    total_pop_served = df["pop_served_cum"].iloc[-1] if "pop_served_cum" in df.columns else np.nan
    total_od_volume = df["od_volume_total_cum"].iloc[-1] if "od_volume_total_cum" in df.columns else np.nan

    print("\nOVERALL TOTALS (All Ranked Sites)")
    print("-" * 80)
    print(f"Total Population Served     : {total_pop_served:,.0f}")
    print(f"Total OD Volume (Trips)     : {total_od_volume:,.0f}")

    # ----- Selected K values -----
    rows = []

    for k in k_values:
        sub = df[df["k_sites"] == k]
        if sub.empty:
            continue

        row = sub.iloc[0]

        rows.append({
            "k_sites": k,
            "pop_served_cum": row.get("pop_served_cum", 0),
            "pop_served_frac": row.get("pop_served_cum_frac", np.nan),
            "od_volume_total_cum": row.get("od_volume_total_cum", 0),
            "od_volume_frac": row.get("od_volume_total_cum_frac", np.nan),
        })

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        print("\nNo matching k values found.")
        return

    pd.options.display.float_format = "{:,.4f}".format

    print("\nCUMULATIVE VALUES AT SELECTED K")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print("=" * 80 + "\n")

def save_bg_unique_outputs(
    avg_rank: pd.Series,
    bg_link_dir: str,
    start_date: Optional[str],
    end_date: Optional[str],
    out_dir: str,
    label: str,
    direction: Literal["Destination", "Origin"] = "Destination",
    union_mode: Literal["cap_sum", "binary", "prob_union"] = "cap_sum",
    weight_from: Literal["Volume", "Area", "Population"] = "Volume",
    min_weight: float = 0.0,

    # BG-union weighting options
    bg_weight_mode: Literal["share", "intensity_cap", "intensity_exp"] = "share",
    tau: float = 1.0,
    eps_pop: float = 100.0,

    pop_served: Optional[pd.Series] = None,
        pop_served_total_override: Optional[float] = None,

) -> None:

    """
    V2 portfolio cumulative outputs (BG-link based):

    - pop_served: additive across sites (assumes sewersheds do not overlap).
    - pop_reached_unique / area_reached_unique: BG-level union to remove double counting across sites.
    - od_volume_total: additive site-total (no BG union), so cumulative fraction reaches 100% when all sites are included.

    Writes:
      - bg_portfolio_cumulative_{label}.csv
      - bg_portfolio_cumulative_{label}.png
      - bg_portfolio_cumulative_fraction_{label}.png
      - bg_portfolio_marginal_gains_{label}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    df = build_unique_cumulative_table_bg(
        avg_rank=avg_rank,
        bg_link_dir=bg_link_dir,
        start_date=start_date,
        end_date=end_date,
        direction=direction,
        union_mode=union_mode,
        weight_from=weight_from,
        min_weight=min_weight,

        bg_weight_mode=bg_weight_mode,
        tau=tau,
        eps_pop=eps_pop,

        pop_served=pop_served,
        pop_served_total_override=pop_served_total_override,

    )

    if df.empty:
        print(f"[V2 Portfolio] No BG links found for window {start_date} to {end_date}.")
        return

    csv_path = os.path.join(out_dir, f"bg_portfolio_cumulative_{label}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[V2 Portfolio] CSV saved: {csv_path}")
    print_portfolio_summary_at_k(df, k_values=(20, 40, 60))

    _plot_bg_unique_cumulative(df, os.path.join(out_dir, f"bg_portfolio_cumulative_{label}.png"), title_suffix=label)
    _plot_bg_unique_fractions(df, os.path.join(out_dir, f"bg_portfolio_cumulative_fraction_{label}.png"), title_suffix=label)
    _plot_bg_unique_marginal(df, os.path.join(out_dir, f"bg_portfolio_marginal_gains_{label}.png"), title_suffix=label)


# ============================================================
# Selected-site portfolio summary (one-row CSV)
# ============================================================

from typing import Sequence, Union

def save_selected_sites_summary(
    selected_ordered: Sequence[Union[str, int]],
    features: pd.DataFrame,
    out_dir: str,
    label: str = "selected",
    include_fraction_of_all: bool = True,
) -> str:
    """
    Write a one-row CSV summarizing TOTAL coverage for a pre-selected site set.

    This is intentionally separate from the global cumulative CSV, because the
    cumulative workflow is based on the full ranked universe (all sites), while
    `selected_ordered` may be a trimmed/filled subset (e.g., top-N after subnetwork logic).

    Parameters
    ----------
    selected_ordered
        List of selected WWTP names (any casing). Order is not used for totals.
    features
        DataFrame indexed by WWTP name/ID (typically wwtp_clean). Expected columns (if available):
          - pop_served
          - od_volume_total
          - pop_covered_by_od
          - area_reached
          - risk_score (optional)
    out_dir
        Output folder.
    label
        File label suffix.
    include_fraction_of_all
        If True, also write fraction-of-all totals for each metric.

    Returns
    -------
    csv_path : str
        Path to the written CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Canonicalize keys for robust matching
    feat = features.copy()
    feat.index = feat.index.astype(str).str.strip().str.lower()

    sel = [str(s).strip().lower() for s in list(selected_ordered)]
    sel = [s for s in sel if s]  # drop empty
    sel_unique = list(dict.fromkeys(sel))  # stable unique

    matched = [s for s in sel_unique if s in feat.index]
    missing = [s for s in sel_unique if s not in feat.index]

    sub = feat.reindex(matched).fillna(0.0)

    metric_cols = [c for c in ["pop_served", "od_volume_total", "pop_covered_by_od", "area_reached", "risk_score"] if c in sub.columns]

    totals = {f"{c}_total": float(pd.to_numeric(sub[c], errors="coerce").fillna(0.0).sum()) for c in metric_cols}

    out = {
        "label": label,
        "n_selected_input": len(sel),
        "n_selected_unique": len(sel_unique),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "missing_sites": ";".join(missing[:200]) if missing else "",
    }
    out.update(totals)

    if include_fraction_of_all and len(metric_cols) > 0:
        all_sum = {c: float(pd.to_numeric(feat[c], errors="coerce").fillna(0.0).sum()) for c in metric_cols if c in feat.columns}
        for c in metric_cols:
            denom = all_sum.get(c, 0.0)
            out[f"{c}_fraction_of_all"] = (totals.get(f"{c}_total", 0.0) / denom) if denom > 0 else np.nan

    df_out = pd.DataFrame([out])

    csv_path = os.path.join(out_dir, f"selected_sites_summary_{label}.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[Selected Summary] CSV saved: {csv_path}")

    return csv_path
