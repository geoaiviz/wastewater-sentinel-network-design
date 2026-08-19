"""Select the best Level 2 contribution output for Level 1 comparison.

Rows are filtered by disease and mode, ranked by validation metrics, and
resolved with an explicit time tie-break.
"""

from __future__ import annotations

import os
import glob
from typing import Optional, Literal
from datetime import datetime

import pandas as pd


def _find_latest_grid_results_csv(default_dir: str) -> Optional[str]:
    """Find newest grid_results*.csv in the folder."""
    cand = glob.glob(os.path.join(default_dir, "grid_results*.csv"))
    if not cand:
        return None
    cand.sort(key=os.path.getmtime, reverse=True)
    return cand[0]


def _pick_best_row_from_grid(
    grid_csv: str,
    disease: Optional[str] = None,
    mode: Optional[str] = None,
    primary: Literal["rmse", "mae", "corr"] = "rmse",
    secondary: Optional[Literal["rmse", "mae", "corr"]] = "mae",
    tie_break: Literal["time_newest", "time_oldest"] = "time_newest",
) -> pd.Series:
    """Pick best row from a grid_results CSV using metric rules."""
    df = pd.read_csv(grid_csv).copy()

    # optional filters
    if disease is not None and "disease" in df.columns:
        df = df[df["disease"].astype(str).str.upper() == disease.upper()]
    if mode is not None and "mode" in df.columns:
        df = df[df["mode"].astype(str) == mode]

    # parse time for deterministic tie-break
    if "time" in df.columns:
        df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time_dt"] = pd.NaT

    # ensure primary exists
    if primary not in df.columns:
        raise ValueError(f"Primary metric '{primary}' not found in {grid_csv}. "
                         f"Available: {list(df.columns)}")

    df = df.dropna(subset=[primary]).copy()
    if df.empty:
        raise ValueError(f"No valid rows after filtering/dropping NaNs in '{primary}'.")

    # sort logic: rmse/mae lower is better; corr higher is better
    sort_cols = [primary]
    asc = [False] if primary == "corr" else [True]

    if secondary and secondary in df.columns and secondary != primary:
        sort_cols.append(secondary)
        asc.append(False if secondary == "corr" else True)

    sort_cols.append("time_dt")
    asc.append(False if tie_break == "time_newest" else True)

    df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    return df.iloc[0]


def glob_best_txraw_csv(
    default_dir: str = "./run_COVID",
    disease: Optional[str] = "COVID",
    mode: Optional[str] = None,
    primary: Literal["rmse", "mae", "corr"] = "rmse",
    secondary: Optional[Literal["rmse", "mae", "corr"]] = "mae",
    tie_break: Literal["time_newest", "time_oldest"] = "time_newest",
    verbose: bool = True,
) -> Optional[str]:
    """
    Pick the 'best' TX RAW table CSV based on the BEST row in grid_results*.csv.

    Steps:
      1) find newest grid_results*.csv in default_dir
      2) pick best row by (primary, secondary, time tie-break)
      3) use that row's 'tag' to find sentinel_RAW_TX_{tag}_table*.csv
      4) if multiple matches, choose newest by mtime

    Returns: file path or None.
    """

    grid_csv = _find_latest_grid_results_csv(default_dir)
    if grid_csv is None:
        if verbose:
            print(f"[Picker] No grid_results*.csv found in: {default_dir}")
        return None

    best = _pick_best_row_from_grid(
        grid_csv=grid_csv,
        disease=disease,
        mode=mode,
        primary=primary,
        secondary=secondary,
        tie_break=tie_break,
    )

    tag = str(best.get("tag", "")).strip()
    if not tag:
        if verbose:
            print(f"[Picker] Best row in {grid_csv} has empty 'tag'.")
        return None

    # Look for TX raw table generated for that tag
    pats = [
        os.path.join(default_dir, f"sentinel_RAW_TX_{tag}_table.csv"),
        os.path.join(default_dir, f"sentinel_RAW_TX_{tag}_table*.csv"),
        # fallback if files embed tag with extra tokens
        os.path.join(default_dir, f"sentinel_RAW_TX_*{tag}*_table*.csv"),
    ]

    cands: list[str] = []
    for p in pats:
        cands.extend(glob.glob(p))

    # de-dup while preserving order
    seen = set()
    cands = [x for x in cands if not (x in seen or seen.add(x))]

    if not cands:
        if verbose:
            print(f"[Picker] Found grid file: {grid_csv}")
            print(f"[Picker] Best tag: {tag}")
            print("[Picker] BUT no matching TX raw table CSV found for that tag.")
            print("[Picker] Searched patterns:")
            for p in pats:
                print("  -", p)
        return None

    # pick newest matching output by mtime
    cands.sort(key=os.path.getmtime, reverse=True)
    chosen = cands[0]

    if verbose:
        def _fmt_time(p: str) -> str:
            return datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")

        print(f"[Picker] Using grid file: {grid_csv}")
        print(f"[Picker] Best row: tag={tag} | {primary}={best.get(primary)}"
              + (f" | {secondary}={best.get(secondary)}" if secondary else "")
              + (f" | time={best.get('time')}" if 'time' in best else ""))

        print("\n[Picker] Matching TX raw candidates:")
        for f in cands[:10]:
            print(f"  - {os.path.basename(f)} | mtime={_fmt_time(f)}")

        print(f"\n[Picker] Selected TX raw CSV: {chosen}")

    return chosen
