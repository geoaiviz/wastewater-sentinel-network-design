"""Convert ACS ZIP/ZCTA commute times to approximate one-way distances."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPEED_MPH = {"Urban": 30.0, "Suburban": 40.0, "Rural": 55.0}


def classify_area(ruca_code: float) -> str:
    """Group RUCA primary codes into the three travel-speed categories."""
    if ruca_code == 1:
        return "Urban"
    if ruca_code in (2, 4):
        return "Suburban"
    return "Rural"


def derive_commute_distance(
    commute_time_path: Path,
    ruca_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Join ACS commute times to RUCA codes and calculate distance estimates."""
    commute = pd.read_csv(commute_time_path)
    if "mean_commute_minutes" not in commute.columns:
        raise ValueError("Commute-time input must contain 'mean_commute_minutes'.")
    if "ZCTA5" not in commute.columns:
        if "GEO_ID" not in commute.columns:
            raise ValueError("Commute-time input must contain 'ZCTA5' or 'GEO_ID'.")
        commute["ZCTA5"] = commute["GEO_ID"].astype(str).str.extract(r"(\d{5})")
    commute["ZCTA5"] = commute["ZCTA5"].astype(str).str.zfill(5)

    ruca = pd.read_excel(ruca_path, sheet_name="Data", dtype={"ZIP_CODE": str})
    ruca = ruca[["ZIP_CODE", "RUCA1"]].rename(columns={"ZIP_CODE": "ZCTA5"})
    ruca["ZCTA5"] = ruca["ZCTA5"].astype(str).str.zfill(5)
    ruca["area_type"] = ruca["RUCA1"].apply(classify_area)

    result = commute.merge(ruca, on="ZCTA5", how="left")
    result["average_speed_mph"] = result["area_type"].map(SPEED_MPH).fillna(30.0)
    result["commute_distance_miles"] = (
        pd.to_numeric(result["mean_commute_minutes"], errors="coerce")
        / 60.0
        * result["average_speed_mph"]
    )
    result["commute_distance_km"] = result["commute_distance_miles"] * 1.60934

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACS ZIP/ZCTA commute times to approximate distances."
    )
    parser.add_argument("--commute-time", required=True, type=Path)
    parser.add_argument("--ruca", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = derive_commute_distance(args.commute_time, args.ruca, args.output)
    print(f"Saved {len(result):,} commute-distance records to {args.output}")


if __name__ == "__main__":
    main()
