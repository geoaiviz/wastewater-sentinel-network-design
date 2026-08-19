"""Aggregate licensed daily origin-destination records to weekly summaries.

Weekly values use the mean of available daily records, matching the reported
workflow. Files that cannot be parsed are reported and skipped.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

def aggregate_od_weekly(daily_dir, output_dir):
    """Aggregate daily OD records by sewershed, county, direction, and week."""
    import glob
    import numpy as np

    all_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    weekly_bins = {}
    for f in all_files:
        try:
            date_str = os.path.basename(f).split("_")[-1].replace(".csv", "")
            date = datetime.strptime(date_str, "%Y%m%d")
            week_start = date - timedelta(days=date.weekday())
            key = week_start.strftime("%Y-%m-%d")
            weekly_bins.setdefault(key, []).append(f)
        except Exception as e:
            print(f" Skipping file {f}: {e}")
            continue
    for week, file_list in weekly_bins.items():
        dfs = []
        for file in file_list:
            try:
                df = pd.read_csv(file)

                dfs.append(df)
            except Exception as e:
                print(f"️ Skipping record from {file}: {e}")
                continue

        if dfs:
            week_df = pd.concat(dfs, ignore_index=True)

            origin_df = week_df[week_df["direction"] == "Origin"].copy()
            origin_agg = (
                origin_df.groupby(["Sewershed-O", "County"], as_index=False)[["Volume","Area", "Population"]]
                .mean()
                .assign(week=week, direction="Origin")
            )
            origin_agg.to_csv(os.path.join(output_dir, f"weekly_o_{week}.csv"), index=False)
            dest_df = week_df[week_df["direction"] == "Destination"].copy()
            dest_agg = (
                dest_df.groupby(["Sewershed-D", "County"], as_index=False)[["Volume","Area", "Population"]]
                .mean()
                .assign(week=week, direction="Destination")
            )

            dest_agg.to_csv(os.path.join(output_dir, f"weekly_d_{week}.csv"), index=False)
        print(f" Saved weekly OD for {week}")

# === FUNCTION: Aggregate COVID Cases Weekly ===
def aggregate_covid_cases(daily_fp, output_fp):
    df = pd.read_excel(daily_fp)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"County": "LABEL"})
    df["LABEL"] = df["LABEL"].str.strip().str.lower()
    df["week"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek, unit='d')
    weekly = df.groupby(["LABEL", "week"], as_index=False)["County_cases_3dayavg_r100Kutil"].mean()
    weekly.to_csv(output_fp, index=False)
    print(" Saved weekly COVID case data")


import geopandas as gpd
import pandas as pd

def population_outside_wwtp(block_group_boundary_fp, sewershed_boundary_fp, output_csv):
    """
    Calculate population living outside WWTP boundaries, grouped by county.
    Saves CSV with: FIPS, COUNTYNAME, POP
    """
    # Load shapefiles
    block_groups = gpd.read_file(block_group_boundary_fp)
    sewersheds = gpd.read_file(sewershed_boundary_fp)

    # Reproject for accurate area calculations
    target_crs = "EPSG:26913"
    block_groups = block_groups.to_crs(target_crs)
    sewersheds = sewersheds.to_crs(target_crs)

    # Compute BG area
    block_groups["bg_area"] = block_groups.geometry.area

    # Intersect BG with WWTP boundaries
    bg_sewershed_intersection = gpd.overlay(block_groups, sewersheds, how="intersection")
    bg_sewershed_intersection["intersect_area"] = bg_sewershed_intersection.geometry.area

    #  Remove bg_area from intersection before merging
    if "bg_area" in bg_sewershed_intersection.columns:
        bg_sewershed_intersection = bg_sewershed_intersection.drop(columns=["bg_area"])

    # Merge population and bg_area from block_groups
    bg_sewershed_intersection = bg_sewershed_intersection.merge(
        block_groups[["FIPS", "bg_area", "POPULATION"]],
        on="FIPS",
        how="left"
    )

    # Calculate proportion covered
    bg_sewershed_intersection["percent_covered"] = (
        bg_sewershed_intersection["intersect_area"] / bg_sewershed_intersection["bg_area"]
    )

    # Sum coverage per BG (handles multiple overlaps)
    coverage_per_bg = (
        bg_sewershed_intersection.groupby("FIPS")["percent_covered"].sum().reset_index()
    )
    coverage_per_bg["percent_covered"] = coverage_per_bg["percent_covered"].clip(upper=1.0)

    # Merge back to all BGs, fill missing with 0 for no-intersection BGs
    block_groups = block_groups.merge(coverage_per_bg, on="FIPS", how="left").fillna({"percent_covered": 0})

    # Calculate outside population
    block_groups["outside_pop"] = block_groups["POPULATION"] * (1 - block_groups["percent_covered"])

    # Group by county FIPS
    result = block_groups.groupby("STCOFIPS")["outside_pop"].sum().reset_index()

    # Initialize the county-name field for an optional later spatial join.
    result["COUNTYNAME"] = ""

    # Format and save
    result = result.rename(columns={"STCOFIPS": "FIPS", "outside_pop": "POP"})
    result["POP"] = result["POP"].round(0).astype(int)
    result[["FIPS", "COUNTYNAME", "POP"]].to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


def run_aggr():

    # === CONFIGURATION ===
    daily_dir = "../ZoneSelection/Input/ODData/Daily"
    # covid_case_daily_fp = "../ZoneSelection/Input/Viral/WW_clinicaldata_COVID_DU.xlsx"
    # covid_case_weekly_fp = "../ZoneSelection/Outfile/ODData/Weekly/weekly_covid_cases.csv"
    weekly_od_dir = "../ZoneSelection/Outfile/ODData/Weekly"
    os.makedirs(weekly_od_dir, exist_ok=True)

    aggregate_od_weekly(daily_dir, weekly_od_dir)
    # aggregate_covid_cases(covid_case_daily_fp, covid_case_weekly_fp)

    # population_outside_wwtp(
    #     block_group_boundary_fp="../ZoneSelection/Input/Census/COBlockGroup.shp",
    #     sewershed_boundary_fp="../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp",
    #     output_csv="../ZoneSelection/Input/Census/county_uncovered_pop.csv"
    # )

if __name__ == "__main__":
    run_aggr()

# run_aggr()

# This module intentionally reports weekly OD summaries only. Population not
# represented by the authorized OD product is evaluated in the Level 1 stage.
