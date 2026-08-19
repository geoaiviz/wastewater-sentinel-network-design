"""
Process utility_county_adj.csv with ArcPy.

Method
------
1. Project WWTP sewersheds, Census population polygons, and counties to EPSG:5070.
2. Intersect sewersheds with Census population polygons.
3. Allocate each source polygon's population by overlap-area fraction.
4. Aggregate allocated population by Utility x CountyFIPS.
5. Calculate PctOfUtility and PctOfCounty and attach county_name.

Recommended population geography: Census blocks (most accurate), otherwise block groups.
The population field should contain total resident population for each source polygon.

Example (one line)
------------------
python Process_UtilityCountyAdj_ArcPy.py --utilities "../ZoneSelection/Input/WWTP_WI/WWTP_Select.shp" --population "../ZoneSelection/Input/Census/WI_blocks_2020.shp" --counties "../ZoneSelection/Input/Census/WI_County.shp" --output "../ZoneSelection/Input/WWTP_WI/utility_county_adj.csv" --utility-field wwtp --population-field P0010001
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import arcpy
except ImportError as exc:
    raise SystemExit(
        "ArcPy is required. Run this script from the ArcGIS Pro Python environment."
    ) from exc


TARGET_WKID = 5070  # NAD83 / Conus Albers; area-preserving for the contiguous U.S.

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION — edit these paths once, then run the script directly.
# Colorado is the default test case. Command-line arguments remain optional and
# can override any value below.
# -----------------------------------------------------------------------------
DEFAULT_UTILITIES = r"..\ZoneSelection\Input\WWTP_CO\WWTP_Select.shp"
DEFAULT_POPULATION = r"..\ZoneSelection\Input\Census\COBlockGroup.shp"
DEFAULT_COUNTIES = r"..\ZoneSelection\Input\Census\COCounty.shp"
DEFAULT_OUTPUT = r"..\ZoneSelection\Input\WWTP_CO\utility_county_adj_test.csv"
DEFAULT_UTILITY_FIELD = "wwtp"
DEFAULT_POPULATION_FIELD = "POPULATION"
DEFAULT_MIN_PIECE_AREA_M2 = 1.0



def _field_names(dataset: str) -> dict[str, str]:
    return {f.name.lower(): f.name for f in arcpy.ListFields(dataset)}


def _resolve_field(dataset: str, requested: str | None, candidates: list[str]) -> str:
    fields = _field_names(dataset)
    if requested:
        hit = fields.get(requested.lower())
        if hit:
            return hit
        raise ValueError(f"Field '{requested}' not found in {dataset}. Available: {list(fields.values())}")
    for candidate in candidates:
        hit = fields.get(candidate.lower())
        if hit:
            return hit
    raise ValueError(f"None of {candidates} found in {dataset}. Available: {list(fields.values())}")


def _ensure_polygon(dataset: str, label: str) -> None:
    desc = arcpy.Describe(dataset)
    if getattr(desc, "shapeType", "").lower() != "polygon":
        raise ValueError(f"{label} must be a polygon feature class: {dataset}")


def _project(in_fc: str, out_fc: str) -> str:
    sr = arcpy.SpatialReference(TARGET_WKID)
    desc = arcpy.Describe(in_fc)
    if desc.spatialReference and desc.spatialReference.factoryCode == TARGET_WKID:
        arcpy.management.CopyFeatures(in_fc, out_fc)
    else:
        arcpy.management.Project(in_fc, out_fc, sr)
    return out_fc


def _county_fips_from_row(statefp, countyfp, geoid) -> str:
    if geoid not in (None, ""):
        s = str(geoid).strip().split(".")[0]
        if s and s.lower() != "none":
            return s.zfill(5)[:5]
    st = str(statefp).strip().split(".")[0].zfill(2)
    co = str(countyfp).strip().split(".")[0].zfill(3)
    return st + co


def process_utility_county_adj(
    utilities: str,
    population: str,
    counties: str,
    output_csv: str,
    utility_field: str | None = None,
    population_field: str | None = None,
    min_piece_area_m2: float = 1.0,
) -> None:
    arcpy.env.overwriteOutput = True

    for path, label in [(utilities, "Utilities"), (population, "Population polygons"), (counties, "Counties")]:
        if not arcpy.Exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        _ensure_polygon(path, label)

    utility_field = _resolve_field(utilities, utility_field, ["wwtp", "utility", "name", "facility"])
    population_field = _resolve_field(
        population,
        population_field,
        ["P0010001", "P1_001N", "POP20", "POPULATION", "TOTAL_POP", "POP"],
    )

    county_name_field = _resolve_field(counties, None, ["NAME", "NAMELSAD", "COUNTYNAME", "COUNTY"])
    county_fields = _field_names(counties)
    county_geoid_field = county_fields.get("geoid") or county_fields.get("geoid20") or county_fields.get("countyfips")
    statefp_field = county_fields.get("statefp") or county_fields.get("statefp20")
    countyfp_field = county_fields.get("countyfp") or county_fields.get("countyfp20")
    if not county_geoid_field and not (statefp_field and countyfp_field):
        raise ValueError("County layer needs GEOID/CountyFIPS or STATEFP + COUNTYFP fields.")

    scratch = arcpy.env.scratchGDB
    if not scratch:
        raise RuntimeError("ArcPy scratchGDB is unavailable.")

    util_p = os.path.join(scratch, "uca_util_5070")
    pop_p = os.path.join(scratch, "uca_pop_5070")
    county_p = os.path.join(scratch, "uca_county_5070")
    pop_county = os.path.join(scratch, "uca_pop_county")
    pieces = os.path.join(scratch, "uca_pieces")

    for fc in [util_p, pop_p, county_p, pop_county, pieces]:
        if arcpy.Exists(fc):
            arcpy.management.Delete(fc)

    print("[1/6] Projecting inputs to EPSG:5070...")
    _project(utilities, util_p)
    _project(population, pop_p)
    _project(counties, county_p)

    # Add stable IDs and source areas before intersection.
    util_id = "UCA_UID"
    pop_id = "UCA_PID"
    pop_area = "SRC_A_M2"
    for fc, field in [(util_p, util_id), (pop_p, pop_id)]:
        if field.lower() not in _field_names(fc):
            arcpy.management.AddField(fc, field, "LONG")
        arcpy.management.CalculateField(fc, field, "!OBJECTID!", "PYTHON3")

    if pop_area.lower() not in _field_names(pop_p):
        arcpy.management.AddField(pop_p, pop_area, "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(pop_p, [[pop_area, "AREA"]], area_unit="SQUARE_METERS")

    # Spatially attach county attributes to population polygons. Census blocks/BGs
    # normally nest within counties, but HAVE_THEIR_CENTER_IN avoids boundary slivers.
    print("[2/6] Assigning each population polygon to a county...")
    arcpy.analysis.SpatialJoin(
        target_features=pop_p,
        join_features=county_p,
        out_feature_class=pop_county,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_COMMON",
        match_option="HAVE_THEIR_CENTER_IN",
    )

    # Resolve fields again after SpatialJoin because duplicate names can be suffixed.
    pc_fields = _field_names(pop_county)
    pop_field_joined = pc_fields.get(population_field.lower())
    pop_area_joined = pc_fields.get(pop_area.lower())
    pop_id_joined = pc_fields.get(pop_id.lower())
    county_name_joined = pc_fields.get(county_name_field.lower())
    geoid_joined = pc_fields.get(county_geoid_field.lower()) if county_geoid_field else None
    statefp_joined = pc_fields.get(statefp_field.lower()) if statefp_field else None
    countyfp_joined = pc_fields.get(countyfp_field.lower()) if countyfp_field else None

    required = [pop_field_joined, pop_area_joined, pop_id_joined, county_name_joined]
    if any(x is None for x in required):
        raise RuntimeError("Required fields were lost or renamed unexpectedly during SpatialJoin.")

    print("[3/6] Intersecting sewersheds with population polygons...")
    arcpy.analysis.PairwiseIntersect([util_p, pop_county], pieces, "ALL", None, "INPUT")

    piece_area = "PCS_A_M2"
    alloc_pop = "ALLOC_POP"
    if piece_area.lower() not in _field_names(pieces):
        arcpy.management.AddField(pieces, piece_area, "DOUBLE")
    if alloc_pop.lower() not in _field_names(pieces):
        arcpy.management.AddField(pieces, alloc_pop, "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(pieces, [[piece_area, "AREA"]], area_unit="SQUARE_METERS")

    piece_fields = _field_names(pieces)
    util_joined = piece_fields.get(utility_field.lower())
    pop_joined = piece_fields.get(pop_field_joined.lower())
    src_area_joined = piece_fields.get(pop_area_joined.lower())
    piece_area_joined = piece_fields.get(piece_area.lower())
    county_name_piece = piece_fields.get(county_name_joined.lower())
    geoid_piece = piece_fields.get(geoid_joined.lower()) if geoid_joined else None
    statefp_piece = piece_fields.get(statefp_joined.lower()) if statefp_joined else None
    countyfp_piece = piece_fields.get(countyfp_joined.lower()) if countyfp_joined else None

    if not util_joined:
        # PairwiseIntersect may suffix the utility field if names collide.
        candidates = [v for k, v in piece_fields.items() if k.startswith(utility_field.lower())]
        util_joined = candidates[0] if candidates else None
    if not util_joined:
        raise RuntimeError(f"Could not locate utility field '{utility_field}' after intersection.")

    print("[4/6] Allocating population by overlap-area fraction...")
    expression = (
        f"0.0 if !{src_area_joined}! in (None, 0) else "
        f"float(!{pop_joined}! or 0) * min(1.0, max(0.0, float(!{piece_area_joined}! or 0) / float(!{src_area_joined}!)))"
    )
    arcpy.management.CalculateField(pieces, alloc_pop, expression, "PYTHON3")

    # County totals come from full source population polygons, not only sewered areas.
    county_population = defaultdict(float)
    county_names: dict[str, str] = {}
    county_cursor_fields = [pop_field_joined, county_name_joined]
    if geoid_joined:
        county_cursor_fields.append(geoid_joined)
    else:
        county_cursor_fields.extend([statefp_joined, countyfp_joined])

    with arcpy.da.SearchCursor(pop_county, county_cursor_fields) as cur:
        for row in cur:
            pop_value = float(row[0] or 0.0)
            county_name = str(row[1] or "").strip()
            if geoid_joined:
                fips = _county_fips_from_row(None, None, row[2])
            else:
                fips = _county_fips_from_row(row[2], row[3], None)
            county_population[fips] += pop_value
            county_names[fips] = county_name

    utility_county_population = defaultdict(float)
    read_fields = [util_joined, alloc_pop, piece_area_joined, county_name_piece]
    if geoid_piece:
        read_fields.append(geoid_piece)
    else:
        read_fields.extend([statefp_piece, countyfp_piece])

    with arcpy.da.SearchCursor(pieces, read_fields) as cur:
        for row in cur:
            utility = str(row[0] or "").strip()
            allocated = float(row[1] or 0.0)
            area = float(row[2] or 0.0)
            county_name = str(row[3] or "").strip()
            if not utility or area < float(min_piece_area_m2):
                continue
            if geoid_piece:
                fips = _county_fips_from_row(None, None, row[4])
            else:
                fips = _county_fips_from_row(row[4], row[5], None)
            utility_county_population[(utility, fips)] += allocated
            if county_name:
                county_names[fips] = county_name

    utility_totals = defaultdict(float)
    for (utility, _fips), value in utility_county_population.items():
        utility_totals[utility] += value

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    print("[5/6] Writing output CSV...")
    rows = []
    for (utility, fips), pop_value in utility_county_population.items():
        util_total = utility_totals.get(utility, 0.0)
        county_total = county_population.get(fips, 0.0)
        rows.append({
            "Utility": utility,
            "CountyFIPS": fips,
            "Pop": pop_value,
            "PctOfUtility": (pop_value / util_total) if util_total > 0 else 0.0,
            "PctOfCounty": (pop_value / county_total) if county_total > 0 else 0.0,
            "county_name": county_names.get(fips, ""),
        })

    rows.sort(key=lambda r: (r["Utility"].lower(), -r["PctOfUtility"], r["CountyFIPS"]))
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Utility", "CountyFIPS", "Pop", "PctOfUtility", "PctOfCounty", "county_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Validation report.
    pct_sums = defaultdict(float)
    for row in rows:
        pct_sums[row["Utility"]] += row["PctOfUtility"]
    bad = {u: s for u, s in pct_sums.items() if abs(s - 1.0) > 1e-6 and utility_totals[u] > 0}

    print("[6/6] Validation")
    print(f"  Utilities represented: {len(utility_totals)}")
    print(f"  Utility-county rows:   {len(rows)}")
    print(f"  Output:                {output_csv}")
    if bad:
        print(f"  WARNING: {len(bad)} utilities have PctOfUtility sums not equal to 1.0")
        for utility, value in list(bad.items())[:10]:
            print(f"    {utility}: {value:.8f}")
    else:
        print("  All nonzero utilities have PctOfUtility sums equal to 1.0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process utility_county_adj.csv using ArcPy area-weighted population allocation."
    )
    parser.add_argument(
        "--utilities",
        default=DEFAULT_UTILITIES,
        help=f"WWTP/sewershed polygon feature class (default: {DEFAULT_UTILITIES})",
    )
    parser.add_argument(
        "--population",
        default=DEFAULT_POPULATION,
        help=f"Census block or block-group polygon feature class (default: {DEFAULT_POPULATION})",
    )
    parser.add_argument(
        "--counties",
        default=DEFAULT_COUNTIES,
        help=f"County polygon feature class (default: {DEFAULT_COUNTIES})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--utility-field",
        default=DEFAULT_UTILITY_FIELD,
        help=f"Utility name field (default: {DEFAULT_UTILITY_FIELD}; use empty value only for auto-detection)",
    )
    parser.add_argument(
        "--population-field",
        default=DEFAULT_POPULATION_FIELD,
        help=f"Population field (default: {DEFAULT_POPULATION_FIELD}; use empty value only for auto-detection)",
    )
    parser.add_argument(
        "--min-piece-area-m2",
        type=float,
        default=DEFAULT_MIN_PIECE_AREA_M2,
        help=f"Ignore intersection slivers smaller than this area (default: {DEFAULT_MIN_PIECE_AREA_M2})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        process_utility_county_adj(
            utilities=args.utilities,
            population=args.population,
            counties=args.counties,
            output_csv=args.output,
            utility_field=args.utility_field,
            population_field=args.population_field,
            min_piece_area_m2=args.min_piece_area_m2,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        try:
            print(arcpy.GetMessages(2), file=sys.stderr)
        except Exception:
            pass
        raise
