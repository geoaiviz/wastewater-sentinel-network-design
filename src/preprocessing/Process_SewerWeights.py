# -*- coding: utf-8 -*-
"""
ArcPy script: Build BG–WWTP OD weights + county uncovered population

Outputs:
  1) bg_wwtp_weights.csv
     Fields:
       BG_FIPS, WWTP_ID, outside_prop, bg_area, bg_pop, county_fips, intersect_area

  2) county_uncovered_pop.csv
     Fields:
       FIPS, COUNTYNAME, POP   (POP = uncovered population outside UNION(sewersheds))

  3) sewershed_pop_served_bg.csv
     Fields:
       WWTP_ID, pop_served_bg, area_served_bg_m2
     Notes:
       pop_served_bg = Σ_BG (bg_pop * Area(BG∩WWTP)/Area(BG))
       area_served_bg_m2 = Σ_BG (bg_area * Area(BG∩WWTP)/Area(BG))

Design notes:
- BG–WWTP weights (outside_prop) are computed per WWTP sewershed:
    outside_prop = 1 - (Area(BG∩WWTP) / Area(BG)), clipped to [0,1]

- County uncovered population is computed as BG population outside the UNION of all sewersheds:
    covered_frac_u = Area(BG∩UNION(sewersheds)) / Area(BG), clipped to [0,1]
    uncovered_pop  = bg_pop * (1 - covered_frac_u)

Sanity checks added:
A) Per (BG,WWTP): covered_frac + outside_prop ≈ 1
B) Statewide (UNION): covered + uncovered ≈ 100% for AREA and POP proxies

Author: (you)
"""

import arcpy
import os
import csv

# --------------------------
# USER CONFIG (edit paths/fields)
# --------------------------
bg_fc = r"..\ZoneSelection\Input\Census\COBlockGroup.shp"
wwtp_fc = r"..\ZoneSelection\Input\WWTP_CO\WWTP_Select.shp"
out_dir = r"..\ZoneSelection\Outfile\ArcGIS_Weights"
county_fc = r"..\ZoneSelection\Input\Census\COCounty.shp"  # optional; county names only

# Field names (EDIT if needed)
BG_ID_FIELD     = "FIPS"         # BG GEOID / unique id
BG_POP_FIELD    = "POPULATION"   # BG population
BG_COUNTY_FIELD = "STCOFIPS"     # County FIPS
WWTP_ID_FIELD   = "wwtp"         # normalized WWTP identifier matching the OD targets

# CRS for area
TARGET_SR = arcpy.SpatialReference(26913)  # NAD83 / UTM zone 13N (good for CO)

# Optional: skip WWTPs whose name contains this substring (case-insensitive)
SKIP_WWTP_NAME_SUBSTRINGS = ["historic"]  # set [] to disable

# Outputs
weights_csv = os.path.join(out_dir, "bg_wwtp_weights.csv")
county_csv  = os.path.join(out_dir, "county_uncovered_pop.csv")
sewershed_pop_csv = os.path.join(out_dir, "sewershed_pop_served_bg.csv")
utility_county_csv = os.path.join(out_dir, "utility_county_adj.csv")

# Workspace GDB
gdb_name = "od_weights.gdb"

# Tolerance for fraction checks
TOL = 1e-6


# --------------------------
# Helpers
# --------------------------
def add_field_if_missing(fc, field_name, field_type, field_length=None):
    existing = [f.name for f in arcpy.ListFields(fc)]
    if field_name not in existing:
        if field_length is not None and field_type.upper() == "TEXT":
            arcpy.management.AddField(fc, field_name, field_type, field_length=field_length)
        else:
            arcpy.management.AddField(fc, field_name, field_type)


def find_stat_field(table, contains_lower):
    """Find first field whose name contains substring (case-insensitive)."""
    contains_lower = contains_lower.lower()
    for f in arcpy.ListFields(table):
        if contains_lower in f.name.lower():
            return f.name
    return None


def safe_zfill5(v):
    s = "" if v is None else str(v).strip()
    return s.zfill(5)


def _msg(s: str):
    """ArcGIS-friendly message + console print fallback."""
    try:
        arcpy.AddMessage(s)
    except Exception:
        pass
    print(s)


def _assert_has_fields(fc, fields, label):
    existing = {f.name for f in arcpy.ListFields(fc)}
    missing = [f for f in fields if f not in existing]
    if missing:
        raise RuntimeError(f"[{label}] Missing field(s) {missing} in {fc}")



def load_county_name_lookup(county_feature_class):
    """
    Optional lookup used only to populate county_name.

    Population allocation does not depend on county polygons. County FIPS comes
    directly from BG_COUNTY_FIELD in the block-group layer.
    """
    if not county_feature_class or not arcpy.Exists(county_feature_class):
        _msg("[County names] Optional county layer not found; county_name will be blank.")
        return {}

    fields = {f.name.lower(): f.name for f in arcpy.ListFields(county_feature_class)}
    fips_field = (
        fields.get("us_fips")
        or fields.get("geoid")
        or fields.get("countyfips")
        or fields.get("cnty_fips")
        or fields.get("fips")
    )
    name_field = (
        fields.get("county")
        or fields.get("countyname")
        or fields.get("name")
        or fields.get("namelsad")
    )

    if not fips_field or not name_field:
        _msg("[County names] No usable FIPS/name fields found; county_name will be blank.")
        return {}

    lookup = {}
    with arcpy.da.SearchCursor(county_feature_class, [fips_field, name_field]) as cur:
        for fips, name in cur:
            fips5 = safe_zfill5(fips)
            if fips5:
                lookup[fips5] = "" if name is None else str(name).strip()

    _msg(f"[County names] Loaded {len(lookup)} names using {fips_field} -> {name_field}")
    return lookup


def export_utility_county_adj(
    dissolved_bg_wwtp_fc,
    bg_projected_fc,
    output_csv,
    county_feature_class=None,
    wwtp_id_field=WWTP_ID_FIELD,
    bg_county_field=BG_COUNTY_FIELD,
    bg_pop_field=BG_POP_FIELD,
    pop_part_field="pop_served_part",
):
    """
    Create utility_county_adj.csv from the existing BG-WWTP weighted results.

    Pop = sum(bg_pop * covered_frac) for each Utility × CountyFIPS
    PctOfUtility = Utility–County Pop / Utility total weighted Pop
    PctOfCounty = Utility–County Pop / County total BG Pop

    No county intersection is performed.
    """
    _msg("=== 10D) Export Utility × County weighted population ===")

    pair_pop = {}
    utility_totals = {}

    with arcpy.da.SearchCursor(
        dissolved_bg_wwtp_fc,
        [wwtp_id_field, bg_county_field, pop_part_field],
    ) as cur:
        for utility, county_fips, pop_part in cur:
            utility = "" if utility is None else str(utility).strip()
            county = safe_zfill5(county_fips)
            pop = 0.0 if pop_part is None else float(pop_part)
            if not utility or not county:
                continue

            pair_pop[(utility, county)] = pair_pop.get((utility, county), 0.0) + pop
            utility_totals[utility] = utility_totals.get(utility, 0.0) + pop

    county_totals = {}
    with arcpy.da.SearchCursor(
        bg_projected_fc,
        [bg_county_field, bg_pop_field],
    ) as cur:
        for county_fips, bg_pop in cur:
            county = safe_zfill5(county_fips)
            pop = 0.0 if bg_pop is None else float(bg_pop)
            if county:
                county_totals[county] = county_totals.get(county, 0.0) + pop

    county_names = load_county_name_lookup(county_feature_class)

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Utility", "CountyFIPS", "Pop", "PctOfUtility", "PctOfCounty", "county_name"]
        )

        for (utility, county), pop in sorted(
            pair_pop.items(),
            key=lambda item: (item[0][0].lower(), item[0][1]),
        ):
            utility_total = utility_totals.get(utility, 0.0)
            county_total = county_totals.get(county, 0.0)

            writer.writerow([
                utility,
                county,
                pop,
                pop / utility_total if utility_total > 0 else 0.0,
                pop / county_total if county_total > 0 else 0.0,
                county_names.get(county, ""),
            ])

    max_sum_error = 0.0
    for utility, total in utility_totals.items():
        if total <= 0:
            continue
        pct_sum = sum(
            pop / total
            for (u, _), pop in pair_pop.items()
            if u == utility
        )
        max_sum_error = max(max_sum_error, abs(pct_sum - 1.0))

    _msg(f"Saved utility-county weighted CSV: {output_csv}")
    _msg(f"PctOfUtility max sum error: {max_sum_error:.3e}")


def main():
    # ---------- Robust workspace setup ----------
    arcpy.env.overwriteOutput = True

    out_dir_abs = os.path.abspath(out_dir)
    if not os.path.exists(out_dir_abs):
        os.makedirs(out_dir_abs)

    gdb = os.path.join(out_dir_abs, gdb_name)
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(out_dir_abs, gdb_name)

    arcpy.env.workspace = gdb
    _msg(f"Workspace: {gdb}")

    # ---------- Copy inputs into GDB ----------
    _msg("=== 1) Copy inputs into GDB ===")
    bg_in = arcpy.conversion.FeatureClassToFeatureClass(bg_fc, gdb, "bg_raw")
    wwtp_in = arcpy.conversion.FeatureClassToFeatureClass(wwtp_fc, gdb, "wwtp_raw")

    # Field presence checks early (helpful when schemas change)
    _assert_has_fields(bg_in, [BG_ID_FIELD, BG_POP_FIELD, BG_COUNTY_FIELD], "BG input")
    _assert_has_fields(wwtp_in, [WWTP_ID_FIELD], "WWTP input")

    # ---------- Repair geometry ----------
    _msg("=== 2) Repair geometry ===")
    arcpy.management.RepairGeometry(bg_in)
    arcpy.management.RepairGeometry(wwtp_in)

    # ---------- Optional: filter WWTPs (skip 'historic') ----------
    _msg("=== 3) Optional filter WWTPs ===")
    wwtp_filtered = os.path.join(gdb, "wwtp_filtered")
    if SKIP_WWTP_NAME_SUBSTRINGS:
        field_delim = arcpy.AddFieldDelimiters(wwtp_in, WWTP_ID_FIELD)
        clauses = []
        for sub in SKIP_WWTP_NAME_SUBSTRINGS:
            sub = sub.strip()
            if not sub:
                continue
            clauses.append(f"UPPER({field_delim}) NOT LIKE '%{sub.upper()}%'")
        where = " AND ".join(clauses) if clauses else None
        if where:
            arcpy.analysis.Select(wwtp_in, wwtp_filtered, where)
        else:
            arcpy.management.CopyFeatures(wwtp_in, wwtp_filtered)
    else:
        arcpy.management.CopyFeatures(wwtp_in, wwtp_filtered)

    # ---------- Project both layers ----------
    _msg("=== 4) Project to target CRS ===")
    bg_proj = os.path.join(gdb, "bg_proj")
    wwtp_proj = os.path.join(gdb, "wwtp_proj")
    arcpy.management.Project(bg_in, bg_proj, TARGET_SR)
    arcpy.management.Project(wwtp_filtered, wwtp_proj, TARGET_SR)

    # ---------- Compute BG area ----------
    _msg("=== 5) Compute BG area ===")
    add_field_if_missing(bg_proj, "bg_area", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        bg_proj,
        [["bg_area", "AREA"]],
        area_unit="SQUARE_METERS"
    )

    # ============================================================
    # PART A: BG–WWTP weights (per WWTP)
    # ============================================================
    _msg("=== 6) Intersect BG × WWTP ===")
    inter_fc = os.path.join(gdb, "bg_wwtp_intersect")
    arcpy.analysis.Intersect([bg_proj, wwtp_proj], inter_fc, "ALL", None, "INPUT")

    _msg("=== 7) Compute intersection area ===")
    add_field_if_missing(inter_fc, "int_area", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        inter_fc,
        [["int_area", "AREA"]],
        area_unit="SQUARE_METERS"
    )

    _msg("=== 8) Dissolve by (BG, WWTP), sum intersection area ===")
    diss_fc = os.path.join(gdb, "bg_wwtp_diss")
    arcpy.management.Dissolve(
        inter_fc,
        diss_fc,
        dissolve_field=[BG_ID_FIELD, WWTP_ID_FIELD],
        statistics_fields=[["int_area", "SUM"]],
        multi_part="MULTI_PART"
    )

    # Find SUM field name (usually SUM_int_area)
    sum_int_field = None
    for f in arcpy.ListFields(diss_fc):
        if f.name.lower().startswith("sum_") and "int_area" in f.name.lower():
            sum_int_field = f.name
            break
    if not sum_int_field:
        sum_int_field = find_stat_field(diss_fc, "sum_")
    if not sum_int_field:
        raise RuntimeError("Could not find SUM(int_area) field in dissolved output.")

    _msg("=== 9) Join BG attrs (bg_area, pop, county) ===")
    try:
        arcpy.management.AddIndex(bg_proj, BG_ID_FIELD, "idx_bgid")
    except Exception:
        pass

    arcpy.management.JoinField(
        diss_fc,
        BG_ID_FIELD,
        bg_proj,
        BG_ID_FIELD,
        ["bg_area", BG_POP_FIELD, BG_COUNTY_FIELD]
    )

    _msg("=== 10) Compute covered_frac and outside_prop (clipped 0..1) ===")
    add_field_if_missing(diss_fc, "covered_frac", "DOUBLE")
    add_field_if_missing(diss_fc, "outside_prop", "DOUBLE")

    with arcpy.da.UpdateCursor(diss_fc, [sum_int_field, "bg_area", "covered_frac", "outside_prop"]) as cur:
        for s_int, bg_area, _, _ in cur:
            s_int = 0.0 if s_int is None else float(s_int)
            bg_area = 0.0 if bg_area is None else float(bg_area)

            if bg_area <= 0.0:
                covered = 0.0
            else:
                covered = s_int / bg_area

            # clip
            if covered < 0.0:
                covered = 0.0
            if covered > 1.0:
                covered = 1.0

            outside = 1.0 - covered
            cur.updateRow((s_int, bg_area, covered, outside))


    # ------------------------------------------------------------
    # Per-WWTP population served derived from BG intersections
    # ------------------------------------------------------------
    _msg("=== 10B) Compute per-row pop_served_part and area_served_part ===")
    add_field_if_missing(diss_fc, "pop_served_part", "DOUBLE")
    add_field_if_missing(diss_fc, "area_served_part", "DOUBLE")

    with arcpy.da.UpdateCursor(
        diss_fc,
        [BG_POP_FIELD, "bg_area", "covered_frac", "pop_served_part", "area_served_part"]
    ) as cur:
        for bg_pop, bg_area, covered, _, _ in cur:
            bg_pop = 0.0 if bg_pop is None else float(bg_pop)
            bg_area = 0.0 if bg_area is None else float(bg_area)
            covered = 0.0 if covered is None else float(covered)

            # clip just in case
            if covered < 0.0:
                covered = 0.0
            if covered > 1.0:
                covered = 1.0

            pop_part = bg_pop * covered
            area_part = bg_area * covered
            cur.updateRow((bg_pop, bg_area, covered, pop_part, area_part))

    _msg("=== 10C) Aggregate pop_served_part to WWTP and export CSV ===")
    wwtp_pop_table = os.path.join(gdb, "wwtp_pop_served_bg")
    arcpy.analysis.Statistics(
        diss_fc,
        wwtp_pop_table,
        statistics_fields=[["pop_served_part", "SUM"], ["area_served_part", "SUM"]],
        case_field=[WWTP_ID_FIELD]
    )

    pop_sum_field = None
    area_sum_field = None
    for f in arcpy.ListFields(wwtp_pop_table):
        n = f.name.lower()
        if n.startswith("sum_") and "pop_served_part" in n:
            pop_sum_field = f.name
        if n.startswith("sum_") and "area_served_part" in n:
            area_sum_field = f.name

    if not pop_sum_field or not area_sum_field:
        raise RuntimeError("Could not find SUM fields for pop_served_part / area_served_part in wwtp_pop_served_bg.")

    with open(sewershed_pop_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["WWTP_ID", "pop_served_bg", "area_served_bg_m2"])
        with arcpy.da.SearchCursor(wwtp_pop_table, [WWTP_ID_FIELD, pop_sum_field, area_sum_field]) as cur:
            for wwtp_id, pop_sum, area_sum in cur:
                wwtp_id = "" if wwtp_id is None else str(wwtp_id).strip()
                pop_sum = 0.0 if pop_sum is None else float(pop_sum)
                area_sum = 0.0 if area_sum is None else float(area_sum)
                w.writerow([wwtp_id, pop_sum, area_sum])

    _msg(f"Saved sewershed pop-served CSV: {sewershed_pop_csv}")


    export_utility_county_adj(
        dissolved_bg_wwtp_fc=diss_fc,
        bg_projected_fc=bg_proj,
        output_csv=utility_county_csv,
        county_feature_class=county_fc,
    )


    # ---- Sanity check A: per-(BG,WWTP) fractions sum to 1 ----
    _msg("=== Sanity check A: per-(BG,WWTP) covered_frac + outside_prop ≈ 1 ===")
    n_bad = 0
    n_total = 0
    max_err = 0.0
    with arcpy.da.SearchCursor(diss_fc, ["covered_frac", "outside_prop"]) as cur:
        for covered, outside in cur:
            covered = 0.0 if covered is None else float(covered)
            outside = 0.0 if outside is None else float(outside)
            s = covered + outside
            err = abs(s - 1.0)
            n_total += 1
            if err > max_err:
                max_err = err
            if err > TOL:
                n_bad += 1
    _msg(f"Per-pair check: rows={n_total}, bad_rows={n_bad}, max_abs_error={max_err:.3e}")

    _msg("=== 11) Export BG–WWTP weights CSV ===")
    out_fields = [BG_ID_FIELD, WWTP_ID_FIELD, "outside_prop", "bg_area", BG_POP_FIELD, BG_COUNTY_FIELD, sum_int_field]
    with open(weights_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["BG_FIPS", "WWTP_ID", "outside_prop", "bg_area", "bg_pop", "county_fips", "intersect_area"])
        with arcpy.da.SearchCursor(diss_fc, out_fields) as cur:
            for bg_id, wwtp_id, outside_prop, bg_area, bg_pop, county_fips, s_int in cur:
                bg_id = "" if bg_id is None else str(bg_id).strip()
                wwtp_id = "" if wwtp_id is None else str(wwtp_id).strip()
                county_fips = safe_zfill5(county_fips)
                w.writerow([bg_id, wwtp_id, outside_prop, bg_area, bg_pop, county_fips, s_int])
    _msg(f"Saved weights CSV: {weights_csv}")

    # ============================================================
    # PART B: County uncovered population (OVERLAP-SAFE via UNION)
    # ============================================================
    _msg("=== 12) Build UNION of all sewersheds (overlap-safe) ===")
    wwtp_union = os.path.join(gdb, "wwtp_union")
    arcpy.management.Dissolve(wwtp_proj, wwtp_union, multi_part="MULTI_PART")

    _msg("=== 13) Intersect BG with UNION(sewersheds) ===")
    bg_union_inter = os.path.join(gdb, "bg_union_intersect")
    arcpy.analysis.Intersect([bg_proj, wwtp_union], bg_union_inter, "ALL", None, "INPUT")

    _msg("=== 14) Compute covered area per BG (union) ===")
    add_field_if_missing(bg_union_inter, "cov_area", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        bg_union_inter,
        [["cov_area", "AREA"]],
        area_unit="SQUARE_METERS"
    )

    # Summarize coverage area by BG
    bg_cov_table = os.path.join(gdb, "bg_cov_sum")
    arcpy.analysis.Statistics(
        bg_union_inter,
        bg_cov_table,
        statistics_fields=[["cov_area", "SUM"]],
        case_field=[BG_ID_FIELD]
    )

    cov_sum_field = None
    for f in arcpy.ListFields(bg_cov_table):
        if f.name.lower().startswith("sum_") and "cov_area" in f.name.lower():
            cov_sum_field = f.name
            break
    if not cov_sum_field:
        cov_sum_field = find_stat_field(bg_cov_table, "sum_")
    if not cov_sum_field:
        raise RuntimeError("Could not find SUM(cov_area) field in bg_cov_sum table.")

    _msg("=== 15) Join BG coverage back to BGs and compute uncovered_pop ===")
    bg_with_cov = os.path.join(gdb, "bg_with_cov")
    arcpy.management.CopyFeatures(bg_proj, bg_with_cov)
    arcpy.management.JoinField(bg_with_cov, BG_ID_FIELD, bg_cov_table, BG_ID_FIELD, [cov_sum_field])

    add_field_if_missing(bg_with_cov, "uncovered_pop", "DOUBLE")
    add_field_if_missing(bg_with_cov, "covered_frac_u", "DOUBLE")  # union-based covered fraction

    with arcpy.da.UpdateCursor(bg_with_cov, [cov_sum_field, "bg_area", BG_POP_FIELD, "covered_frac_u", "uncovered_pop"]) as cur:
        for cov_area, bg_area, bg_pop, _, _ in cur:
            cov_area = 0.0 if cov_area is None else float(cov_area)
            bg_area = 0.0 if bg_area is None else float(bg_area)
            bg_pop  = 0.0 if bg_pop is None else float(bg_pop)

            if cov_area < 0.0:
                cov_area = 0.0
            if cov_area > bg_area and bg_area > 0:
                cov_area = bg_area

            if bg_area <= 0.0:
                covered = 0.0
            else:
                covered = cov_area / bg_area

            # clip
            if covered < 0.0:
                covered = 0.0
            if covered > 1.0:
                covered = 1.0

            uncovered = bg_pop * (1.0 - covered)
            cur.updateRow((cov_area, bg_area, bg_pop, covered, uncovered))

    # ---- Sanity check B: UNION-based statewide totals (covered + uncovered = 100%) ----
    _msg("=== Sanity check B: UNION(sewersheds) statewide coverage sums (should be 100%) ===")
    total_bg_area = 0.0
    total_cov_area = 0.0
    total_bg_pop = 0.0
    total_uncovered_pop = 0.0

    with arcpy.da.SearchCursor(bg_with_cov, [cov_sum_field, "bg_area", BG_POP_FIELD, "uncovered_pop"]) as cur:
        for cov_area, bg_area, bg_pop, unc_pop in cur:
            bg_area = 0.0 if bg_area is None else float(bg_area)
            bg_pop  = 0.0 if bg_pop  is None else float(bg_pop)
            unc_pop = 0.0 if unc_pop is None else float(unc_pop)
            cov_area = 0.0 if cov_area is None else float(cov_area)

            if cov_area < 0.0:
                cov_area = 0.0
            if cov_area > bg_area and bg_area > 0:
                cov_area = bg_area

            total_bg_area += bg_area
            total_cov_area += cov_area
            total_bg_pop += bg_pop
            total_uncovered_pop += unc_pop

    total_uncovered_area = total_bg_area - total_cov_area
    total_covered_pop = total_bg_pop - total_uncovered_pop

    area_cov_frac = (total_cov_area / total_bg_area) if total_bg_area > 0 else 0.0
    area_unc_frac = (total_uncovered_area / total_bg_area) if total_bg_area > 0 else 0.0

    pop_unc_frac = (total_uncovered_pop / total_bg_pop) if total_bg_pop > 0 else 0.0
    pop_cov_frac = (total_covered_pop / total_bg_pop) if total_bg_pop > 0 else 0.0

    _msg(
        f"STATE AREA: total={total_bg_area:.3f} m^2 | "
        f"covered={total_cov_area:.3f} ({area_cov_frac*100:.2f}%) | "
        f"uncovered={total_uncovered_area:.3f} ({area_unc_frac*100:.2f}%) | "
        f"sum={(area_cov_frac+area_unc_frac)*100:.2f}%"
    )
    _msg(
        f"STATE POP : total={total_bg_pop:.1f} | "
        f"covered={total_covered_pop:.1f} ({pop_cov_frac*100:.2f}%) | "
        f"uncovered={total_uncovered_pop:.1f} ({pop_unc_frac*100:.2f}%) | "
        f"sum={(pop_cov_frac+pop_unc_frac)*100:.2f}%"
    )

    if abs((area_cov_frac + area_unc_frac) - 1.0) > 1e-4:
        _msg("WARNING: Area covered+uncovered does not sum to 100% within tolerance.")
    if abs((pop_cov_frac + pop_unc_frac) - 1.0) > 1e-4:
        _msg("WARNING: Pop covered+uncovered does not sum to 100% within tolerance.")

    _msg("=== 16) Aggregate uncovered_pop to county and export CSV ===")
    county_table = os.path.join(gdb, "county_uncovered")
    arcpy.analysis.Statistics(
        bg_with_cov,
        county_table,
        statistics_fields=[["uncovered_pop", "SUM"], [BG_POP_FIELD, "SUM"]],
        case_field=[BG_COUNTY_FIELD]
    )

    unc_field = None
    tot_field = None
    for f in arcpy.ListFields(county_table):
        n = f.name.lower()
        if n.startswith("sum_") and "uncovered_pop" in n:
            unc_field = f.name
        if n.startswith("sum_") and BG_POP_FIELD.lower() in n:
            tot_field = f.name
    if not unc_field or not tot_field:
        raise RuntimeError("Could not find county SUM fields for uncovered_pop / total_pop.")

    # Export county-level uncovered population in the required schema.
    with open(county_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["FIPS", "COUNTYNAME", "POP"])  # required schema

        with arcpy.da.SearchCursor(county_table, [BG_COUNTY_FIELD, unc_field]) as cur:
            for county_fips, unc in cur:
                fips5 = str(county_fips).strip().zfill(5)
                pop = 0 if unc is None else int(round(float(unc)))
                writer.writerow([fips5, "", pop])  # COUNTYNAME left blank by design

    _msg(f"Saved county uncovered pop CSV: {county_csv}")
    _msg("DONE.")


if __name__ == "__main__":
    main()
