"""Aggregate authorized OD inputs and create mobility-flow summaries.

The public repository does not include the licensed OD records consumed by
these functions. Plotting helpers are retained for downstream risk workflows.
"""


from Process_ODData_Aggr import aggregate_od_weekly
import geopandas as gpd
import pandas as pd
import folium
import os
import branca.colormap as cm
import re

import matplotlib.pyplot as plt

def safe_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))

def draw_weekly_od_lines_with_layercontrol(
    weekly_od_dir,
    shapefile_path,
    county_fp,
    covid_weekly_fp,
    output_folder="outputs/od_maps_perweek"
):

    os.makedirs(output_folder, exist_ok=True)

    # Load base data
    wwtp = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    counties = gpd.read_file(county_fp).to_crs(epsg=4326)
    covid = pd.read_csv(covid_weekly_fp)
    covid["week"] = pd.to_datetime(covid["week"])
    covid["LABEL"] = covid["LABEL"].str.strip().str.lower()

    wwtp["wwtp"] = wwtp["wwtp"].astype(str).str.strip()
    wwtp["centroid"] = wwtp.geometry.centroid
    sewer_centroids = wwtp.drop_duplicates("wwtp").set_index("wwtp")["centroid"]

    counties["LABEL"] = counties["LABEL"].str.strip().str.lower()
    counties["US_FIPS"] = counties["US_FIPS"].astype(str).str.zfill(5)
    counties["centroid"] = counties.geometry.centroid
    county_centroids = counties.set_index("US_FIPS")["centroid"]
    county_geo = counties[["LABEL", "geometry"]].copy()

    weeks = sorted(set(
        f.replace("weekly_o_", "").replace("weekly_d_", "").replace(".csv", "")
        for f in os.listdir(weekly_od_dir)
        if f.startswith("weekly_o_") or f.startswith("weekly_d_")
    ))

    for week in weeks:
        print("Processing week:", week)
        m = folium.Map(location=[39.0, -105.5], zoom_start=7, tiles="cartodbpositron")
        total = 0

        # COVID choropleth
        covid_group = folium.FeatureGroup(name="Weekly COVID Rate", show=True)
        covid_week = covid[covid["week"] == pd.to_datetime(week)]
        covid_joined = county_geo.merge(covid_week, on="LABEL", how="left")

        # Fix serialization
        covid_joined["County_cases_3dayavg_r100Kutil"] = covid_joined["County_cases_3dayavg_r100Kutil"].astype(float)
        if "week" in covid_joined.columns:
            covid_joined = covid_joined.drop(columns=["week"], errors="ignore")

        values = covid_week["County_cases_3dayavg_r100Kutil"].dropna()
        if len(values) < 5:
            bins = [0, 10, 20, 50, 100, 200]
        else:
            bins = values.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()

        color_scale = ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"]

        def assign_color(val):
            if pd.isnull(val):
                return "#ffffff"
            for i in range(len(bins) - 1):
                if bins[i] <= val < bins[i + 1]:
                    return color_scale[i]
            return color_scale[-1]

        covid_joined["color"] = covid_joined["County_cases_3dayavg_r100Kutil"].apply(assign_color)

        def covid_style_function(feature):
            return {
                "fillColor": feature["properties"].get("color", "#ffffff"),
                "color": "gray",
                "weight": 0.3,
                "fillOpacity": 0.6
            }

        folium.GeoJson(
            covid_joined,
            style_function=covid_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["LABEL", "County_cases_3dayavg_r100Kutil"],
                aliases=["County", "Weekly Avg Rate"],
                localize=True
            ),
            name="Weekly COVID Rate"
        ).add_to(covid_group)
        m.add_child(covid_group)

        def add_color_legend(map_obj, bins, colors, label="COVID Rate per 100K"):
            legend_html = f"""
            <div style='position: fixed; bottom: 30px; left: 30px; z-index:9999;
                        font-size:14px; background:white; padding:8px; border:1px solid gray'>
                <b>{label}</b><br>
            """
            for i in range(len(colors)):
                low = round(bins[i], 1)
                high = round(bins[i + 1], 1)
                legend_html += f"""
                <div style='display: flex; align-items: center;'>
                    <div style='width: 18px; height: 18px; background:{colors[i]}; margin-right:6px;'></div>
                    {low} – {high}
                </div>
                """
            legend_html += "</div>"
            map_obj.get_root().html.add_child(folium.Element(legend_html))

        add_color_legend(m, bins, color_scale)

        # Add WWTP polygons
        wwtp_group = folium.FeatureGroup(name="Sewersheds", show=True)
        folium.GeoJson(
            wwtp.geometry,
            style_function=lambda x: {"color": "blue", "weight": 1, "fillOpacity": 0.1}
        ).add_to(wwtp_group)
        m.add_child(wwtp_group)

        # OD lines
        for direction_code in ["o", "d"]:
            fpath = os.path.join(weekly_od_dir, f"weekly_{direction_code}_{week}.csv")
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath)
            df["County"] = df["County"].astype(str).str.zfill(5)
            if 'weekly_o' in fpath:
                df["Sewershed-O"] = df.get("Sewershed-O", "").astype(str).str.strip()
            else:
                df["Sewershed-D"] = df.get("Sewershed-D", "").astype(str).str.strip()

            key_col = "Sewershed-O" if direction_code == "o" else "Sewershed-D"
            for wwtp_id, group in df.groupby(key_col):
                wwtp_id = str(wwtp_id).strip()
                fg = folium.FeatureGroup(name=f"{direction_code.upper()}: {wwtp_id}", show=False)

                for _, row in group.iterrows():
                    direction = row["direction"]
                    county = row["County"]
                    volume = row["Volume"]
                    try:
                        weight = max(1.0, min(10.0, (volume ** 0.5) / 2))
                    except:
                        weight = 1.0

                    if direction == "Origin":
                        orig = sewer_centroids.get(wwtp_id)
                        dest = county_centroids.get(county)
                    elif direction == "Destination":
                        orig = county_centroids.get(county)
                        dest = sewer_centroids.get(wwtp_id)
                    else:
                        continue

                    if orig is None or dest is None:
                        continue

                    folium.PolyLine(
                        locations=[[orig.y, orig.x], [dest.y, dest.x]],
                        color="grey",
                        weight=weight,
                        opacity=0.8,
                        popup=f"{direction}: {wwtp_id} ↔ {county}, Vol={volume:.0f}, Wt={weight:.1f}"
                    ).add_to(fg)
                    total += 1

                m.add_child(fg)

        folium.LayerControl(collapsed=False).add_to(m)
        out_path = os.path.join(output_folder, f"od_map_{week}.html")
        m.save(out_path)
        print("Map saved:", out_path, "| Lines drawn:", total)


# ------------------------------------------------------------
# Helper: compute weekly flows & fips lookup
# ------------------------------------------------------------
def compute_weekly_flows(weekly_od_dir, fips_lookup=None):
    weekly_results = {}
    weeks = sorted(set(
        f.replace("weekly_o_", "").replace("weekly_d_", "").replace(".csv", "")
        for f in os.listdir(weekly_od_dir)
        if f.startswith("weekly_o_") or f.startswith("weekly_d_")
    ))

    # Create FIPS lookup if not provided
    if fips_lookup is None:
        tiger_url = "../ZoneSelection/Input/Census/st08_co_cou.txt"
        tiger_df = pd.read_csv(tiger_url, header=None, dtype=str)
        tiger_df.columns = ["State", "StateFP", "CountyFP", "CountyName", "ClassCode"]
        tiger_df["CountyName"] = tiger_df["CountyName"].str.strip().str.lower()
        tiger_df["CountyFIPS"] = tiger_df["StateFP"] + tiger_df["CountyFP"]
        fips_lookup = dict(
            zip(
                tiger_df["CountyName"].str.replace(" county", "", regex=False).str.strip(),
                tiger_df["CountyFIPS"]
            )
        )

    for week in weeks:
        origin_path = os.path.join(weekly_od_dir, f"weekly_o_{week}.csv")
        dest_path = os.path.join(weekly_od_dir, f"weekly_d_{week}.csv")

        o_df = pd.read_csv(origin_path) if os.path.exists(origin_path) else pd.DataFrame()
        d_df = pd.read_csv(dest_path) if os.path.exists(dest_path) else pd.DataFrame()

        flows = pd.DataFrame()

        if not o_df.empty:
            o_df["Sewershed-O"] = o_df["Sewershed-O"].astype(str).str.strip()
            o_sum = o_df.groupby("Sewershed-O")["Volume"].sum().reset_index()
            o_sum.columns = ["wwtp", "total_out"]
            flows = o_sum

        if not d_df.empty:
            d_df["Sewershed-D"] = d_df["Sewershed-D"].astype(str).str.strip()
            d_sum = d_df.groupby("Sewershed-D")["Volume"].sum().reset_index()
            d_sum.columns = ["wwtp", "total_in"]
            flows = pd.merge(flows, d_sum, on="wwtp", how="outer")

        flows = flows.fillna(0)
        flows["net_flow"] = flows["total_out"] - flows["total_in"]
        flows["total_flow"] = flows["total_out"] + flows["total_in"]

        # County-to-sewershed flow matrix
        full_matrix = (
            pd.pivot_table(
                d_df,
                index="Sewershed-D",
                columns="County",
                values="Volume",
                aggfunc="sum",
                fill_value=0.0,
            )
            .astype(float)
        )

        flipped_matrix = (
            pd.pivot_table(
                o_df,
                index="Sewershed-O",
                columns="County",
                values="Volume",
                aggfunc="sum",
                fill_value=0.0,
            )
            .astype(float)
        )

        # Fill any residual missing values from sparse weekly inputs.
        full_matrix = full_matrix.fillna(0.0)
        flipped_matrix = flipped_matrix.fillna(0.0)

        flows.attrs = {
            "county_to_wwtp": full_matrix,  # C → W (NO NaNs)
            "wwtp_to_county": flipped_matrix  # W → C (NO NaNs)
        }

        weekly_results[week] = flows

    return weekly_results, fips_lookup


# Generate a single HTML map with inflow, outflow, total-flow, and per-capita-flow layers.
# for each week, using pop_served from shapefile and excluding historic WWTPs
def plot_weekly_flow_maps_html(wwtp_shapefile, weekly_od_dir, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Load WWTP shapefile, filter out historic
    wwtp_gdf = gpd.read_file(wwtp_shapefile).to_crs(epsg=4326)
    wwtp_gdf['wwtp'] = wwtp_gdf['wwtp'].str.lower().str.strip()
    wwtp_gdf = wwtp_gdf[~wwtp_gdf['wwtp'].str.contains('historic', na=False)]

    # Identify all unique weeks from file names
    weeks = sorted({fname.replace('weekly_o_', '').replace('weekly_d_', '').replace('.csv', '')
                    for fname in os.listdir(weekly_od_dir) if fname.endswith('.csv')})

    for week in weeks:
        inflow_data = {}
        outflow_data = {}

        inflow_path = os.path.join(weekly_od_dir, f"weekly_d_{week}.csv")
        if os.path.exists(inflow_path):
            df_in = pd.read_csv(inflow_path)
            df_in['Sewershed-D'] = df_in['Sewershed-D'].astype(str).str.lower().str.strip()
            inflow_data = df_in.groupby('Sewershed-D')['Volume'].sum().to_dict()

        outflow_path = os.path.join(weekly_od_dir, f"weekly_o_{week}.csv")
        if os.path.exists(outflow_path):
            df_out = pd.read_csv(outflow_path)
            df_out['Sewershed-O'] = df_out['Sewershed-O'].astype(str).str.lower().str.strip()
            outflow_data = df_out.groupby('Sewershed-O')['Volume'].sum().to_dict()

        all_wwtps = set(inflow_data.keys()) | set(outflow_data.keys())
        flow_df = pd.DataFrame({'wwtp': list(all_wwtps)})
        flow_df['inflow'] = flow_df['wwtp'].map(inflow_data).fillna(0)
        flow_df['outflow'] = flow_df['wwtp'].map(outflow_data).fillna(0)
        flow_df['total_flow'] = flow_df['inflow'] + flow_df['outflow']

        merged_gdf = wwtp_gdf.merge(flow_df, on='wwtp', how='left')
        merged_gdf['flow_pop_ratio'] = merged_gdf['total_flow'] / merged_gdf['pop_served']

        m = folium.Map(location=[39.0, -105.5], zoom_start=7, tiles="cartodbpositron")

        def add_polygon_layer(gdf, data_col, name, cmap_name):
            values = gdf[data_col].dropna()
            if values.empty:
                return
            min_val, max_val = float(values.min()), float(values.max())
            if min_val == max_val:
                max_val = min_val + 1e-9  # avoid identical thresholds

            try:
                colormap = getattr(cm.linear, cmap_name).scale(min_val, max_val)
            except AttributeError:
                raise ValueError(f"Colormap '{cmap_name}' not found in branca.colormap.linear")

            colormap.caption = name
            fg = folium.FeatureGroup(name=name, show=False)

            for _, row in gdf.iterrows():
                fill_col = '#ffffff' if pd.isna(row[data_col]) else colormap(row[data_col])
                folium.GeoJson(
                    row['geometry'],
                    style_function=lambda x, fill_col=fill_col: {
                        'fillColor': fill_col,
                        'color': 'black',
                        'weight': 0.3,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.Tooltip(f"WWTP: {row['wwtp']}<br>{name}: {row[data_col]:.2f}")
                ).add_to(fg)

            fg.add_to(m)
            colormap.add_to(m)

        add_polygon_layer(merged_gdf, 'inflow', 'Inflow Volume', 'Blues_09')
        add_polygon_layer(merged_gdf, 'outflow', 'Outflow Volume', 'Oranges_09')
        add_polygon_layer(merged_gdf, 'total_flow', 'Total Flow', 'Purples_09')
        add_polygon_layer(merged_gdf, 'flow_pop_ratio', 'Flow/Population Ratio', 'YlGnBu_09')

        folium.LayerControl(collapsed=False).add_to(m)

        out_html = os.path.join(output_folder, f"flow_maps_{week}.html")
        m.save(out_html)
        print(f"Saved {out_html}")




# ------------------------------------------------------------
# OD traffic statistics
# ------------------------------------------------------------
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def _canonical_wwtp(values):
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _canonical_county(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def _parse_date_or_none(value):
    if value in (None, "", "None", "none"):
        return None
    return pd.to_datetime(value, errors="raise").normalize()


def _weekly_keys(weekly_od_dir):
    return sorted({
        fname.replace("weekly_o_", "").replace("weekly_d_", "").replace(".csv", "")
        for fname in os.listdir(weekly_od_dir)
        if fname.startswith("weekly_o_") or fname.startswith("weekly_d_")
    })


def load_weekly_county_edges(weekly_od_dir, start_date=None, end_date=None):
    """
    Read weekly_o_* and weekly_d_* files into a common long table.

    Volume is already a weekly mean-daily OD volume. The date window is
    applied to the Monday week-start date encoded in each filename.
    """
    start = _parse_date_or_none(start_date)
    end = _parse_date_or_none(end_date)
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date.")

    rows = []
    for week_key in _weekly_keys(weekly_od_dir):
        week = pd.to_datetime(week_key, errors="coerce")
        if pd.isna(week):
            print(f"[WARN] Skipping unrecognized week key: {week_key}")
            continue
        week = week.normalize()
        if start is not None and week < start:
            continue
        if end is not None and week > end:
            continue

        direction_specs = [
            ("d", "Sewershed-D", "county_to_wwtp"),
            ("o", "Sewershed-O", "wwtp_to_county"),
        ]
        for code, wwtp_col, direction in direction_specs:
            fp = os.path.join(weekly_od_dir, f"weekly_{code}_{week_key}.csv")
            if not os.path.exists(fp):
                continue

            df = pd.read_csv(fp)
            required = {wwtp_col, "County", "Volume"}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"Missing columns {sorted(missing)} in {fp}")

            for col in ["Volume", "Area", "Population"]:
                if col not in df.columns:
                    df[col] = float("nan")
                df[col] = pd.to_numeric(df[col], errors="coerce")

            tmp = df[[wwtp_col, "County", "Volume", "Area", "Population"]].copy()
            tmp = tmp.rename(columns={wwtp_col: "wwtp", "County": "county_fips"})
            tmp["wwtp"] = _canonical_wwtp(tmp["wwtp"])
            tmp["county_fips"] = _canonical_county(tmp["county_fips"])
            tmp["direction"] = direction
            tmp["week"] = week
            rows.append(tmp)

    if not rows:
        raise ValueError(
            "No weekly OD records were found in the requested date window. "
            "Confirm that weekly_o_* and weekly_d_* files exist for those years."
        )

    edges = pd.concat(rows, ignore_index=True)
    edges = (
        edges.groupby(
            ["week", "direction", "wwtp", "county_fips"],
            as_index=False,
            dropna=False,
        )[["Volume", "Area", "Population"]]
        .sum(min_count=1)
        .sort_values(["week", "direction", "wwtp", "county_fips"])
        .reset_index(drop=True)
    )

    print(
        f"[OD stats] Loaded {edges['week'].nunique()} weeks from "
        f"{edges['week'].min().date()} through {edges['week'].max().date()}."
    )
    return edges


def build_weekly_wwtp_totals(weekly_edges):
    volume = (
        weekly_edges.groupby(["week", "wwtp", "direction"], as_index=False)["Volume"]
        .sum(min_count=1)
    )
    totals = (
        volume.pivot(index=["week", "wwtp"], columns="direction", values="Volume")
        .reset_index()
    )
    totals.columns.name = None
    totals = totals.rename(columns={
        "county_to_wwtp": "total_in",
        "wwtp_to_county": "total_out",
    })
    for col in ["total_in", "total_out"]:
        if col not in totals.columns:
            totals[col] = 0.0
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0.0)

    totals["total_flow"] = totals["total_in"] + totals["total_out"]
    totals["net_flow"] = totals["total_out"] - totals["total_in"]
    return totals.sort_values(["wwtp", "week"]).reset_index(drop=True)


def _add_period_fields(df):
    out = df.copy()
    out["week"] = pd.to_datetime(out["week"]).dt.normalize()
    out["year"] = out["week"].dt.year
    out["month"] = out["week"].dt.month
    out["year_month"] = out["week"].dt.to_period("M").astype(str)
    out["quarter"] = "Q" + out["week"].dt.quarter.astype(str)
    out["year_quarter"] = out["year"].astype(str) + "-" + out["quarter"]
    out["season"] = out["month"].map(SEASON_MAP)
    out["season_year"] = out["year"]
    out.loc[out["month"] == 12, "season_year"] += 1
    out["year_season"] = out["season_year"].astype(str) + "-" + out["season"]
    return out


def _summarize_wwtp_traffic(df, group_cols):
    return (
        df.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            total_in_mean=("total_in", "mean"),
            total_in_min=("total_in", "min"),
            total_in_max=("total_in", "max"),
            total_out_mean=("total_out", "mean"),
            total_out_min=("total_out", "min"),
            total_out_max=("total_out", "max"),
            total_flow_mean=("total_flow", "mean"),
            total_flow_min=("total_flow", "min"),
            total_flow_max=("total_flow", "max"),
            n_weeks=("week", "nunique"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_wwtp_period_stats(weekly_totals):
    data = _add_period_fields(weekly_totals)
    return {
        "monthly": _summarize_wwtp_traffic(
            data, ["wwtp", "year", "month", "year_month"]
        ),
        "quarterly": _summarize_wwtp_traffic(
            data, ["wwtp", "year", "quarter", "year_quarter"]
        ),
        "seasonal": _summarize_wwtp_traffic(
            data, ["wwtp", "season_year", "season", "year_season"]
        ),
        "annual": _summarize_wwtp_traffic(data, ["wwtp", "year"]),
    }


def _complete_county_week_grid(weekly_edges):
    """
    Add zero-volume rows for county links absent in an observed WWTP-direction week.

    This keeps the sum of county-level monthly means consistent with the WWTP
    monthly total mean. An absent link in an existing site-week is interpreted
    as zero traffic.
    """
    pieces = []
    for (direction, wwtp), sub in weekly_edges.groupby(["direction", "wwtp"], sort=True):
        site_weeks = pd.Index(sorted(pd.to_datetime(sub["week"]).unique()), name="week")
        counties = pd.Index(sorted(sub["county_fips"].astype(str).unique()), name="county_fips")
        grid = pd.MultiIndex.from_product([site_weeks, counties], names=["week", "county_fips"])

        indexed = sub.set_index(["week", "county_fips"])
        indexed.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(indexed.index.get_level_values(0)),
                indexed.index.get_level_values(1),
            ],
            names=["week", "county_fips"],
        )
        reindexed = indexed.reindex(grid)
        reindexed["Volume"] = pd.to_numeric(reindexed["Volume"], errors="coerce").fillna(0.0)
        reindexed["direction"] = direction
        reindexed["wwtp"] = wwtp
        pieces.append(reindexed.reset_index())

    return pd.concat(pieces, ignore_index=True)


def load_county_name_lookup(county_boundary_fp=None):
    """Return a county_fips -> county_name mapping for the county-source exports."""
    if not county_boundary_fp or not os.path.exists(county_boundary_fp):
        print("[WARN] County boundary file not found; county_name will be blank.")
        return {}

    county_gdf = gpd.read_file(county_boundary_fp)

    fips_candidates = ["US_FIPS", "GEOID", "FIPS", "COUNTYFP"]
    name_candidates = ["LABEL", "NAME", "County", "COUNTYNAME"]

    fips_col = next((c for c in fips_candidates if c in county_gdf.columns), None)
    name_col = next((c for c in name_candidates if c in county_gdf.columns), None)

    if fips_col is None or name_col is None:
        print(
            "[WARN] Could not identify county FIPS/name fields in county boundary; "
            "county_name will be blank."
        )
        return {}

    lookup = county_gdf[[fips_col, name_col]].drop_duplicates().copy()
    lookup["county_fips"] = _canonical_county(lookup[fips_col])
    lookup["county_name"] = lookup[name_col].astype(str).str.strip()

    return dict(zip(lookup["county_fips"], lookup["county_name"]))


def _build_county_direction_monthly_stats(
    weekly_edges,
    direction,
    county_name_lookup=None,
):
    """
    Build monthly county-level statistics for one OD direction.

    direction="county_to_wwtp": inbound source counties -> sewershed
    direction="wwtp_to_county": outbound sewershed -> destination counties
    """
    subset = weekly_edges[weekly_edges["direction"] == direction].copy()
    if subset.empty:
        raise ValueError(f"No {direction} weekly records were found.")

    complete = _add_period_fields(_complete_county_week_grid(subset))
    county_name_lookup = county_name_lookup or {}

    if direction == "county_to_wwtp":
        mean_col = "trips_to_sewershed_mean_daily"
        min_col = "trips_to_sewershed_min_weekly_mean_daily"
        max_col = "trips_to_sewershed_max_weekly_mean_daily"
    elif direction == "wwtp_to_county":
        mean_col = "trips_to_county_mean_daily"
        min_col = "trips_to_county_min_weekly_mean_daily"
        max_col = "trips_to_county_max_weekly_mean_daily"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    result = (
        complete.groupby(
            ["wwtp", "county_fips", "year", "month", "year_month"],
            as_index=False,
            dropna=False,
        )
        .agg(
            **{
                mean_col: ("Volume", "mean"),
                min_col: ("Volume", "min"),
                max_col: ("Volume", "max"),
                "n_weeks": ("week", "nunique"),
            }
        )
        .sort_values(["wwtp", "year", "month", "county_fips"])
        .reset_index(drop=True)
    )

    result["county_name"] = result["county_fips"].map(county_name_lookup).fillna("")

    return result[[
        "wwtp",
        "county_fips",
        "county_name",
        "year",
        "month",
        "year_month",
        mean_col,
        min_col,
        max_col,
        "n_weeks",
    ]]


def build_county_source_monthly_stats(weekly_edges, county_name_lookup=None):
    """Inbound county -> sewershed monthly statistics."""
    return _build_county_direction_monthly_stats(
        weekly_edges,
        direction="county_to_wwtp",
        county_name_lookup=county_name_lookup,
    )


def build_county_destination_monthly_stats(weekly_edges, county_name_lookup=None):
    """Outbound sewershed -> county monthly statistics."""
    return _build_county_direction_monthly_stats(
        weekly_edges,
        direction="wwtp_to_county",
        county_name_lookup=county_name_lookup,
    )


def combine_county_direction_monthly_stats(
    county_sources_monthly,
    county_destinations_monthly,
):
    """
    Combine inbound and outbound monthly county statistics into one wide table.

    Output grain:
        one WWTP/sewershed x county x year_month

    Inbound fields describe county -> sewershed traffic.
    Outbound fields describe sewershed -> county traffic and use the
    clearer trips_to_county_* naming convention.
    """
    keys = ["wwtp", "county_fips", "year", "month", "year_month"]

    inbound = county_sources_monthly.copy().rename(
        columns={
            "county_name": "county_name_inbound",
            "n_weeks": "n_weeks_to_sewershed",
        }
    )
    outbound = county_destinations_monthly.copy().rename(
        columns={
            "county_name": "county_name_outbound",
            "n_weeks": "n_weeks_to_county",
        }
    )

    combined = inbound.merge(
        outbound,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    combined["county_name"] = (
        combined["county_name_inbound"]
        .replace("", pd.NA)
        .combine_first(
            combined["county_name_outbound"].replace("", pd.NA)
        )
        .fillna("")
    )

    numeric_cols = [
        "trips_to_sewershed_mean_daily",
        "trips_to_sewershed_min_weekly_mean_daily",
        "trips_to_sewershed_max_weekly_mean_daily",
        "trips_to_county_mean_daily",
        "trips_to_county_min_weekly_mean_daily",
        "trips_to_county_max_weekly_mean_daily",
        "n_weeks_to_sewershed",
        "n_weeks_to_county",
    ]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(
            combined.get(col), errors="coerce"
        ).fillna(0.0)

    combined["n_weeks_to_sewershed"] = (
        combined["n_weeks_to_sewershed"].round().astype(int)
    )
    combined["n_weeks_to_county"] = (
        combined["n_weeks_to_county"].round().astype(int)
    )

    return combined[[
        "wwtp",
        "county_fips",
        "county_name",
        "year",
        "month",
        "year_month",
        "trips_to_sewershed_mean_daily",
        "trips_to_sewershed_min_weekly_mean_daily",
        "trips_to_sewershed_max_weekly_mean_daily",
        "n_weeks_to_sewershed",
        "trips_to_county_mean_daily",
        "trips_to_county_min_weekly_mean_daily",
        "trips_to_county_max_weekly_mean_daily",
        "n_weeks_to_county",
    ]].sort_values(
        ["wwtp", "year", "month", "county_fips"]
    ).reset_index(drop=True)


def _find_target_wwtp(values, requested):
    if requested in (None, ""):
        return None
    requested_clean = _canonical_wwtp(pd.Series([requested])).iloc[0]
    available = sorted(pd.Series(values).dropna().astype(str).unique())
    if requested_clean in available:
        return requested_clean

    contains = [name for name in available if requested_clean in name or name in requested_clean]
    if len(contains) == 1:
        print(f"[OD stats] Target matched to: {contains[0]}")
        return contains[0]

    print(f"[WARN] Target WWTP '{requested_clean}' was not uniquely matched.")
    candidates = [name for name in available if "metro" in name or "clear creek" in name]
    if candidates:
        print("[WARN] Possible matches:")
        for name in candidates:
            print("   ", name)
    return None


def export_od_traffic_statistics(
    weekly_od_dir,
    output_dir="outputs/od_traffic_stats",
    start_date=None,
    end_date=None,
    target_wwtp=None,
    county_boundary_fp=None,
):
    """
    Export weekly totals and monthly/quarterly/seasonal/annual statistics.

    Mean/min/max statistics are calculated across weekly mean-daily values;
    they are not extrema of individual daily observations.
    """
    os.makedirs(output_dir, exist_ok=True)
    by_sewershed_dir = os.path.join(output_dir, "by_sewershed")
    os.makedirs(by_sewershed_dir, exist_ok=True)

    weekly_edges = load_weekly_county_edges(
        weekly_od_dir=weekly_od_dir,
        start_date=start_date,
        end_date=end_date,
    )
    weekly_totals = build_weekly_wwtp_totals(weekly_edges)
    period_stats = build_wwtp_period_stats(weekly_totals)
    county_name_lookup = load_county_name_lookup(county_boundary_fp)
    county_sources_monthly = build_county_source_monthly_stats(
        weekly_edges,
        county_name_lookup=county_name_lookup,
    )
    county_destinations_monthly = build_county_destination_monthly_stats(
        weekly_edges,
        county_name_lookup=county_name_lookup,
    )
    county_traffic_monthly = combine_county_direction_monthly_stats(
        county_sources_monthly,
        county_destinations_monthly,
    )

    weekly_edges.to_csv(
        os.path.join(output_dir, "sewershed_county_weekly_mean_daily.csv"),
        index=False,
    )
    weekly_totals.to_csv(
        os.path.join(output_dir, "wwtp_weekly_mean_daily_totals.csv"),
        index=False,
    )

    for period, table in period_stats.items():
        table.to_csv(
            os.path.join(output_dir, f"wwtp_{period}_traffic_stats.csv"),
            index=False,
        )

    # All-sewershed inbound and outbound county tables.
    county_sources_monthly.to_csv(
        os.path.join(output_dir, "county_to_sewershed_monthly_traffic_stats.csv"),
        index=False,
    )
    county_destinations_monthly.to_csv(
        os.path.join(output_dir, "sewershed_to_county_monthly_traffic_stats.csv"),
        index=False,
    )

    # Combined county-level table for all sewersheds.
    county_traffic_monthly.to_csv(
        os.path.join(
            output_dir,
            "sewershed_county_monthly_traffic_stats.csv",
        ),
        index=False,
    )

    # One combined inbound + outbound file per sewershed.
    manifest = []
    for wwtp in sorted(county_traffic_monthly["wwtp"].unique()):
        safe_wwtp = safe_filename(wwtp)
        combined_fp = os.path.join(
            by_sewershed_dir,
            f"county_traffic_{safe_wwtp}.csv",
        )

        county_traffic_monthly[
            county_traffic_monthly["wwtp"] == wwtp
        ].to_csv(combined_fp, index=False)

        manifest.append({
            "wwtp": wwtp,
            "county_traffic_filepath": combined_fp,
        })

    pd.DataFrame(manifest).to_csv(
        os.path.join(output_dir, "county_traffic_file_manifest.csv"),
        index=False,
    )

    monthly_totals = period_stats["monthly"]

    # QA 1: inbound county means should sum to monthly total_in_mean.
    county_source_sum = (
        county_sources_monthly.groupby(
            ["wwtp", "year", "month", "year_month"],
            as_index=False,
        )["trips_to_sewershed_mean_daily"]
        .sum()
        .rename(columns={
            "trips_to_sewershed_mean_daily": "county_source_sum"
        })
    )
    qa_in = monthly_totals[
        ["wwtp", "year", "month", "year_month", "total_in_mean"]
    ].merge(
        county_source_sum,
        on=["wwtp", "year", "month", "year_month"],
        how="outer",
        validate="one_to_one",
    )
    qa_in["difference"] = qa_in["county_source_sum"] - qa_in["total_in_mean"]
    qa_in.to_csv(
        os.path.join(output_dir, "county_source_vs_total_in_QA.csv"),
        index=False,
    )

    # QA 2: outbound county means should sum to monthly total_out_mean.
    county_destination_sum = (
        county_destinations_monthly.groupby(
            ["wwtp", "year", "month", "year_month"],
            as_index=False,
        )["trips_to_county_mean_daily"]
        .sum()
        .rename(columns={
            "trips_to_county_mean_daily": "county_destination_sum"
        })
    )
    qa_out = monthly_totals[
        ["wwtp", "year", "month", "year_month", "total_out_mean"]
    ].merge(
        county_destination_sum,
        on=["wwtp", "year", "month", "year_month"],
        how="outer",
        validate="one_to_one",
    )
    qa_out["difference"] = (
        qa_out["county_destination_sum"] - qa_out["total_out_mean"]
    )
    qa_out.to_csv(
        os.path.join(output_dir, "county_destination_vs_total_out_QA.csv"),
        index=False,
    )

    target = _find_target_wwtp(weekly_totals["wwtp"], target_wwtp)
    if target is not None:
        target_dir = os.path.join(output_dir, f"target_{safe_filename(target)}")
        os.makedirs(target_dir, exist_ok=True)
        for period, table in period_stats.items():
            table[table["wwtp"] == target].to_csv(
                os.path.join(target_dir, f"{safe_filename(target)}_{period}_traffic_stats.csv"),
                index=False,
            )
        county_traffic_monthly[
            county_traffic_monthly["wwtp"] == target
        ].to_csv(
            os.path.join(
                target_dir,
                f"county_traffic_{safe_filename(target)}.csv",
            ),
            index=False,
        )

    max_diff_in = pd.to_numeric(
        qa_in["difference"], errors="coerce"
    ).abs().max()
    max_diff_out = pd.to_numeric(
        qa_out["difference"], errors="coerce"
    ).abs().max()
    print(f"[OD stats] Saved outputs to: {output_dir}")
    print(
        f"[OD stats QA] max |county source sum - total_in_mean| = {max_diff_in}"
    )
    print(
        f"[OD stats QA] max |county destination sum - total_out_mean| = {max_diff_out}"
    )
    return {
        "weekly_edges": weekly_edges,
        "weekly_totals": weekly_totals,
        "period_stats": period_stats,
        "county_sources_monthly": county_sources_monthly,
        "county_destinations_monthly": county_destinations_monthly,
        "county_traffic_monthly": county_traffic_monthly,
        "qa_in": qa_in,
        "qa_out": qa_out,
    }



def plot_yoy_flow_lines(
    weekly_od_dir,
    output_folder="outputs/yoy_flow_lines",
    start_date=None,
    end_date=None,
):
    """
    Per-WWTP monthly mean-daily inflow/outflow/total-flow lines across years.

    Monthly values use MEAN across weekly mean-daily values, not SUM.
    """
    os.makedirs(output_folder, exist_ok=True)

    weekly_edges = load_weekly_county_edges(
        weekly_od_dir=weekly_od_dir,
        start_date=start_date,
        end_date=end_date,
    )
    weekly_totals = _add_period_fields(build_weekly_wwtp_totals(weekly_edges))

    monthly = (
        weekly_totals.groupby(["wwtp", "year", "month"], as_index=False)
        .agg(
            inflow=("total_in", "mean"),
            outflow=("total_out", "mean"),
            total_flow=("total_flow", "mean"),
            n_weeks=("week", "nunique"),
        )
        .sort_values(["wwtp", "year", "month"])
    )

    monthly_out = os.path.join(output_folder, "yoy_monthly_mean_daily_flows.csv")
    monthly.to_csv(monthly_out, index=False)

    month_idx = list(range(1, 13))
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    for wwtp_id, group in monthly.groupby("wwtp"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        metrics = ["inflow", "outflow", "total_flow"]
        titles = ["Inbound", "Outbound", "Total"]

        for ax, metric, title in zip(axes, metrics, titles):
            for year, yearly in group.groupby("year"):
                yearly = yearly.set_index("month").reindex(month_idx).reset_index()
                ax.plot(yearly["month"], yearly[metric], marker="o", label=str(year))
            ax.set_title(title)
            ax.set_xlabel("Month")
            ax.set_xticks(month_idx)
            ax.set_xticklabels(month_labels)
            ax.set_ylabel("Mean-daily trip volume")
            ax.grid(True, alpha=0.3)
            if metric == "total_flow":
                ax.legend(title="Year", ncol=2, fontsize=9)

        fig.suptitle(f"Monthly mean-daily traffic — {wwtp_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = os.path.join(output_folder, f"yoy_flow_{safe_filename(wwtp_id)}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved:", out_path)

    print("Monthly mean-daily CSV saved:", monthly_out)


# === MAIN CALL ===
def run_od_analysis():
    # ============================================================
    # EDIT SETTINGS HERE — no command-line arguments are required.
    # ============================================================
    DAILY_DIR = "../ZoneSelection/Input/ODData/Daily"
    WEEKLY_OD_DIR = "../ZoneSelection/Outfile/ODData/Weekly"
    STATS_OUTPUT_DIR = "outputs/od_traffic_stats"
    YOY_PLOT_DIR = "outputs/yoy_flow_lines"
    COUNTY_BOUNDARY_FP = "../ZoneSelection/Input/Census/COCounty.shp"

    # Use None to include every available year. Examples:
    # START_DATE = "2023-01-01"
    # END_DATE = "2025-12-31"
    START_DATE = None
    END_DATE = None

    # Optional convenience export. All-WWTP summary files are always created.
    TARGET_WWTP = "metro - clear creek"

    # Set True after Process_ODData_fixed.py creates/replaces the daily files.
    # This rebuilds weekly_o_* and weekly_d_* from Daily using weekly daily means.
    REBUILD_WEEKLY_FROM_DAILY = False

    RUN_TRAFFIC_STATS = True
    RUN_YOY_PLOTS = True

    os.makedirs(WEEKLY_OD_DIR, exist_ok=True)

    if REBUILD_WEEKLY_FROM_DAILY:
        print("[OD] Rebuilding weekly files from daily county-WWTP files...")
        aggregate_od_weekly(DAILY_DIR, WEEKLY_OD_DIR)

    if RUN_TRAFFIC_STATS:
        export_od_traffic_statistics(
            weekly_od_dir=WEEKLY_OD_DIR,
            output_dir=STATS_OUTPUT_DIR,
            start_date=START_DATE,
            end_date=END_DATE,
            target_wwtp=TARGET_WWTP,
            county_boundary_fp=COUNTY_BOUNDARY_FP,
        )

    if RUN_YOY_PLOTS:
        plot_yoy_flow_lines(
            weekly_od_dir=WEEKLY_OD_DIR,
            output_folder=YOY_PLOT_DIR,
            start_date=START_DATE,
            end_date=END_DATE,
        )


if __name__ == "__main__":
    run_od_analysis()
