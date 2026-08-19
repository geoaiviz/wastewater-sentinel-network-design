"""Run Level 2 DCRNN evaluation, OD verification, and output visualization."""
# Now includes: (A) RAW transmission sentinel ranking (no row-normalization, diag skipped)
#               (B) SELF-IMPACT-only sentinel ranking (diag only)

import os, csv, itertools, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from statsmodels.tools.sm_exceptions import SpecificationWarning
import geopandas as gpd
warnings.filterwarnings("ignore", category=SpecificationWarning)
from Model_DCRNN_Simple_newdata import CountyWWTP_DCRNN
try:
    import Model_DCRNN_Viz as viz
except Exception:
    viz = None

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

CORR_THRESH_DEFAULT = 0.40

# -------------------- WWTP↔County ranking helpers (legacy) --------------------
def _safe_row_norm(A):
    A = np.array(A, dtype=float)
    rs = A.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return A / rs

def _split_nodes(node_names):
    idx_W = [i for i,n in enumerate(node_names) if str(n).startswith("W_")]
    idx_C = [i for i,n in enumerate(node_names) if str(n).startswith("C_")]
    W_names = [node_names[i][2:] for i in idx_W]  # strip "W_"
    C_names = [node_names[i][2:] for i in idx_C]  # strip "C_"
    return idx_W, idx_C, W_names, C_names


def _load_wwtp_pop_map(wwtp_shapefile):
    """
    Build {WWTPName->pop_served} from the WWTP shapefile.
    Prefers 'pop_Served' but falls back to common variants.
    Name matching is normalized to Title Case for robustness.
    """
    try:
        gdf = gpd.read_file(wwtp_shapefile)
    except Exception:
        return {}

    # pick a reasonable "name" column
    name_cols = [c for c in ["WWTP","wwtp","Facility","FACILITY","NAME","Name","Plant","plant","fac_name","site_name"]
                 if c in gdf.columns]
    name_col = name_cols[0] if name_cols else next((c for c in gdf.columns if gdf[c].dtype == object), None)
    if name_col is None:
        return {}

    # prefer pop_Served; allow common variants
    pop_candidates = ["pop_Served","pop_served","Pop_Served","POP_SERVED","population_served","POP","Pop"]
    pop_col = next((c for c in pop_candidates if c in gdf.columns), None)
    if pop_col is None:
        # pick first numeric as a last resort
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        if not num_cols:
            return {}
        pop_col = num_cols[0]

    df = gdf[[name_col, pop_col]].copy()
    df["NameClean"] = df[name_col].astype(str).str.replace("_"," ").str.strip().str.title()
    df["PopServed"] = pd.to_numeric(df[pop_col], errors="coerce")
    df = df.dropna(subset=["PopServed"])
    return dict(zip(df["NameClean"], df["PopServed"].astype(float)))


def _load_county_pop_map(county_pop_csv, county_polygon_shp=None):
    """
    Load county population into a {CountyName -> POP} map.

    Robust to pop CSVs that ONLY contain FIPS + POP (no CountyName).
    If CountyName is missing/empty, we try to map FIPS->CountyName using:
      1) county_polygon_shp (preferred, local)
      2) CO Census county codes URL (runtime internet)
    """
    import os
    import pandas as pd

    if (not county_pop_csv) or (not os.path.exists(county_pop_csv)):
        return {}

    df = pd.read_csv(county_pop_csv)

    # detect FIPS
    fips_col = next((c for c in ["FIPS","fips","GEOID","GEOID10","CountyFIPS","county_fips","FIPS5"] if c in df.columns), None)
    name_col = next((c for c in ["CountyName","county_name","NAME","County","NAMELSAD"] if c in df.columns), None)

    # detect POP
    pop_col = next((c for c in ["POP","Pop","population","Population","Total_Population","TOTAL_POPULATION","pop"] if c in df.columns), None)
    if pop_col is None:
        num_cols = [c for c in df.columns if c not in (fips_col, name_col) and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return {}
        pop_col = num_cols[0]

    df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")
    df = df.dropna(subset=[pop_col]).copy()

    # build fips->name map if needed
    fips_to_name = {}
    if fips_col is not None:
        df[fips_col] = (df[fips_col].astype(str).str.strip().str.replace(r"\.0+$","",regex=True).str.zfill(5))

        # 1) county polygon shapefile (preferred)
        try:
            import geopandas as gpd
            if county_polygon_shp and os.path.exists(county_polygon_shp):
                gdf = gpd.read_file(county_polygon_shp)
                g_name = next((c for c in ["NAME","CountyName","COUNTY","COUNTYNAME","NAMELSAD"] if c in gdf.columns), None)
                g_fips = next((c for c in ["GEOID","GEOID10","FIPS","FIPS5"] if c in gdf.columns), None)
                if g_fips is None and ("STATEFP" in gdf.columns) and ("COUNTYFP" in gdf.columns):
                    gdf["__FIPS5"] = (gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)).astype(str)
                    g_fips = "__FIPS5"
                if g_name and g_fips:
                    tmp = gdf[[g_fips, g_name]].copy()
                    tmp[g_fips] = tmp[g_fips].astype(str).str.replace(r"\.0+$","",regex=True).str.zfill(5)
                    tmp[g_name] = (
                        tmp[g_name].astype(str).str.strip()
                        .str.replace(r"\s+county$","",case=False,regex=True)
                        .str.strip().str.title()
                    )
                    fips_to_name = dict(zip(tmp[g_fips], tmp[g_name]))
        except Exception:
            pass

        # 2) fallback URL
        if not fips_to_name:
            try:
                look = pd.read_csv(
                    "https://www2.census.gov/geo/docs/reference/codes/files/st08_co_cou.txt",
                    header=None, dtype=str
                )
                look.columns = ["State", "StateFP", "CountyFP", "CountyName", "ClassCode"]
                look["FIPS"] = (look["StateFP"] + look["CountyFP"]).astype(str).str.zfill(5)
                look["CountyClean"] = (
                    look["CountyName"].str.strip()
                    .str.replace(r"\s+county$","",case=False,regex=True)
                    .str.strip().str.title()
                )
                fips_to_name = dict(zip(look["FIPS"], look["CountyClean"]))
            except Exception:
                pass

    # county name (prefer explicit column if usable)
    if name_col is not None:
        df["CountyNameClean"] = (
            df[name_col].astype(str).str.strip()
            .str.replace(r"\s+county$","",case=False,regex=True)
            .str.replace(r"\s+parish$","",case=False,regex=True)
            .str.replace(r"\s+borough$","",case=False,regex=True)
            .str.replace(r"\s+municipality$","",case=False,regex=True)
            .str.strip().str.title()
        )
    elif fips_col is not None and fips_to_name:
        df["CountyNameClean"] = df[fips_col].map(fips_to_name).fillna(df[fips_col])
    else:
        return {}

    out = df[["CountyNameClean", pop_col]].dropna().copy()
    return dict(zip(out["CountyNameClean"], out[pop_col].astype(float)))


def _maybe_weight_by_stability(block, agree_block, std_block):
    w = np.ones_like(block)
    if (agree_block is not None) and (std_block is not None):
        A = np.asarray(agree_block, float)
        S = np.asarray(std_block, float)
        finite_std = S[np.isfinite(S)]
        if finite_std.size:
            q80 = np.quantile(finite_std, 0.80)
            std_w = 1.0 - np.clip(S / (q80 if q80 > 0 else 1.0), 0, 1)
        else:
            std_w = np.ones_like(S)
        w = np.clip(A, 0, 1) * std_w
    return w

def _maybe_weight_by_population(block, names_axis, pop_map, axis="target"):
    if not pop_map:
        return np.ones_like(block)
    if axis == "target":
        vec = np.array([pop_map.get(c.replace("_"," ").title(), np.nan) for c in names_axis], float)
    else:
        vec = np.array([pop_map.get(c.replace("_"," ").title(), np.nan) for c in names_axis], float)
    if np.isfinite(vec).any():
        vec = np.nan_to_num(vec, nan=np.nanmedian(vec))
        vec = vec / (vec.mean() + 1e-9)
        return vec[None,:] if axis=="target" else vec[:,None]
    return np.ones_like(block)

def _maybe_weight_by_mobility(block, flow_adj_ref, idx_rows, idx_cols, axis="row"):
    if flow_adj_ref is None:
        return np.ones_like(block)
    F = np.array(flow_adj_ref, float).copy()
    np.fill_diagonal(F, 0.0)
    F_row = _safe_row_norm(F)
    F_sub = F_row[np.ix_(idx_rows, idx_cols)]
    return F_sub

def save_wwtp_rank_both(
    out_dir,
    node_names,
    risk_matrix,
    risk_std=None,
    agree=None,
    flow_adj_ref=None,
    county_pop_csv=None,
    wwtp_shapefile=None,   # supplies WWTP population-served attributes
    tau=0.02,
    tag="", county_polygon_shp = None,
):
    """
    Save OUTBOUND and INBOUND WWTP ranks.
    Weights applied:
      - Stability (agreement/std)
      - Mobility (row-normalized flow adjacency on the corresponding block)
      - Population:
          * OUTBOUND rows (WWTP)   -> pop_Served from shapefile
          * OUTBOUND cols (County) -> county_uncovered_pop.csv
          * INBOUND  rows (County) -> county_uncovered_pop.csv
          * INBOUND  cols (WWTP)   -> pop_Served from shapefile
    """
    import os
    import numpy as np
    import pandas as pd

    # The following ranking helpers are defined in this module:
    # _split_nodes, _safe_row_norm, _maybe_weight_by_stability,
    # _maybe_weight_by_population, _maybe_weight_by_mobility, _load_county_pop_map

    # Fallback local loader for WWTP pop if a global one isn't present
    def _load_wwtp_pop_map_local(wwtp_shp):
        try:
            import geopandas as gpd
        except Exception:
            return {}
        try:
            gdf = gpd.read_file(wwtp_shp)
        except Exception:
            return {}
        # pick a reasonable "name" column
        name_cols = [c for c in ["WWTP","wwtp","Facility","FACILITY","NAME","Name","Plant","plant","fac_name","site_name"]
                     if c in gdf.columns]
        name_col = name_cols[0] if name_cols else next((c for c in gdf.columns if gdf[c].dtype == object), None)
        if name_col is None:
            return {}
        # prefer pop_Served; allow common variants
        pop_candidates = ["pop_Served","pop_served","Pop_Served","POP_SERVED","population_served","POP","Pop"]
        pop_col = next((c for c in pop_candidates if c in gdf.columns), None)
        if pop_col is None:
            # first numeric as last resort
            num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
            if not num_cols:
                return {}
            pop_col = num_cols[0]
        df = gdf[[name_col, pop_col]].copy()
        df["NameClean"] = df[name_col].astype(str).str.replace("_"," ").str.strip().str.title()
        df["PopServed"] = pd.to_numeric(df[pop_col], errors="coerce")
        df = df.dropna(subset=["PopServed"])
        return dict(zip(df["NameClean"], df["PopServed"].astype(float)))

    os.makedirs(out_dir, exist_ok=True)

    # split nodes
    idx_W, idx_C, W_names_raw, C_names_raw = _split_nodes(node_names)
    # normalized names for lookup
    W_names = [str(n).replace("_"," ").strip().title() for n in W_names_raw]
    C_names = [str(n).replace("_"," ").strip().title() for n in C_names_raw]

    R = np.asarray(risk_matrix, float)

    # load population maps
    pop_map_county = _load_county_pop_map(county_pop_csv, county_polygon_shp=county_polygon_shp) if county_pop_csv else {}
    if callable(globals().get("_load_wwtp_pop_map", None)):
        pop_map_wwtp = _load_wwtp_pop_map(wwtp_shapefile) if wwtp_shapefile else {}
    else:
        pop_map_wwtp = _load_wwtp_pop_map_local(wwtp_shapefile) if wwtp_shapefile else {}

    # ==================== OUTBOUND: WWTP (rows) -> County (cols) ====================
    R_wc = R[np.ix_(idx_W, idx_C)]
    A_wc = agree[np.ix_(idx_W, idx_C)] if agree is not None else None
    S_wc = risk_std[np.ix_(idx_W, idx_C)] if risk_std is not None else None

    w_stab_wc = _maybe_weight_by_stability(R_wc, A_wc, S_wc)  # agreement/std-based
    w_mob_wc  = _maybe_weight_by_mobility(R_wc, flow_adj_ref, idx_W, idx_C, axis="row")

    # Rows use WWTP population served; columns use county population.
    w_pop_row_wc = _maybe_weight_by_population(R_wc, W_names, pop_map_wwtp,   axis="source")
    w_pop_col_wc = _maybe_weight_by_population(R_wc, C_names, pop_map_county, axis="target")

    W_wc = w_stab_wc * w_mob_wc * w_pop_row_wc * w_pop_col_wc

    Rn_wc = _safe_row_norm(R_wc)
    Rw_wc = _safe_row_norm(R_wc * W_wc)

    # OUTBOUND summary table (rank WWTP by outbound impact)
    out_df = pd.DataFrame({
        "wwtp": W_names,
        "score_unweighted": Rn_wc.sum(axis=1),
        "score_weighted":   Rw_wc.sum(axis=1),
        f"diversity_ge_{tau:g}": (Rn_wc >= float(tau)).sum(axis=1)
    }).sort_values("score_weighted", ascending=False).reset_index(drop=True)

    prefix_o = f"wwtp_rank_OUT_{tag}_" if tag else "wwtp_rank_OUT_"
    out_df.to_csv(os.path.join(out_dir, f"{prefix_o}table.csv"), index=False)
    pd.DataFrame(R_wc,  index=W_names, columns=C_names).to_csv(os.path.join(out_dir, f"{prefix_o}risk_W_to_C_raw.csv"))
    pd.DataFrame(Rn_wc, index=W_names, columns=C_names).to_csv(os.path.join(out_dir, f"{prefix_o}risk_W_to_C_rownorm.csv"))
    pd.DataFrame(Rw_wc, index=W_names, columns=C_names).to_csv(os.path.join(out_dir, f"{prefix_o}risk_W_to_C_WEIGHTED.csv"))

    # ==================== INBOUND: County (rows) -> WWTP (cols) =====================
    R_cw = R[np.ix_(idx_C, idx_W)]
    A_cw = agree[np.ix_(idx_C, idx_W)] if agree is not None else None
    S_cw = risk_std[np.ix_(idx_C, idx_W)] if risk_std is not None else None

    w_stab_cw = _maybe_weight_by_stability(R_cw, A_cw, S_cw)
    w_mob_cw  = _maybe_weight_by_mobility(R_cw, flow_adj_ref, idx_C, idx_W, axis="row")

    # Rows use county population; columns use WWTP population served.
    w_pop_row_cw = _maybe_weight_by_population(R_cw, C_names, pop_map_county, axis="source")
    w_pop_col_cw = _maybe_weight_by_population(R_cw, W_names, pop_map_wwtp,   axis="target")

    W_cw = w_stab_cw * w_mob_cw * w_pop_row_cw * w_pop_col_cw

    Rn_cw = _safe_row_norm(R_cw)
    Rw_cw = _safe_row_norm(R_cw * W_cw)

    inbound_unweighted = Rn_cw.sum(axis=0)
    inbound_weighted   = Rw_cw.sum(axis=0)

    in_df = pd.DataFrame({
        "wwtp": W_names,
        "score_unweighted": inbound_unweighted,
        "score_weighted":   inbound_weighted
    }).sort_values("score_weighted", ascending=False).reset_index(drop=True)

    prefix_i = f"wwtp_rank_IN_{tag}_" if tag else "wwtp_rank_IN_"
    in_df.to_csv(os.path.join(out_dir, f"{prefix_i}table.csv"), index=False)
    pd.DataFrame(R_cw,  index=C_names, columns=W_names).to_csv(os.path.join(out_dir, f"{prefix_i}risk_C_to_W_raw.csv"))
    pd.DataFrame(Rn_cw, index=C_names, columns=W_names).to_csv(os.path.join(out_dir, f"{prefix_i}risk_C_to_W_rownorm.csv"))
    pd.DataFrame(Rw_cw, index=C_names, columns=W_names).to_csv(os.path.join(out_dir, f"{prefix_i}risk_C_to_W_WEIGHTED.csv"))

    # Print a concise diagnostic summary.
    print("[WWTP OUTBOUND] top 10 (score_weighted):")
    print(out_df.head(10))
    print("[WWTP INBOUND] top 10 (score_weighted):")
    print(in_df.head(10))

# Raw-adjacency and self-only sensitivity helpers.
def rank_raw_transmission_no_norm(risk_matrix, node_names, top_k=20):
    """Raw, absolute magnitude: WWTP->County block, sum by row (WWTP outbound).
       No row-normalization; diagonal already skipped by using the block."""
    idx_W, idx_C, W_names, C_names = _split_nodes(node_names)
    R = np.asarray(risk_matrix, float)
    R_wc = R[np.ix_(idx_W, idx_C)]  # this excludes diag automatically
    scores = R_wc.sum(axis=1)       # outbound absolute magnitude
    df = pd.DataFrame({"wwtp": W_names, "score": scores})
    return df.sort_values("score", ascending=False).head(int(top_k)), R_wc, W_names, C_names

def rank_self_impact_only(risk_matrix_keepdiag, node_names, top_k=20):
    """Self-impact = diagonal of RAW matrix (keep-diagonal run). WWTP entries only."""
    idx_W, idx_C, W_names, _ = _split_nodes(node_names)
    R = np.asarray(risk_matrix_keepdiag, float)
    diag_vals = np.diag(R)
    ww_self = np.array([diag_vals[i] for i in idx_W], dtype=float)
    df = pd.DataFrame({"wwtp": W_names, "self_impact": ww_self})
    return df.sort_values("self_impact", ascending=False).head(int(top_k))

# -------------------- Small IO helpers --------------------
def _append_row(out_dir, row, header):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "grid_results.csv")
    write_hdr = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(header)
        w.writerow(row)

def _load_done_tags(out_dir):
    path = os.path.join(out_dir, "grid_results.csv")
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["tag"])
        return set(df["tag"].astype(str).tolist())
    except Exception:
        return set()

def _tag(conf_model, conf_graph, conf_unreg):
    sc = f"sc{int(round(conf_graph['cap_self']*100)):02d}"
    bf = "bfu" if conf_graph["backfill"].startswith("uni") else "bfprop"
    tp = f"tp{int(round(conf_graph['teleport']*100)):02d}"
    a  = f"a{int(round(conf_unreg['alpha']*10)):02d}"
    pc = f"pc{int(conf_unreg['percent_change'])}"
    vn = f"vn{int(conf_unreg['variance_normalize'])}"
    nm = f"nm{int(conf_unreg['normalize'])}"
    arn= f"arn{int(conf_graph['adj_row_norm'])}"
    mdl= f"sl{conf_model['seq_len']}_hd{conf_model['hidden_dim']}_l{conf_model['rnn_layers']}_do{int(100*conf_model['dropout'])}_l2{int(1e6*conf_model['l2_reg'])}_cos{int(conf_model['lr_cosine'])}"
    return f"{sc}_{bf}_{tp}_{a}_{pc}_{vn}_{nm}_{arn}_{mdl}"

def _iter_product(dcts):
    keys = list(dcts.keys())
    vals = list(dcts.values())
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}

# -------------------- Shared config builders --------------------
def build_shared_io(dtype):
    return dict(
        weekly_od_dir="../ZoneSelection/Outfile/ODData/Weekly/",
        county_pop_csv_uncovered="../ZoneSelection/Input/Census/county_uncovered_pop.csv",
        county_pop_csv_full="../ZoneSelection/Input/Census/CO_County_Population_FIPS5.csv",
        # backward-compatible alias (points to FULL pop)
        county_pop_csv="../ZoneSelection/Input/Census/CO_County_Population_FIPS5.csv",
        wwtp_shapefile="../ZoneSelection/Input/WWTP_CO/WWTP_Select.shp",
        wwtp_metrics_csv=f"../ZoneSelection/Input/Viral/wwtp_metrics_all_{dtype}.csv",
        county_polygon_shp="../ZoneSelection/Input/Census/COCounty.shp",
        agg="week",
        seq_len=4,
        hidden_dim=128,
        rnn_layers=2,
        epochs=300,
        batch_size=8,
        start_week="2024-01-08",
        end_week="2024-12-30",
        hosp_missing_as_zero=True,
        auto_pick_dynamic=True,
        force_keep_all_counties=True,
        min_weeks_per_node=2,
    )

def disease_cfgs(dtype):
    return {
        "Influenza": dict(
            disease="Influenza",
            flu_rsv_hosp_file="../ZoneSelection/Input/Viral/WW_clinicalhospitalized_FluRSV_DU.xlsx",
            flu_rsv_rate_col="Hospitalized_CaseCount_r100Kutil",
            wval_fp="../ZoneSelection/Input/Viral/NWSS FluA WVAL.xlsx",
            clinical_target="hospitalization",
            feature_mode="case_only",
            output_dir=f"s_outputs_flu_{dtype}",
        ),
        "RSV": dict(
            disease="RSV",
            flu_rsv_hosp_file="../ZoneSelection/Input/Viral/WW_clinicalhospitalized_FluRSV_DU.xlsx",
            flu_rsv_rate_col="Hospitalized_CaseCount_r100Kutil",
            wval_fp="../ZoneSelection/Input/Viral/NWSS RSV WVAL.xlsx",
            clinical_target="hospitalization",
            feature_mode="case_only",
            output_dir=f"s_outputs_rsv_{dtype}",
        ),
        "COVID": dict(
            disease="COVID",
            covid_case_csv="../ZoneSelection/Input/Viral/WW_clinicaldata_COVID_DU.xlsx",
            covid_rate_col="County_hosp_3dayavg_r100Kutil",
            county_wval_csv="../ZoneSelection/Input/Viral/county_weighted_viral_load.csv",
            county_wval_col="mean_ww_index_normed_ln_lin",
            wval_fp="../ZoneSelection/Input/Viral/NWSS WVAL.xlsx",
            clinical_target="hospitalization",
            feature_mode="case_only",
            output_dir=f"s_outputs_covid_{dtype}",
        ),
    }


# Output-tag helper.
def run_full_pipeline_with_viz(pipe, tag="_run1"):
    """
    Full evaluation + visualization.

    Assumes:
      - pipe.create_sequences(...) and pipe.build_model(...) already called
      - pipe.train_model(...) already run ONCE in run_one()

    Does:
      - Save pred vs true for the FULL timeline (train + test)
      - Make grouped prediction plots
      - Save metrics for train / test / all
    """
    import Model_DCRNN_Viz as viz

    # 1. Pred vs true for FULL period (80% train + 20% test)
    predtrue_csv = pipe.save_predicted_and_true(use="all", tag=tag)

    # 2. Grouped prediction plots using the full timeline
    manifest_csv = viz.plot_predictions_grouped_by_range_from_csv(
        pred_true_csv=predtrue_csv,
        out_dir=pipe.output_dir,
        names=pipe.node_names,
        title_prefix="Pred_vs_True_grouped",
        q_span=2,
        q_center=3,
        cols=4,
    )

    # 3. Metrics
    metrics_train = pipe.evaluate_predictions(use="train", output_tag=f"{tag}_train")
    metrics_test  = pipe.evaluate_predictions(use="val",   output_tag=f"{tag}_test")
    metrics_all   = pipe.evaluate_predictions(use="all",   output_tag=f"{tag}_all")

    return {
        "predtrue_csv": predtrue_csv,
        "group_manifest_csv": manifest_csv,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "metrics_all": metrics_all,
        "output_dir": pipe.output_dir,
    }

# -------------------- One run --------------------
def run_one(cfg_all, shared, corr_thresh, out_dir_override=None, viz_enabled=True,
            seed_for_tag=42, use_trend_risk=True, trend_window=3,
            trend_method="ucm", season_period_weeks=52,
            do_verify_od=True, verify_top_weeks=3):

    conf_model, conf_graph, conf_unreg, pipe_overrides = (
        cfg_all["MODEL"], cfg_all["GRAPH"], cfg_all["UNREG"], cfg_all["PIPE"]
    )

    # init
    run_kwargs = {
        **shared,
        **pipe_overrides,
        "feature_mode": pipe_overrides.get("feature_mode", "combined"),
        "seq_len": conf_model["seq_len"],
        "hidden_dim": conf_model["hidden_dim"],
        "rnn_layers": conf_model["rnn_layers"],
    }
    # pop legacy key to avoid unexpected __init__ kwarg
    run_kwargs.pop("county_pop_csv", None)
    # run_kwargs.pop("county_polygon_shp", None)

    pipe = CountyWWTP_DCRNN(**run_kwargs)
    base_out = out_dir_override or getattr(pipe, "output_dir", None) or "runs_disease"
    os.makedirs(base_out, exist_ok=True)
    pipe.output_dir = base_out

    # load
    pipe.load_data()

    # OD verify (optional)
    if do_verify_od:
        try:
            pipe.verify_od_flows(
                top_weeks=int(verify_top_weeks),
                tag=f"_{pipe_overrides['disease'].lower()}_{pipe_overrides['feature_mode'].lower()}",
                start_week=shared.get("start_week"),
                end_week=shared.get("end_week"),
                save_aggregate=True
            )
        except Exception as e:
            print("[verify_od] skipped due to error:", e)

    # features + adjacency
    pipe.build_features_and_adj()
    flow_adj_ref = np.array(pipe.adj_static, copy=True)

    # graph tweaks
    try:
        pipe.cap_self_loops(cap=float(conf_graph["cap_self"]), backfill=conf_graph["backfill"])
    except TypeError:
        pipe.cap_self_loops(cap=float(conf_graph["cap_self"]))
    except Exception:
        pass

    try:
        if float(conf_graph["teleport"]) > 0 and hasattr(pipe, "apply_adj_teleport"):
            pipe.apply_adj_teleport(alpha=float(conf_graph["teleport"]))
    except Exception:
        pass

    if conf_graph["adj_row_norm"]:
        A = np.array(pipe.adj_static, dtype=float)
        rs = A.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        pipe.adj_static = A / rs

    # trend config
    pipe.trend_window = int(trend_window)
    pipe.trend_method = trend_method
    pipe.season_period_weeks = int(season_period_weeks)
    # wwtp_gdf = viz.load_wwtp_points(shared["wwtp_shapefile"])
    # sequences + model
    pipe.create_sequences(gap_tolerance=2, require_strict=False)
    pipe.build_model(
        dropout=conf_model["dropout"],
        l2_reg=conf_model["l2_reg"],
        use_huber=conf_model["use_huber"],
        train_in_original_space=True,
        lr_cosine=conf_model["lr_cosine"],
        use_flow_reg=False,
        proxy_align_lambda=0.0,
    )

    # ---- Train once on the 80% training windows ----
    pipe.train_model(patience=20)

    # ---- Quick metrics for train & test (20%) ----
    eval_res = pipe.evaluate_predictions(
        use="train",
        output_tag=f"_{pipe_overrides['disease'].lower()}_{pipe_overrides['feature_mode']}"
    )
    pipe.evaluate_predictions(use="val", output_tag="_test")  # "val" alias -> test split

    # Optionally retain a separate evaluation-period CSV.
    pipe.save_predicted_and_true(use="val", tag="_test")

    # ---- Full-period viz & metrics (train + test) ----
    results = run_full_pipeline_with_viz(pipe, tag="_exp01")
    print("Main outputs:")
    for k, v in results.items():
        print(f"{k}: {v}")


    mae, rmse = (np.nan, np.nan)
    if isinstance(eval_res, dict) and "overall" in eval_res:
        mae, rmse = map(float, eval_res["overall"])

    # === (Legacy) transmission risk with consensus; row-normalized path kept intact ===
    risk_mean, risk_std, agree = pipe.compute_transmission_matrix(
        use="train",
        alpha=conf_unreg["alpha"],
        percent_change=conf_unreg["percent_change"],
        variance_normalize=conf_unreg["variance_normalize"],
        normalize=conf_unreg["normalize"],
        self_mode=conf_unreg["self_mode"],        # usually "skip"
        renorm_offdiag=conf_unreg["renorm_offdiag"],
        use_trend=bool(use_trend_risk),
        mc_samples=16, enable_dropout=True, return_std=True,
        raw_trend_consensus=True, consensus_rule="min", agree_tau=0.10
    )

    # Row normalization retained for compatibility with the reported ranking outputs.
    R = np.asarray(risk_mean, dtype=float)
    row_sums = R.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
    risk_unreg = R / row_sums  # row-normalized values used by the reported outputs

    # flow correlation gate on legacy risk
    A = np.array(flow_adj_ref, dtype=float)
    Fsym = 0.5 * (A + A.T); np.fill_diagonal(Fsym, 0.0)
    iu = np.triu_indices_from(Fsym, k=1)
    x, y = risk_unreg[iu], Fsym[iu]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    m = np.isfinite(x) & np.isfinite(y)
    r_flow = float(np.dot(x[m], y[m]) / max(m.sum(), 1)) if m.sum() >= 5 else np.nan

    if not (np.isfinite(r_flow) and (r_flow >= corr_thresh)):
        print(f"[skip-log] corr {r_flow:.3f} < thresh {corr_thresh:.2f}")
        return r_flow, mae, rmse, None

    # tagging
    method_tag = ("UCM" if (use_trend_risk and str(trend_method).lower() == "ucm")
                  else "ROLL" if (use_trend_risk and str(trend_method).lower() == "rolling")
                  else "RAW")
    base_tag = _tag(conf_model, conf_graph, conf_unreg)
    tag = f"{base_tag}_{method_tag}_seed{seed_for_tag}"

    # save legacy risk layers
    risk_fname = f"{tag}_unreg_risk_{pipe_overrides['disease']}.csv"
    std_fname = f"{tag}_unreg_risk_std_{pipe_overrides['disease']}.csv"
    agree_fname = f"{tag}_raw_trend_agree_{pipe_overrides['disease']}.csv"

    pd.DataFrame(risk_unreg, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, risk_fname))
    pd.DataFrame(risk_std, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, std_fname))
    pd.DataFrame(agree, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, agree_fname))

    # Stable mask used by the compatibility path
    AGREE_MIN = 0.70
    STD_MAX_Q = 0.80
    std_thr = np.quantile(risk_std[np.isfinite(risk_std)], STD_MAX_Q) if np.isfinite(risk_std).any() else np.inf
    stable_mask = (agree >= AGREE_MIN) & (risk_std <= std_thr)
    risk_stable = np.where(stable_mask, risk_unreg, 0.0)

    pd.DataFrame(risk_stable, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, f"{tag}_unreg_risk_STABLE_{pipe_overrides['disease']}.csv"))

    # legacy ranking (left as-is)
    risk_for_rank = risk_stable if (np.isfinite(risk_stable).any() and (risk_stable > 0).sum() > 0) else risk_unreg
    try:
        save_wwtp_rank_both(
            out_dir=pipe.output_dir,
            node_names=pipe.node_names,
            risk_matrix=risk_for_rank,
            risk_std=risk_std,
            agree=agree,
            flow_adj_ref=flow_adj_ref,
            county_pop_csv=pipe.county_pop_csv_full,
            wwtp_shapefile=pipe.wwtp_shapefile,
            tau=0.02,
            tag=tag,
            county_polygon_shp =pipe.county_polygon_shp

        )

    except Exception as e:
        print("[rank] WWTP inbound/outbound ranking skipped:", e)

    # Raw (no row normalization) and self-only sensitivity branches.
    risk_raw_skip, risk_std_skip, agree_skip = pipe.compute_transmission_matrix(
        use="train",
        alpha=conf_unreg["alpha"],
        percent_change=conf_unreg["percent_change"],
        variance_normalize=conf_unreg["variance_normalize"],
        normalize=conf_unreg["normalize"],
        self_mode="skip",
        renorm_offdiag=False,
        use_trend=bool(use_trend_risk),
        mc_samples=16, enable_dropout=True, return_std=True,
        raw_trend_consensus=True, consensus_rule="min", agree_tau=0.10
    )

    risk_raw_keep, risk_std_keep, agree_keep = pipe.compute_transmission_matrix(
        use="train",
        alpha=conf_unreg["alpha"],
        percent_change=conf_unreg["percent_change"],
        variance_normalize=conf_unreg["variance_normalize"],
        normalize=conf_unreg["normalize"],
        self_mode="keep",
        renorm_offdiag=False,
        use_trend=bool(use_trend_risk),
        mc_samples=16, enable_dropout=True, return_std=True,
        raw_trend_consensus=True, consensus_rule="min", agree_tau=0.10
    )

    # Save RAW matrices (no row-normalization applied)
    pd.DataFrame(risk_raw_skip, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, f"{tag}_RAW_NOTNORM_SKIPDIAG_{pipe_overrides['disease']}.csv"))
    pd.DataFrame(risk_raw_keep, index=pipe.node_names, columns=pipe.node_names) \
        .to_csv(os.path.join(pipe.output_dir, f"{tag}_RAW_NOTNORM_KEEPDIAG_{pipe_overrides['disease']}.csv"))

    # --- RAW transmission ranking (absolute magnitude; no normalization)
    df_tx, R_wc_raw, W_names, C_names = rank_raw_transmission_no_norm(
        risk_raw_skip, pipe.node_names, top_k=20
    )
    df_tx.to_csv(os.path.join(pipe.output_dir, f"sentinel_RAW_TX_{tag}_table.csv"), index=False)

    # --- SELF-IMPACT ranking (diagonal only; no normalization)
    df_self = rank_self_impact_only(risk_raw_keep, pipe.node_names, top_k=20)
    df_self.to_csv(os.path.join(pipe.output_dir, f"sentinel_SELF_ONLY_{tag}_table.csv"), index=False)

    # Visualizations for the raw-transmission and self-impact tracks
    if viz_enabled and viz is not None:
        try:
            wwtp_gdf = viz.load_wwtp_points(shared["wwtp_shapefile"])
        except Exception as e:
            wwtp_gdf = None
            print("[viz] WWTP shapefile load failed; skipping maps:", e)

        # county centroids (optional spokes) already handled above
        # County boundaries and optional sewersheds provide map context.
        county_polys_gdf = None
        try:
            county_poly_path = shared.get("county_polygon_shp")
            if county_poly_path:
                county_polys_gdf = viz.load_polygons(county_poly_path)
        except Exception as e:
            print("[viz] county polygons load failed:", e)

        # RAW_TX: spokes + top-sites map using raw (no-norm) WWTP->County block
        try:
            if wwtp_gdf is not None:
                # county centroids (optional spokes)
                try:
                    county_poly_path = shared.get("county_polygon_shp")
                    county_centroids_gdf, gdf_proj = viz.load_county_centroids(county_poly_path) if county_poly_path else None
                except Exception:
                    county_centroids_gdf, gdf_proj = None, None

                # spoke map (optional)
                if county_centroids_gdf is not None:
                    viz.plot_wwtp_spokes(
                        R_wc_raw, [w.lower() for w in W_names], [c.lower() for c in C_names],
                        wwtp_gdf=wwtp_gdf,
                        county_centroids_gdf=county_centroids_gdf,
                        polygons_gdf = gdf_proj,
                        out_dir=pipe.output_dir,
                        tag=f"{tag}_RAW_TX",
                        K=10, M=5
                    )

                # top sites map
                df_tx_map = df_tx.rename(columns={"score":"score"})
                viz.save_top_sentinel_sites(df_tx_map, wwtp_gdf, pipe.output_dir, tag=f"{tag}_RAW_TX")
                viz.plot_top_sites_map(
                    df_top=df_tx_map,
                    wwtp_gdf=wwtp_gdf,
                    title=f"Top WWTP Sentinels (RAW transmission, no row-normalization)",
                    out_dir=pipe.output_dir,
                    tag=f"{tag}_RAW_TX",
                    annotate=True,
                    county_polys_gdf=county_polys_gdf,
                )
                # (optional) also make the HTML
                try:
                    viz.plot_top_sites_map_html(
                        df_top=df_tx_map,
                        wwtp_gdf=wwtp_gdf,
                        out_dir=pipe.output_dir,
                        tag=f"{tag}_RAW_TX",
                        county_polys_gdf=county_polys_gdf,
                        title="Top WWTP Sentinels (interactive; RAW TX)"
                    )
                except Exception as e:
                    print("[viz][RAW_TX][html] skipped:", e)

        except Exception as e:
            print("[viz][RAW_TX] skipped:", e)

        # SELF_ONLY: point map of top sites (no edges)
        try:
            if wwtp_gdf is not None:
                df_self_map = df_self.rename(columns={"self_impact":"score"})
                viz.save_top_sentinel_sites(df_self_map, wwtp_gdf, pipe.output_dir, tag=f"{tag}_SELF_ONLY")
                viz.plot_top_sites_map(
                    df_top=df_self_map,
                    wwtp_gdf=wwtp_gdf,
                    title=f"Top WWTP Sentinels (SELF impact only, no normalization)",
                    out_dir=pipe.output_dir,
                    tag=f"{tag}_SELF_ONLY",
                    annotate=True,
                    county_polys_gdf=county_polys_gdf,
                )
                try:
                    viz.plot_top_sites_map_html(
                        df_top=df_self_map,
                        wwtp_gdf=wwtp_gdf,
                        out_dir=pipe.output_dir,
                        tag=f"{tag}_SELF_ONLY",
                        county_polys_gdf=county_polys_gdf,
                        title="Top WWTP Sentinels (interactive; SELF ONLY)"
                    )
                except Exception as e:
                    print("[viz][SELF_ONLY][html] skipped:", e)

        except Exception as e:
            print("[viz][SELF_ONLY] skipped:", e)

    # ---- risk–signal correlations (legacy)
    for sig, mode in [("clinical","target"),("clinical","source"),("wval","target"),("wval","source")]:
        try:
            pipe.save_risk_signal_correlations(
                use="train", signal=sig, mode=mode,
                tag=f"_{tag}"
            )
        except Exception as e:
            print("[viz-corr] skipped:", e)

    # ---- grid log
    header = ["time","disease","mode","seed","tag","mae","rmse","corr",
              "seq_len","hidden_dim","rnn_layers","dropout","l2_reg","lr_cosine",
              "cap_self","backfill","teleport","adj_row_norm",
              "alpha","percent_change","variance_normalize","normalize","self_mode","renorm_offdiag",
              "use_trend","trend_method","trend_win","season_period_weeks","verify_od"]

    row = [
        datetime.now().isoformat(timespec="seconds"),
        pipe_overrides["disease"], pipe_overrides["feature_mode"], seed_for_tag, tag,
        f"{mae:.6f}", f"{rmse:.6f}", f"{r_flow:.6f}",
        conf_model["seq_len"], conf_model["hidden_dim"], conf_model["rnn_layers"],
        conf_model["dropout"], conf_model["l2_reg"], conf_model["lr_cosine"],
        conf_graph["cap_self"], conf_graph["backfill"], conf_graph["teleport"], conf_graph["adj_row_norm"],
        conf_unreg["alpha"], conf_unreg["percent_change"], conf_unreg["variance_normalize"], conf_unreg["normalize"],
        conf_unreg["self_mode"], conf_unreg["renorm_offdiag"],
        int(bool(use_trend_risk)), trend_method, int(trend_window), int(season_period_weeks), int(bool(do_verify_od))
    ]
    _append_row(pipe.output_dir, row, header)

    return r_flow, mae, rmse, tag

# -------------------- CLI --------------------
def main():
    p = argparse.ArgumentParser(description="Quick DCRNN runner with structural trend + viz + OD verification.")
    p.add_argument("--disease", choices=["COVID","RSV","Influenza"], default="COVID")
    p.add_argument("--feature_mode", choices=["combined","case_only","wval_only"], default=None)
    p.add_argument("--type", default="rate")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corr_thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--no_viz", action="store_true")

    # Trend options
    p.add_argument("--use_trend_risk", action="store_true", default=True)
    p.add_argument("--trend_method", choices=["ucm", "rolling"], default="ucm")
    p.add_argument("--trend_window", type=int, default=3)
    p.add_argument("--season_period_weeks", type=int, default=52)

    # OD verification
    p.add_argument("--verify_od", action="store_true", help="Run OD-flow verification in the date window.")
    p.add_argument("--verify_top_weeks", type=int, default=3)

    # Grid toggles/overrides
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seq_len", type=int)
    p.add_argument("--hidden_dim", type=int)
    p.add_argument("--rnn_layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--l2_reg", type=float)
    p.add_argument("--lr_cosine", type=int, choices=[0,1])
    p.add_argument("--cap_self", type=float)
    p.add_argument("--backfill", choices=["uniform","proportional"])
    p.add_argument("--teleport", type=float)
    p.add_argument("--adj_row_norm", type=int, choices=[0,1])
    p.add_argument("--alpha", type=float)
    p.add_argument("--percent_change", type=int, choices=[0,1])
    p.add_argument("--variance_normalize", type=int, choices=[0,1])
    p.add_argument("--normalize", type=int, choices=[0,1])

    args = p.parse_args()

    # Seeds
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    shared = build_shared_io(args.type)
    all_d = disease_cfgs(args.type)
    if args.disease not in all_d:
        raise SystemExit(f"Unknown disease {args.disease}")
    base_cfg = all_d[args.disease].copy()
    d2folder = {"COVID": "run_COVID", "Influenza": "run_FLU", "RSV": "run_RSV"}

    out_dir_override = args.out_dir or d2folder.get(base_cfg["disease"], f"runs_{base_cfg['disease']}")
    corr_thresh = float(args.corr_thresh)
    done = _load_done_tags(out_dir_override)

    if args.feature_mode is not None:
        base_cfg["feature_mode"] = args.feature_mode

    MODEL_GRID = {
        "seq_len":    [args.seq_len] if args.seq_len else [4, 6] if args.quick else [4],
        "hidden_dim": [args.hidden_dim] if args.hidden_dim else [64, 128] if args.quick else [64],
        "rnn_layers": [args.rnn_layers] if args.rnn_layers else [1],
        "dropout":    [args.dropout] if args.dropout is not None else [0.0, 0.2] if args.quick else [0.0],
        "l2_reg":     [args.l2_reg] if args.l2_reg is not None else [1e-4],
        "use_huber":  [True],
        "lr_cosine":  [bool(args.lr_cosine)] if args.lr_cosine is not None else [False],
    }
    GRAPH_GRID = {
        "cap_self":     [args.cap_self] if args.cap_self is not None else [0.00, 0.05] if args.quick else [0.1],
        "backfill":     [args.backfill] if args.backfill else ["uniform"],
        "teleport":     [args.teleport] if args.teleport is not None else [0.00],
        "adj_row_norm": [bool(args.adj_row_norm)] if args.adj_row_norm is not None else [True],
    }
    UNREG_GRID = {
        "alpha":              [args.alpha] if args.alpha is not None else [0.30, 0.50] if args.quick else [0.40],
        "percent_change":     [bool(args.percent_change)] if args.percent_change is not None else [True],
        "variance_normalize": [bool(args.variance_normalize)] if args.variance_normalize is not None else [True],
        "normalize":          [bool(args.normalize)] if args.normalize is not None else [True],
        "self_mode":          ["skip"],
        "renorm_offdiag":     [True],
    }

    combos = []
    for m in _iter_product(MODEL_GRID):
        for g in _iter_product(GRAPH_GRID):
            for u in _iter_product(UNREG_GRID):
                combos.append({"MODEL": m, "GRAPH": g, "UNREG": u, "PIPE": base_cfg})

    print(f"[info] Running {len(combos)} combo(s) | disease={base_cfg['disease']} | mode={base_cfg['feature_mode']} | seed={args.seed}")
    for c in combos:
        base_tag = _tag(c["MODEL"], c["GRAPH"], c["UNREG"])
        method_tag = ("UCM" if args.use_trend_risk and args.trend_method.lower() == "ucm"
                      else "ROLL" if args.use_trend_risk and args.trend_method.lower() == "rolling"
                      else "RAW")
        tag = f"{base_tag}_{method_tag}_seed{args.seed}"
        # if tag in done:
        #     print(f"[skip] already logged: {tag}")
        #     continue
        print(f"\n=== {base_cfg['disease']} | mode={base_cfg['feature_mode']} | {tag} ===")
        r_flow, mae, rmse, logged_tag = run_one(
            c, shared, corr_thresh,
            out_dir_override=out_dir_override, viz_enabled=not args.no_viz,
            seed_for_tag=args.seed,
            use_trend_risk=args.use_trend_risk,
            trend_window=args.trend_window,
            trend_method=args.trend_method,
            season_period_weeks=args.season_period_weeks,
            do_verify_od=args.verify_od,
            verify_top_weeks=args.verify_top_weeks
        )
        if logged_tag:
            print(f"[run] {logged_tag}  MAE={mae:.4f}  RMSE={rmse:.4f}  corr={r_flow:.3f}")

if __name__ == "__main__":
    main()
