"""Diffusion-GRU implementation for Level 2 disease-dynamic refinement.

Contribution scores produced after perturbation are model-based sensitivity
measures and must not be interpreted as causal transmission effects.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from AnalysisViz_ODFlowPure import compute_weekly_flows
import Model_DCRNN_Viz as viz

from Process_ViralLoader import (
    load_county_weighted_wval,
    load_wwtp_wval,
    load_wwtp_shapes
)

from Process_ViralLoader_V2 import (
    load_county_clinical_fixed,
    load_wwtp_clinical_from_metrics_fixed,
)




# --- Repro defaults (doesn't guarantee determinism on all GPUs, but helps) ---
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism = False  # enable for strict deterministic execution


class CountyWWTP_DCRNN:
    """
    DCRNN pipeline with flexible clinical target and unified WWTP metrics.

    Feature channels per node (counties then WWTPs):
      [0] clinical_rate     – case/hospitalization; NaN if missing
      [1] wval              – wastewater viral load; NaN if missing
      [2] clinical_missing  – 1 if clinical_rate missing, else 0
      [3] wval_missing      – 1 if WVAL missing, else 0
      [4] pop               – population
      [5] od_flow_count     – # distinct OD partners with positive flow
      [6] node_mask         – 1 if (clinical or wval) present at this time, else 0
      [7] d_clinical        – first difference of clinical_rate
      [8] d_wval            – first difference of WVAL
      [9] spike_d_clinical  – 1 if |zscore(d_clinical)| > 2
      [10] spike_d_wval     – 1 if |zscore(d_wval)| > 2
      (delta-time appended in create_sequences())

    Modes:
      • 'wval_only' : keep [1,3,4,5,6,8,10]
      • 'case_only' : keep [0,2,4,5,6,7,9]
      • 'combined'  : keep [0..10] (clinical + WVAL)
    """

    def __init__(
            self,
            weekly_od_dir,
            # County-level clinical (cases for COVID; hospitalization for Flu/RSV)
            covid_case_csv=None,
            covid_rate_col=None,  # e.g., "County_hosp_3dayavg_r100Kutil"f
            flu_rsv_rate_col=None,
            county_wval_csv=None,
            county_wval_col=None,
            county_polygon_shp = None,
            # WWTP-level viral load
            wval_fp=None,
            # Unified WWTP metrics (case/hospitalization)
            wwtp_metrics_csv=None,
            clinical_target="case",  # "case" or "hospitalization"
            hosp_missing_as_zero=False,
            # Pop + geoms
            county_pop_csv_full=None,
            county_pop_csv_uncovered = None,
            wwtp_shapefile=None,
            # Model/seq config
            agg="week",
            seq_len=3,
            hidden_dim=32,
            rnn_layers=1,  # one or two diffusion-GRU layers
            epochs=300,
            batch_size=8,
            max_self_prop_county=0.5,
            max_self_prop_wwtp=0.5,
            output_dir="outputs",
            feature_mode="combined",
            start_week=None,
            end_week=None,
            disease="COVID",  # "COVID" | "Influenza" | "RSV"
            # Flu/RSV hospitalized file
            flu_rsv_hosp_file=None,
            auto_pick_dynamic=True,
            # Retention/robustness for counties
            force_keep_all_counties=True,  # keep counties even if sparse; helps ensure they appear in CSV/plots
            min_weeks_per_node=2,  # relaxed default so sparse counties remain
    ):
        assert feature_mode in {"wval_only", "case_only", "combined"}
        assert agg in {"week", "month"}
        self.weekly_od_dir = weekly_od_dir

        self.disease = str(disease)
        self.covid_case_csv = covid_case_csv
        self.covid_rate_col = covid_rate_col
        self.flu_rsv_rate_col = flu_rsv_rate_col
        self.county_wval_csv = county_wval_csv
        self.county_wval_col = county_wval_col
        self.flu_rsv_hosp_file = flu_rsv_hosp_file

        self.wval_fp = wval_fp
        self.wwtp_metrics_csv = wwtp_metrics_csv
        self.clinical_target = str(clinical_target).lower().strip()
        assert self.clinical_target in {"case", "hospitalization"}
        self.hosp_missing_as_zero = bool(hosp_missing_as_zero)

        # self.county_pop_csv = ##county_pop_csv
        self.county_polygon_shp =county_polygon_shp
        # Population CSVs:
        #   - FULL: used for per-capita clinical rates + population feature channels
        #   - UNCOVERED: used for adjacency self-loop (self-potential)
        self.county_pop_csv_uncovered = county_pop_csv_uncovered
        self.county_pop_csv_full = county_pop_csv_full
        # Backward-compatible alias (kept as uncovered)
        self.county_pop_csv = county_pop_csv_uncovered
        self.wwtp_shapefile = wwtp_shapefile

        self.agg = agg
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.rnn_layers = int(rnn_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_self_prop_county = float(max_self_prop_county)
        self.max_self_prop_wwtp = float(max_self_prop_wwtp)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_mode = feature_mode
        self.start_week = pd.to_datetime(start_week) if start_week else None
        self.end_week = pd.to_datetime(end_week) if end_week else None

        self.auto_pick_dynamic = bool(auto_pick_dynamic)
        self.force_keep_all_counties = bool(force_keep_all_counties)
        self.min_weeks_per_node = int(min_weeks_per_node)

    def apply_adj_teleport(self, alpha=0.1):
        self._adj_before_teleport = np.array(self.adj_static, dtype=np.float32)

        A = np.array(self.adj_static, dtype=np.float32)
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A = A / row_sum
        N = A.shape[1]
        J = np.ones_like(A, dtype=np.float32) / float(N)
        self.adj_static = (1.0 - float(alpha)) * A + float(alpha) * J
        self._adj_after_teleport = self.adj_static.copy()

        return self.adj_static

    def _rolling_mean_3d(self, X, win=3):
        # Rolling-window helper
        import numpy as np, pandas as pd
        T, N, D = X.shape
        assert D == 1
        out = np.zeros_like(X, dtype=np.float32)
        for j in range(N):
            s = pd.Series(X[:, j, 0])
            out[:, j, 0] = s.rolling(win, min_periods=1).mean().astype(np.float32).values
        return out


    def _state_space_trend_3d(self, X, season=52):
        """
        Structural time series trend via statsmodels Unobserved Components:
        local level + local trend (+ optional seasonal). Falls back to rolling if it fails.
        """
        import numpy as np, pandas as pd
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except Exception as e:
            raise RuntimeError(
                "statsmodels is required for trend_method='ucm'. Install: pip install statsmodels"
            ) from e

        T, N, D = X.shape
        assert D == 1
        out = np.zeros_like(X, dtype=np.float32)
        for j in range(N):
            s = pd.Series(X[:, j, 0]).astype(float).fillna(0.0)
            try:
                model = UnobservedComponents(
                    s.values,
                    level="lltrend",                # local level + local trend
                    seasonal=season if season and season > 1 else None,
                    stochastic_level=True,
                    stochastic_trend=True
                )
                res = model.fit(disp=False)
                level = getattr(res, "level_smoothed", None)
                if level is not None:
                    level = getattr(level, "values", level)
                    out[:, j, 0] = np.asarray(level, dtype=np.float32)
                else:
                    out[:, j, 0] = s.rolling(3, min_periods=1).mean().astype(np.float32).values
            except Exception:
                out[:, j, 0] = s.rolling(3, min_periods=1).mean().astype(np.float32).values
        return out

    def cap_self_loops(self, cap=0.25, backfill="proportional"):
        """
        Limit how much of each node's row-sum can come from its self-loop.

        Parameters
        ----------
        cap : float
            Maximum fraction of row weight that can be self-loop.
        backfill : {"proportional","uniform"}
            How to redistribute the excess self weight:
            - "proportional": spread to neighbors proportional to their existing weights
            - "uniform": spread evenly to all non-self neighbors
        """
        self._adj_before_cap = self.adj_static.copy()

        A = self.adj_static.copy()
        N = A.shape[0]
        for i in range(N):
            row_sum = A[i].sum()
            if row_sum <= 0:
                continue
            max_self = cap * row_sum
            if A[i, i] > max_self:
                excess = A[i, i] - max_self
                A[i, i] = max_self
                if backfill == "proportional":
                    # scale up other entries proportionally
                    if row_sum - A[i, i] > 0:
                        A[i, :] = A[i, :] + excess * (A[i, :] / (row_sum - A[i, i]))
                        A[i, i] = max_self
                elif backfill == "uniform":
                    neighbors = [j for j in range(N) if j != i]
                    if neighbors:
                        A[i, neighbors] += excess / len(neighbors)
        # renormalize row sums to 1
        # AFTER computing A and renormalizing:
        rs = A.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        A_norm = A / rs
        self.adj_static = A_norm
        self._adj_after_cap = A_norm.copy()
        return self.adj_static

    # -------------------- Loading --------------------
    def load_data(self):
        """
        Unified data loading for counties and WWTPs using Process_ViralLoader_V2.
        Loads:
          - Weekly OD matrices (mobility)
          - County-level clinical data (COVID/Flu/RSV)
          - County and WWTP population info
          - WWTP-level wastewater viral load
          - WWTP-level clinical metrics (matching disease)
        """
        # 1. Weekly OD matrices
        self.weekly_results, _ = compute_weekly_flows(self.weekly_od_dir)

        # 2. County population lookup
        self._load_county_population_maps()


        # 3. County clinical (all-ages aggregate)
        try:
            self.county_clinical_df = load_county_clinical_fixed(
                disease=self.disease,
                county_shp=self.county_polygon_shp,
                county_pop_csv=self.county_pop_csv_full,
                metric="hospitalization",
                value_kind="rate",
            )
        except Exception as e:
            print(f"[warn] County clinical load failed: {e}")
            self.county_clinical_df = pd.DataFrame(columns=["week", "County", "clinical_rate"])


        # 4. County WVAL (COVID only)
        self.county_wval_df = pd.DataFrame(columns=["week", "County", "county_wval"])
        if self.disease.upper() == "COVID" and self.county_wval_csv:
            self.county_wval_df = load_county_weighted_wval(
                self.county_wval_csv,
                value_col=self.county_wval_col,
                fips_to_name=getattr(self, "county_name_map", None)
            )

        # 5. WWTP viral load + population
        self.viral_df = load_wwtp_wval(self.wval_fp)
        self.wwtp_pop_map = load_wwtp_shapes(self.wwtp_shapefile)

        # 6. WWTP clinical metrics (from fixed-source mapping)
        try:
            self.wwtp_clin_df = load_wwtp_clinical_from_metrics_fixed(
                disease=self.disease,
                wwtp_shp=self.wwtp_shapefile,
                metric=self.clinical_target,
                value_kind="rate"
            )
        except Exception as e:
            print(f"[warn] WWTP clinical load failed: {e}")
            self.wwtp_clin_df = pd.DataFrame(columns=["week", "wwtp", "clinical_rate"])

        # --- Normalize naming conventions (critical for joins) ---
        if not self.county_clinical_df.empty and "County" in self.county_clinical_df.columns:
            self.county_clinical_df["County"] = (
                self.county_clinical_df["County"].astype(str).str.strip().str.lower()
            )

        if not self.wwtp_clin_df.empty and "wwtp" in self.wwtp_clin_df.columns:
            self.wwtp_clin_df["wwtp"] = (
                self.wwtp_clin_df["wwtp"].astype(str).str.strip().str.lower()
            )

        if hasattr(self, "county_wval_df") and not self.county_wval_df.empty:
            self.county_wval_df["County"] = (
                self.county_wval_df["County"].astype(str).str.strip().str.lower()
            )

        if hasattr(self, "viral_df") and not self.viral_df.empty and "wwtp" in self.viral_df.columns:
            self.viral_df["wwtp"] = (
                self.viral_df["wwtp"].astype(str).str.strip().str.lower()
            )

        # print(self.wwtp_clin_df["clinical_rate"].isna().sum())
        # print(self.county_wval_df["clinical_rate"].isna().sum())

        # 7. Build node lists (counties + WWTPs)
        self._init_node_lists()

    def _load_county_population_maps(self):
        """Load BOTH county population baselines and build robust FIPS↔name maps.

        Principle:
          - FULL population (county_pop_csv_full) is used for per-capita clinical rates and the population feature channel.
          - UNCOVERED population (county_pop_csv_uncovered) is used ONLY for adjacency self-loops ("selfpotential").

        Important robustness:
          - Your pop CSVs may NOT include county names. We therefore build FIPS↔name maps from the county polygon
            shapefile (preferred), falling back to a Census lookup URL if available at runtime.
          - Backward-compatible attributes (used elsewhere) point to UNCOVERED:
              self.county_pop_df, self.county_pop_map, self.county_pop_map_by_name
        """
        import os
        import pandas as pd

        def _read_pop(csv_path):
            """Return standardized df with columns: FIPS, POP (CountyName may be empty)."""
            if (not csv_path) or (not os.path.exists(csv_path)):
                return pd.DataFrame(columns=["FIPS", "CountyName", "POP"])

            df = pd.read_csv(csv_path)

            # FIPS-like key
            key_col = None
            for col in ["FIPS", "fips", "CountyFIPS", "county_fips", "GEOID",
                        "County_FIPS", "COUNTY_FIPS", "County_Fips", "CountyFIPS5", "FIPS5", "GEOID10"]:
                if col in df.columns:
                    key_col = col
                    break
            if key_col is None:
                # try first column that looks like integer-like ids
                key_col = df.columns[0]

            df[key_col] = (
                df[key_col].astype(str).str.strip()
                .str.replace(r"\.0+$", "", regex=True)
                .str.zfill(5)
            )

            # Population
            pop_col = None
            for col in ["POP", "Pop", "population", "Population", "Total_Population", "TOTAL_POPULATION", "pop"]:
                if col in df.columns:
                    pop_col = col
                    break
            if pop_col is None:
                num_cols = [c for c in df.columns if c != key_col and pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    raise ValueError(f"Population CSV must contain a numeric column: {csv_path}")
                pop_col = num_cols[0]

            df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")
            df = df.dropna(subset=[pop_col]).copy()

            # Optional county name column (may exist but be empty; keep if present)
            name_col = None
            for col in ["CountyName", "COUNTYNAME", "countyname", "County", "NAME", "NAMELSAD"]:
                if col in df.columns:
                    name_col = col
                    break
            if name_col is not None:
                df["CountyName"] = (
                    df[name_col].astype(str).str.strip()
                    .replace("nan", "")
                )
            else:
                df["CountyName"] = ""

            return df[[key_col, "CountyName", pop_col]].rename(columns={key_col: "FIPS", pop_col: "POP"}).copy()

        # ---------- Build FIPS↔Name maps (prefer county polygon shapefile) ----------
        fips_to_name, name_to_fips = {}, {}

        # 1) Preferred: local county polygon shapefile
        shp_path = getattr(self, "county_shp", None) or getattr(self, "county_polygon_shp", None)
        try:
            import geopandas as gpd
            if shp_path and os.path.exists(shp_path):
                gdf = gpd.read_file(shp_path)
                # find a name column
                name_col = next((c for c in ["NAME", "CountyName", "COUNTY", "COUNTYNAME", "NAMELSAD"] if c in gdf.columns), None)
                # find a FIPS / GEOID column (5-digit)
                fips_col = next((c for c in ["GEOID", "GEOID10", "FIPS", "FIPS5"] if c in gdf.columns), None)
                if fips_col is None:
                    # sometimes STATEFP + COUNTYFP exist
                    if ("STATEFP" in gdf.columns) and ("COUNTYFP" in gdf.columns):
                        gdf["__FIPS5"] = (gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)).astype(str)
                        fips_col = "__FIPS5"
                if name_col and fips_col:
                    tmp = gdf[[fips_col, name_col]].copy()
                    tmp[fips_col] = tmp[fips_col].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(5)
                    tmp[name_col] = (
                        tmp[name_col].astype(str).str.strip()
                        .str.replace(r"\s+county$", "", case=False, regex=True)
                        .str.replace(r"\s+parish$", "", case=False, regex=True)
                        .str.replace(r"\s+borough$", "", case=False, regex=True)
                        .str.replace(r"\s+municipality$", "", case=False, regex=True)
                        .str.strip().str.title()
                    )
                    fips_to_name = dict(zip(tmp[fips_col], tmp[name_col]))
                    name_to_fips = dict(zip(tmp[name_col], tmp[fips_col]))
        except Exception:
            pass

        # 2) Fallback: Census URL (only if runtime has internet)
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
                    .str.replace(r"\s+county$", "", case=False, regex=True)
                    .str.strip().str.title()
                )
                fips_to_name = dict(zip(look["FIPS"], look["CountyClean"]))
                name_to_fips = dict(zip(look["CountyClean"], look["FIPS"]))
            except Exception:
                pass

        # ---------- Read uncovered + full ----------
        uncovered_path = getattr(self, "county_pop_csv_uncovered", None) or getattr(self, "county_pop_csv", None)
        full_path = getattr(self, "county_pop_csv_full", None)

        df_uncovered = _read_pop(uncovered_path)
        df_full = _read_pop(full_path)

        # Fill CountyName if missing/empty using FIPS↔name map (best effort)
        for df_ in (df_uncovered, df_full):
            if df_.empty:
                continue
            miss = df_["CountyName"].astype(str).str.strip().eq("")
            if miss.any() and fips_to_name:
                df_.loc[miss, "CountyName"] = df_.loc[miss, "FIPS"].map(fips_to_name).fillna("")

        # Store: uncovered baseline (for selfpotential)
        self.county_pop_df_uncovered = df_uncovered.copy()
        self.county_pop_map_uncovered = dict(zip(df_uncovered["FIPS"], df_uncovered["POP"].astype(float))) if not df_uncovered.empty else {}
        self.county_pop_map_by_name_uncovered = dict(zip(df_uncovered["CountyName"], df_uncovered["POP"].astype(float))) if not df_uncovered.empty else {}

        # Store: full baseline (for rates/features)
        self.county_pop_df_full = df_full.copy()
        self.county_pop_map_full = dict(zip(df_full["FIPS"], df_full["POP"].astype(float))) if not df_full.empty else {}
        self.county_pop_map_by_name_full = dict(zip(df_full["CountyName"], df_full["POP"].astype(float))) if not df_full.empty else {}

        # Backward-compatible aliases (point to uncovered)
        self.county_pop_df = self.county_pop_df_uncovered.copy()
        self.county_pop_map = dict(self.county_pop_map_uncovered)
        self.county_pop_map_by_name = dict(self.county_pop_map_by_name_uncovered)

        # Name maps
        self.county_name_map = dict(fips_to_name) if fips_to_name else {}
        self.county_fips_map = dict(name_to_fips) if name_to_fips else {}


        # -------------------- Node lists --------------------


    # -------------------- Node lists --------------------
    def _init_node_lists(self):

        # Counties from clinical file (names)
        counties_from_clin = []
        if (
            hasattr(self, "county_clinical_df")
            and self.county_clinical_df is not None
            and (not self.county_clinical_df.empty)
            and ("County" in self.county_clinical_df.columns)
        ):
            counties_from_clin = (
                self.county_clinical_df["County"]
                .astype(str).str.strip().str.lower()
                .tolist()
            )

        # Counties from FULL population list (FIPS -> name if mapping exists)
        counties_from_pop = []
        pop_df = getattr(self, "county_pop_df_full", None)
        if pop_df is not None and (not pop_df.empty) and ("FIPS" in pop_df.columns):
            fips_list = pop_df["FIPS"].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(5).tolist()
            f2n = getattr(self, "county_name_map", {}) or {}
            if f2n:
                counties_from_pop = [str(f2n.get(f, f)).strip().lower() for f in fips_list]
            else:
                # Fallback: keep FIPS as county ids (not ideal, but avoids empty list)
                counties_from_pop = [str(f).strip().lower() for f in fips_list]

        # Canonical county identifiers used in the model are COUNTY NAMES (lowercase) when available.
        # If fips->name is available, pop-derived counties will be names too.
        self.all_counties = sorted(set([c for c in (counties_from_clin + counties_from_pop) if c and c != "international"]))


        # WWTPs: restrict to those in shapefile, drop "Historic"
        wwtps = []
        if hasattr(self, "wwtp_pop_map") and isinstance(self.wwtp_pop_map, dict):
            wwtps = list(self.wwtp_pop_map.keys())

        # normalize to lowercase and filter
        wwtps = [str(w).strip().lower() for w in wwtps if isinstance(w, str)]
        wwtps = [w for w in wwtps if "historic" not in w.lower()]
        wwtps = sorted(set(wwtps))

        self.all_wwtps = wwtps

        # Name ↔ index
        self.county_index = {c: i for i, c in enumerate(self.all_counties)}
        self.wwtp_index = {w: i for i, w in enumerate(self.all_wwtps)}

        self.C = len(self.all_counties)
        self.W = len(self.all_wwtps)
        self.N = self.C + self.W
        self.node_names = [f"C_{c}" for c in self.all_counties] + [f"W_{w}" for w in self.all_wwtps]

    # -------------------- Accessors --------------------
    def get_wwtp_wval(self, wwtp, week):
        if self.viral_df is None or self.viral_df.empty:
            return np.nan
        row = self.viral_df[(self.viral_df["wwtp"] == str(wwtp).lower()) &
                            (self.viral_df["week"] == pd.to_datetime(week))]
        return float(row["ww_index_normed_ln_lin"].values[0]) if not row.empty else np.nan

    def verify_od_flows(self, top_weeks: int = 3, tag: str = "",
                        start_week=None, end_week=None, save_aggregate=True):
        """
        Verify OD flows within an optional [start_week, end_week] window.
        - Weeks are filtered to the window (inclusive)
        - Per-week: build bipartite adjacency, remove self, row-normalize; compute stats
        - Optional: aggregate flows across the whole window and save one matrix (C↔W + W↔C)
        """
        import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- resolve date window ----
        sw = pd.to_datetime(start_week) if start_week is not None else getattr(self, "start_week", None)
        ew = pd.to_datetime(end_week) if end_week is not None else getattr(self, "end_week", None)

        # canonicalize keys
        weeks_all = sorted(self.weekly_results.keys(), key=lambda x: pd.to_datetime(x))
        weeks = []
        for w in weeks_all:
            wt = pd.to_datetime(w)
            if (sw is None or wt >= sw) and (ew is None or wt <= ew):
                weeks.append(w)

        if not weeks:
            print("[verify_od] No weeks in the requested window.")
            return

        # ---- make per-week buckets for the filtered weeks ----
        buckets = {wk: [wk] for wk in weeks}

        # 1) flow blocks (handles label normalization internally)
        flow_blocks = self.build_flow_blocks(buckets=buckets)

        # 2) adjacencies (pure bipartite for verification)
        adj_time, adj_static = self.build_flow_adjacency(flow_blocks, beta_cc=0.3, beta_ww=0.3)

        C, W, N = self.C, self.W, self.N
        stats, week_mats = [], {}

        # ---- per-week stats in window ----
        for w in weeks:
            adj = adj_time[w].copy()

            # remove self & row-normalize
            np.fill_diagonal(adj, 0.0)
            rs = adj.sum(axis=1, keepdims=True);
            rs[rs == 0] = 1.0
            Arow = adj / rs

            # off-diagonal stats
            mask = ~np.eye(N, dtype=bool)
            off = Arow[mask].ravel()
            density = 100.0 * float(np.count_nonzero(off)) / max(off.size, 1)
            std_off = float(np.nanstd(off))

            # normalized row entropy
            P = Arow.copy()
            rs2 = P.sum(axis=1, keepdims=True) + 1e-12
            P = P / rs2
            support = (P > 0).sum(axis=1).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                H = -np.nansum(np.where(P > 0, P * np.log(P + 1e-12), 0.0), axis=1)
                Hnorm = np.divide(H, np.log(np.maximum(support, 1.0)), out=np.zeros_like(H), where=support > 1)
            row_entropy_mean = float(np.nanmean(Hnorm))

            stats.append(dict(
                week=w,
                std_off=std_off,
                density_pct=density,
                mean_row_entropy=row_entropy_mean,
            ))
            week_mats[w] = Arow

        # filename tag includes the window
        def _fmt(d):
            return pd.to_datetime(d).strftime("%Y-%m-%d")

        win_tag = f"_win_{_fmt(weeks[0])}_to_{_fmt(weeks[-1])}"
        auto_tag = f"_{str(getattr(self, 'disease', 'unknown')).lower()}_{str(getattr(self, 'feature_mode', 'unknown')).lower()}"
        final_tag = (tag or auto_tag) + win_tag

        # save per-week table + preview top variance weeks
        df = pd.DataFrame(stats).sort_values("std_off", ascending=False)
        out_csv = os.path.join(self.output_dir, f"od_verification_by_week{final_tag}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[verify_od] Saved per-week stats → {out_csv}")
        print(df.head(min(top_weeks, len(df))))

        # # quick viz for top weeks
        try:
            import Model_DCRNN_Viz as viz
            picks = list(df.head(min(top_weeks, len(df)))["week"])
            for w in picks:
                M = week_mats[w]
                viz.plot_risk_heatmap(
                    M, self.node_names,
                    title=f"Week {w} – Row-normalized OD (no self)",
                    output_dir=self.output_dir, skip_self=False, highlight_self=True,
                )
        except Exception as e:
            print("[verify_od] plotting skipped:", e)

        # ---- aggregate across the window (optional) ----
        if save_aggregate:
            # sum flow blocks over all weeks in window, then build one adjacency
            import pandas as pd, numpy as np
            mat_c2w = pd.DataFrame(0.0, index=self.all_counties, columns=self.all_wwtps)
            mat_w2c = pd.DataFrame(0.0, index=self.all_wwtps, columns=self.all_counties)
            for wk in weeks:
                c2w_df, w2c_df = flow_blocks[wk]
                mat_c2w = mat_c2w.add(c2w_df, fill_value=0.0)
                mat_w2c = mat_w2c.add(w2c_df, fill_value=0.0)

            adj = np.zeros((N, N), dtype=np.float32)
            c2w = mat_c2w.values.astype(np.float32)
            w2c = mat_w2c.values.astype(np.float32)
            adj[0:C, C:C + W] = c2w
            adj[C:C + W, 0:C] = w2c

            # save raw (unnormalized) aggregate adjacency + a row-normalized view
            agg_csv = os.path.join(self.output_dir, f"od_flow_aggregate_matrix{final_tag}.csv")
            pd.DataFrame(adj, index=self.node_names, columns=self.node_names).to_csv(agg_csv)
            print(f"[verify_od] Saved aggregate OD matrix → {agg_csv}")

            A = adj.copy()
            np.fill_diagonal(A, 0.0)
            rs = A.sum(axis=1, keepdims=True);
            rs[rs == 0] = 1.0
            Arow = A / rs

            # simple stats on the aggregate
            mask = ~np.eye(N, dtype=bool)
            off = Arow[mask].ravel()
            agg_stats = dict(
                agg_density_pct=100.0 * float(np.count_nonzero(off)) / max(off.size, 1),
                agg_std_off=float(np.nanstd(off)),
            )
            pd.DataFrame([agg_stats]).to_csv(
                os.path.join(self.output_dir, f"od_flow_aggregate_stats{final_tag}.csv"), index=False
            )

    def get_wwtp_clinical(self, wwtp, week):
        if self.wwtp_clin_df is None or self.wwtp_clin_df.empty:
            return np.nan
        row = self.wwtp_clin_df[(self.wwtp_clin_df["wwtp"] == str(wwtp).lower()) &
                                (self.wwtp_clin_df["week"] == pd.to_datetime(week))]
        return float(row["clinical_rate"].values[0]) if not row.empty else np.nan

    def _normalize_od_labels_to_names(self, df, axis_kind="both", kind="county"):
        """
        Colorado-only simplifier:
          - If a label matches 8xxx or 08xxx (optionally with .0), convert to 5-digit FIPS and map to county name.
          - Otherwise treat as already a name (lowercased).
          - WWTP labels are simply lowercased.
        This scans whichever axes you pass via axis_kind ("rows", "cols", "both").
        """
        import pandas as pd, re
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        f2n = getattr(self, "county_name_map", {}) or {}

        def _fix_one(label, treat_as_county: bool) -> str:
            s = str(label).strip()
            if not treat_as_county:
                return s.lower()

            # 4-digit beginning with 8 (e.g., 8031 or 8031.0) -> 08xxx
            m4 = re.match(r"^\s*(8\d{3})(?:\.0+)?\s*$", s)
            if m4:
                fips = "0" + m4.group(1)
                name = f2n.get(fips, fips)
                return str(name).strip().lower()

            # 5-digit 08xxx (optionally .0)
            m5 = re.match(r"^\s*(08\d{3})(?:\.0+)?\s*$", s)
            if m5:
                fips = m5.group(1)
                name = f2n.get(fips, fips)
                return str(name).strip().lower()

            # Not numeric-ish: assume it's already a county name
            return s.lower()

        df2 = df.copy()

        # Normalize 8xxx-like labels on the requested axes.
        if axis_kind in ("rows", "both"):
            # County rows can be normalized explicitly; WWTP labels are lowercased otherwise.
            df2.index = [_fix_one(v, True) for v in df2.index]
        if axis_kind in ("cols", "both"):
            # Same idea for columns
            df2.columns = [_fix_one(v, True) for v in df2.columns]

        return df2

    def build_flow_blocks(self, buckets=None):
        """
        Build county→WWTP (C×W) and WWTP→county (W×C) flow blocks for each bucket.
        Pure OD—no features, no self-loops, no row-norm.
        Returns:
            flow_blocks: dict[bucket_id] -> (mat_c2w[C×W] DataFrame, mat_w2c[W×C] DataFrame)
        """
        import pandas as pd

        # choose buckets: passed-in, or self.buckets, or 1 bucket per week
        if buckets is None:
            buckets = getattr(self, "buckets", None)
            if buckets is None:
                import pandas as pd
                weeks = sorted(self.weekly_results.keys(), key=lambda x: pd.to_datetime(x))
                buckets = {wk: [wk] for wk in weeks}

        all_counties = self.all_counties
        all_wwtps = self.all_wwtps

        # Alias to the configured label normalizer when available
        def _norm(df, axis_kind="both", kind="county"):
            if hasattr(self, "_normalize_od_labels_to_names"):
                return self._normalize_od_labels_to_names(df, axis_kind=axis_kind, kind=kind)
            return df

        flow_blocks = {}
        for bid, wk_keys in buckets.items():
            # aggregate over weeks in this bucket
            mat_c2w = pd.DataFrame(0.0, index=all_counties, columns=all_wwtps)
            mat_w2c = pd.DataFrame(0.0, index=all_wwtps, columns=all_counties)

            for wk in wk_keys:
                flows = self.weekly_results[wk]
                mc = getattr(flows, "attrs", {}).get("county_to_wwtp", pd.DataFrame())
                mw = getattr(flows, "attrs", {}).get("wwtp_to_county", pd.DataFrame())

                mc = self._normalize_od_labels_to_names(mc, axis_kind="both", kind="county")
                mw = self._normalize_od_labels_to_names(mw, axis_kind="both", kind="wwtp")

                if isinstance(mc, pd.DataFrame) and not mc.empty:
                    block = mc.T.reindex(index=all_counties, columns=all_wwtps).fillna(0.0)
                    mat_c2w = mat_c2w.add(block, fill_value=0.0)
                if isinstance(mw, pd.DataFrame) and not mw.empty:
                    block = mw.T.reindex(index=all_wwtps, columns=all_counties).fillna(0.0)
                    mat_w2c = mat_w2c.add(block, fill_value=0.0)

            flow_blocks[bid] = (mat_c2w, mat_w2c)

        return flow_blocks

    def build_flow_adjacency(self, flow_blocks, beta_cc: float = 0.0, beta_ww: float = 0.0):
        """
        Convert flow blocks into adjacencies.
        - Bipartite by default (C→W and W→C only)
        - Optionally add within-layer projections:
            CC = (C→W) @ (W→C),  WW = (W→C) @ (C→W)
          scaled by beta_cc / beta_ww in [0..1]
        - No self-loops or row-normalization here; that’s up to caller.

        Returns:
            adj_time: dict[bucket_id] -> adjacency np.ndarray [N, N]
            adj_static: np.ndarray [N, N] average across buckets
        """
        import numpy as np

        C, W, N = self.C, self.W, self.N
        adj_time = {}

        for bid, (mat_c2w, mat_w2c) in flow_blocks.items():
            adj = np.zeros((N, N), dtype=np.float32)
            c2w = mat_c2w.values.astype(np.float32)  # [C, W]
            w2c = mat_w2c.values.astype(np.float32)  # [W, C]

            # bipartite blocks
            adj[0:C, C:C + W] = c2w
            adj[C:C + W, 0:C] = w2c

            # optional within-layer projections
            if beta_cc > 0.0 or beta_ww > 0.0:
                CC = c2w @ w2c  # [C, C]
                WW = w2c @ c2w  # [W, W]
                np.fill_diagonal(CC, 0.0);
                np.fill_diagonal(WW, 0.0)
                if beta_cc > 0.0 and CC.max() > 0:
                    adj[0:C, 0:C] += beta_cc * (CC / CC.max())
                if beta_ww > 0.0 and WW.max() > 0:
                    adj[C:C + W, C:C + W] += beta_ww * (WW / WW.max())

            adj_time[bid] = adj

        # static = mean across buckets
        if len(adj_time) > 0:
            A_stack = np.stack([adj_time[k] for k in adj_time.keys()], axis=0)
            adj_static = A_stack.mean(axis=0).astype(np.float32)
        else:
            adj_static = np.zeros((N, N), dtype=np.float32)

        return adj_time, adj_static

    def build_features_and_adj(self):
        # Normalize weekly_results keys to YYYY-MM-DD strings
        self.weekly_results = {pd.to_datetime(k).strftime("%Y-%m-%d"): v for k, v in self.weekly_results.items()}

        weeks_sorted = sorted(self.weekly_results.keys(), key=lambda x: pd.to_datetime(x))

        # Apply start/end week filter
        if self.start_week is not None:
            weeks_sorted = [w for w in weeks_sorted if pd.to_datetime(w) >= self.start_week]
        if self.end_week is not None:
            weeks_sorted = [w for w in weeks_sorted if pd.to_datetime(w) <= self.end_week]

        # Buckets (week or month)
        if self.agg == "month":
            time_index = sorted({pd.to_datetime(w).to_period("M").to_timestamp("M") for w in weeks_sorted})
            buckets = {
                t: [pd.to_datetime(w).strftime("%Y-%m-%d")
                    for w in weeks_sorted if pd.to_datetime(w).to_period("M").to_timestamp("M") == t]
                for t in time_index
            }
        else:
            time_index = [pd.to_datetime(w) for w in weeks_sorted]
            buckets = {t: [t.strftime("%Y-%m-%d")] for t in time_index}

        features_all, adjs_all = [], []

        # Apply the configured label normalizer when available.
        def _norm(df, axis_kind="both", kind="county"):
            if hasattr(self, "_normalize_od_labels_to_names"):
                return self._normalize_od_labels_to_names(df, axis_kind=axis_kind, kind=kind)
            return df

        for _, wk_keys in buckets.items():
            # ---------- Aggregate OD for this bucket ----------
            mat_c2w = pd.DataFrame(0.0, index=self.all_counties, columns=self.all_wwtps)
            mat_w2c = pd.DataFrame(0.0, index=self.all_wwtps, columns=self.all_counties)

            for wk in wk_keys:
                flows = self.weekly_results[wk]
                mc = getattr(flows, "attrs", {}).get("county_to_wwtp", pd.DataFrame())
                mw = getattr(flows, "attrs", {}).get("wwtp_to_county", pd.DataFrame())

                # NORMALIZE (critical for 4-digit FIPS like "8031" / "8031.0")
                mc = self._normalize_od_labels_to_names(mc, axis_kind="both", kind="county")
                mw = self._normalize_od_labels_to_names(mw, axis_kind="both", kind="wwtp")

                if isinstance(mc, pd.DataFrame) and not mc.empty:
                    block = mc.T.reindex(index=self.all_counties, columns=self.all_wwtps).fillna(0.0)
                    mat_c2w = mat_c2w.add(block, fill_value=0.0)
                if isinstance(mw, pd.DataFrame) and not mw.empty:
                    block = mw.reindex(index=self.all_wwtps, columns=self.all_counties).fillna(0.0)
                    mat_w2c = mat_w2c.add(block, fill_value=0.0)

            C, W, N = self.C, self.W, self.N
            adj = np.zeros((N, N), dtype=np.float32)
            c2w = mat_c2w.values.astype(np.float32)
            w2c = mat_w2c.values.astype(np.float32)
            adj[0:C, C:C + W] = c2w
            adj[C:C + W, 0:C] = w2c

            # --- Directional OD stats (counts and volumes) ---
            county_out_count = (c2w > 0).sum(axis=1).astype(np.float32)
            county_in_count = (w2c.T > 0).sum(axis=1).astype(np.float32)
            county_out_vol = c2w.sum(axis=1).astype(np.float32)
            county_in_vol = w2c.T.sum(axis=1).astype(np.float32)

            wwtp_out_count = (w2c > 0).sum(axis=1).astype(np.float32)
            wwtp_in_count = (c2w.T > 0).sum(axis=1).astype(np.float32)
            wwtp_out_vol = w2c.sum(axis=1).astype(np.float32)
            wwtp_in_vol = c2w.T.sum(axis=1).astype(np.float32)

            # Self-loops from population (then cap and row-normalize)
            for ci, county in enumerate(self.all_counties):
                # Self-loop for counties uses UNCOVERED population (selfpotential).
                c_str = str(county).strip()
                if (c_str.isdigit() and len(c_str)==5):
                    fips = c_str
                else:
                    fips = (getattr(self, "county_fips_map", {}) or {}).get(c_str.title(), None)
                pop = 0.0
                if fips:
                    pop = float((getattr(self, "county_pop_map_uncovered", {}) or getattr(self, "county_pop_map", {}) or {}).get(str(fips).zfill(5), 0.0))
                adj[ci, ci] = float(pop)
            for wi, wwtp in enumerate(self.all_wwtps):
                adj[C + wi, C + wi] = float(self.wwtp_pop_map.get(wwtp, 0.0))

            # Cap diagonal by type and row-normalize
            for i in range(N):
                rs = adj[i].sum()
                if rs > 0:
                    cap = self.max_self_prop_county if i < C else self.max_self_prop_wwtp
                    if adj[i, i] / rs > cap:
                        adj[i, i] = cap * rs
                    denom = adj[i].sum()
                    if denom > 0:
                        adj[i] = adj[i] / denom
            adjs_all.append(adj.astype(np.float32))

            # ---------- Node features ----------
            # COUNTY: clinical + county_wval (COVID only) + pop + OD channels
            period_df = self.county_clinical_df[
                self.county_clinical_df["week"].isin([pd.to_datetime(w) for w in wk_keys])]
            clin_s = period_df.groupby("County")["clinical_rate"].mean()
            county_clin = clin_s.reindex(self.all_counties).astype(float).to_numpy()
            county_clin_missing = np.isnan(county_clin).astype(np.float32)
            county_clin = np.where(np.isnan(county_clin), 0.0, county_clin)

            county_wval = np.full(C, np.nan, dtype=np.float32)
            county_wval_missing = np.ones(C, dtype=np.float32)
            if hasattr(self, "county_wval_df") and not self.county_wval_df.empty:
                cw = self.county_wval_df[self.county_wval_df["week"].isin([pd.to_datetime(w) for w in wk_keys])]
                wval_s = cw.groupby("County")["county_wval"].mean()
                arr = wval_s.reindex(self.all_counties).astype(float).to_numpy()
                county_wval = arr.astype(np.float32)
                county_wval_missing = np.isnan(county_wval).astype(np.float32)

            # Population feature uses "uncovered" baseline (rates/pop channels).
            county_pop = np.array(
                [
                    float(
                        # 1) FULL pop by FIPS (preferred)
                        (self.county_pop_map_uncovered or {}).get(
                            self.county_fips_map.get(c.title(), ""), 0.0
                        )
                        # 2) FULL pop by county name (fallback)
                        or (self.county_pop_map_by_name_uncovered or {}).get(c.title(), 0.0)
                    )
                    for c in self.all_counties
                ],
                dtype=np.float32,
            )

            county_mask_any = (~np.isnan(county_clin) | ~np.isnan(county_wval)).astype(np.float32)

            county_in_vol_per_cap = county_in_vol / (county_pop + 1e-9)

            county_feat = np.column_stack([
                county_clin, county_wval,
                county_clin_missing, county_wval_missing,
                county_pop,
                county_in_count, county_out_count, county_in_vol, county_out_vol,
                county_mask_any,
                county_in_vol_per_cap,
            ])

            # WWTP: clinical+WVAL + pop + OD channels
            Wn = self.W
            wwtp_clin = np.full(Wn, np.nan, dtype=np.float32)
            wwtp_wval = np.full(Wn, np.nan, dtype=np.float32)
            wwtp_wval_missing = np.ones(Wn, dtype=np.float32)
            wwtp_pop = np.array([float(self.wwtp_pop_map.get(w, 0.0)) for w in self.all_wwtps], dtype=np.float32)

            for wi, w in enumerate(self.all_wwtps):
                vals_c = [self.get_wwtp_clinical(w, x) for x in wk_keys]
                vals_c = [v for v in vals_c if pd.notna(v)]
                if vals_c:
                    wwtp_clin[wi] = float(np.nanmean(vals_c))
                vals_w = [self.get_wwtp_wval(w, x) for x in wk_keys]
                vals_w = [v for v in vals_w if pd.notna(v)]
                if vals_w:
                    wwtp_wval[wi] = float(np.mean(vals_w))
                    wwtp_wval_missing[wi] = 0.0

            wwtp_clin_missing = np.isnan(wwtp_clin).astype(np.float32)
            wwtp_clin = np.where(np.isnan(wwtp_clin), 0.0, wwtp_clin)
            wwtp_mask_any = (~np.isnan(wwtp_clin) | ~np.isnan(wwtp_wval)).astype(np.float32)

            wwtp_in_vol_per_cap = wwtp_in_vol / (wwtp_pop + 1e-9)

            wwtp_feat = np.column_stack([
                wwtp_clin, wwtp_wval,
                wwtp_clin_missing, wwtp_wval_missing,
                wwtp_pop,
                wwtp_in_count, wwtp_out_count, wwtp_in_vol, wwtp_out_vol,
                wwtp_mask_any,
                wwtp_in_vol_per_cap,
            ])

            feat_mat = np.vstack([county_feat, wwtp_feat]).astype(np.float32)
            features_all.append(feat_mat)

        # Stack and store
        self.features_arr_raw = np.stack(features_all)  # [T, N, F]

        # existing: self.features_arr_raw shape [T, N, F]
        F = self.features_arr_raw.shape[-1]

        # Indices of the appended per-capita flow channels
        F = self.features_arr_raw.shape[-1]
        ch_tourism_percap = F - 1
        tourism_percap = self.features_arr_raw[..., ch_tourism_percap:ch_tourism_percap + 1]

        # County and WWTP channels may be concatenated or represented by one composite channel.

        # first difference + z-score spikes (per node)
        d_tour = np.diff(tourism_percap, axis=0, prepend=tourism_percap[:1])
        mu_t = np.nanmean(d_tour, axis=0, keepdims=True)
        sd_t = np.nanstd(d_tour, axis=0, keepdims=True) + 1e-6
        z_t = (d_tour - mu_t) / sd_t
        spike_tour = (np.abs(z_t) > 2.0).astype(np.float32)

        # append to features (d_tour, spike_tour)
        self.features_arr_raw = np.concatenate([self.features_arr_raw, d_tour, spike_tour], axis=-1)

        self.adj_time = np.stack(adjs_all).astype(np.float32)  # [T, N, N]
        self.adj_static = np.mean(self.adj_time, axis=0)  # keep this for the cell
        self.time_index = list(buckets.keys())

        # First-diff + spike flags over [clinical, wval]
        dyn = self.features_arr_raw[..., :2]
        d_dyn = np.diff(dyn, axis=0, prepend=dyn[:1])
        mu = np.nanmean(d_dyn, axis=0, keepdims=True)
        std = np.nanstd(d_dyn, axis=0, keepdims=True) + 1e-6
        z = (d_dyn - mu) / std
        spike = (np.abs(z) > 2.0).astype(np.float32)
        self.features_arr_raw = np.concatenate([self.features_arr_raw, d_dyn, spike], axis=-1)

        # Node mask + fill
        isfinite_any = np.any(np.isfinite(self.features_arr_raw[..., :2]), axis=-1).astype(np.float32)
        self.features_arr_raw[..., 9] = isfinite_any
        self.features_arr = np.nan_to_num(self.features_arr_raw, nan=0.0)

        # Target = clinical
        self._target_ch = 0
        self.y_raw_all = self.features_arr_raw[..., self._target_ch].astype(np.float32)

    # -------------------- Feature-mode mask --------------------
    def _apply_feature_mode(self, X):
        keep_map = {
            "wval_only": [1, 3, 4, 5, 6, 7, 8, 9, 11, 13],
            "case_only": [0, 2, 4, 5, 6, 7, 8, 9, 10, 12],
            "combined":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
        keep = keep_map[self.feature_mode][:]

        # If we created trend channels, keep them regardless of mode
        extra = getattr(self, "_trend_keep_idx", [])
        keep = sorted(set(keep + [i for i in extra if i is not None]))

        F = X.shape[-1]
        mask_vec = np.zeros((F,), dtype=np.float32)
        for ch in keep:
            if 0 <= ch < F:
                mask_vec[ch] = 1.0
        return X * mask_vec[None, None, None, :]


    def create_sequences(self, gap_tolerance=2, require_strict=False,
                         x_log1p_channels=(0, 1, 7, 8, 10),
                         x_scale_channels=(0, 1, 4, 5, 6, 7, 8, 10, 11),
                         y_log1p=True):
        import numpy as np
        times = np.array(self.time_index)
        dt_weeks = np.r_[0, np.diff(times).astype('timedelta64[D]').astype(int) / 7.0]
        dt_weeks = dt_weeks[:, None, None]

        # === Dynamic mobility features + trend channels ===
        A_time = getattr(self, "adj_time", None)
        if A_time is not None and A_time.shape[0] == self.features_arr.shape[0]:
            A_dyn = A_time.astype(np.float32).copy()

            # edge dropout
            p = getattr(self, "mobility_edge_dropout_p", 0.10)
            if p and p > 0:
                mask = (np.random.rand(*A_dyn.shape) > float(p)).astype(np.float32)
                A_dyn *= mask

            # row-normalize per t
            row_sum = A_dyn.sum(axis=-1, keepdims=True)
            A_dyn = A_dyn / (row_sum + 1e-8)

            # 1-hop / 2-hop diffusion feats
            AX  = np.einsum("tij,tjf->tif", A_dyn, self.features_arr)
            A2  = np.einsum("tij,tjk->tik", A_dyn, A_dyn)
            A2X = np.einsum("tij,tjf->tif", A2, self.features_arr)

            Xall = np.concatenate([self.features_arr, AX, A2X], axis=-1)

            # === Trend features on clinical/wval ===
            ch_clin = 0
            ch_wval = 1
            trend_win = getattr(self, "trend_window", 3)
            clin_series = Xall[..., ch_clin:ch_clin + 1]
            wval_series = Xall[..., ch_wval:ch_wval + 1]

            trend_method = getattr(self, "trend_method", "ucm")
            if trend_method == "ucm":
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~trend")
                trend_clin = self._state_space_trend_3d(clin_series, season=getattr(self, "season_period_weeks", 52))
                trend_wval = self._state_space_trend_3d(wval_series, season=getattr(self, "season_period_weeks", 52))
            else:  # "rolling"
                trend_clin = self._rolling_mean_3d(clin_series, win=trend_win)
                trend_wval = self._rolling_mean_3d(wval_series, win=trend_win)

            Xall = np.concatenate([Xall, trend_clin, trend_wval], axis=-1)
            self._idx_trend_clin = Xall.shape[-1] - 2
            self._idx_trend_wval = Xall.shape[-1] - 1
            self._trend_keep_idx = [self._idx_trend_clin, self._idx_trend_wval]
        else:
            Xall = self.features_arr

        # Fourier seasonality
        def _fourier_time_features(T, period=52, n_harmonics=3):
            import numpy as np
            t = np.arange(T, dtype=np.float32)
            cols = []
            for k in range(1, n_harmonics + 1):
                cols.append(np.sin(2.0 * np.pi * k * t / float(period)))
                cols.append(np.cos(2.0 * np.pi * k * t / float(period)))
            return np.stack(cols, axis=-1) if cols else np.zeros((T, 0), dtype=np.float32)

        F = _fourier_time_features(Xall.shape[0], period=getattr(self, "season_period_weeks", 52),
                                   n_harmonics=getattr(self, "season_harmonics", 3))
        if F.shape[1] > 0:
            F_b = np.repeat(F[:, None, :], Xall.shape[1], axis=1)
            Xall = np.concatenate([Xall, F_b], axis=-1)

        # --- Node retention policy ---
        dyn_mask_any = np.any(np.isfinite(self.features_arr_raw[..., :2]), axis=-1).astype(np.float32)
        node_ok = (dyn_mask_any.sum(axis=0) >= self.min_weeks_per_node)
        if not self.force_keep_all_counties:
            pass  # use node_ok directly
        else:
            # Always keep counties; apply threshold only to WWTPs
            keep = np.zeros_like(node_ok, dtype=bool)
            keep[:self.C] = True
            keep[self.C:] = node_ok[self.C:]
            node_ok = keep

        # Apply node filter
        Xall = Xall[:, node_ok, :]
        self.y_raw_all = self.y_raw_all[:, node_ok]
        self.node_names = [n for n, k in zip(self.node_names, node_ok) if k]
        self.num_nodes = len(self.node_names)

        # Windows
        X, Y, SW = [], [], []
        for start in range(len(times) - self.seq_len):
            end = start + self.seq_len
            gaps = np.diff(times[start:end]).astype('timedelta64[D]').astype(int) / 7.0
            if require_strict and (gaps > 1.01).any():
                continue
            if (gaps > (gap_tolerance + 1e-6)).any():
                continue
            X.append(Xall[start:end])
            y_end = self.y_raw_all[end]  # [N]
            sw_end = np.isfinite(y_end).astype(np.float32)
            y_end = np.nan_to_num(y_end, nan=0.0)
            Y.append(y_end[:, None])
            SW.append(sw_end[:, None])

        X = np.array(X, dtype=np.float32)  # [B,T,N,F]
        Y = np.array(Y, dtype=np.float32)  # [B,N,1]
        SW = np.array(SW, dtype=np.float32)  # [B,N,1]
        if X.size == 0:
            raise ValueError("No training windows created. Expand date range or relax gap filters.")

        # Mode mask
        X = self._apply_feature_mode(X)

        # log1p on dynamics
        for ch in x_log1p_channels:
            if ch < X.shape[-1]:
                X[..., ch] = np.log1p(np.clip(X[..., ch], a_min=0.0, a_max=None))

        # Standardize selected channels
        idx = [ch for ch in x_scale_channels if ch < X.shape[-1]]
        if idx:
            xm = X[..., idx].mean(axis=(0, 1, 2), keepdims=True)
            xs = X[..., idx].std(axis=(0, 1, 2), keepdims=True) + 1e-6
            X[..., idx] = (X[..., idx] - xm) / xs
            self._x_scale = {"idx": idx, "mean": xm, "std": xs}

        # Target transform (log1p + per-node z)
        if y_log1p:
            Y = np.log1p(np.clip(Y, a_min=0.0, a_max=None))

        obs_sum = (Y * SW).sum(axis=0)
        obs_cnt = SW.sum(axis=0) + 1e-6
        y_mu = obs_sum / obs_cnt
        y_std = np.sqrt(((SW * (Y - y_mu) ** 2).sum(axis=0) / obs_cnt)) + 1e-6
        Y = (Y - y_mu) / y_std

        self._y_mu, self._y_std = y_mu, y_std

        # ------------------------------------------------------------------
        # Simple 80/20 time-based split over windows (past seq_len → next 1)
        # ------------------------------------------------------------------
        B = X.shape[0]  # number of windows
        # Chronological 80/20 development/held-out split reported in the SI.
        split_idx = int(0.8 * B)
        self._split_idx = split_idx  # remember for later (weeks in CSV)

        if split_idx < 1 or split_idx >= B:
            # Fallback: if too few windows, keep everything as train
            self.X_train, self.Y_train, self.SW_train = X, Y, SW
            self.X_test, self.Y_test, self.SW_test = (
                np.empty((0,) + X.shape[1:], X.dtype),
                np.empty((0,) + Y.shape[1:], Y.dtype),
                np.empty((0,) + SW.shape[1:], SW.dtype),
            )
            print(f"[split] Not enough windows ({B}) for 80/20; using all as training.")
        else:
            # First 80% windows (earlier in time) → train
            # Last 20% (later in time)           → test
            self.X_train, self.Y_train, self.SW_train = (
                X[:split_idx], Y[:split_idx], SW[:split_idx]
            )
            self.X_test, self.Y_test, self.SW_test = (
                X[split_idx:], Y[split_idx:], SW[split_idx:]
            )
            print(f"[split] 80/20 time split: "
                  f"train windows={self.X_train.shape[0]}, "
                  f"test windows={self.X_test.shape[0]}")

        # Helpful logs
        print(f"Nodes kept: {len(self.node_names)} | "
              f"Counties={sum(n.startswith('C_') for n in self.node_names)} "
              f"| WWTPs={sum(n.startswith('W_') for n in self.node_names)}")
        print("X_train:", self.X_train.shape, "Y_train:", self.Y_train.shape, "SW_train:", self.SW_train.shape)
        print("X_test:",  self.X_test.shape,  "Y_test:",  self.Y_test.shape,  "SW_test:",  self.SW_test.shape)


    # -------------------- Model --------------------
    def build_model(
            self,
            dropout=0.2,
            l2_reg=2e-4,
            optimizer="adam",
            use_huber=True,
            train_in_original_space=True,
            lr_cosine=False,
            # --- regularization knobs ---
            use_flow_reg=True,
            flow_reg_lambda=1e-3,  # try 1e-4 ~ 1e-2
            proxy_align_csv=None,  # path to N×N csv aligned to self.node_names
            proxy_align_lambda=1e-4,  # try 1e-5 ~ 5e-4
            proxy_align_row_normalize=True,
    ):
        import tensorflow as tf
        import numpy as np
        import pandas as pd

        input_dim = self.X_train.shape[-1]
        num_nodes = int(self.num_nodes)
        adj = tf.convert_to_tensor(self.adj_static, dtype=tf.float32)

        # Diffusion-GRU stack
        class DiffusionGRUCell(tf.keras.layers.Layer):
            def __init__(self, units, adj, num_nodes, feature_dim=None, use_residual=True):
                super().__init__()
                self.units = int(units)
                self.num_nodes = int(num_nodes)
                self.adj = adj
                self.feature_dim = int(feature_dim) if feature_dim is not None else None
                self.use_residual = bool(use_residual)
                self.gru_cell = tf.keras.layers.GRUCell(self.units)
                self.residual_proj = tf.keras.layers.Dense(self.units) if self.use_residual else None
                self.state_size = [self.units] * self.num_nodes
                self.output_size = self.num_nodes * self.units

            def build(self, input_shape):
                feat_dim = self.feature_dim if self.feature_dim is not None else int(input_shape[-1])
                self._cell_input_dim = feat_dim + self.units
                self.gru_cell.build(tf.TensorShape([None, self._cell_input_dim]))
                if self.residual_proj is not None:
                    self.residual_proj.build(tf.TensorShape([None, feat_dim]))
                super().build(input_shape)

            def call(self, inputs, states, training=None):
                h_prev = tf.stack(states, axis=1)  # [B,N,U]
                x_diff = tf.einsum("ij,bjf->bif", self.adj, inputs)
                h_diff = tf.einsum("ij,bjf->bif", self.adj, h_prev)
                B = tf.shape(x_diff)[0]
                x_cat = tf.concat([x_diff, h_diff], axis=-1)
                x_flat = tf.reshape(x_cat, [B * self.num_nodes, -1])
                h_flat = tf.reshape(h_prev, [B * self.num_nodes, self.units])
                out_flat, [h_new_flat] = self.gru_cell(x_flat, [h_flat], training=training)
                if self.use_residual and self.residual_proj is not None:
                    rx = self.residual_proj(tf.reshape(x_diff, [B * self.num_nodes, -1]))
                    out_flat = out_flat + rx
                h_new = tf.reshape(h_new_flat, [B, self.num_nodes, self.units])
                new_states = tf.unstack(h_new, axis=1)
                out = tf.reshape(out_flat, [B, self.num_nodes * self.units])
                return out, new_states

        class DCRNNStack(tf.keras.Model):
            def __init__(self, num_nodes, hidden_dim, output_dim, adj, dropout, l2_reg, layers=1):
                super().__init__()
                self.num_nodes = num_nodes
                self.hidden_dim = hidden_dim
                self.num_layers = int(layers)

                self.rnn1 = tf.keras.layers.RNN(
                    DiffusionGRUCell(hidden_dim, adj, num_nodes, feature_dim=input_dim),
                    return_sequences=(self.num_layers > 1),
                )
                self.rnn2 = None
                if self.num_layers > 1:
                    self.rnn2 = tf.keras.layers.RNN(
                        DiffusionGRUCell(hidden_dim, adj, num_nodes, feature_dim=hidden_dim),
                        return_sequences=False,
                    )

                self.dropout = tf.keras.layers.Dropout(dropout) if dropout and dropout > 0 else None
                self.head = tf.keras.layers.Dense(1, activation=None,
                                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

            def call(self, inputs, training=False):
                # inputs: [B, T, N, F]
                x = self.rnn1(inputs, training=training)
                if self.num_layers > 1:
                    B = tf.shape(x)[0];
                    T = tf.shape(x)[1]
                    x = tf.reshape(x, [B, T, self.num_nodes, self.hidden_dim])
                    x = self.rnn2(x, training=training)  # -> [B, N*hidden]
                x = tf.reshape(x, [-1, self.num_nodes, self.hidden_dim])  # [B, N, hidden]
                if self.dropout is not None:
                    x = self.dropout(x, training=training)
                return self.head(x)  # [B, N, 1]

        base = DCRNNStack(num_nodes, self.hidden_dim, 1, adj, dropout, l2_reg, layers=self.rnn_layers)

        # -------- Functional wrapper so we can add losses on tensors (fixes inbound-nodes error) --------
        inp = tf.keras.Input(shape=(self.seq_len, num_nodes, input_dim), name="seq_input")
        y_pred_n = base(inp)  # normalized-space predictions [B,N,1]

        # ----------------- Loss in original space (existing behavior) -----------------
        mu = tf.constant(self._y_mu, dtype=tf.float32)  # [N,1]
        sd = tf.constant(self._y_std, dtype=tf.float32)  # [N,1]

        @tf.function
        def _inv(y_n):
            y = y_n * sd + mu
            y = tf.clip_by_value(y, -50.0, 50.0)
            return tf.math.expm1(y)

        # keep mu, sd, and _inv as-is
        def mae_original_elementwise(y_true_n, y_pred_n):
            y_true = _inv(y_true_n)
            y_pred = _inv(y_pred_n)
            # return per-element errors so sample_weight can be used
            return tf.abs(y_pred - y_true)

        def mae_original_mean(y_true_n, y_pred_n):
            # for logging only
            return tf.reduce_mean(mae_original_elementwise(y_true_n, y_pred_n))

        # Optional Huber loss in the original scale can reduce sensitivity to spikes.
        def huber_original_elementwise(delta=1.0):
            def _loss(y_true_n, y_pred_n):
                y_true = _inv(y_true_n)
                y_pred = _inv(y_pred_n)
                e = y_true - y_pred
                ae = tf.abs(e)
                quad = 0.5 * tf.square(e)
                lin = delta * (ae - 0.5 * delta)
                return tf.where(ae <= delta, quad, lin)  # element-wise

            return _loss

        if train_in_original_space:
            loss_fn = huber_original_elementwise(delta=1.0) if use_huber else mae_original_elementwise
            metrics = [mae_original_mean]
        else:
            loss_fn = tf.keras.losses.Huber() if use_huber else "mse"
            metrics = ["mae"]

        # Flow-Laplacian regularizer
        if use_flow_reg and (flow_reg_lambda or 0) > 0:
            A = tf.convert_to_tensor(self.adj_static, dtype=tf.float32)
            A = A - tf.linalg.tensor_diag(tf.linalg.diag_part(A))  # zero self
            rowsum = tf.reduce_sum(A, axis=1, keepdims=True) + 1e-9
            Arow = A / rowsum
            D = tf.linalg.tensor_diag(tf.reduce_sum(Arow, axis=1))
            L = D - Arow  # graph Laplacian

            def flow_reg(y_pred_tensor):  # y_pred_tensor: [B,N,1]
                y = tf.squeeze(y_pred_tensor, axis=-1)  # [B,N]
                Ly = tf.matmul(y, L)  # [B,N]
                return tf.reduce_mean(tf.reduce_sum(Ly * y, axis=1))

            # attach as a symbolic loss on the graph
            flow_loss = tf.keras.layers.Lambda(lambda z: flow_reg(z), name="flow_reg")(y_pred_n)
            # Keras requires the loss scalar to be added via add_loss:
            # multiply inside add_loss so it is tracked
            model_out = tf.keras.layers.Lambda(lambda z: z, name="identity_out")(y_pred_n)
            tmp_model = tf.keras.Model(inp, [model_out, flow_loss])
            tmp_model.add_loss(flow_reg_lambda * flow_loss)
            y_pred_n = tmp_model.outputs[0]  # continue with the main output

        # Optional proxy-alignment regularizer
        if proxy_align_csv is not None and (proxy_align_lambda or 0) > 0:
            try:
                import pandas as pd, numpy as np
                P = pd.read_csv(proxy_align_csv, index_col=0).reindex(index=self.node_names, columns=self.node_names)
                P = P.fillna(0.0).values.astype(np.float32)
                if proxy_align_row_normalize:
                    rs = P.sum(axis=1, keepdims=True);
                    rs[rs == 0] = 1.0
                    P = P / rs
                P = tf.constant(P, dtype=tf.float32)  # [N,N]

                def proxy_align_loss(y_pred_tensor):
                    y = tf.squeeze(y_pred_tensor, axis=-1)  # [B,N]
                    y = y - tf.reduce_mean(y, axis=0, keepdims=True)  # center per node
                    cov = tf.matmul(y, y, transpose_a=True) / (tf.cast(tf.shape(y)[0], tf.float32) + 1e-6)  # [N,N]
                    cov = tf.nn.relu(cov)
                    if proxy_align_row_normalize:
                        row_sum = tf.reduce_sum(cov, axis=1, keepdims=True) + 1e-9
                        cov = cov / row_sum
                    return tf.reduce_mean(tf.square(cov - P))

                pal = tf.keras.layers.Lambda(lambda z: proxy_align_loss(z), name="proxy_align")(y_pred_n)
                model_out2 = tf.keras.layers.Lambda(lambda z: z, name="identity_out2")(y_pred_n)
                tmp_model2 = tf.keras.Model(inp, [model_out2, pal])
                tmp_model2.add_loss(proxy_align_lambda * pal)
                y_pred_n = tmp_model2.outputs[0]
            except Exception as e:
                print("[proxy_align] skipped due to error:", e)

        # Final functional model (now y_pred_n has any add_loss attached)
        final_model = tf.keras.Model(inp, y_pred_n, name="dcrnn_stack")

        # -------------------------- Optimizer --------------------------
        if lr_cosine:
            lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3, first_decay_steps=200, t_mul=1.5, m_mul=0.9, alpha=1e-5
            )
            opt = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0) if optimizer == "adam" else optimizer

        # === BEGIN PATCH: robust Huber loss ===
        def huber_loss(delta=1.0):
            import tensorflow as tf
            def _loss(y_true, y_pred):
                e = y_true - y_pred
                ae = tf.abs(e)
                quad = 0.5 * tf.square(e)
                lin = delta * (ae - 0.5 * delta)
                return tf.where(ae <= delta, quad, lin)

            return _loss

        # Compile with the configured AdamW optimizer.
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=getattr(self, "lr", 1e-3))
        # self.model.compile(optimizer=opt, loss=huber_loss(delta=getattr(self, "huber_delta", 1.0)))
        # === END PATCH ===

        final_model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
        self.model = final_model

    def train_model(self, patience=20):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=int(patience), min_delta=1e-4,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.7, patience=8, min_lr=1e-5),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        self.model.fit(
            x=self.X_train, y=self.Y_train, sample_weight=self.SW_train,
            epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks,
            verbose=0, shuffle=False
        )

    # -------------------- Transmission --------------------
    def _pick_present_dynamic_channels(self, X, candidate=(0, 1, 7, 8), var_eps=1e-12):
        F = X.shape[-1]
        picks = []
        for ch in candidate:
            if 0 <= ch < F:
                arr = X[..., ch]
                if np.any(np.abs(arr) > 0) and (np.nanstd(arr) > var_eps):
                    picks.append(ch)
        if not picks:
            for ch in range(F):
                arr = X[..., ch]
                if np.any(np.abs(arr) > 0) and (np.nanstd(arr) > var_eps):
                    picks.append(ch)
        return sorted(set(picks))

    def _gate_by_flow_mask(self, M, A_row, thr):
        """
        Gates M by zeroing entries where A_row < thr.
        Works for M with shape (N,N) or (B,N,N).
        """
        import numpy as np
        mask = (A_row >= thr).astype(M.dtype)  # [N,N]
        if M.ndim == 3:
            return M * mask[None, :, :]
        return M * mask

    # -------------------- Outputs --------------------
    def save_results(self, risk_matrices, tag=""):
        for mode, mat in risk_matrices.items():
            pd.DataFrame(mat, index=self.node_names, columns=self.node_names).to_csv(
                os.path.join(self.output_dir, f"transmission_risk_matrix_{mode}{tag}.csv")
            )

    def inverse_transform_y(self, y_hat):
        y = y_hat * self._y_std + self._y_mu
        y = np.expm1(np.clip(y, a_min=-50, a_max=None))
        return np.maximum(y, 0.0)

    def save_predicted_and_true(self, use="train", tag=""):
        import numpy as np
        import pandas as pd
        import os

        use = str(use).lower()

        # ----------------- select split -----------------
        if use == "train":
            X, Yn, SW = self.X_train, self.Y_train, self.SW_train
        elif use in ("val", "test"):
            X  = self.X_test  if self.X_test.size  else self.X_train
            Yn = self.Y_test  if self.Y_test.size  else self.Y_train
            SW = self.SW_test if self.SW_test.size else self.SW_train
        elif use == "all":
            # concat train + test in time order (if test exists)
            if self.X_test.size:
                X  = np.concatenate([self.X_train,  self.X_test],  axis=0)
                Yn = np.concatenate([self.Y_train, self.Y_test],  axis=0)
                SW = np.concatenate([self.SW_train, self.SW_test], axis=0)
            else:
                X, Yn, SW = self.X_train, self.Y_train, self.SW_train
        else:
            raise ValueError(f"Unknown split '{use}' (expected 'train','val','test','all').")

        # ----------------- predictions -----------------
        yhat_n = self.model.predict(X, verbose=0)
        yhat = self.inverse_transform_y(yhat_n)[:, :, 0]
        ytrue = self.inverse_transform_y(Yn)[:, :, 0]
        mask = SW[:, :, 0] if SW is not None else np.ones_like(ytrue)

        # ----------------- week index -----------------
        # Train windows correspond to targets at indices [seq_len ... seq_len+train-1]
        # Test  windows correspond to targets at indices [seq_len+split_idx ...]
        if use in ("val", "test"):
            start_offset = getattr(self, "_split_idx", 0)
        else:
            # "train" or "all": start at first target window
            start_offset = 0

        weeks = [
            pd.to_datetime(self.time_index[self.seq_len + start_offset + i]).strftime("%Y-%m-%d")
            for i in range(X.shape[0])
        ]

        df_pred = pd.DataFrame(yhat, columns=[f"pred_{n}" for n in self.node_names], index=weeks)
        df_true = pd.DataFrame(ytrue, columns=[f"true_{n}" for n in self.node_names], index=weeks)
        df_mask = pd.DataFrame(mask, columns=[f"mask_{n}" for n in self.node_names], index=weeks)

        out = pd.concat([df_pred, df_true, df_mask], axis=1)
        csv_path = os.path.join(self.output_dir, f"pred_vs_true_{use}{tag}.csv")
        out.to_csv(csv_path)
        print(f"[Saved] {csv_path}")
        return csv_path

    def evaluate_predictions(self, use="train", output_tag=""):
        import numpy as np
        import pandas as pd
        import os

        use = str(use).lower()

        # ----------------- select split -----------------
        if use == "train":
            X, Yn, SW = self.X_train, self.Y_train, self.SW_train
        elif use in ("val", "test"):
            X  = self.X_test  if self.X_test.size  else self.X_train
            Yn = self.Y_test  if self.Y_test.size  else self.Y_train
            SW = self.SW_test if self.SW_test.size else self.SW_train
        elif use == "all":
            if self.X_test.size:
                X  = np.concatenate([self.X_train,  self.X_test],  axis=0)
                Yn = np.concatenate([self.Y_train, self.Y_test],  axis=0)
                SW = np.concatenate([self.SW_train, self.SW_test], axis=0)
            else:
                X, Yn, SW = self.X_train, self.Y_train, self.SW_train
        else:
            raise ValueError(f"Unknown split '{use}' (expected 'train','val','test','all').")

        # ----------------- predictions -----------------
        yhat = self.inverse_transform_y(self.model.predict(X, verbose=0))
        ytrue = self.inverse_transform_y(Yn)
        mask = SW if SW is not None else np.ones_like(ytrue)

        denom = mask.sum(axis=0) + 1e-6
        mae_node = (np.abs((yhat - ytrue)) * mask).sum(axis=0) / denom
        rmse_node = np.sqrt((np.square((yhat - ytrue)) * mask).sum(axis=0) / denom)

        df = pd.DataFrame({"node": self.node_names, "MAE": mae_node[:, 0], "RMSE": rmse_node[:, 0]})
        csv_path = os.path.join(self.output_dir, f"per_node_metrics_{use}{output_tag}.csv")
        df.to_csv(csv_path, index=False)

        overall_mae = (np.abs((yhat - ytrue)) * mask).sum() / (mask.sum() + 1e-6)
        overall_rmse = np.sqrt((np.square((yhat - ytrue)) * mask).sum() / (mask.sum() + 1e-6))
        print(f"Overall {use}  MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}")
        return {"df": df, "overall": (overall_mae, overall_rmse)}

    def _predict_inference(self, X, training: bool = False):
        """Run the model with optional dropout active."""
        import numpy as np, tensorflow as tf
        # Keras Functional models can be called directly to force training=True
        y = self.model(X, training=bool(training))
        return y.numpy() if isinstance(y, (tf.Tensor,)) else np.asarray(y)

    def compute_transmission_matrix(
            self, use="train", mask_features=None,
            perturb="relative", alpha=0.2,
            variance_normalize=True, percent_change=True,
            normalize=True, self_mode="skip", self_dampen=0.5,
            renorm_offdiag=True, return_timewise=False,
            per_batch_normalize=False,
            use_trend=False,
            # ---- stability knobs expected by Main_quick ----
            mc_samples: int = 0,  # e.g., 16
            enable_dropout: bool = False,  # keep dropout active during MC
            return_std: bool = False,  # return (mean, std[, agree])
            raw_trend_consensus: bool = False,  # compute RAW & TREND, then combine
            consensus_rule: str = "min",  # {"min","mean","geom"}
            agree_tau: float = 0.10  # (kept for API parity; thresholding done upstream)
    ):
        """
        Transmission sensitivity via node-wise feature perturbation.

        Returns:
          - if return_timewise:
              (risk_time[B,N,N], risk_std[B,N,N], weeks[list], agree[N,N]?) depending on flags
          - else:
              (risk_mean[N,N], risk_std[N,N], agree[N,N]?) depending on flags
        """
        import numpy as np
        import pandas as pd

        use = str(use).lower()

        # ----- select split -----
        if use == "train":
            X = self.X_train
        elif use in ("val", "test"):
            X = self.X_test if self.X_test.size else self.X_train
        elif use == "all":
            if self.X_test.size:
                X = np.concatenate([self.X_train, self.X_test], axis=0)
            else:
                X = self.X_train
        else:
            raise ValueError(f"Unknown split '{use}' (expected 'train','val','test','all').")

        if not isinstance(X, np.ndarray) or X.size == 0:
            raise ValueError("No data for prediction.")

        # ----- helpers -----
        def _predict(Xinp, training_flag: bool):
            y = self.model(Xinp, training=bool(training_flag))
            y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
            return y  # [B,N,1] in normalized space

        def _pick_mask_idx(Xarr, use_trend_local: bool):
            F = Xarr.shape[-1]
            if use_trend_local:
                cand = []
                if hasattr(self, "_idx_trend_clin") and self._idx_trend_clin is not None:
                    cand.append(self._idx_trend_clin)
                if hasattr(self, "_idx_trend_wval") and self._idx_trend_wval is not None:
                    cand.append(self._idx_trend_wval)
                idx = [i for i in cand if 0 <= i < F]
                if not idx:
                    idx = self._pick_present_dynamic_channels(Xarr, candidate=(0, 1, 7, 8))
            else:
                if mask_features is None and getattr(self, "auto_pick_dynamic", True):
                    idx = self._pick_present_dynamic_channels(Xarr, candidate=(0, 1, 7, 8))
                else:
                    default = [0, 1, 7, 8]
                    idx = [i for i in (default if mask_features is None else mask_features) if 0 <= i < F]
            if not idx:
                raise ValueError("No valid mask features for transmission sensitivity.")
            return sorted(set(idx))

        def _one_pass(use_trend_local: bool, training_flag: bool):
            """
            One Monte Carlo sample:
            - predict BASE
            - perturb node i features on chosen channels
            - accumulate |Δ| (with optional variance/percent normalization)
            - zero self, row-normalize if requested
            - optional global normalization
            """
            B, N = X.shape[0], int(self.num_nodes)
            base = _predict(X, training_flag)  # [B,N,1]
            mask_idx = _pick_mask_idx(X, use_trend_local)

            # variance normalize across time if requested
            if variance_normalize:
                dst_std = np.std(base, axis=0, keepdims=True) + 1e-6  # [1,N,1]

            risk_time = np.zeros((B, N, N), dtype=np.float32)

            for i in range(N):
                Xm = X.copy()
                if perturb == "zero":
                    Xm[:, :, i, mask_idx] = 0.0
                else:  # relative shrink
                    Xm[:, :, i, mask_idx] *= (1.0 - float(alpha))
                pred = _predict(Xm, training_flag)  # [B,N,1]
                delta = np.abs(base - pred)  # [B,N,1]
                if variance_normalize:
                    delta = delta / dst_std
                if percent_change:
                    denom = np.abs(base) + 1e-6
                    delta = delta / denom
                risk_time[:, i, :] = delta[:, :, 0]  # place along row i

            # self handling + row renorm (off-diagonal) to make rows comparable
            if self_mode == "skip":
                for b in range(B):
                    np.fill_diagonal(risk_time[b], 0.0)
                    if renorm_offdiag:
                        rs = risk_time[b].sum(axis=1, keepdims=True) + 1e-9
                        risk_time[b] = risk_time[b] / rs

            # global scale to [0,1] if requested
            if normalize:
                g = risk_time.max()
                if g > 0:
                    risk_time = risk_time / g

            # optional dampen diagonal if included
            if (self_mode != "skip") and (self_dampen not in (None, 1.0)):
                for b in range(B):
                    d = np.diag(risk_time[b]) * float(self_dampen)
                    np.fill_diagonal(risk_time[b], d)

            return risk_time  # [B,N,N]

        # ----- MC stack (with optional RAW↔TREND consensus) -----
        S = max(1, int(mc_samples))
        training_flag = bool(enable_dropout)

        if raw_trend_consensus:
            raw_stack = np.stack([_one_pass(False, training_flag) for _ in range(S)], axis=0)  # [S,B,N,N]
            trd_stack = np.stack([_one_pass(True, training_flag) for _ in range(S)], axis=0)   # [S,B,N,N]

            if consensus_rule == "min":
                samples = np.minimum(raw_stack, trd_stack)
            elif consensus_rule == "geom":
                samples = np.sqrt(np.maximum(raw_stack, 0.0) * np.maximum(trd_stack, 0.0))
            else:  # "mean"
                samples = 0.5 * (raw_stack + trd_stack)

            eps = 1e-9
            agree = 1.0 - np.abs(raw_stack - trd_stack) / (np.maximum(raw_stack, trd_stack) + eps)
            agree = agree.mean(axis=(0, 1))  # [N,N]
        else:
            samples = np.stack([_one_pass(bool(use_trend), training_flag) for _ in range(S)], axis=0)  # [S,B,N,N]
            agree = None

        # ----- return shapes: timewise or collapsed -----
        if return_timewise:
            risk_mean = samples.mean(axis=0)  # [B,N,N]
            risk_std = samples.std(axis=0, ddof=1)  # [B,N,N]
            weeks = [pd.to_datetime(t).strftime("%Y-%m-%d")
                     for t in self.time_index[self.seq_len:self.seq_len + risk_mean.shape[0]]]
            if return_std and agree is not None:
                return (risk_mean, risk_std, weeks, agree)
            elif return_std:
                return (risk_mean, risk_std, weeks)
            else:
                return (risk_mean, weeks)

        risk_mean = samples.mean(axis=(0, 1))  # [N,N]
        risk_std = samples.std(axis=(0, 1), ddof=1)  # [N,N]

        if return_std and agree is not None:
            return (risk_mean, risk_std, agree)
        elif return_std:
            return (risk_mean, risk_std)
        else:
            return risk_mean

    def save_risk_signal_correlations(
        self,
        use="train",
        signal="clinical",      # {"clinical","wval"}
        mode="target",          # {"target","source"}
        corr_abs=True,
        min_periods=2,
        tag=""
    ):
        """
        Correlate time-resolved transmission risk with a node signal (clinical or WVAL).
        Outputs an N×N CSV (correlation matrix).
        - mode="target": corr(risk_i→j, signal_j)
        - mode="source": corr(risk_i→j, signal_i)
        """
        import numpy as np
        import pandas as pd
        import os

        # 1) Risk cube
        risk_time, weeks = self.compute_transmission_matrix(
            use=use, return_timewise=True, renorm_offdiag=True
        )  # [B,N,N]

        # 2) Signals (train/val/all consistent with 'use')
        use_l = str(use).lower()
        if use_l == "train":
            X, Yn = self.X_train, self.Y_train
        elif use_l in ("val", "test"):
            X  = self.X_test  if self.X_test.size  else self.X_train
            Yn = self.Y_test  if self.Y_test.size  else self.Y_train
        elif use_l == "all":
            if self.X_test.size:
                X  = np.concatenate([self.X_train,  self.X_test],  axis=0)
                Yn = np.concatenate([self.Y_train, self.Y_test],  axis=0)
            else:
                X, Yn = self.X_train, self.Y_train
        else:
            raise ValueError(f"Unknown split '{use}' (expected 'train','val','test','all').")

        if signal == "clinical":
            S = self.inverse_transform_y(Yn)[:, :, 0]  # [B,N]
        elif signal == "wval":
            S = X[:, -1, :, 1]
        else:
            raise ValueError("signal must be 'clinical' or 'wval'")

        B, N, _ = risk_time.shape

        def _pair_corr(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < int(min_periods):
                return 0.0
            r = np.corrcoef(x[m], y[m])[0, 1]
            return abs(r) if corr_abs else r

        out = np.zeros((N, N), dtype=np.float32)
        if mode == "target":
            for i in range(N):
                for j in range(N):
                    out[i, j] = _pair_corr(risk_time[:, i, j], S[:, j])
        elif mode == "source":
            for i in range(N):
                for j in range(N):
                    out[i, j] = _pair_corr(risk_time[:, i, j], S[:, i])
        else:
            raise ValueError("mode must be 'target' or 'source'")

        csv = os.path.join(self.output_dir, f"corr_risk_vs_{signal}_{mode}{tag}.csv")
        pd.DataFrame(out, index=self.node_names, columns=self.node_names).to_csv(csv)
        print(f"[Saved] {csv}")

        try:
            import Model_DCRNN_Viz as viz
            viz.plot_risk_heatmap(
                out, self.node_names,
                title=f"Corr(risk vs {signal}, {mode}){tag}",
                output_dir=self.output_dir,
                skip_self=False, highlight_self=True,
            )
        except Exception as e:
            print("[warn] plotting skipped:", e)

    def save_transmission_pred_true_heatmaps(
            self,
            use="train",
            self_mode="include",
            tag="",
            # --- Proxy construction knobs ---
            save_clinical_only_proxy=True,
            save_mobility_weighted_proxy=True,
            flow_symmetrize="mean",  # {"mean","sum","max"}
            flow_gamma=1.0,  # >1 highlights high-flow links (power transform)
            flow_min_q=None,  # e.g., 0.20 -> zero edges below this global quantile
            corr_abs=True,
            corr_min_periods=2,
            # Mapping and cluster-visualization settings
            make_cluster_maps=False,  # turn on to also make WWTP cluster maps for all three
            cluster_k=None,  # None=auto-k, or set an int (e.g., 4)
            cluster_top_q=0.90,  # sparsify before spectral clustering
            cluster_k_min=2,
            cluster_k_max=8,
    ):
        """
        Saves three heatmaps/CSVs and (optionally) WWTP cluster maps:
          1) Predicted transmission (model sensitivity)
          2) TRUE proxy: clinical-only correlation (optional)
          3) TRUE proxy: clinical × mobility weighted (optional)
        If make_cluster_maps=True, also generates WWTP cluster maps for each case.
        """

        # ------------------------------------------------------------------
        # (A) PREDICTED (model-based) transmission
        # ------------------------------------------------------------------
        pred_mat = self.compute_transmission_matrix(use=use, self_mode=self_mode, renorm_offdiag=True)
        pred_csv = os.path.join(self.output_dir, f"transmission_predicted{tag}.csv")
        pd.DataFrame(pred_mat, index=self.node_names, columns=self.node_names).to_csv(pred_csv)

        viz.plot_risk_heatmap(
            pred_mat, self.node_names,
            title=f"Transmission_PREDICTED_all_nodes{tag}",
            output_dir=self.output_dir,
            skip_self=(self_mode == "skip"),
            highlight_self=(self_mode != "skip"),
        )

        # We’ll need this for cluster maps (fallback similarity & timestamps)
        pred_true_csv_path = os.path.join(self.output_dir, f"pred_vs_true_{use}{tag}.csv")

        # ------------------------------------------------------------------
        # Fetch observed clinical series (for proxy construction)
        # ------------------------------------------------------------------
        X = self.X_train if use != "val" else (self.X_test if self.X_test.size else self.X_train)
        Yn = self.Y_train if use != "val" else (self.Y_test if self.Y_test.size else self.Y_train)
        SW = self.SW_train if use != "val" else (self.SW_test if self.SW_test.size else self.SW_train)

        ytrue = self.inverse_transform_y(Yn)[:, :, 0]  # [B,N]
        mask = SW[:, :, 0] if SW is not None else np.ones_like(ytrue)
        mask_df = pd.DataFrame(mask, columns=self.node_names).astype(bool)
        Ydf = pd.DataFrame(ytrue, columns=self.node_names)
        Ydf = Ydf.where(mask_df, np.nan)

        # ------------------------------------------------------------------
        # (B) TRUE proxy: clinical-only correlation
        # ------------------------------------------------------------------
        clinical_csv = None
        if save_clinical_only_proxy:
            clinical_corr = Ydf.corr(method="pearson", min_periods=corr_min_periods).fillna(0.0).values
            if corr_abs:
                clinical_corr = np.abs(clinical_corr)

            clinical_csv = os.path.join(self.output_dir, f"transmission_TRUE_proxy_corr{tag}.csv")
            pd.DataFrame(clinical_corr, index=self.node_names, columns=self.node_names).to_csv(clinical_csv)

            viz.plot_risk_heatmap(
                clinical_corr, self.node_names,
                title=f"Transmission_TRUE_proxy_corr_all_nodes{tag}",
                output_dir=self.output_dir,
                skip_self=False, highlight_self=True,
            )

        # ------------------------------------------------------------------
        # (C) TRUE proxy: clinical × mobility weighted
        # ------------------------------------------------------------------
        proxy_mob_csv = None
        if save_mobility_weighted_proxy:
            clinical_corr2 = Ydf.corr(method="pearson", min_periods=corr_min_periods).fillna(0.0).values
            if corr_abs:
                clinical_corr2 = np.abs(clinical_corr2)

            A = np.array(self.adj_static, dtype=np.float32)
            if flow_symmetrize == "sum":
                F = A + A.T
            elif flow_symmetrize == "max":
                F = np.maximum(A, A.T)
            else:
                F = 0.5 * (A + A.T)
            np.fill_diagonal(F, 0.0)

            if flow_min_q is not None:
                pos = F[F > 0]
                thr = np.quantile(pos, flow_min_q) if pos.size else 0.0
                F = np.where(F >= thr, F, 0.0)

            if flow_gamma != 1.0:
                F = np.power(F, float(flow_gamma))
            mF = F.max()
            if mF > 0:
                F = F / mF

            proxy_mob = clinical_corr2 * F
            # Row renorm off-diagonal (helps visual comparability)
            row_sum = proxy_mob.sum(axis=1, keepdims=True) + 1e-9
            proxy_mob = np.divide(proxy_mob, row_sum, out=np.zeros_like(proxy_mob), where=row_sum > 0)

            proxy_mob_csv = os.path.join(self.output_dir, f"transmission_TRUE_proxy_corr_x_mobility{tag}.csv")
            pd.DataFrame(proxy_mob, index=self.node_names, columns=self.node_names).to_csv(proxy_mob_csv)

            viz.plot_risk_heatmap(
                proxy_mob, self.node_names,
                title=f"Transmission_TRUE_proxy_corr×Mobility_all_nodes{tag}",
                output_dir=self.output_dir,
                skip_self=False, highlight_self=True,
            )

        # ------------------------------------------------------------------
        # (D) Optional: WWTP cluster maps for each matrix
        # ------------------------------------------------------------------
        if make_cluster_maps:
            try:

                wwtp_gdf = viz.load_wwtp_points(self.wwtp_shapefile)

                # predicted matrix cluster map
                viz.proxy_group_colors_on_map(
                    pred_true_csv=pred_true_csv_path,
                    wwtp_gdf=wwtp_gdf,
                    out_dir=self.output_dir,
                    tag=tag + "_PRED",
                    matrix_csv=pred_csv,
                    k=cluster_k,
                    top_q=cluster_top_q,
                    k_min=cluster_k_min,
                    k_max=cluster_k_max,
                )

                # clinical-only proxy cluster map
                if clinical_csv is not None:
                    viz.proxy_group_colors_on_map(
                        pred_true_csv=pred_true_csv_path,
                        wwtp_gdf=wwtp_gdf,
                        out_dir=self.output_dir,
                        tag=tag + "_PROXY_CLIN",
                        matrix_csv=clinical_csv,
                        k=cluster_k,
                        top_q=cluster_top_q,
                        k_min=cluster_k_min,
                        k_max=cluster_k_max,
                    )

                # clinical×mobility proxy cluster map
                if proxy_mob_csv is not None:
                    viz.proxy_group_colors_on_map(
                        pred_true_csv=pred_true_csv_path,
                        wwtp_gdf=wwtp_gdf,
                        out_dir=self.output_dir,
                        tag=tag + "_PROXY_CLINxMOB",
                        matrix_csv=proxy_mob_csv,
                        k=cluster_k,
                        top_q=cluster_top_q,
                        k_min=cluster_k_min,
                        k_max=cluster_k_max,
                    )

            except Exception as e:
                print("[Cluster Maps] Skipped due to error:", e)
