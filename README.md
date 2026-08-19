# Wastewater Sentinel Network Design

Code and derived outputs for a mobility-informed framework for designing fixed-capacity wastewater sentinel surveillance networks.

This repository accompanies the manuscript **“Designing Wastewater Sentinel Surveillance Networks Using Mobility-Informed Coverage and Disease-Risk Dynamics”** (under review). It is intended to document the analytical workflow, reported configuration, and public-safe aggregate derived outputs. It does not redistribute licensed mobility records, restricted clinical data, or confidential facility identifiers.

The workflow has two components:

1. **Level 1: multi-criteria site selection.** Candidate wastewater treatment plants (WWTPs) are ranked using population served, mobility activity, mobility-connected population and area, mobility-weighted infection risk, spatial representation, and subnetwork redundancy.
2. **Level 2: disease-dynamic refinement.** A diffusion recurrent neural network estimates model-based WWTP contributions to predicted regional disease dynamics. These scores complement rather than replace Level 1 selection.

## Practical interpretation

The workflow is a decision-support tool, not an automatic site-replacement algorithm. Level 1 supports recruitment, retention, expansion, and fixed-capacity planning using interpretable site and network criteria. Level 2 provides complementary evidence about whether information associated with a site contributes to predicted regional disease patterns. Local feasibility, laboratory capacity, sampling agreements, equity, and public-health judgment remain necessary inputs to final decisions.

## Repository contents

```text
config/                         Reported analysis settings
docs/                           Data schema, availability, and workflow notes
derived_results/                Shareable monthly derived figures
src/preprocessing/              Commute-distance and spatial preprocessing
src/level1/                     Level 1 selection and baseline evaluation
src/level2/                     Level 2 model, risk, and visualization modules
environment.yml                 Open-source Python environment
```

## Data availability

This repository does **not** contain licensed StreetLight mobility data or restricted Colorado Department of Public Health and Environment data. The public code therefore cannot reproduce the Colorado case study end to end without authorized access to those inputs. See [`docs/data_availability.md`](docs/data_availability.md) and [`docs/input_schema.md`](docs/input_schema.md).

## Environment

The open-source portion of the workflow was developed for Python 3.10. Create the environment with:

```bash
conda env create -f environment.yml
conda activate wastewater-sentinel
```

`Process_UtilityCountyAdj_ArcPy.py` is optional and requires a separate ArcGIS Pro Python environment with ArcPy. ArcPy is not installed by `environment.yml`.

## Execution order

The scripts expect authorized input files to be arranged according to `docs/input_schema.md`. Some source modules retain the relative directory structure used in the study; update project-root paths locally before execution.

The relationship between repository outputs and the manuscript is summarized in [`docs/manuscript_output_crosswalk.md`](docs/manuscript_output_crosswalk.md).
The purpose and execution role of every retained Python script are listed in [`docs/script_inventory.md`](docs/script_inventory.md).

### 1. Optional spatial preprocessing

```bash
python src/preprocessing/Process_CommuteTimeDistance.py \
  --commute-time path/to/zip_code_commute_time.csv \
  --ruca path/to/RUCA2010zipcode.xlsx \
  --output path/to/zip_code_commute_distance_adjusted.csv
```

If utility-to-county spatial allocation must be regenerated, run `Process_UtilityCountyAdj_ArcPy.py` in ArcGIS Pro. Then use `Process_SewerWeights.py` to create overlap-aware sewershed weights.

### 2. Generate mobility-weighted clinical-risk inputs

Run from `src/level2` so the local module imports resolve:

```bash
cd src/level2
python Analysis_ODDiffusionRisk_newdata_clinicalonly.py
```

### 3. Run Level 1 selection and baseline comparisons

Run from `src/level1` using the reported main configuration:

```bash
cd ../level1
python AnalysisViz_NetworkCreation_Ranking_Level1Compare.py \
  --selection_fraction 0.25 \
  --total_N 20 \
  --singleton_top_pct 0.30 \
  --singleton_drop_km 100 \
  --singleton_bottom_pct 0.20 \
  --singleton_reserve_k 6 \
  --singleton_reserve_frac 0.0 \
  --isolation_bonus_km 100 \
  --risk_weight 0.5 \
  --coverage_weight 0.5
```

The same settings are recorded in `config/level1_main.yaml`. The analysis is repeated for each month of 2024 using the corresponding monthly input fields.

### 4. Run Level 2 model evaluation

From `src/level2`:

```bash
python Model_DCRNN_Main_quick_raw_newdata.py \
  --disease COVID \
  --feature_mode case_only \
  --type rate \
  --seed 42 \
  --verify_od
```

The script uses a chronological split, with the first 80% of weekly observations used for model development and the final 20% retained for evaluation. Hyperparameter settings evaluated in the study are summarized in `config/level2_model.yaml`.

## Derived monthly results

`derived_results/` contains a compact visual record of the complete 12-month analysis. The `network_maps/base/` figures report Level 1-only rankings; `network_maps/risk/` adds the Level 2 disease-dynamic contribution/transmission result. Each monthly folder-equivalent set includes a privacy-safe static ranked-network map, a rank heatmap, and a cumulative-benefit plot. `level1_summary/` contains the monthly Level 1 benefit-bar and spatial-comparison figures.

The interactive HTML maps were converted to static PNGs for stable GitHub display. Facility names are omitted from the public static maps; county-based labels and letter suffixes in the heatmaps are geographic/anonymized labels rather than a facility-name crosswalk. Spreadsheet tables, HTML files, intermediate summaries, and site-frequency outputs are intentionally excluded. Licensed or restricted source records and the confidential facility-name crosswalk are not included.

Maps retain only the geographic detail already reported in the manuscript. If an applicable data-use agreement also restricts facility locations, remove the spatial-map files before release.

## Reuse

Users applying the workflow elsewhere should replace the restricted Colorado inputs with locally authorized data matching the documented schemas. CBSA-based and population-only comparisons can be implemented without commercial mobility data, although mobility-connected metrics cannot be interpreted equivalently when high-resolution origin-destination data are unavailable.

The repository is provided for peer review and scholarly examination. Reuse, redistribution, modification, or incorporation into another product or publication requires prior written permission; see `LICENSE`.

## Citation

Please cite the associated manuscript when using this code. Full bibliographic information and the DOI will be added after publication.

Li, J.; Weisbeck, K.; Girgente, G.; Cook, R.; Wheeler, A.; Nicklay, T.; Lengsfeld, C. *Designing Wastewater Sentinel Surveillance Networks Using Mobility-Informed Coverage and Disease-Risk Dynamics.* Manuscript under review.
