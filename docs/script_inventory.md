# Script inventory and execution role

The repository retains only scripts that support the reported preprocessing, Level 1 analysis, Level 2 analysis, or public derived outputs. Optional visualization helpers remain because they are called by the reported workflow. Debug-only adjacency export and the unused early-detection-speed output were removed from the public release.

## Preprocessing

- `Process_CommuteTimeDistance.py`: converts ACS ZIP-level commute-time context to estimated commute distance using RUCA-based assumptions.
- `Process_UtilityCountyAdj_ArcPy.py`: creates overlap-aware utility-to-county spatial allocation in an ArcGIS Pro environment.
- `Process_SewerWeights.py`: constructs population and area weights for overlapping sewershed and census geometries.

## Level 1

- `AnalysisViz_NetworkCreation_Ranking_Level1Compare.py`: main multi-criteria ranking and subnetwork-aware fixed-capacity selection workflow.
- `AnalysisViz_NetworkCreation_FileGen.py`: exports network edges, nodes, sentinel tables, and contribution-site tables.
- `AnalysisViz_NetworkRanking_DiffusionFilePicker.py`: selects a Level 2 contribution result using explicit validation metrics.
- `AnalysisViz_Level1BaselineComparison.py`: evaluates population-only, spatial-only, CBSA-based, existing-network, and proposed Level 1 strategies at the same network size.
- `AnalysisViz_CumulativeCoverageV1.py`: calculates cumulative and marginal surveillance-benefit summaries.

## Level 2 and mobility-risk preparation

- `Process_ODData_Aggr.py`: aggregates authorized daily OD records to weekly summaries.
- `AnalysisViz_ODFlowPure.py`: creates weekly directional mobility-flow matrices and optional maps.
- `Process_ViralLoader.py`: provides shared weekly clinical and optional wastewater preprocessing.
- `Process_ViralLoader_V2.py`: harmonizes county and WWTP clinical inputs for the case-only Level 2 workflow.
- `Analysis_ODDiffusionRisk_newdata_clinicalonly.py`: computes mobility-weighted clinical-risk inputs.
- `Analysis_ODDiffusionRisk_newdata_FileGen.py`: exports weekly and monthly risk summaries.
- `Model_DCRNN_Simple_newdata.py`: implements the county-WWTP diffusion-GRU and perturbation analysis.
- `Model_DCRNN_Main_quick_raw_newdata.py`: runs training, chronological evaluation, sensitivity variants, and contribution export.
- `Model_DCRNN_Viz.py`: creates held-out prediction and contribution diagnostics.

## Public-release cleanup principles

- No licensed or restricted inputs are included.
- No personal absolute file paths, API keys, or credentials are included.
- Facility names in public result tables use anonymous `WWTP_###` identifiers.
- Model contribution scores are documented as sensitivity measures, not causal effects.
- The reported Level 2 configuration uses hospitalization indicators; optional viral-load channels are not used in the reported case-only implementation.
