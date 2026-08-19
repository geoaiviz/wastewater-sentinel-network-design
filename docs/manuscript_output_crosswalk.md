# Manuscript and repository crosswalk

This guide links the public repository to the analyses described in the manuscript and Supporting Information (SI). File availability does not imply that restricted source data are redistributed.

## Level 1 site selection

- Reported configuration: `config/level1_main.yaml`
- Main selection workflow: `src/level1/AnalysisViz_NetworkCreation_Ranking_Level1Compare.py`
- Fixed-size reference comparisons: `src/level1/AnalysisViz_Level1BaselineComparison.py`
- Cumulative and marginal benefit calculations: `src/level1/AnalysisViz_CumulativeCoverageV1.py`
- Corresponding manuscript content: Materials and Methods, “Two-Level Sentinel Site-Selection Framework” and “Comparative Evaluation”; Results and Discussion, “Level one Selection, Comparison, and Cumulative Benefit”; SI Texts S2 and S4.

## Level 2 disease-dynamic refinement

- Reported model settings: `config/level2_model.yaml`
- Model training and evaluation: `src/level2/Model_DCRNN_Main_quick_raw_newdata.py`
- Perturbation and contribution analysis: `src/level2/Model_DCRNN_Simple_newdata.py` and `src/level2/Model_DCRNN_Viz.py`
- Corresponding manuscript content: Materials and Methods, Level two description; Results and Discussion, “Level two Disease-Dynamic Refinement and Site Contribution”; SI Texts S3 and S5.

## Public derived outputs

- Level 1-only monthly ranked maps, rank heatmaps, and cumulative-benefit figures: `derived_results/network_maps/base/`
- Monthly ranked maps, rank heatmaps, and cumulative-benefit figures after adding the Level 2 result: `derived_results/network_maps/risk/`
- Monthly Level 1 benefit bars and spatial comparison maps: `derived_results/level1_summary/`

Tabular intermediates, cutoff summaries, site-frequency outputs, marginal-gain plots, and interactive HTML are not included in the public release. The HTML maps were converted to static PNGs with facility names suppressed. County-based labels and letter suffixes in rank heatmaps are public geographic/anonymized labels; the confidential facility-name crosswalk is not included.

## Reproducibility boundary

The repository supports review of the code, configuration, workflow sequence, and reported nonidentifiable outputs. Complete end-to-end regeneration of the Colorado case study requires separately authorized access to StreetLight origin-destination data and restricted clinical and sewershed inputs. This limitation is described in `data_availability.md` and in the manuscript Data Availability Statement.
