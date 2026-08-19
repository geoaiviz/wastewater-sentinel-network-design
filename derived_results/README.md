# Derived visual results

This directory contains the retained, public-safe figures for all 12 months of 2024.

## Directory guide

- `network_maps/base/`: Level 1-only results. Each month has a static ranked-network map, a rank heatmap, and a cumulative-fraction plot.
- `network_maps/risk/`: Level 1 results augmented with the Level 2 disease-dynamic contribution/transmission result. Each month has the same three figure types.
- `level1_summary/`: monthly Level 1 benefit-bar and spatial-comparison figures.

The `base`/`risk` distinction therefore indicates whether the Level 2 result is absent or included; it does not indicate a different source-data release.

## Privacy and file format

Interactive HTML maps were converted to PNG for stable display on GitHub. The static maps intentionally omit WWTP names. Rank heatmaps retain county-based names and anonymized letter suffixes (for example, `El Paso C`); these are not a facility-name crosswalk. No CSV, Excel, HTML, restricted raw data, or confidential name crosswalk is included.

## Monthly file naming

The filename contains the month and analysis mode, for example:

- `ranked_network_map_2024-01_base.png`
- `rank_heatmap_2024-01_risk.png`
- `cumulative_fraction_2024-01_risk.png`
- `benefit_bar_2024-01_level1_N20.png`
- `spatial_map_2024-01_level1_N20.png`
