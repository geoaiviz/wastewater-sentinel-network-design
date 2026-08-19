# Input schema

This document describes the minimum analytical fields expected by the workflow. Exact filenames may be changed locally if the corresponding paths in the scripts are updated.

## WWTP/sewershed table or spatial layer

Required fields:

- `wwtp`: stable wastewater treatment plant identifier or name.
- `geometry`: WWTP point or sewershed polygon geometry.
- `population_served`: estimated residential population served.
- `county_fips`: associated county FIPS code where applicable.
- `commute_distance_km`: estimated one-way commute-distance threshold assigned to the sewershed.

## Origin-destination mobility table

Required fields:

- `origin_id`: origin census block group or county identifier.
- `destination_wwtp`: destination WWTP/sewershed identifier.
- `week`: weekly observation date or epidemiological week.
- `trip_count`: aggregated origin-destination trip count.
- `direction`: inbound or outbound when stored in long form.

Mobility inputs used in the study are licensed and are not distributed.

## Clinical surveillance table

Required fields:

- `county_fips`: county FIPS code.
- `week`: weekly observation date or epidemiological week.
- `pathogen`: SARS-CoV-2, influenza, or RSV.
- `hospitalization_rate`: weekly hospitalization rate per 100,000 population.

## Census block-group coverage layer

Required fields:

- `bg_geoid`: census block-group GEOID.
- `population`: block-group population.
- `area`: consistent area measurement.
- `geometry`: block-group polygon geometry.

## Derived Level 1 table

Expected fields include:

- WWTP identifier.
- population served.
- total weekly trip volume.
- mobility-connected population reached.
- mobility-connected area reached.
- mobility-weighted import/export risk.
- subnetwork identifier and singleton status.
- normalized ranks or scores used for selection.

## Derived Level 2 graph/model inputs

- county and WWTP node identifiers.
- weekly node-feature sequences.
- county-WWTP mobility edge weights.
- chronological training/evaluation indices.
- model-derived WWTP contribution scores.

