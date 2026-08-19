# Data availability

The analysis combines public, licensed, and restricted data.

## Public inputs

- U.S. Census Bureau geographic boundaries, population estimates, and American Community Survey commute-time data.
- USDA Economic Research Service Rural-Urban Commuting Area classifications.
- Public Core-Based Statistical Area definitions used for the CBSA comparator.

## Inputs not redistributed here

- StreetLight origin-destination mobility products are licensed commercial data.
- Sewershed boundaries and clinical surveillance records used in the Colorado case study were obtained under applicable agreements with the Colorado Department of Public Health and Environment.

These inputs are not included in the repository. Their absence is intentional and is not a missing-file error. Users must obtain authorized data and format them according to `input_schema.md` to rerun the complete case study.

## Facility confidentiality

Facility identifiers in the public derived tables have been anonymized as `WWTP_###`. The mapping to participating facility names is confidential and is not distributed. Figures that displayed facility names as text labels were removed from the public package. Geographic maps should be retained only if the applicable agreements permit publication of the mapped facility locations.

## Shareable materials

The repository provides analysis code, configuration files, input-field descriptions, and nonidentifiable derived monthly outputs. Additional aggregated outputs may be available from the corresponding author subject to licensing and data-use restrictions.
