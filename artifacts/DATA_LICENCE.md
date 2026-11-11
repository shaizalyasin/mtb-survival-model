# Licensing of the files in this directory

The code in this repository is AGPL-3.0, like the rest of the project. The
artefacts here are not all covered by it, because some are derived from a
licensed dataset.

`reference_cohort.parquet` and `baseline_hazard.parquet` are derived from
MSK-MET (`msk_met_2021`), which cBioPortal publishes under the **Open Data
Commons Open Database License (ODbL) v1.0**. ODbL is share-alike: an adapted
database must be offered under the same licence, with attribution. Those two
files are therefore distributed under ODbL v1.0, not AGPL-3.0.

`model.json`, `feature_schema.json` and `gene_panels.json` are model parameters
and column definitions rather than a database, and are covered by the
repository's AGPL-3.0 licence.

## Attribution

Nguyen, B. et al. Genomic characterization of metastatic patterns from
prospective clinical sequencing of 25,000 patients. *Cell* (2022).
cBioPortal study identifier `msk_met_2021`.

Gene panel definitions (IMPACT341, IMPACT410, IMPACT468) are retrieved from the
public cBioPortal API and cached in `gene_panels.json`.
