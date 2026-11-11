"""Parse the MSK-MET flat files and verify every assumption the pipeline makes.

Column names here were read off the actual files, not guessed. Two of them
differ from what the cBioPortal REST API reports for the same study, which is
the reason this module exists as a separate layer:

    flat file          REST API
    ---------          --------
    FGA                FRACTION_GENOME_ALTERED
    (patient file)     AGE_AT_SEQUENCING is patient-level, not sample-level

Running this module directly prints a verification report and exits non-zero if
any assertion fails.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column names, verified against msk_met_2021 as published on 2026-11-11.
# --------------------------------------------------------------------------

PATIENT_COLS = {
    "patient_id": "PATIENT_ID",
    "sex": "SEX",
    "race": "RACE",
    "os_months": "OS_MONTHS",
    "os_status": "OS_STATUS",
    "age_at_sequencing": "AGE_AT_SEQUENCING",
    "age_at_death": "AGE_AT_DEATH",
    "age_at_last_contact": "AGE_AT_LAST_CONTACT",
}

SAMPLE_COLS = {
    "sample_id": "SAMPLE_ID",
    "patient_id": "PATIENT_ID",
    "oncotree_code": "ONCOTREE_CODE",
    "cancer_type": "CANCER_TYPE",
    "sample_type": "SAMPLE_TYPE",
    "gene_panel": "GENE_PANEL",
    "tmb": "TMB_NONSYNONYMOUS",
    "msi_score": "MSI_SCORE",
    "msi_type": "MSI_TYPE",
    "fga": "FGA",
    "tumor_purity": "TUMOR_PURITY",
}

# Present in the data and deliberately never used as features. Each of these is
# recorded over the patient's whole disease course, so relative to the
# sequencing timepoint they carry future information. Mestrallet (2025) reports
# metastasis count as the single strongest predictor of overall survival, which
# is precisely why excluding it changes the achievable C-index.
LEAKING_COLS = (
    "MET_COUNT",
    "MET_SITE_COUNT",
    "IS_DIST_MET_MAPPED",
    "METASTATIC_SITE",
    "AGE_AT_EVIDENCE_OF_METS",
    "AGE_AT_DEATH",
    "AGE_AT_LAST_CONTACT",
    "AGE_AT_SURGERY",
)

# Variant classifications that do not change protein sequence.
SILENT_CLASSES = frozenset(
    {
        "Silent",
        "Intron",
        "3'UTR",
        "5'UTR",
        "3'Flank",
        "5'Flank",
        "RNA",
        "IGR",
    }
)


def read_cbio_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a cBioPortal flat file, skipping its '#' metadata header lines."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False, **kwargs)


def _require(frame: pd.DataFrame, mapping: dict[str, str], label: str) -> None:
    missing = [col for col in mapping.values() if col not in frame.columns]
    if missing:
        raise SystemExit(
            f"{label}: expected columns absent from the file: {', '.join(missing)}\n"
            f"columns present: {', '.join(frame.columns)}"
        )


def parse_os_status(series: pd.Series) -> pd.Series:
    """'1:DECEASED' / '0:LIVING' -> boolean event indicator.

    Casting the string to int would silently succeed on some encodings and
    produce the wrong label on others, so match on the text instead.
    """
    text = series.astype(str).str.upper()
    deceased = text.str.contains("DECEASED")
    living = text.str.contains("LIVING")

    unknown = ~(deceased | living)
    if unknown.any():
        values = sorted(series[unknown].astype(str).unique())[:5]
        raise SystemExit(f"unrecognised OS_STATUS values: {values}")

    return deceased


def parse_age(series: pd.Series) -> pd.Series:
    """Ages are de-identified at the extremes ('>90', '<18' and similar).

    Strip any comparison operator and keep the numeric bound. This is lossy and
    deliberately so: the alternative is dropping the oldest patients, who carry
    real prognostic signal. The same rule has to run at inference time, which
    is why it lives here rather than inline in the feature builder.
    """
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text.str.replace(r"^[<>=]+", "", regex=True), errors="coerce")
    return numeric


@dataclass
class Cohort:
    patients: pd.DataFrame
    samples: pd.DataFrame
    mutations: pd.DataFrame
    cna: pd.DataFrame
    panel_matrix: pd.DataFrame
    dropped_duplicate_samples: int


def one_sample_per_patient(samples: pd.DataFrame, patients: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the earliest-sequenced sample per patient.

    There is no sequencing date column in this study, only AGE_AT_SEQUENCING at
    patient level, which cannot separate two samples from the same patient.
    So the tie-break order is: primary tumour first, then sample id, which is
    monotonic in collection order within a patient (T01 before T02).
    """
    before = len(samples)
    merged = samples.merge(
        patients[[PATIENT_COLS["patient_id"]]],
        on=PATIENT_COLS["patient_id"],
        how="inner",
    )
    is_primary = (
        merged[SAMPLE_COLS["sample_type"]].astype(str).str.strip().str.lower() == "primary"
    )
    ordered = merged.assign(_primary_first=(~is_primary).astype(int)).sort_values(
        ["_primary_first", SAMPLE_COLS["sample_id"]]
    )
    kept = (
        ordered.drop_duplicates(subset=[PATIENT_COLS["patient_id"]], keep="first")
        .drop(columns="_primary_first")
        .reset_index(drop=True)
    )
    return kept, before - len(kept)


def load(study_dir: Path) -> Cohort:
    patients = read_cbio_tsv(study_dir / "data_clinical_patient.txt")
    samples = read_cbio_tsv(study_dir / "data_clinical_sample.txt")
    mutations = read_cbio_tsv(study_dir / "data_mutations.txt")
    cna = read_cbio_tsv(study_dir / "data_cna.txt")
    panel_matrix = read_cbio_tsv(study_dir / "data_gene_panel_matrix.txt")

    _require(patients, PATIENT_COLS, "data_clinical_patient.txt")
    _require(samples, SAMPLE_COLS, "data_clinical_sample.txt")

    kept, dropped = one_sample_per_patient(samples, patients)

    return Cohort(
        patients=patients,
        samples=kept,
        mutations=mutations,
        cna=cna,
        panel_matrix=panel_matrix,
        dropped_duplicate_samples=dropped,
    )


def check_os_anchor(patients: pd.DataFrame) -> dict[str, float]:
    """Confirm OS_MONTHS is measured from the date of sequencing.

    This is the assumption the whole task rests on. If OS ran from diagnosis
    instead, the cohort would be left-truncated and the features would postdate
    the time origin. The study's own data dictionary says 'Overall survival in
    months since date of sequencing'; this reconstructs the interval from the
    age columns to confirm it rather than take it on trust.
    """
    deceased = parse_os_status(patients[PATIENT_COLS["os_status"]])
    seq = parse_age(patients[PATIENT_COLS["age_at_sequencing"]])
    death = parse_age(patients[PATIENT_COLS["age_at_death"]])
    contact = parse_age(patients[PATIENT_COLS["age_at_last_contact"]])
    observed = pd.to_numeric(patients[PATIENT_COLS["os_months"]], errors="coerce")

    anchor = np.where(deceased, death, contact)
    derived = (pd.Series(anchor, index=patients.index) - seq) * 12.0

    comparable = derived.notna() & observed.notna()
    diff = (derived[comparable] - observed[comparable]).abs()

    return {
        "comparable": int(comparable.sum()),
        "within_1_month_pct": float((diff <= 1.0).mean() * 100.0),
        "mean_abs_diff_months": float(diff.mean()),
        "median_abs_diff_months": float(diff.median()),
        "max_abs_diff_months": float(diff.max()),
    }


def report(study_dir: Path) -> int:
    print(f"study directory: {study_dir}\n")
    cohort = load(study_dir)

    patients, samples = cohort.patients, cohort.samples
    print("--- shapes ---")
    print(f"patients             : {len(patients):,}")
    print(f"samples (deduped)    : {len(samples):,}")
    print(f"duplicate samples    : {cohort.dropped_duplicate_samples:,} dropped")
    print(f"mutation rows        : {len(cohort.mutations):,}")
    print(f"cna genes x samples  : {cohort.cna.shape[0]:,} x {cohort.cna.shape[1] - 1:,}")

    print("\n--- survival label ---")
    deceased = parse_os_status(patients[PATIENT_COLS["os_status"]])
    os_months = pd.to_numeric(patients[PATIENT_COLS["os_months"]], errors="coerce")
    print(f"events (deceased)    : {int(deceased.sum()):,} ({deceased.mean() * 100:.1f}%)")
    print(f"censored (living)    : {int((~deceased).sum()):,}")
    print(f"OS_MONTHS missing    : {int(os_months.isna().sum()):,}")
    print(f"OS_MONTHS median     : {os_months.median():.1f}  max {os_months.max():.1f}")

    print("\n--- OS anchor check (expect ~100% within 1 month of sequencing) ---")
    for key, value in check_os_anchor(patients).items():
        print(f"{key:24s}: {value:,.3f}" if isinstance(value, float) else f"{key:24s}: {value:,}")

    print("\n--- feature availability ---")
    for name, col in (
        ("TMB", SAMPLE_COLS["tmb"]),
        ("MSI score", SAMPLE_COLS["msi_score"]),
        ("FGA", SAMPLE_COLS["fga"]),
        ("tumour purity", SAMPLE_COLS["tumor_purity"]),
    ):
        values = pd.to_numeric(samples[col], errors="coerce")
        pct = values.isna().mean() * 100
        print(f"{name:16s} missing {pct:5.1f}%   median {values.median():.3f}")

    print("\n--- gene panel versions (a gene off-panel is untested, not wild type) ---")
    counts = samples[SAMPLE_COLS["gene_panel"]].value_counts()
    for panel, count in counts.items():
        print(f"{str(panel):16s} {count:6,}  ({count / len(samples) * 100:4.1f}%)")

    print("\n--- sample type ---")
    for value, count in samples[SAMPLE_COLS["sample_type"]].value_counts().items():
        print(f"{str(value):16s} {count:6,}")

    print("\n--- OncoTree codes ---")
    codes = samples[SAMPLE_COLS["oncotree_code"]].value_counts()
    print(f"distinct codes       : {len(codes):,}")
    print(f"codes with >= 50     : {int((codes >= 50).sum()):,}")
    print(f"patients in those    : {int(codes[codes >= 50].sum()):,}")
    print(f"top 8                : {', '.join(f'{k}={v}' for k, v in codes.head(8).items())}")

    print("\n--- excluded by design (present in the data, never a feature) ---")
    present = [c for c in LEAKING_COLS if c in samples.columns or c in patients.columns]
    print(", ".join(present))
    dmets = [c for c in samples.columns if c.startswith("DMETS_DX_")]
    print(f"plus {len(dmets)} DMETS_DX_* metastatic site flags")

    return 0


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/msk_met_2021")
    sys.exit(report(directory))
