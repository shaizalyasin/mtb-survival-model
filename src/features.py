"""The one feature builder, used by both training and serving.

Train/serve skew is the failure mode this module exists to prevent, so there is
exactly one function that turns a patient into a row: `build_row`. The training
path calls it 25,775 times rather than having a faster vectorised twin, because
a second implementation is a second thing that can drift.

Both paths first normalise their very different inputs into `PatientFeatures`:

    flat files  -> from_flat_files()   -> PatientFeatures -> build_row()
    API payloads -> from_api_payloads() -> PatientFeatures -> build_row()

Missing values are never imputed. They stay NaN and XGBoost routes them down
its learned missing branch. See `panels.py` for why an off-panel gene is NaN
rather than 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from load import PATIENT_COLS, SAMPLE_COLS, SILENT_CLASSES, parse_age
from panels import panel_sets

SCHEMA_VERSION = "1.0.0"

OTHER_CODE = "OTHER"

MIN_PATIENTS_PER_FEATURE = 50


# --------------------------------------------------------------------------
# Normalised single-patient input
# --------------------------------------------------------------------------


@dataclass
class PatientFeatures:
    """Everything the model is allowed to know about one patient.

    Every field here is retrievable for a new patient at the moment of
    sequencing. Nothing derived from follow-up appears, by construction.
    """

    patient_id: str
    sample_id: str
    panel: str | None = None
    age: float | None = None
    sex: str | None = None
    sample_type: str | None = None
    oncotree_code: str | None = None
    tmb: float | None = None
    msi_score: float | None = None
    fga: float | None = None
    mutated_genes: frozenset[str] = field(default_factory=frozenset)
    amplified_genes: frozenset[str] = field(default_factory=frozenset)
    deleted_genes: frozenset[str] = field(default_factory=frozenset)
    # Notes about how the input was resolved, surfaced to the UI. These are not
    # features and never reach the model; they record where a served study's
    # attribute means something slightly different from the trained one.
    caveats: tuple[str, ...] = ()

    # Leakage probe, used by one ablation arm and never served. This counts
    # metastatic sites recorded over the patient's entire disease course, so
    # relative to sequencing it is future information. It is populated only
    # from the flat files; `from_api_payloads` leaves it None by construction,
    # so a served schema physically cannot carry it.
    met_site_count: float | None = None


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


MET_PROBE_COLUMN = "met_site_count_LEAKAGE_PROBE"


@dataclass
class FeatureSchema:
    version: str
    oncotree_codes: list[str]
    mutation_genes: list[str]
    amp_genes: list[str]
    del_genes: list[str]
    panels: dict[str, list[str]]
    columns: list[str]
    # Only ever True for the ablation arm that measures how much the published
    # result depends on a feature unavailable at sequencing time. A schema
    # written with this set must not be served.
    include_mets_probe: bool = False

    @property
    def numeric_columns(self) -> list[str]:
        return list(NUMERIC_COLUMNS)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        return cls(**json.loads(Path(path).read_text()))


NUMERIC_COLUMNS = (
    "age",
    "sex_female",
    "sample_is_metastasis",
    "tmb",
    "msi_score",
    "fga",
    # Retained for the serving path: MSK-MET has complete TMB and MSI, so these
    # are constant 0 in training. They exist because another panel may not
    # report them, and the UI needs to say which value was absent.
    "tmb_missing",
    "msi_score_missing",
)


def _columns(
    oncotree_codes: list[str],
    mutation_genes: list[str],
    amp_genes: list[str],
    del_genes: list[str],
    include_mets_probe: bool = False,
) -> list[str]:
    return (
        list(NUMERIC_COLUMNS)
        + [f"onco_{code}" for code in oncotree_codes]
        + [f"mut_{gene}" for gene in mutation_genes]
        + [f"amp_{gene}" for gene in amp_genes]
        + [f"del_{gene}" for gene in del_genes]
        + ([MET_PROBE_COLUMN] if include_mets_probe else [])
    )


def build_schema(
    rows: list[PatientFeatures],
    panels: dict[str, list[str]],
    min_patients: int = MIN_PATIENTS_PER_FEATURE,
    include_mets_probe: bool = False,
) -> FeatureSchema:
    """Derive the column set from the training rows only.

    Counting these thresholds over the whole cohort would leak the test split
    into the column set, so this is only ever called with training patients.
    """
    code_counts: dict[str, int] = {}
    mut_counts: dict[str, int] = {}
    amp_counts: dict[str, int] = {}
    del_counts: dict[str, int] = {}

    for row in rows:
        if row.oncotree_code:
            code_counts[row.oncotree_code] = code_counts.get(row.oncotree_code, 0) + 1
        for gene in row.mutated_genes:
            mut_counts[gene] = mut_counts.get(gene, 0) + 1
        for gene in row.amplified_genes:
            amp_counts[gene] = amp_counts.get(gene, 0) + 1
        for gene in row.deleted_genes:
            del_counts[gene] = del_counts.get(gene, 0) + 1

    def retained(counts: dict[str, int]) -> list[str]:
        return sorted(key for key, count in counts.items() if count >= min_patients)

    oncotree_codes = retained(code_counts) + [OTHER_CODE]
    mutation_genes = retained(mut_counts)
    amp_genes = retained(amp_counts)
    del_genes = retained(del_counts)

    return FeatureSchema(
        version=SCHEMA_VERSION,
        oncotree_codes=oncotree_codes,
        mutation_genes=mutation_genes,
        amp_genes=amp_genes,
        del_genes=del_genes,
        panels={name: sorted(genes) for name, genes in panels.items()},
        columns=_columns(
            oncotree_codes, mutation_genes, amp_genes, del_genes, include_mets_probe
        ),
        include_mets_probe=include_mets_probe,
    )


# --------------------------------------------------------------------------
# The shared row builder
# --------------------------------------------------------------------------


def build_row(patient: PatientFeatures, schema: FeatureSchema) -> np.ndarray:
    """Turn one patient into one row, in exactly `schema.columns` order."""
    sets = panel_sets(schema.panels)
    tested = sets.get(patient.panel) if patient.panel else None

    def gene_value(gene: str, observed: frozenset[str]) -> float:
        if gene in observed:
            return 1.0
        # Not observed. Distinguish "tested and negative" from "never tested".
        if tested is not None and gene not in tested:
            return np.nan
        return 0.0

    values: list[float] = []

    # Coerce here as well as at the boundaries: build_row is the shared
    # contract, so it must not depend on which path normalised its input.
    sex = (_optional_str(patient.sex) or "").lower()
    values.append(float(patient.age) if patient.age is not None else np.nan)
    values.append(1.0 if sex == "female" else 0.0 if sex == "male" else np.nan)
    sample_type = (_optional_str(patient.sample_type) or "").lower()
    values.append(
        1.0 if sample_type == "metastasis" else 0.0 if sample_type == "primary" else np.nan
    )
    values.append(float(patient.tmb) if patient.tmb is not None else np.nan)
    values.append(float(patient.msi_score) if patient.msi_score is not None else np.nan)
    values.append(float(patient.fga) if patient.fga is not None else np.nan)
    values.append(1.0 if patient.tmb is None else 0.0)
    values.append(1.0 if patient.msi_score is None else 0.0)

    code = patient.oncotree_code if patient.oncotree_code in schema.oncotree_codes else OTHER_CODE
    for candidate in schema.oncotree_codes:
        values.append(1.0 if candidate == code else 0.0)

    for gene in schema.mutation_genes:
        values.append(gene_value(gene, patient.mutated_genes))
    for gene in schema.amp_genes:
        values.append(gene_value(gene, patient.amplified_genes))
    for gene in schema.del_genes:
        values.append(gene_value(gene, patient.deleted_genes))

    if schema.include_mets_probe:
        values.append(
            float(patient.met_site_count)
            if patient.met_site_count is not None
            else np.nan
        )

    row = np.asarray(values, dtype=np.float64)
    if row.shape[0] != len(schema.columns):
        raise AssertionError(
            f"built {row.shape[0]} values for {len(schema.columns)} schema columns"
        )
    return row


def build_matrix(rows: list[PatientFeatures], schema: FeatureSchema) -> pd.DataFrame:
    """Stack `build_row` over many patients. No separate implementation."""
    matrix = np.empty((len(rows), len(schema.columns)), dtype=np.float64)
    for index, row in enumerate(rows):
        matrix[index] = build_row(row, schema)
    return pd.DataFrame(
        matrix,
        columns=schema.columns,
        index=[row.patient_id for row in rows],
    )


# --------------------------------------------------------------------------
# Training input: flat files
# --------------------------------------------------------------------------


def _mutated_by_sample(mutations: pd.DataFrame) -> dict[str, frozenset[str]]:
    keep = ~mutations["Variant_Classification"].isin(SILENT_CLASSES)
    subset = mutations.loc[keep, ["Tumor_Sample_Barcode", "Hugo_Symbol"]].dropna()
    grouped = subset.groupby("Tumor_Sample_Barcode")["Hugo_Symbol"].apply(
        lambda genes: frozenset(genes)
    )
    return grouped.to_dict()


def _cna_by_sample(cna: pd.DataFrame) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """GISTIC-style discrete calls. Only +/-2 are confident, per the spec."""
    indexed = cna.set_index("Hugo_Symbol")
    indexed = indexed.loc[~indexed.index.duplicated(keep="first")]
    genes = np.asarray(indexed.index)
    values = indexed.to_numpy(dtype=np.float64, na_value=np.nan)

    amp_mask = values == 2.0
    del_mask = values == -2.0

    amplified: dict[str, frozenset[str]] = {}
    deleted: dict[str, frozenset[str]] = {}
    for position, sample_id in enumerate(indexed.columns):
        amplified[sample_id] = frozenset(genes[amp_mask[:, position]])
        deleted[sample_id] = frozenset(genes[del_mask[:, position]])
    return amplified, deleted


def _optional_float(value) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if number is None or pd.isna(number):
        return None
    return float(number)


def _optional_str(value) -> str | None:
    """Absent categoricals arrive as NaN floats from pandas and as None or the
    literal 'NA' from the API. All three mean the same thing."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.upper() not in ("NA", "NAN", "NONE", "UNKNOWN") else None


def from_flat_files(cohort, panels: dict[str, list[str]]) -> list[PatientFeatures]:
    """Build normalised inputs for every patient in the cohort."""
    mutated = _mutated_by_sample(cohort.mutations)
    amplified, deleted = _cna_by_sample(cohort.cna)

    patients = cohort.patients.set_index(PATIENT_COLS["patient_id"])
    ages = parse_age(patients[PATIENT_COLS["age_at_sequencing"]])

    known_panels = set(panels)
    rows: list[PatientFeatures] = []

    for record in cohort.samples.to_dict("records"):
        patient_id = record[SAMPLE_COLS["patient_id"]]
        sample_id = record[SAMPLE_COLS["sample_id"]]
        if patient_id not in patients.index:
            continue

        panel = record.get(SAMPLE_COLS["gene_panel"])
        panel = panel if isinstance(panel, str) and panel in known_panels else None

        rows.append(
            PatientFeatures(
                patient_id=str(patient_id),
                sample_id=str(sample_id),
                panel=panel,
                age=_optional_float(ages.get(patient_id)),
                sex=_optional_str(
                    patients.at[patient_id, PATIENT_COLS["sex"]]
                    if PATIENT_COLS["sex"] in patients.columns
                    else None
                ),
                sample_type=_optional_str(record.get(SAMPLE_COLS["sample_type"])),
                oncotree_code=_optional_str(record.get(SAMPLE_COLS["oncotree_code"])),
                tmb=_optional_float(record.get(SAMPLE_COLS["tmb"])),
                msi_score=_optional_float(record.get(SAMPLE_COLS["msi_score"])),
                fga=_optional_float(record.get(SAMPLE_COLS["fga"])),
                mutated_genes=mutated.get(sample_id, frozenset()),
                amplified_genes=amplified.get(sample_id, frozenset()),
                deleted_genes=deleted.get(sample_id, frozenset()),
                met_site_count=_optional_float(record.get("MET_SITE_COUNT")),
            )
        )

    return rows


# --------------------------------------------------------------------------
# Serving input: cBioPortal API payloads
# --------------------------------------------------------------------------

# Attribute ids differ between studies for the same quantity, so serving a
# study other than the training one needs a mapping rather than a hardcoded
# name. msk_chord_2024 has no SEX at all and calls age CURRENT_AGE_DEID.
CLINICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "age": ("AGE_AT_SEQUENCING", "CURRENT_AGE_DEID", "AGE_AT_DOS_BX", "AGE"),
    "sex": ("SEX", "GENDER"),
    "sample_type": ("SAMPLE_TYPE",),
    "oncotree_code": ("ONCOTREE_CODE",),
    "tmb": ("TMB_NONSYNONYMOUS",),
    "msi_score": ("MSI_SCORE",),
    "fga": ("FRACTION_GENOME_ALTERED", "FGA"),
    "panel": ("GENE_PANEL",),
}

# The first alias for each field is the one the model was trained on. A later
# alias is accepted but is not always the same quantity, and the difference has
# to reach the UI rather than being absorbed silently. CURRENT_AGE_DEID is the
# clearest case: it is the patient's age now, not their age when the sample was
# sequenced, so for a patient sequenced years ago it runs high.
ALIAS_CAVEATS: dict[str, str] = {
    "CURRENT_AGE_DEID": (
        "age taken from CURRENT_AGE_DEID (age today), while the model was "
        "trained on age at sequencing; for a sample sequenced years ago this "
        "overstates age"
    ),
    "AGE_AT_DOS_BX": (
        "age taken from AGE_AT_DOS_BX (age at biopsy), a close but not "
        "identical match to the trained age at sequencing"
    ),
    "AGE": ("age taken from a generic AGE attribute of unspecified timepoint"),
}


def _pick(clinical: dict[str, str], field_name: str) -> tuple[str | None, str | None]:
    """Return the first matching value and which alias supplied it."""
    for alias in CLINICAL_ALIASES[field_name]:
        if alias in clinical and clinical[alias] not in ("", "NA", None):
            return clinical[alias], alias
    return None, None


def from_api_payloads(
    clinical_data: list[dict],
    mutations: list[dict],
    discrete_cna: list[dict],
    schema: FeatureSchema,
) -> PatientFeatures:
    """Normalise the three cBioPortal responses for one sample.

    `clinical_data` is the concatenation of the sample-level and patient-level
    clinical-data responses; both have clinicalAttributeId / value pairs.
    """
    clinical = {
        str(item.get("clinicalAttributeId")): str(item.get("value"))
        for item in clinical_data
    }

    patient_id = next(
        (str(item["patientId"]) for item in clinical_data if item.get("patientId")), "unknown"
    )
    sample_id = next(
        (str(item["sampleId"]) for item in clinical_data if item.get("sampleId")), "unknown"
    )

    caveats: list[str] = []

    panel, _ = _pick(clinical, "panel")
    if panel not in schema.panels:
        if panel:
            caveats.append(
                f"gene panel {panel!r} is not one the model knows, so no gene "
                f"could be marked untested; absent genes are treated as negative"
            )
        panel = None

    def gene_symbol(item: dict) -> str | None:
        gene = item.get("hugoGeneSymbol")
        if gene:
            return str(gene)
        nested = item.get("gene") or {}
        symbol = nested.get("hugoGeneSymbol")
        return str(symbol) if symbol else None

    mutated = {
        symbol
        for item in mutations
        if str(item.get("mutationType") or item.get("variantClassification") or "")
        not in SILENT_CLASSES
        and (symbol := gene_symbol(item))
    }

    amplified: set[str] = set()
    deleted: set[str] = set()
    for item in discrete_cna:
        symbol = gene_symbol(item)
        if not symbol:
            continue
        alteration = pd.to_numeric(item.get("alteration"), errors="coerce")
        if alteration == 2:
            amplified.add(symbol)
        elif alteration == -2:
            deleted.add(symbol)

    age_text, age_alias = _pick(clinical, "age")
    if age_alias in ALIAS_CAVEATS:
        caveats.append(ALIAS_CAVEATS[age_alias])

    sex, _ = _pick(clinical, "sex")
    sample_type, _ = _pick(clinical, "sample_type")
    oncotree_code, _ = _pick(clinical, "oncotree_code")
    tmb, _ = _pick(clinical, "tmb")
    msi_score, _ = _pick(clinical, "msi_score")
    fga, _ = _pick(clinical, "fga")

    return PatientFeatures(
        patient_id=patient_id,
        sample_id=sample_id,
        panel=panel,
        age=_optional_float(parse_age(pd.Series([age_text])).iloc[0]) if age_text else None,
        sex=_optional_str(sex),
        sample_type=_optional_str(sample_type),
        oncotree_code=_optional_str(oncotree_code),
        tmb=_optional_float(tmb),
        msi_score=_optional_float(msi_score),
        fga=_optional_float(fga),
        mutated_genes=frozenset(mutated),
        amplified_genes=frozenset(amplified),
        deleted_genes=frozenset(deleted),
        caveats=tuple(caveats),
    )


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def survival_labels(cohort, rows: list[PatientFeatures]) -> pd.DataFrame:
    """OS months and the event indicator, aligned to `rows`."""
    from load import parse_os_status

    patients = cohort.patients.set_index(PATIENT_COLS["patient_id"])
    months = pd.to_numeric(patients[PATIENT_COLS["os_months"]], errors="coerce")
    event = parse_os_status(patients[PATIENT_COLS["os_status"]])

    index = [row.patient_id for row in rows]
    return pd.DataFrame(
        {
            "os_months": months.reindex(index).to_numpy(),
            "event": event.reindex(index).to_numpy(),
            "oncotree_code": [row.oncotree_code for row in rows],
        },
        index=index,
    )
