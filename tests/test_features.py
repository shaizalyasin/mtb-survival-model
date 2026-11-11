"""Tests for the failure modes that cost the most time to find later.

The first test is the important one: it reconstructs a patient from
API-shaped JSON and asserts the resulting row is identical to the one the
training path builds from the flat files. That is the train/serve skew check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as feat  # noqa: E402
from features import FeatureSchema, PatientFeatures  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    path = ARTIFACTS / "feature_schema.json"
    if not path.exists():
        pytest.skip("run `make train` first; no feature_schema.json")
    return FeatureSchema.load(path)


def _training_style_patient() -> PatientFeatures:
    return PatientFeatures(
        patient_id="P-TEST",
        sample_id="P-TEST-T01-IM6",
        panel="IMPACT468",
        age=64.0,
        sex="Female",
        sample_type="Metastasis",
        oncotree_code="LUAD",
        tmb=9.4,
        msi_score=0.31,
        fga=0.22,
        mutated_genes=frozenset({"TP53", "KRAS"}),
        amplified_genes=frozenset({"MYC"}),
        deleted_genes=frozenset({"CDKN2A"}),
    )


def _same_patient_as_api_payloads() -> tuple[list[dict], list[dict], list[dict]]:
    clinical = [
        {"clinicalAttributeId": "ONCOTREE_CODE", "value": "LUAD", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "SAMPLE_TYPE", "value": "Metastasis", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "GENE_PANEL", "value": "IMPACT468", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "TMB_NONSYNONYMOUS", "value": "9.4", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "MSI_SCORE", "value": "0.31", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "FRACTION_GENOME_ALTERED", "value": "0.22", "sampleId": "P-TEST-T01-IM6"},
        {"clinicalAttributeId": "SEX", "value": "Female", "patientId": "P-TEST"},
        {"clinicalAttributeId": "AGE_AT_SEQUENCING", "value": "64", "patientId": "P-TEST"},
    ]
    mutations = [
        {"hugoGeneSymbol": "TP53", "mutationType": "Missense_Mutation"},
        {"hugoGeneSymbol": "KRAS", "mutationType": "Missense_Mutation"},
        # Must be filtered out, so it cannot change the row.
        {"hugoGeneSymbol": "EGFR", "mutationType": "Silent"},
    ]
    cna = [
        {"gene": {"hugoGeneSymbol": "MYC"}, "alteration": 2},
        {"gene": {"hugoGeneSymbol": "CDKN2A"}, "alteration": -2},
        # Shallow calls are not confident and must be ignored.
        {"gene": {"hugoGeneSymbol": "PTEN"}, "alteration": -1},
        {"gene": {"hugoGeneSymbol": "ERBB2"}, "alteration": 1},
    ]
    return clinical, mutations, cna


def test_train_and_serve_paths_agree(schema):
    """The row built from flat-file shapes equals the row built from API JSON."""
    from_training = feat.build_row(_training_style_patient(), schema)
    clinical, mutations, cna = _same_patient_as_api_payloads()
    from_api = feat.build_row(
        feat.from_api_payloads(clinical, mutations, cna, schema), schema
    )

    assert from_training.shape == from_api.shape
    np.testing.assert_array_equal(
        np.isnan(from_training), np.isnan(from_api), "NaN positions diverge"
    )
    finite = ~np.isnan(from_training)
    mismatched = [
        schema.columns[i]
        for i in np.flatnonzero(finite)
        if from_training[i] != from_api[i]
    ]
    assert not mismatched, f"train/serve skew in columns: {mismatched[:10]}"


def test_column_order_matches_schema_exactly(schema):
    row = feat.build_row(_training_style_patient(), schema)
    assert len(row) == len(schema.columns)
    frame = feat.build_matrix([_training_style_patient()], schema)
    assert list(frame.columns) == list(schema.columns)


def test_unseen_oncotree_code_maps_to_other(schema):
    patient = _training_style_patient()
    patient.oncotree_code = "NOT_A_REAL_CODE"
    row = feat.build_row(patient, schema)

    other_index = schema.columns.index(f"onco_{feat.OTHER_CODE}")
    assert row[other_index] == 1.0
    onco_columns = [i for i, c in enumerate(schema.columns) if c.startswith("onco_")]
    assert sum(row[i] for i in onco_columns) == 1.0


def test_unseen_gene_is_ignored_not_added(schema):
    patient = _training_style_patient()
    patient.mutated_genes = frozenset({"TP53", "A_GENE_THAT_DOES_NOT_EXIST"})
    row = feat.build_row(patient, schema)
    assert len(row) == len(schema.columns)


def test_missing_msi_propagates_as_nan(schema):
    patient = _training_style_patient()
    patient.msi_score = None
    row = feat.build_row(patient, schema)

    msi_index = schema.columns.index("msi_score")
    flag_index = schema.columns.index("msi_score_missing")
    assert np.isnan(row[msi_index]), "missing MSI must stay NaN, never become 0"
    assert row[flag_index] == 1.0


def test_off_panel_gene_is_nan_not_zero(schema):
    """A gene the sample's panel never covered is untested, not wild type."""
    small = set(schema.panels["IMPACT341"])
    large = set(schema.panels["IMPACT468"])
    off_panel = sorted(
        gene for gene in schema.mutation_genes if gene in large and gene not in small
    )
    if not off_panel:
        pytest.skip("no schema gene distinguishes the two panels")
    gene = off_panel[0]
    index = schema.columns.index(f"mut_{gene}")

    narrow = _training_style_patient()
    narrow.panel = "IMPACT341"
    assert np.isnan(feat.build_row(narrow, schema)[index])

    wide = _training_style_patient()
    wide.panel = "IMPACT468"
    assert feat.build_row(wide, schema)[index] == 0.0


def test_age_alias_difference_is_reported(schema):
    """Serving a study that only has CURRENT_AGE_DEID must say so."""
    clinical = [
        {"clinicalAttributeId": "ONCOTREE_CODE", "value": "LUAD", "sampleId": "S1"},
        {"clinicalAttributeId": "CURRENT_AGE_DEID", "value": "71", "patientId": "P1"},
    ]
    patient = feat.from_api_payloads(clinical, [], [], schema)
    assert patient.age == 71.0
    assert any("CURRENT_AGE_DEID" in note for note in patient.caveats)


def test_censored_age_is_parsed_not_dropped(schema):
    clinical = [
        {"clinicalAttributeId": "ONCOTREE_CODE", "value": "LUAD", "sampleId": "S1"},
        {"clinicalAttributeId": "AGE_AT_SEQUENCING", "value": ">90", "patientId": "P1"},
    ]
    patient = feat.from_api_payloads(clinical, [], [], schema)
    assert patient.age == 90.0
