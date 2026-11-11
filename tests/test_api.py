"""Service-level tests, run in-process rather than against a live server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if not (ROOT / "artifacts" / "model.json").exists():
    pytest.skip("run `make train` first; no model.json", allow_module_level=True)

from api import app  # noqa: E402

client = TestClient(app)

CLINICAL = [
    {"clinicalAttributeId": "ONCOTREE_CODE", "value": "LUAD", "sampleId": "S1"},
    {"clinicalAttributeId": "SAMPLE_TYPE", "value": "Primary", "sampleId": "S1"},
    {"clinicalAttributeId": "GENE_PANEL", "value": "IMPACT468", "sampleId": "S1"},
    {"clinicalAttributeId": "TMB_NONSYNONYMOUS", "value": "7.2", "sampleId": "S1"},
    {"clinicalAttributeId": "FRACTION_GENOME_ALTERED", "value": "0.18", "sampleId": "S1"},
    {"clinicalAttributeId": "SEX", "value": "Male", "patientId": "P1"},
    {"clinicalAttributeId": "AGE_AT_SEQUENCING", "value": "61", "patientId": "P1"},
]

MUTATIONS = [{"hugoGeneSymbol": "TP53", "mutationType": "Missense_Mutation"}]


def test_health_reports_a_loaded_model():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["stateless"] is True
    assert body["features"] > 0


def test_predict_returns_a_percentile_and_reference_cohort():
    response = client.post(
        "/predict",
        json={"clinical_data": CLINICAL, "mutations": MUTATIONS, "discrete_cna": []},
    )
    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["percentile_within_cancer_type"] <= 100
    assert body["risk_tertile"] in {"low", "intermediate", "high"}
    assert body["reference_cohort"]["oncotree_code"] == "LUAD"
    assert body["reference_cohort"]["n"] >= 100
    assert body["attributions"], "expected non-empty attributions"
    assert "not for clinical decision-making" in body["disclaimer"]


def test_absent_msi_is_reported_as_missing_not_imputed():
    """MSI is absent from CLINICAL, so the response must say so."""
    body = client.post(
        "/predict",
        json={"clinical_data": CLINICAL, "mutations": MUTATIONS, "discrete_cna": []},
    ).json()
    assert "msi_score" in body["missing_features"]


def test_predict_without_clinical_data_is_rejected():
    response = client.post(
        "/predict", json={"clinical_data": [], "mutations": [], "discrete_cna": []}
    )
    assert response.status_code == 422


def test_unknown_cancer_type_falls_back_to_pan_cancer_and_says_so():
    clinical = [
        {"clinicalAttributeId": "ONCOTREE_CODE", "value": "NOT_A_CODE", "sampleId": "S1"},
        {"clinicalAttributeId": "AGE_AT_SEQUENCING", "value": "61", "patientId": "P1"},
    ]
    body = client.post(
        "/predict",
        json={"clinical_data": clinical, "mutations": [], "discrete_cna": []},
    ).json()
    assert body["reference_cohort"]["is_pan_cancer_fallback"] is True


def test_reference_km_has_three_tertiles_and_at_risk_counts():
    body = client.get("/reference-km/LUAD").json()
    assert [curve["tertile"] for curve in body["curves"]] == [
        "low",
        "intermediate",
        "high",
    ]
    for curve in body["curves"]:
        assert len(curve["time"]) == len(curve["survival"]) == len(curve["at_risk"])
        assert curve["survival"][0] <= 1.0
    # The model must rank: more deaths in the high tertile than the low one.
    by_name = {curve["tertile"]: curve for curve in body["curves"]}
    assert by_name["high"]["events"] > by_name["low"]["events"]
