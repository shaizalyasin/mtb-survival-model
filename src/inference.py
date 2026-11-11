"""Single-patient inference: score, percentile, attributions, KM reference.

The raw Cox score is not interpretable on its own, so the percentile within the
patient's own cancer type is the primary output and the score is reported only
for completeness.

Attributions use TreeSHAP via XGBoost's own `pred_contribs`, which is the same
algorithm as `shap.TreeExplainer` on a tree model but without the extra
dependency. Contributions are in log-hazard space and sum, with the bias term,
to the reported risk score.

Nothing here touches cBioPortal. The caller supplies the payloads it already
fetched under the end user's own session, so this service can never read a
patient the user is not entitled to see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

import features as feat
from features import FeatureSchema, PatientFeatures

MODEL_VERSION = "1.0.0"

# Below this, a cancer-type-specific reference distribution is too small to
# quote a percentile against, so we fall back to pan-cancer and say so.
MIN_REFERENCE_COHORT = 100

TOP_ATTRIBUTIONS = 8

PAN_CANCER = "PAN_CANCER"


@dataclass
class Prediction:
    risk_score: float
    risk_tertile: str
    percentile_within_cancer_type: int
    reference_cohort: dict
    attributions: list[dict]
    missing_features: list[str]
    input_caveats: list[str]
    model_version: str

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_tertile": self.risk_tertile,
            "percentile_within_cancer_type": self.percentile_within_cancer_type,
            "reference_cohort": self.reference_cohort,
            "attributions": self.attributions,
            "missing_features": self.missing_features,
            "input_caveats": self.input_caveats,
            "model_version": self.model_version,
            "scale": "log hazard (Cox margin); attributions are additive on this scale",
        }


def human_label(column: str, value: float) -> str:
    """Column names are not clinician-readable. These are."""
    if column.startswith("mut_"):
        return f"{column[4:]} mutation"
    if column.startswith("amp_"):
        return f"{column[4:]} amplification"
    if column.startswith("del_"):
        return f"{column[4:]} deep deletion"
    if column.startswith("onco_"):
        return f"Cancer type {column[5:]}"

    if np.isnan(value):
        return {
            "age": "Age not reported",
            "sex_female": "Sex not reported",
            "tmb": "TMB not reported",
            "msi_score": "MSI score not reported",
            "fga": "Fraction of genome altered not reported",
            "sample_is_metastasis": "Sample type not reported",
        }.get(column, f"{column} not reported")

    return {
        "age": f"Age {value:.0f}",
        "sex_female": "Female" if value == 1.0 else "Male",
        "sample_is_metastasis": (
            "Metastatic sample" if value == 1.0 else "Primary tumour sample"
        ),
        "tmb": f"TMB {value:.1f}/Mb",
        "msi_score": f"MSI score {value:.2f}",
        "fga": f"Fraction of genome altered {value:.2f}",
        "tmb_missing": "TMB unavailable" if value == 1.0 else "TMB available",
        "msi_score_missing": (
            "MSI score unavailable" if value == 1.0 else "MSI score available"
        ),
    }.get(column, column)


class RiskModel:
    """Loaded once at application startup, never per request."""

    def __init__(self, artifacts: Path = Path("artifacts")):
        self.artifacts = Path(artifacts)
        self.booster = xgb.Booster()
        self.booster.load_model(self.artifacts / "model.json")
        self.schema = FeatureSchema.load(self.artifacts / "feature_schema.json")
        self.reference = pd.read_parquet(self.artifacts / "reference_cohort.parquet")
        self.card = json.loads((self.artifacts / "model_card.json").read_text())
        self._columns = list(self.schema.columns)

    # ---------------------------------------------------------------- scoring

    def _reference_slice(self, oncotree_code: str | None) -> tuple[pd.DataFrame, str]:
        if oncotree_code:
            subset = self.reference[self.reference["oncotree_code"] == oncotree_code]
            if len(subset) >= MIN_REFERENCE_COHORT:
                return subset, oncotree_code
        return self.reference, PAN_CANCER

    def predict(self, patient: PatientFeatures) -> Prediction:
        row = feat.build_row(patient, self.schema)
        frame = pd.DataFrame([row], columns=self._columns)
        matrix = xgb.DMatrix(frame, missing=np.nan)

        score = float(self.booster.predict(matrix, output_margin=True)[0])
        contributions = self.booster.predict(matrix, pred_contribs=True)[0]

        subset, reference_code = self._reference_slice(patient.oncotree_code)
        scores = subset["risk_score"].to_numpy()
        percentile = int(round(float((scores < score).mean() * 100)))

        lower, upper = np.quantile(scores, [1 / 3, 2 / 3])
        tertile = "low" if score < lower else "high" if score >= upper else "intermediate"

        # pred_contribs appends the bias term, which is not a feature.
        feature_contributions = contributions[: len(self._columns)]
        ranked = np.argsort(np.abs(feature_contributions))[::-1][:TOP_ATTRIBUTIONS]
        attributions = [
            {
                "feature": human_label(self._columns[index], row[index]),
                "column": self._columns[index],
                "contribution": round(float(feature_contributions[index]), 4),
                "direction": (
                    "increases risk" if feature_contributions[index] > 0 else "decreases risk"
                ),
            }
            for index in ranked
            if abs(feature_contributions[index]) > 1e-9
        ]

        missing = [
            self._columns[index]
            for index in range(len(self._columns))
            if np.isnan(row[index]) and not self._columns[index].startswith(("mut_", "amp_", "del_"))
        ]
        untested_genes = int(
            sum(
                1
                for index, column in enumerate(self._columns)
                if column.startswith(("mut_", "amp_", "del_")) and np.isnan(row[index])
            )
        )
        if untested_genes:
            missing.append(f"{untested_genes} genes not on panel {patient.panel or 'unknown'}")

        return Prediction(
            risk_score=score,
            risk_tertile=tertile,
            percentile_within_cancer_type=percentile,
            reference_cohort={
                "oncotree_code": reference_code,
                "n": int(len(subset)),
                "is_pan_cancer_fallback": reference_code == PAN_CANCER,
            },
            attributions=attributions,
            missing_features=missing,
            input_caveats=list(patient.caveats),
            model_version=MODEL_VERSION,
        )

    # ----------------------------------------------------------- KM reference

    @lru_cache(maxsize=128)
    def reference_km(self, oncotree_code: str | None) -> dict:
        """Kaplan-Meier curves for the reference cohort, split into tertiles.

        Built from out-of-fold risk scores, so the separation shown is what the
        model achieves on patients it did not see, not the flattering in-sample
        version.
        """
        subset, reference_code = self._reference_slice(oncotree_code)
        scores = subset["risk_score"].to_numpy()
        lower, upper = np.quantile(scores, [1 / 3, 2 / 3])

        group = np.where(scores < lower, 0, np.where(scores >= upper, 2, 1))
        names = ["low", "intermediate", "high"]
        event = subset["event"].to_numpy().astype(bool)
        months = subset["os_months"].to_numpy()

        curves = []
        for index, name in enumerate(names):
            mask = group == index
            times, survival = kaplan_meier_estimator(event[mask], months[mask])
            at_risk = [int((months[mask] >= t).sum()) for t in times]
            curves.append(
                {
                    "tertile": name,
                    "n": int(mask.sum()),
                    "events": int(event[mask].sum()),
                    "time": [round(float(t), 2) for t in times],
                    "survival": [round(float(s), 4) for s in survival],
                    "at_risk": at_risk,
                }
            )

        structured = np.empty(len(months), dtype=[("event", bool), ("time", float)])
        structured["event"] = event
        structured["time"] = months
        try:
            statistic, p_value = compare_survival(structured, group)
        except Exception:
            statistic, p_value = float("nan"), float("nan")

        return {
            "reference_cohort": {
                "oncotree_code": reference_code,
                "n": int(len(subset)),
                "is_pan_cancer_fallback": reference_code == PAN_CANCER,
            },
            "tertile_thresholds": [round(float(lower), 4), round(float(upper), 4)],
            "curves": curves,
            "logrank": {
                "statistic": None if np.isnan(statistic) else round(float(statistic), 3),
                "p_value": None if np.isnan(p_value) else float(p_value),
            },
            "max_follow_up_months": round(float(months.max()), 1),
            "model_version": MODEL_VERSION,
        }

    def available_cancer_types(self) -> list[dict]:
        counts = self.reference["oncotree_code"].value_counts()
        return [
            {"oncotree_code": str(code), "n": int(count)}
            for code, count in counts.items()
            if count >= MIN_REFERENCE_COHORT
        ]


def predict_from_payloads(
    model: RiskModel,
    clinical_data: list[dict],
    mutations: list[dict],
    discrete_cna: list[dict],
) -> Prediction:
    patient = feat.from_api_payloads(clinical_data, mutations, discrete_cna, model.schema)
    return model.predict(patient)
