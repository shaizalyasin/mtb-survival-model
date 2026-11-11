"""FastAPI service wrapping the inference module.

Deliberately stateless. No database, no session store, no request logging of
payloads, nothing written to disk after startup. Patient genomic data is held
only for the lifetime of the request. That is not tidiness: it turns the
data-protection question from "we store patient data, here are our controls"
into "we retain nothing", which is true by construction rather than by policy.

The service is never given a study or sample identifier to go and fetch. The
caller sends payloads it already retrieved under the end user's own portal
session, so this service has no credentials of its own and cannot be used to
read a patient the user is not entitled to see.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import MODEL_VERSION, RiskModel, predict_from_payloads

ARTIFACTS = Path(os.environ.get("MTB_ARTIFACTS", "artifacts"))

# The portal tab calls this service from the browser. Behind the auth proxy
# both share an origin and no CORS is involved; in the plain demo stack the
# service sits on its own port, so the portal's origin has to be allowed.
# Listed explicitly rather than "*", because a wildcard here would let any page
# the clinician visits post patient payloads to this service.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("MTB_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Loaded once, at startup, never per request.
_model: RiskModel | None = None


def model() -> RiskModel:
    global _model
    if _model is None:
        _model = RiskModel(ARTIFACTS)
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI):
    model()
    yield


app = FastAPI(
    title="MTB survival risk model",
    version=MODEL_VERSION,
    description=(
        "Retrospective overall-survival risk model trained on MSK-MET. "
        "Research use only, not for clinical decision-making."
    ),
    lifespan=lifespan,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        # No cookies are involved: the service has no session and no identity
        # of its own, so credentialed requests would serve no purpose.
        allow_credentials=False,
    )


class PredictRequest(BaseModel):
    """The three cBioPortal payloads for one sample.

    `clinical_data` is the sample-level and patient-level clinical-data
    responses concatenated; both carry clinicalAttributeId / value pairs.
    """

    clinical_data: list[dict] = Field(default_factory=list)
    mutations: list[dict] = Field(default_factory=list)
    discrete_cna: list[dict] = Field(default_factory=list)


DISCLAIMER = (
    "Retrospective model trained on MSK-MET (n = 24,561 after exclusions, "
    "MSK-IMPACT panel sequencing). Not prospectively validated. "
    "Research use only, not for clinical decision-making."
)


@app.get("/health")
def health() -> dict:
    loaded = model()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "schema_version": loaded.schema.version,
        "features": len(loaded.schema.columns),
        "reference_cohort_size": int(len(loaded.reference)),
        "stateless": True,
    }


@app.get("/model-card")
def model_card() -> dict:
    return {**model().card, "disclaimer": DISCLAIMER}


@app.get("/cancer-types")
def cancer_types() -> dict:
    """OncoTree codes with a reference cohort large enough to quote against."""
    return {"cancer_types": model().available_cancer_types()}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if not request.clinical_data:
        raise HTTPException(
            status_code=422,
            detail="clinical_data is required; without it there is no cancer type "
            "to build a reference cohort from",
        )
    prediction = predict_from_payloads(
        model(), request.clinical_data, request.mutations, request.discrete_cna
    )
    return {**prediction.to_dict(), "disclaimer": DISCLAIMER}


@app.get("/reference-km/{oncotree_code}")
def reference_km(oncotree_code: str) -> dict:
    """Out-of-fold KM curves by risk tertile, for the tab to draw.

    These are static per cancer type, so the tab can cache them and only the
    patient's own tertile marker changes between patients.
    """
    return {**model().reference_km(oncotree_code), "disclaimer": DISCLAIMER}
