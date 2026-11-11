"""Score one real patient fetched live from a cBioPortal instance.

This is the end-to-end check the spec asks for before anything is tuned: the
serving path runs against payloads in the shape the portal actually returns,
not against a row lifted out of the training frame.

It deliberately targets a study the model was NOT trained on (msk_chord_2024),
same MSK-IMPACT panel family but unseen patients, so the demo never shows a
patient the model has already memorised.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import features as feat
from inference import RiskModel

DEFAULT_PORTAL = "https://www.cbioportal.org"
DEFAULT_STUDY = "msk_chord_2024"


def _get(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _post(url: str, body: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def fetch_patient(portal: str, study: str, patient_id: str | None) -> dict:
    """The three payloads Stage 5 takes as input, for one sample."""
    base = f"{portal}/api"

    if patient_id is None:
        patients = _get(f"{base}/studies/{study}/patients?pageSize=1&pageNumber=0")
        patient_id = patients[0]["patientId"]

    samples = _get(f"{base}/studies/{study}/patients/{patient_id}/samples")
    if not samples:
        raise SystemExit(f"patient {patient_id} has no samples")
    sample_id = samples[0]["sampleId"]

    profiles = _get(f"{base}/studies/{study}/molecular-profiles")
    mutation_profile = next(
        p["molecularProfileId"]
        for p in profiles
        if p["molecularAlterationType"] == "MUTATION_EXTENDED"
    )
    cna_profile = next(
        (
            p["molecularProfileId"]
            for p in profiles
            if p["molecularAlterationType"] == "COPY_NUMBER_ALTERATION"
            and p.get("datatype") == "DISCRETE"
        ),
        None,
    )

    clinical = _get(f"{base}/studies/{study}/samples/{sample_id}/clinical-data")
    clinical += _get(f"{base}/studies/{study}/patients/{patient_id}/clinical-data")

    mutations = _post(
        f"{base}/molecular-profiles/{mutation_profile}/mutations/fetch"
        "?projection=DETAILED",
        {"sampleIds": [sample_id]},
    )

    cna = []
    if cna_profile:
        try:
            cna = _post(
                f"{base}/molecular-profiles/{cna_profile}/discrete-copy-number/fetch"
                "?discreteCopyNumberEventType=ALL&projection=DETAILED",
                {"sampleIds": [sample_id]},
            )
        except urllib.error.HTTPError as error:
            print(f"  (CNA fetch returned HTTP {error.code}, continuing without it)")

    return {
        "study": study,
        "patient_id": patient_id,
        "sample_id": sample_id,
        "clinical": clinical,
        "mutations": mutations,
        "cna": cna,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal", default=DEFAULT_PORTAL)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--artifacts", default=Path("artifacts"), type=Path)
    parser.add_argument("--save", default=None, type=Path, help="write the payloads as JSON")
    args = parser.parse_args()

    print(f"fetching from {args.portal} study {args.study}")
    payloads = fetch_patient(args.portal, args.study, args.patient)
    print(
        f"  patient {payloads['patient_id']} sample {payloads['sample_id']}: "
        f"{len(payloads['clinical'])} clinical attributes, "
        f"{len(payloads['mutations'])} mutation records, "
        f"{len(payloads['cna'])} CNA records"
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payloads, indent=2))
        print(f"  saved payloads to {args.save}")

    model = RiskModel(args.artifacts)

    patient = feat.from_api_payloads(
        payloads["clinical"], payloads["mutations"], payloads["cna"], model.schema
    )
    print("\n--- normalised input ---")
    print(f"oncotree      : {patient.oncotree_code}")
    print(f"panel         : {patient.panel}")
    print(f"age / sex     : {patient.age} / {patient.sex}")
    print(f"sample type   : {patient.sample_type}")
    print(f"tmb / msi/fga : {patient.tmb} / {patient.msi_score} / {patient.fga}")
    print(f"mutated genes : {len(patient.mutated_genes)}  {sorted(patient.mutated_genes)[:8]}")
    print(f"amplified     : {len(patient.amplified_genes)}  {sorted(patient.amplified_genes)[:6]}")
    print(f"deleted       : {len(patient.deleted_genes)}  {sorted(patient.deleted_genes)[:6]}")

    print("\n--- prediction ---")
    print(json.dumps(model.predict(patient).to_dict(), indent=2))

    km = model.reference_km(patient.oncotree_code)
    print("\n--- reference KM ---")
    print(
        f"cohort {km['reference_cohort']['oncotree_code']} "
        f"n={km['reference_cohort']['n']}  "
        f"logrank p={km['logrank']['p_value']}"
    )
    for curve in km["curves"]:
        print(
            f"  {curve['tertile']:14s} n={curve['n']:5,} events={curve['events']:5,} "
            f"points={len(curve['time']):4,}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
