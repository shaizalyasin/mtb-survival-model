"""Verify the exact HTTP contract the portal tab uses.

The tab does three things from the browser: fetch a patient's payloads from
cBioPortal, post them here, and ask for the reference curves. This drives that
same sequence against a running service, including the CORS preflight, so a
change that would break the tab fails here rather than in the portal.

    MTB_CORS_ORIGINS=http://localhost:9090 make serve     # in one shell
    python src/verify_contract.py                         # in another
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from demo_patient import DEFAULT_PORTAL, DEFAULT_STUDY, fetch_patient

PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"{PASS if ok else FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def request(url: str, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
        return response.status, dict(response.headers), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="http://127.0.0.1:8000")
    parser.add_argument("--portal", default=DEFAULT_PORTAL)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--origin", default="http://localhost:9090")
    args = parser.parse_args()

    print(f"service {args.service}   portal {args.portal}   study {args.study}\n")

    # 1. The service is up and has a model loaded.
    try:
        _, _, body = request(f"{args.service}/health")
        health = json.loads(body)
    except Exception as error:
        check("Service health endpoint responds", False, str(error))
        return 1
    check(
        "Service health endpoint responds",
        health.get("status") == "ok",
        f"model {health.get('model_version')}, {health.get('features')} features",
    )
    check(
        "Service declares itself stateless",
        health.get("stateless") is True,
    )

    # 2. CORS preflight, which is what the browser sends before the POST when
    #    the service is on a different origin from the portal.
    try:
        status, headers, _ = request(
            f"{args.service}/predict",
            method="OPTIONS",
            headers={
                "Origin": args.origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        allowed = headers.get("access-control-allow-origin")
        check(
            "CORS preflight allows the portal origin",
            allowed in (args.origin, "*"),
            f"allow-origin: {allowed}",
        )
    except urllib.error.HTTPError as error:
        check("CORS preflight allows the portal origin", False, f"HTTP {error.code}")

    # 3. The portal fetch, exactly as portalApi.ts does it.
    payloads = fetch_patient(args.portal, args.study, None)
    check(
        "Portal returns clinical, mutation and CNA payloads",
        bool(payloads["clinical"]),
        f"{len(payloads['clinical'])} clinical, {len(payloads['mutations'])} mutations, "
        f"{len(payloads['cna'])} CNA",
    )

    request_body = {
        "clinical_data": payloads["clinical"],
        "mutations": payloads["mutations"],
        "discrete_cna": payloads["cna"],
    }

    # 4. The prediction itself.
    _, _, body = request(f"{args.service}/predict", method="POST", body=request_body)
    prediction = json.loads(body)

    check(
        "Prediction returns a percentile in range",
        0 <= prediction["percentile_within_cancer_type"] <= 100,
        f"{prediction['percentile_within_cancer_type']}th percentile",
    )
    check(
        "Prediction returns a tertile",
        prediction["risk_tertile"] in {"low", "intermediate", "high"},
        prediction["risk_tertile"],
    )
    check(
        "Reference cohort is named and sized",
        prediction["reference_cohort"]["n"] > 0,
        f"{prediction['reference_cohort']['oncotree_code']}, "
        f"n={prediction['reference_cohort']['n']:,}",
    )
    check(
        "Attributions are human readable, not column names",
        bool(prediction["attributions"])
        and not prediction["attributions"][0]["feature"].startswith(("mut_", "onco_")),
        prediction["attributions"][0]["feature"] if prediction["attributions"] else "",
    )
    check(
        "Response carries the model version",
        bool(prediction.get("model_version")),
        prediction.get("model_version", ""),
    )
    check(
        "Response carries a research-use disclaimer",
        "not for clinical decision-making" in prediction.get("disclaimer", ""),
    )
    check(
        "Missing inputs are reported rather than hidden",
        isinstance(prediction.get("missing_features"), list),
        ", ".join(prediction["missing_features"])[:70] or "none missing",
    )

    # 5. The reference curves the tab draws.
    code = prediction["reference_cohort"]["oncotree_code"]
    _, _, body = request(f"{args.service}/reference-km/{code}")
    km = json.loads(body)

    tertiles = [curve["tertile"] for curve in km["curves"]]
    check(
        "Reference curves cover all three tertiles",
        tertiles == ["low", "intermediate", "high"],
        ", ".join(tertiles),
    )
    check(
        "Curve arrays are the same length as their at-risk counts",
        all(
            len(c["time"]) == len(c["survival"]) == len(c["at_risk"])
            for c in km["curves"]
        ),
    )
    by_name = {c["tertile"]: c for c in km["curves"]}
    check(
        "High-risk tertile has more deaths than low-risk",
        by_name["high"]["events"] > by_name["low"]["events"],
        f"{by_name['low']['events']} low vs {by_name['high']['events']} high",
    )
    check(
        "The patient's tertile is one the curves contain",
        prediction["risk_tertile"] in by_name,
    )

    # 6. The property that lets the service stay out of access control.
    try:
        request(f"{args.service}/predict", method="POST", body={"clinical_data": []})
        check("Service refuses a request with no clinical data", False, "accepted it")
    except urllib.error.HTTPError as error:
        check(
            "Service refuses a request with no clinical data",
            error.code == 422,
            f"HTTP {error.code}",
        )

    failures = sum(1 for status, _, _ in results if status == FAIL)
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
