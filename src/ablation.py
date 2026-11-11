"""The four-arm ablation. This is the most important result in the report.

A pan-cancer survival model mostly learns the diagnosis. Pancreatic does worse
than thyroid, the model finds that in a handful of trees, and the C-index looks
respectable while the output tells an oncologist nothing they did not already
know. These four arms decompose that.

    1. clinical        age, sex, cancer type, sample type
    2. genomic         mutations, CNA, TMB, MSI, FGA; no diagnosis, no age, no sex
    3. full            what the service actually serves
    4. full + mets     arm 3 plus MET_SITE_COUNT

Arm 4 is never served. It exists to measure how much of the published 0.76
C-index of Mestrallet (Onco 2025) depends on a feature recorded across the
patient's whole disease course, which is unavailable when a new patient is
sequenced. Reporting a small honest delta is a stronger result than a large
number with no decomposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
from features import MET_PROBE_COLUMN
from train import (
    DEFAULT_PARAMS,
    breslow_baseline,
    c_index,
    fit,
    out_of_fold,
    prepare,
    risk,
)

CLINICAL_PREFIXES = ("onco_",)
CLINICAL_EXACT = ("age", "sex_female", "sample_is_metastasis")
GENOMIC_PREFIXES = ("mut_", "amp_", "del_")
GENOMIC_EXACT = ("tmb", "msi_score", "fga", "tmb_missing", "msi_score_missing")


def columns_for(arm: str, all_columns: list[str]) -> list[str]:
    def matches(column: str, exact, prefixes) -> bool:
        return column in exact or column.startswith(prefixes)

    if arm == "clinical":
        return [
            c for c in all_columns if matches(c, CLINICAL_EXACT, CLINICAL_PREFIXES)
        ]
    if arm == "genomic":
        return [c for c in all_columns if matches(c, GENOMIC_EXACT, GENOMIC_PREFIXES)]
    if arm == "full":
        return [c for c in all_columns if c != MET_PROBE_COLUMN]
    if arm == "full_plus_mets":
        return list(all_columns)
    raise ValueError(f"unknown arm {arm!r}")


ARM_LABELS = {
    "clinical": "Clinical only (age, sex, cancer type, sample type)",
    "genomic": "Genomic only (mutations, CNA, TMB, MSI, FGA)",
    "full": "Full model (served)",
    "full_plus_mets": "Full + metastasis count (leakage probe, never served)",
}


def run_arm(
    arm: str,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    seed: int,
) -> dict:
    columns = columns_for(arm, list(x_train.columns))
    train_subset, test_subset = x_train[columns], x_test[columns]

    print(f"\n--- {arm}: {len(columns):,} columns ---")
    booster, rounds = fit(train_subset, y_train, DEFAULT_PARAMS, seed)
    scores = risk(booster, test_subset, rounds)

    oof = out_of_fold(train_subset, y_train, DEFAULT_PARAMS, seed)
    baseline = breslow_baseline(
        y_train["os_months"].to_numpy(), y_train["event"].to_numpy(), oof
    )

    result = {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "n_features": len(columns),
        "boosting_rounds": rounds,
        "c_index_out_of_fold": round(c_index(y_train, oof), 4),
        **metrics.evaluate_all(y_train, y_test, scores, baseline),
    }
    print(
        f"  test C-index {result['c_index']:.4f}   "
        f"out-of-fold {result['c_index_out_of_fold']:.4f}   "
        f"{rounds} rounds"
    )
    return result, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default="data/msk_met_2021", type=Path)
    parser.add_argument("--out", default=Path("artifacts"), type=Path)
    parser.add_argument("--seed", default=20261111, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    args = parser.parse_args()

    # Build once with the probe column present; each arm then selects columns.
    schema, x_train, y_train, x_test, y_test = prepare(
        args.study_dir, args.seed, args.test_size, include_mets_probe=True
    )
    probe = x_train[MET_PROBE_COLUMN]
    print(
        f"\nleakage probe present: {MET_PROBE_COLUMN}, "
        f"{float(probe.notna().mean()) * 100:.1f}% populated, "
        f"median {float(probe.median()):.1f} sites"
    )

    results = []
    per_type = None
    for arm in ("clinical", "genomic", "full", "full_plus_mets"):
        result, scores = run_arm(arm, x_train, y_train, x_test, y_test, args.seed)
        results.append(result)
        if arm == "full":
            per_type = metrics.c_index_per_cancer_type(y_test, scores)

    by_arm = {r["arm"]: r["c_index"] for r in results}
    deltas = {
        "genomics_on_top_of_clinical": round(by_arm["full"] - by_arm["clinical"], 4),
        "clinical_on_top_of_genomic": round(by_arm["full"] - by_arm["genomic"], 4),
        "metastasis_count_on_top_of_full": round(
            by_arm["full_plus_mets"] - by_arm["full"], 4
        ),
    }

    payload = {
        "seed": args.seed,
        "arms": results,
        "deltas": deltas,
        "c_index_per_cancer_type": per_type.to_dict("records") if per_type is not None else [],
        "published_reference": {
            "source": "Mestrallet, Onco 2025, 10.3390/onco5010008",
            "c_index": 0.76,
            "cohort": "msk_met_2021 merged with msk_ch_2020, n=16,603",
            "included_features_we_exclude": ["treatment", "metastasis count"],
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 72)
    print(f"{'arm':<52} {'features':>9} {'C-index':>9}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['label']:<52} {result['n_features']:>9,} {result['c_index']:>9.4f}"
        )
    print("-" * 72)
    for name, value in deltas.items():
        print(f"{name:<52} {value:>+19.4f}")
    print("=" * 72)

    if per_type is not None and len(per_type):
        print("\nper cancer type, full model, test split (>=100 patients):")
        print(per_type.to_string(index=False))

    print(f"\nwrote {args.out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
