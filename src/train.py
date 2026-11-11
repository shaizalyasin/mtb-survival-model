"""Train the survival model and write the serving artefacts.

Model: XGBoost with objective="survival:cox". The spec allowed scikit-survival
as an alternative, but it is not actually interchangeable here: the feature
matrix carries NaN for untested genes and for absent TMB/MSI, and
GradientBoostingSurvivalAnalysis requires imputation. Imputing those to zero
would assert "no mutation found" for genes nobody sequenced.

Note on scales. For survival:cox, XGBoost's predict() returns exp(margin),
a hazard ratio, while SHAP works in margin space. Everything here uses the
margin (log hazard) so that attributions and the score are additive and
comparable.

The reference cohort used for percentiles and for the displayed Kaplan-Meier
curves is built from out-of-fold predictions, never in-sample ones. In-sample
risk scores are optimistically spread, which would both distort a new patient's
percentile and flatter the KM separation that clinicians are shown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.metrics import concordance_index_censored

import features as feat
import panels as panel_module
from load import load

MODEL_VERSION = "1.0.0"

PANEL_NAMES = ["IMPACT341", "IMPACT410", "IMPACT468"]

DEFAULT_PARAMS = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "tree_method": "hist",
    "nthread": 0,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50


# --------------------------------------------------------------------------
# Feature stage
# --------------------------------------------------------------------------


def cox_label(months: np.ndarray, event: np.ndarray) -> np.ndarray:
    """XGBoost's Cox label: positive time for an event, negative for censored."""
    signed = np.where(event, months, -months)
    return signed


def prepare(
    study_dir: Path, seed: int, test_size: float, include_mets_probe: bool = False
):
    started = time.time()
    cohort = load(study_dir)
    gene_panels = panel_module.load_panels(PANEL_NAMES)
    rows = feat.from_flat_files(cohort, gene_panels)
    labels = feat.survival_labels(cohort, rows)
    print(f"normalised {len(rows):,} patients in {time.time() - started:.1f}s")

    # A patient with no follow-up time carries no survival information and the
    # Cox label cannot encode a zero, so these go before the split.
    missing = labels["os_months"].isna()
    zero = labels["os_months"].notna() & (labels["os_months"] <= 0)
    usable = ~(missing | zero)
    keep = set(labels.index[usable])
    rows = [row for row in rows if row.patient_id in keep]
    labels = labels.loc[[row.patient_id for row in rows]]
    print(
        f"dropped {int(missing.sum()):,} with OS_MONTHS missing and "
        f"{int(zero.sum()):,} with OS_MONTHS <= 0 "
        f"(sequenced at or after the last recorded contact)"
    )

    # Stratify on cancer type crossed with event status, collapsing rare strata
    # so train_test_split can honour them.
    strata = labels["oncotree_code"].fillna("NA") + "|" + labels["event"].astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 10, other="RARE")

    train_ids, test_ids = train_test_split(
        labels.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=strata.to_numpy(),
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    if train_ids & test_ids:
        raise AssertionError("patient appears in both splits")

    train_rows = [row for row in rows if row.patient_id in train_ids]
    test_rows = [row for row in rows if row.patient_id in test_ids]

    schema = feat.build_schema(
        train_rows, gene_panels, include_mets_probe=include_mets_probe
    )
    print(
        f"schema {schema.version}: {len(schema.columns):,} columns "
        f"({len(schema.oncotree_codes)} oncotree, {len(schema.mutation_genes)} mutation, "
        f"{len(schema.amp_genes)} amp, {len(schema.del_genes)} del)"
    )

    started = time.time()
    x_train = feat.build_matrix(train_rows, schema)
    x_test = feat.build_matrix(test_rows, schema)
    print(f"built design matrices in {time.time() - started:.1f}s")

    y_train = labels.loc[x_train.index]
    y_test = labels.loc[x_test.index]
    return schema, x_train, y_train, x_test, y_test


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------


def fit(x: pd.DataFrame, y: pd.DataFrame, params: dict, seed: int, rounds: int | None = None):
    """Fit on x/y, holding back a slice for early stopping if rounds is None."""
    label = cox_label(y["os_months"].to_numpy(), y["event"].to_numpy())

    if rounds is not None:
        matrix = xgb.DMatrix(x, label=label, missing=np.nan)
        booster = xgb.train({**params, "seed": seed}, matrix, num_boost_round=rounds)
        return booster, rounds

    inner_train, inner_valid = train_test_split(
        np.arange(len(x)), test_size=0.15, random_state=seed, stratify=y["event"].to_numpy()
    )
    d_train = xgb.DMatrix(x.iloc[inner_train], label=label[inner_train], missing=np.nan)
    d_valid = xgb.DMatrix(x.iloc[inner_valid], label=label[inner_valid], missing=np.nan)
    booster = xgb.train(
        {**params, "seed": seed},
        d_train,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(d_valid, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    return booster, booster.best_iteration + 1


def risk(booster: xgb.Booster, x: pd.DataFrame, rounds: int | None = None) -> np.ndarray:
    """Log-hazard margin, the additive scale SHAP also works on."""
    matrix = xgb.DMatrix(x, missing=np.nan)
    kwargs = {"output_margin": True}
    if rounds is not None:
        kwargs["iteration_range"] = (0, rounds)
    return booster.predict(matrix, **kwargs)


def c_index(y: pd.DataFrame, scores: np.ndarray) -> float:
    result = concordance_index_censored(
        y["event"].to_numpy().astype(bool),
        y["os_months"].to_numpy(),
        scores,
    )
    return float(result[0])


def out_of_fold(x: pd.DataFrame, y: pd.DataFrame, params: dict, seed: int, folds: int = 5):
    """Out-of-fold risk scores for every training patient.

    These are what the reference distribution and the displayed KM tertiles are
    built from. Using in-sample scores here is the subtle error that makes a
    model look better to a clinician than it is.
    """
    scores = np.full(len(x), np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    strata = y["event"].astype(str).to_numpy()

    for index, (train_idx, hold_idx) in enumerate(splitter.split(x, strata), start=1):
        booster, rounds = fit(x.iloc[train_idx], y.iloc[train_idx], params, seed + index)
        scores[hold_idx] = risk(booster, x.iloc[hold_idx], rounds)
        print(f"  fold {index}/{folds}: {rounds} rounds")

    if np.isnan(scores).any():
        raise AssertionError("out-of-fold scores incomplete")
    return scores


def breslow_baseline(months: np.ndarray, event: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Breslow estimate of the cumulative baseline hazard.

    XGBoost's Cox objective returns a relative risk with no baseline, so
    survival probabilities, and therefore the integrated Brier score, are not
    computable without this.
    """
    order = np.argsort(months)
    months, event, scores = months[order], event[order], scores[order]
    exp_scores = np.exp(scores)

    times, increments = [], []
    for time_point in np.unique(months[event.astype(bool)]):
        at_risk = months >= time_point
        deaths = int(((months == time_point) & event.astype(bool)).sum())
        denominator = exp_scores[at_risk].sum()
        if denominator > 0:
            times.append(float(time_point))
            increments.append(deaths / denominator)

    return pd.DataFrame(
        {"time": times, "cumulative_hazard": np.cumsum(increments)}
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default="data/msk_met_2021", type=Path)
    parser.add_argument("--out", default="artifacts", type=Path)
    parser.add_argument("--seed", default=20261111, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()

    schema, x_train, y_train, x_test, y_test = prepare(
        args.study_dir, args.seed, args.test_size
    )

    nan_fraction = float(np.isnan(x_train.to_numpy()).mean())
    gene_columns = [c for c in schema.columns if c.startswith(("mut_", "amp_", "del_"))]
    off_panel = float(np.isnan(x_train[gene_columns].to_numpy()).mean())
    print(
        f"train {x_train.shape[0]:,} x {x_train.shape[1]:,}   "
        f"test {x_test.shape[0]:,}   events {int(y_train['event'].sum()):,} train / "
        f"{int(y_test['event'].sum()):,} test"
    )
    print(
        f"NaN overall {nan_fraction * 100:.2f}%   "
        f"of gene columns {off_panel * 100:.2f}% (untested, not wild type)"
    )

    if args.features_only:
        return 0

    print("\nfitting (default hyperparameters, early stopping)")
    booster, rounds = fit(x_train, y_train, DEFAULT_PARAMS, args.seed)
    print(f"  {rounds} boosting rounds")

    train_c = c_index(y_train, risk(booster, x_train, rounds))
    test_c = c_index(y_test, risk(booster, x_test, rounds))
    print(f"\nC-index  train {train_c:.4f}   test {test_c:.4f}")

    if test_c > 0.85:
        print(
            "\nWARNING: test C-index above 0.85. Stop and look for leakage "
            "before trusting this.",
            file=sys.stderr,
        )

    print("\nout-of-fold scores for the reference cohort")
    oof = out_of_fold(x_train, y_train, DEFAULT_PARAMS, args.seed)
    print(f"  out-of-fold C-index {c_index(y_train, oof):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.out / "model.json")
    schema.save(args.out / "feature_schema.json")

    reference = pd.DataFrame(
        {
            "patient_id": x_train.index,
            "oncotree_code": y_train["oncotree_code"].to_numpy(),
            "risk_score": oof,
            "os_months": y_train["os_months"].to_numpy(),
            "event": y_train["event"].to_numpy().astype(bool),
        }
    )
    reference.to_parquet(args.out / "reference_cohort.parquet", index=False)

    baseline = breslow_baseline(
        y_train["os_months"].to_numpy(), y_train["event"].to_numpy(), oof
    )
    baseline.to_parquet(args.out / "baseline_hazard.parquet", index=False)

    (args.out / "model_card.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "schema_version": schema.version,
                "trained_on": "msk_met_2021",
                "data_licence": "ODC-ODbL-1.0",
                "seed": args.seed,
                "boosting_rounds": rounds,
                "params": DEFAULT_PARAMS,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "n_features": len(schema.columns),
                "c_index_train": train_c,
                "c_index_test": test_c,
                "c_index_out_of_fold": c_index(y_train, oof),
                "os_anchor": "date of sequencing",
                "excluded_by_design": [
                    "MET_COUNT",
                    "MET_SITE_COUNT",
                    "DMETS_DX_*",
                    "treatment",
                    "AGE_AT_EVIDENCE_OF_METS",
                ],
            },
            indent=2,
        )
    )

    print(f"\nwrote artefacts to {args.out}/")
    for path in sorted(args.out.iterdir()):
        print(f"  {path.name:28s} {path.stat().st_size:>10,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
