"""Random search over XGBoost hyperparameters.

Protocol, which matters more than the search space:

  * Candidates are ranked by **out-of-fold C-index on the training split
    only**. The held-out test split is not read during the search. Selecting on
    the test split is how a tuning run invents an improvement that does not
    survive contact with new data.
  * The current defaults are evaluated as trial 0 under exactly the same
    protocol, so the comparison is like for like rather than against the
    number printed by a previous run with a different fold assignment.
  * Fold-level scores are kept for every trial. A gain smaller than the spread
    between folds is noise, and the summary says so rather than leaving the
    reader to assume the best number is real.

The test split is scored once, at the end, for the default and the winner.
That is the only time it is touched.

    python src/tune.py --trials 20            # about an hour
    python src/tune.py --trials 40 --max-seconds 7200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train import DEFAULT_PARAMS, c_index, fit, prepare, risk

# Kept identical to train.py's argparse defaults so the split this searches
# over is the same split the shipped model was fitted on.
DEFAULT_STUDY_DIR = Path("data/msk_met_2021")
DEFAULT_SEED = 20261111
RESULTS = Path("artifacts/tuning.json")


def sample_params(rng: np.random.Generator) -> dict:
    """One random configuration.

    The ranges are deliberately centred on the current defaults rather than
    spanning everything XGBoost allows: with 752 columns and 19k patients the
    useful variation is in tree depth, shrinkage and regularisation.
    """
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.20)))),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.3, 1.0)),
            "min_child_weight": int(rng.choice([1, 5, 10, 20, 50])),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(30.0)))),
            "reg_alpha": float(rng.choice([0.0, float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0))))])),
            "gamma": float(rng.choice([0.0, float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0))))])),
        }
    )
    return params


def fold_scores(
    x: pd.DataFrame, y: pd.DataFrame, params: dict, seed: int, folds: int
) -> tuple[list[float], list[int]]:
    """C-index on each held-out fold, plus the rounds early stopping chose.

    Per-fold scores rather than one pooled number: the spread is the only
    honest way to tell a real gain from a lucky split.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    strata = y["event"].astype(str).to_numpy()

    scores, rounds_used = [], []
    for index, (train_idx, hold_idx) in enumerate(splitter.split(x, strata), start=1):
        booster, rounds = fit(x.iloc[train_idx], y.iloc[train_idx], params, seed + index)
        held = risk(booster, x.iloc[hold_idx], rounds)
        scores.append(c_index(y.iloc[hold_idx], held))
        rounds_used.append(rounds)
    return scores, rounds_used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=float("inf"),
        help="stop cleanly after this long, keeping every completed trial",
    )
    args = parser.parse_args()

    schema, x_train, y_train, x_test, y_test = prepare(
        args.study_dir, args.seed, args.test_size
    )
    print(f"\ntuning on {len(x_train):,} training patients, {len(schema.columns):,} features")
    print(f"the {len(x_test):,} test patients are not read until the end\n")

    rng = np.random.default_rng(args.seed)
    started = time.time()
    trials: list[dict] = []

    for trial in range(args.trials + 1):
        if time.time() - started > args.max_seconds:
            print(f"time budget reached after {trial} trials")
            break

        # Trial 0 is the incumbent, scored under the identical protocol.
        params = dict(DEFAULT_PARAMS) if trial == 0 else sample_params(rng)
        label = "defaults" if trial == 0 else f"trial {trial}"

        t0 = time.time()
        scores, rounds_used = fold_scores(x_train, y_train, params, args.seed, args.folds)
        mean, spread = float(np.mean(scores)), float(np.std(scores))

        trials.append(
            {
                "trial": trial,
                "is_default": trial == 0,
                "params": params,
                "fold_c_index": scores,
                "mean_c_index": mean,
                "std_c_index": spread,
                "rounds": rounds_used,
                "seconds": round(time.time() - t0, 1),
            }
        )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(trials, indent=2))

        print(
            f"{label:>10}: OOF C-index {mean:.4f} +/- {spread:.4f}  "
            f"depth {params['max_depth']} lr {params['learning_rate']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )

    default = trials[0]
    best = max(trials, key=lambda t: t["mean_c_index"])
    gain = best["mean_c_index"] - default["mean_c_index"]

    print("\n" + "=" * 68)
    print(f"defaults : {default['mean_c_index']:.4f} +/- {default['std_c_index']:.4f}")
    print(f"best     : {best['mean_c_index']:.4f} +/- {best['std_c_index']:.4f}  (trial {best['trial']})")
    print(f"gain     : {gain:+.4f}")

    # One measurement each on the test split, and only now.
    results = {}
    for name, entry in (("defaults", default), ("best", best)):
        booster, rounds = fit(x_train, y_train, entry["params"], args.seed)
        results[name] = c_index(y_test, risk(booster, x_test, rounds))
    print(f"\ntest C-index: defaults {results['defaults']:.4f}, best {results['best']:.4f}")

    # The honest reading. A gain inside the fold-to-fold spread is not a result.
    if gain <= default["std_c_index"]:
        verdict = (
            "The gain is smaller than the spread between folds. That is not "
            "evidence of a better model; keep the defaults."
        )
    else:
        verdict = (
            "The gain exceeds the fold-to-fold spread. Worth adopting, after "
            "checking the test number moved in the same direction."
        )
    print(verdict)

    RESULTS.write_text(
        json.dumps(
            {
                "trials": trials,
                "default_mean": default["mean_c_index"],
                "best_mean": best["mean_c_index"],
                "best_params": best["params"],
                "gain": gain,
                "test_c_index": results,
                "verdict": verdict,
                "folds": args.folds,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
