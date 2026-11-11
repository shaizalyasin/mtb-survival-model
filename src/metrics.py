"""Evaluation metrics for a Cox risk score.

Two of these need more than the risk score itself. XGBoost's Cox objective
returns a relative risk with no baseline hazard, so survival probabilities do
not exist until a Breslow baseline is fitted on the training split. The
integrated Brier score needs those probabilities; the time-dependent AUC needs
only the ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

# Follow-up in MSK-MET tops out around 81 months and thins out well before
# that, so a horizon past this is estimated from very few patients.
DEFAULT_HORIZONS = (12.0, 24.0, 60.0)

MIN_PATIENTS_PER_CANCER_TYPE = 100


def structured(months: np.ndarray, event: np.ndarray) -> np.ndarray:
    array = np.empty(len(months), dtype=[("event", bool), ("time", float)])
    array["event"] = event.astype(bool)
    array["time"] = months
    return array


def c_index(months: np.ndarray, event: np.ndarray, scores: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), months, scores)[0])


def c_index_per_cancer_type(
    labels: pd.DataFrame,
    scores: np.ndarray,
    minimum: int = MIN_PATIENTS_PER_CANCER_TYPE,
) -> pd.DataFrame:
    """C-index within each cancer type, which is the honest view.

    Pan-cancer concordance is inflated by the model simply ranking cancer types
    against each other. Within a single type that advantage is gone, so these
    numbers are lower and more informative.
    """
    rows = []
    frame = labels.assign(score=scores)
    for code, group in frame.groupby("oncotree_code"):
        if len(group) < minimum or group["event"].sum() < 10:
            continue
        try:
            value = c_index(
                group["os_months"].to_numpy(),
                group["event"].to_numpy(),
                group["score"].to_numpy(),
            )
        except Exception:
            continue
        rows.append(
            {
                "oncotree_code": code,
                "n": int(len(group)),
                "events": int(group["event"].sum()),
                "c_index": round(value, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def survival_probabilities(
    baseline: pd.DataFrame, scores: np.ndarray, horizons: np.ndarray
) -> np.ndarray:
    """S(t | x) = exp(-H0(t) * exp(risk)), with H0 the Breslow estimate."""
    cumulative = np.interp(
        horizons,
        baseline["time"].to_numpy(),
        baseline["cumulative_hazard"].to_numpy(),
        left=0.0,
    )
    return np.exp(-np.outer(np.exp(scores), cumulative))


def time_dependent_auc(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    test_scores: np.ndarray,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
) -> dict:
    train = structured(
        train_labels["os_months"].to_numpy(), train_labels["event"].to_numpy()
    )
    test = structured(
        test_labels["os_months"].to_numpy(), test_labels["event"].to_numpy()
    )
    usable = [t for t in horizons if t < test["time"].max()]
    auc, mean_auc = cumulative_dynamic_auc(train, test, test_scores, np.array(usable))
    return {
        "per_horizon": {
            f"{int(t)}_months": round(float(value), 4) for t, value in zip(usable, auc)
        },
        "mean": round(float(mean_auc), 4),
    }


def brier(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    test_scores: np.ndarray,
    baseline: pd.DataFrame,
) -> dict:
    train = structured(
        train_labels["os_months"].to_numpy(), train_labels["event"].to_numpy()
    )
    test = structured(
        test_labels["os_months"].to_numpy(), test_labels["event"].to_numpy()
    )
    # The grid must stay strictly inside the follow-up of both splits or the
    # censoring distribution cannot be estimated at the endpoints.
    upper = min(train["time"].max(), test["time"].max())
    lower = max(1.0, float(np.percentile(test["time"][test["event"]], 5)))
    grid = np.linspace(lower, upper * 0.95, 40)

    probabilities = survival_probabilities(baseline, test_scores, grid)
    score = integrated_brier_score(train, test, probabilities, grid)
    return {
        "integrated_brier_score": round(float(score), 4),
        "grid_months": [round(float(grid[0]), 1), round(float(grid[-1]), 1)],
        "note": "lower is better; 0.25 is the value of always predicting 50%",
    }


def evaluate_all(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    test_scores: np.ndarray,
    baseline: pd.DataFrame | None = None,
) -> dict:
    result = {
        "c_index": round(
            c_index(
                test_labels["os_months"].to_numpy(),
                test_labels["event"].to_numpy(),
                test_scores,
            ),
            4,
        ),
        "n_test": int(len(test_labels)),
        "events_test": int(test_labels["event"].sum()),
    }
    try:
        result["time_dependent_auc"] = time_dependent_auc(
            train_labels, test_labels, test_scores
        )
    except Exception as error:
        result["time_dependent_auc"] = {"error": str(error)}

    if baseline is not None:
        try:
            result.update(brier(train_labels, test_labels, test_scores, baseline))
        except Exception as error:
            result["integrated_brier_score"] = None
            result["brier_error"] = str(error)
    return result
