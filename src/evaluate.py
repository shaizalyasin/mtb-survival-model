"""Figures and metrics for the report.

Currently renders the Kaplan-Meier figure, which is the element the spec puts
first because it carries more conviction with clinicians than any metric. The
full metric suite and the four-arm ablation are added here once the thin slice
is confirmed working.

Every curve is drawn from out-of-fold risk scores. An in-sample KM plot shows
separation the model does not actually achieve on unseen patients, and it is
the one figure where that would be most misleading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inference import RiskModel  # noqa: E402

TERTILE_COLOURS = {
    "low": "#2c7fb8",
    "intermediate": "#8c8c8c",
    "high": "#c0392b",
}


def format_p(p_value: float | None) -> str:
    if p_value is None:
        return "log-rank p: not computed"
    if p_value < 1e-16:
        return "log-rank p < 1e-16"
    return f"log-rank p = {p_value:.2e}"


def plot_km(
    km: dict,
    out: Path,
    patient_tertile: str | None = None,
    patient_percentile: int | None = None,
) -> Path:
    cohort = km["reference_cohort"]
    figure, (axis, risk_axis) = plt.subplots(
        2, 1, figsize=(8, 6.4), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )

    horizon = km["max_follow_up_months"]
    ticks = [t for t in range(0, int(horizon) + 1, 12)]

    for curve in km["curves"]:
        name = curve["tertile"]
        highlighted = patient_tertile == name
        axis.step(
            curve["time"],
            curve["survival"],
            where="post",
            color=TERTILE_COLOURS[name],
            linewidth=3.0 if highlighted else 1.6,
            alpha=1.0 if highlighted or patient_tertile is None else 0.45,
            label=(
                f"{name} risk (n={curve['n']:,}, {curve['events']:,} deaths)"
                + ("  <- this patient" if highlighted else "")
            ),
        )

    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Overall survival")
    axis.set_title(
        f"Reference cohort: {cohort['oncotree_code']} (n = {cohort['n']:,})"
        + ("  [pan-cancer fallback]" if cohort["is_pan_cancer_fallback"] else ""),
        loc="left",
    )
    axis.legend(frameon=False, fontsize=9, loc="lower left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)

    annotation = format_p(km["logrank"]["p_value"])
    if patient_percentile is not None:
        annotation = (
            f"patient at the {patient_percentile}th percentile of risk\n{annotation}"
        )
    axis.annotate(
        annotation,
        xy=(0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )

    # Number at risk, read off each curve at the yearly ticks. The row keeps
    # its x axis so the month ticks stay visible under the counts.
    risk_axis.set_yticks([])
    risk_axis.spines[["top", "left", "right"]].set_visible(False)
    risk_axis.set_ylim(0, 1)
    for row, curve in enumerate(km["curves"]):
        counts = []
        for tick in ticks:
            remaining = [
                at_risk
                for time_point, at_risk in zip(curve["time"], curve["at_risk"])
                if time_point >= tick
            ]
            counts.append(max(remaining) if remaining else 0)
        height = 0.76 - row * 0.26
        risk_axis.text(
            -0.035,
            height,
            curve["tertile"],
            transform=risk_axis.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color=TERTILE_COLOURS[curve["tertile"]],
        )
        for tick, count in zip(ticks, counts):
            risk_axis.text(
                tick,
                height,
                f"{count:,}",
                transform=risk_axis.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
            )
    risk_axis.text(
        -0.035,
        1.02,
        "at risk",
        transform=risk_axis.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#222222",
    )

    axis.set_xticks(ticks)
    axis.set_xlim(0, horizon)
    axis.tick_params(labelbottom=False)
    risk_axis.set_xticks(ticks)
    risk_axis.set_xlim(0, horizon)
    risk_axis.set_xlabel("Months since sequencing")

    figure.text(
        0.012,
        0.015,
        "Out-of-fold predictions on MSK-MET. Research use only, "
        "not for clinical decision-making.",
        fontsize=7.5,
        color="#777777",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    # Leave room on the left for the at-risk row labels.
    figure.tight_layout(rect=(0.055, 0.03, 1, 1))
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=Path("artifacts"), type=Path)
    parser.add_argument("--figures", default=Path("figures"), type=Path)
    parser.add_argument("--oncotree", default="LUAD")
    parser.add_argument("--tertile", default=None)
    parser.add_argument("--percentile", default=None, type=int)
    args = parser.parse_args()

    model = RiskModel(args.artifacts)
    km = model.reference_km(args.oncotree)
    path = plot_km(
        km,
        args.figures / f"km_{km['reference_cohort']['oncotree_code']}.png",
        patient_tertile=args.tertile,
        patient_percentile=args.percentile,
    )
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    for curve in km["curves"]:
        print(
            f"  {curve['tertile']:14s} n={curve['n']:5,} deaths={curve['events']:5,}"
        )
    print(f"  {format_p(km['logrank']['p_value'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
