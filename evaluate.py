"""
WEEK 5 STARTER — evaluation, calibration, and picking an operating point.

A ROC-AUC is not a decision. Somebody has to say "flag members above 0.31" and
defend it. That is the work in this file.

Fill in every section marked TODO. When you run this file it should:
  1. Rebuild the same held-out test set train.py used
  2. Plot ROC, precision-recall, and calibration curves to reports/
  3. Print a threshold sweep table
  4. Print the expected dollar value of your chosen operating point

Run:
    python evaluate.py

Check your work:
    pytest tests/test_evaluation.py -v
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from config import HIGH_COST_THRESHOLD_USD
from train import TARGET

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_DIR = Path("reports")

# Must match train.py so you are scoring on data the model never saw.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# TODO 5 — your operating point.
#
# Pick ONE use case and the threshold that fits it. This choice is yours, it is
# the spine of your final presentation, and there is no single right answer
#
#   "care_management"  a nurse team calls flagged members early. Outreach costs
#                      real money per member and there is a limited number of
#                      nurses. False positives waste calls; false negatives mean
#                      somebody's cancer gets found late.
#   "stop_loss"        the plan sets aside reserve dollars for members likely to
#                      breach the threshold. Under-reserving is a solvency
#                      problem; over-reserving is idle capital.
#   "pricing"          next year's premium is set using predicted risk. Errors
#                      here move money between all members.
#
# Set CHOSEN_USE_CASE to one of those strings, CHOSEN_THRESHOLD to a probability
# between 0 and 1, and write a justification that mentions the trade-off you are
# accepting. "0.5 is the default" is not a justification.
# ---------------------------------------------------------------------------
VALID_USE_CASES = ("care_management", "stop_loss", "pricing")

CHOSEN_USE_CASE = "care_management"
CHOSEN_THRESHOLD = 0.15
THRESHOLD_JUSTIFICATION = (
    "Nurses call members the model flags, so I want to miss as few high-cost "
    "members as possible. At 0.15 the 400-member test set flags 55 members, "
    "reaches 20 of the 23 real high-cost members (recall 0.87) and nets the "
    "most money on the sweep, $66,250 at $4,000 saved per catch and $250 per "
    "call. The mistake I am choosing to make more often is the wasted call: "
    "precision is only 0.36, so about 2 of every 3 calls go to a member who "
    "would not have been high-cost. I accept that because a call costs $250 "
    "and a missed member costs far more, and because going lower to 0.05 "
    "catches just 2 more members for 36 extra calls and a lower net. If the "
    "team can only make about 25 calls, 0.35 is the fallback: it reaches 11 "
    "of 23 members for $37,750."
)

# Rough economics for the care-management case, in dollars. Change them if your
# use case implies different numbers, and say so in your write-up.
OUTREACH_COST_PER_FLAGGED = 250
SAVINGS_PER_TRUE_POSITIVE = 4_000


def load_holdout() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    TODO 1 — rebuild the held-out test set and score it.

    Return (X_test, y_test, y_proba) where y_proba is the model's predicted
    probability of the positive class.

    Load the model with joblib.load(MODEL_PATH) and split with the same
    TEST_SIZE / RANDOM_STATE / stratify=y that train.py used. If your split does
    not match, you are quietly evaluating on rows the model trained on — which
    is the same category of mistake as last week's.

    Hint: the model is a Pipeline, so it takes the raw feature DataFrame.
    Take the feature column names from the pipeline itself rather than
    hard-coding them:
        pipeline["preprocessor"].feature_names_in_
    """
    pipeline = joblib.load(MODEL_PATH)
    summary = pd.read_csv(DATA_PATH)
    features = list(pipeline["preprocessor"].feature_names_in_)
    X = summary[features]
    y = summary[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_proba


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """
    TODO 2 — build the table that turns probabilities into decisions.

    Return a DataFrame with one row per threshold and these columns:
        threshold, n_flagged, tp, fp, fn, precision, recall

    Use thresholds = np.arange(0.05, 1.0, 0.05) when none are passed.
    A member is flagged when y_proba >= threshold.

    Watch out for the threshold where nothing gets flagged: precision is 0/0
    there. Decide what you want to report (0.0 is fine) rather than letting
    numpy hand you a nan and a warning.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    y_true = np.asarray(y_true).astype(bool)
    y_proba = np.asarray(y_proba)

    rows = []
    for threshold in thresholds:
        flagged = y_proba >= threshold
        tp = int((flagged & y_true).sum())
        fp = int((flagged & ~y_true).sum())
        fn = int((~flagged & y_true).sum())
        n_flagged = tp + fp
        rows.append({
            "threshold": round(float(threshold), 2),
            "n_flagged": n_flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / n_flagged if n_flagged else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        })
    return pd.DataFrame(rows)


def expected_value(tp: int, fp: int, outreach_cost: float, savings_per_tp: float) -> float:
    """
    TODO 3 — the dollar value of an operating point.

    Every flagged member (tp + fp) costs `outreach_cost`. Every true positive
    returns `savings_per_tp`. Return the net dollars.

    This is a deliberately crude model of the world. Its job is to make the
    trade-off explicit, not to be right.
    """
    return tp * savings_per_tp - (tp + fp) * outreach_cost


def plot_curves(y_true, y_proba, out_path: Path = REPORT_DIR / "evaluation_curves.png") -> Path:
    """
    TODO 4 — three panels, saved as one PNG. Return the path you wrote.

    Panel 1: ROC curve            sklearn.metrics.roc_curve + roc_auc_score
    Panel 2: precision-recall     sklearn.metrics.precision_recall_curve
             curve                and average_precision_score
    Panel 3: calibration curve    sklearn.calibration.calibration_curve
                                  (n_bins=10, strategy="quantile")

    Mark CHOSEN_THRESHOLD on the ROC and PR panels so a reader can see which
    point on the curve you are actually proposing to operate at.

    On panel 3, plot the diagonal y=x for reference. A point above the diagonal
    means the model under-predicts risk in that bin; below means it over-
    predicts. Note which way yours leans — that matters for the stop-loss and
    pricing use cases much more than for ranking members for outreach.

    Use matplotlib (pip install -r requirements-dev.txt) and remember to
    out_path.parent.mkdir(exist_ok=True) before saving.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    threshold = CHOSEN_THRESHOLD

    fig, (ax_roc, ax_pr, ax_cal) = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="coin flip")
    ax_roc.set(xlabel="False positive rate", ylabel="Recall (true positive rate)", title="ROC")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax_pr.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    ax_pr.axhline(y_true.mean(), color="k", linestyle="--", alpha=0.4, label="coin flip")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")

    if threshold is not None:
        flagged = y_proba >= threshold
        tp = (flagged & (y_true == 1)).sum()
        fp = (flagged & (y_true == 0)).sum()
        ax_roc.scatter(
            [fp / max((y_true == 0).sum(), 1)], [tp / max((y_true == 1).sum(), 1)],
            color="tab:orange", zorder=3, label=f"chosen cut {threshold}",
        )
        ax_pr.scatter(
            [tp / max((y_true == 1).sum(), 1)], [tp / max(flagged.sum(), 1)],
            color="tab:orange", zorder=3, label=f"chosen cut {threshold}",
        )

    frac_positive, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="quantile"
    )
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    ax_cal.plot(mean_predicted, frac_positive, "o-", label="model")
    ax_cal.set(
        xlabel="Mean predicted score", ylabel="Share really high-cost", title="Calibration"
    )

    for ax in (ax_roc, ax_pr, ax_cal):
        ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    X_test, y_test, y_proba = load_holdout()
    print(f"Scored {len(y_test)} held-out members "
          f"({int(y_test.sum())} of them high-cost at ${HIGH_COST_THRESHOLD_USD:,})")

    sweep = threshold_sweep(y_test, y_proba)
    sweep = sweep.assign(
        net_dollars=[
            expected_value(row.tp, row.fp, OUTREACH_COST_PER_FLAGGED, SAVINGS_PER_TRUE_POSITIVE)
            for row in sweep.itertuples()
        ]
    )
    print("\n=== Threshold sweep ===")
    print(sweep.to_string(index=False))

    path = plot_curves(y_test, y_proba)
    print(f"\nCurves written to {path}")

    print(f"\n=== My operating point ===")
    print(f"Use case:      {CHOSEN_USE_CASE}")
    print(f"Threshold:     {CHOSEN_THRESHOLD}")
    print(f"Justification: {THRESHOLD_JUSTIFICATION}")


if __name__ == "__main__":
    main()
