"""
WEEK 6 STARTER — audit the model across demographic groups.

The model is one number per member. A fairness audit asks whether that number
is equally wrong for everybody, and "equally wrong" turns out to have several
mutually incompatible definitions.

Fill in every section marked TODO. When you run this file it should:
  1. Score every member with the trained model
  2. Report per-group rates for sex and age_bucket
  3. Apply the 80% rule
  4. Compare against a model retrained without the demographic columns
  5. Write reports/fairness_report.md

Run:
    python fairness.py

Check your work:
    pytest tests/test_fairness.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import HIGH_COST_THRESHOLD_USD
from train import CATEGORICAL_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_PATH = Path("reports/fairness_report.md")

PROTECTED_COLUMNS = ["sex", "age_bucket"]
DECISION_THRESHOLD = 0.15  # the care_management cut defended in Week 5

# ---------------------------------------------------------------------------
# TODO 6 — your written assessment.
#
# After you have the numbers, fill this in. It is graded on whether the claim
# follows from your own cohort's numbers, not on whether it is reassuring.
# Reference specific rates. Name the group that is worse off. If a subgroup is
# too small to conclude anything, say that instead of reporting a ratio to two
# decimal places as if it meant something.
# ---------------------------------------------------------------------------
ASSESSMENT = (
    "Audited at the Week 5 care_management cut of 0.15 on all 2,000 members, "
    "1,600 of which the model trained on, so catch rates here are higher than "
    "the 0.87 recall on the held-out set. "
    "By sex the model passes the 80% rule: women are flagged 9.7% of the time "
    "and men 8.2% (ratio 0.84). Women are also more often high-cost (6.4% vs "
    "5.2%), so the small gap follows need, and catch rates are about equal "
    "(women 96.8% of 62 high-cost, men 98.1% of 53). "
    "By age the flag-rate ratio FAILS at 0.19: members 60-64 are flagged 20.0% "
    "of the time and 18-29 only 3.8%. That gap tracks need, not bias: 14.4% of "
    "60-64 are actually high-cost against 2.1% of 18-29, so every group is "
    "flagged at a similar 1.4 to 1.8 times its base rate. Under equalized odds "
    "the catch rates are close, 90% to 100% in every age group. The worst off "
    "are members 30-39, where the model missed 2 of 20 high-cost members "
    "(catch rate 90.0%); that is two people, not a pattern. Members 18-29 have "
    "only 6 high-cost members, so I do not judge that group's catch rate, and "
    "the 0.19 ratio rests on a 3.8% flag rate built from small counts. "
    "Because older members really do use more care, I treat demographic "
    "parity as the wrong test here and equalized odds as the fitting one for "
    "care management: a high-cost member should be as likely to get a call "
    "whatever their age or sex. "
    "The blind model barely changes anything. Dropping sex and age_bucket "
    "moves the sex flag ratio from 0.84 to 0.85 and the age ratio from 0.19 to "
    "0.25, which still fails, and the age catch ratio from 0.90 to 0.89. Other "
    "columns still carry age: the share of members on a specialty drug rises "
    "from 11% at 18-29 to 31% at 60-64. Removing a column does not remove what "
    "it told the model."
)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """
    TODO 1 — return {"tp": .., "fp": .., "fn": .., "tn": ..}.

    Write it with vectorized comparisons rather than a loop. Both inputs are
    0/1 arrays or Series.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


def group_metrics(df: pd.DataFrame, group_col: str,
                  y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """
    TODO 2 — one row per group value, with these columns:

        group, n, base_rate, positive_rate, tpr, fpr, fnr, precision

        n              members in the group
        base_rate      share of the group that is actually high-cost
        positive_rate  share of the group the model flags   (demographic parity)
        tpr            of the actually-high-cost, share flagged
        fpr            of the actually-normal, share flagged (equalized odds)
        fnr            1 - tpr; the members the model misses
        precision      of the flagged, share actually high-cost

    Build it on top of confusion_counts(). Sort by group so the output is stable
    across runs. Guard the divisions: a group with zero actual positives has no
    TPR, and reporting 0.0 there is a lie — use np.nan.
    """
    rows = []
    for group, members in df.groupby(group_col, sort=True):
        c = confusion_counts(members[y_true_col], members[y_pred_col])
        n = len(members)
        positives = c["tp"] + c["fn"]
        negatives = c["fp"] + c["tn"]
        flagged = c["tp"] + c["fp"]
        tpr = c["tp"] / positives if positives else np.nan
        rows.append({
            "group": group,
            "n": n,
            "base_rate": positives / n,
            "positive_rate": flagged / n,
            "tpr": tpr,
            "fpr": c["fp"] / negatives if negatives else np.nan,
            "fnr": 1 - tpr,
            "precision": c["tp"] / flagged if flagged else np.nan,
        })
    return pd.DataFrame(rows)


def disparate_impact_ratio(rates: pd.Series) -> float:
    """
    TODO 3 — the four-fifths ratio: the lowest rate divided by the highest.

    Ignore nan rates. Returns a number in [0, 1], where 1.0 is exact parity.
    """
    rates = rates.dropna()
    if rates.empty or rates.max() == 0:
        return 1.0  # nobody is flagged in any group, so there is no gap to report
    return float(rates.min() / rates.max())


def passes_80_percent_rule(ratio: float) -> bool:
    """
    TODO 4 — True when the ratio clears the EEOC four-fifths guideline.

    Careful with the boundary: exactly 0.80 passes.
    """
    return bool(ratio >= 0.80)


def blind_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 5 — retrain without the demographic columns and re-audit.

    `df` arrives with y_true / y_pred already on it from the full model. Score
    the blind model yourself rather than reusing those columns, so the two
    numbers you compare come from two actual models.

    Steps:
      - take the fitted pipeline's feature list, drop PROTECTED_COLUMNS
      - build and fit a pipeline of the same shape on the same training split
      - score, threshold, and run group_metrics() again per protected column
      - return a DataFrame with one row per protected column and the disparate
        impact ratio with the demographic features and without, side by side

    Predict the result before you run it, in one sentence: The sex ratio stays
    about the same and the age ratio still fails, because claim counts and
    specialty-drug fills rise with age and still tell the model who is older.

    Most students expect the disparity to disappear. Whatever you find, the
    interesting question is *why*. Which remaining columns carry the same
    information the demographic columns did? Check with:
        df.groupby("sex")[numeric_cols].mean()
    """
    full_model = joblib.load(MODEL_PATH)
    features = list(full_model["preprocessor"].feature_names_in_)
    blind_features = [c for c in features if c not in PROTECTED_COLUMNS]

    categorical = [c for c in CATEGORICAL_FEATURES if c in blind_features]
    numeric = [c for c in blind_features if c not in categorical]
    blind_model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ])),
        ("classifier", clone(full_model["classifier"])),
    ])

    # Same split as train.py, so the two models learn from the same members.
    y = df[TARGET].astype(int)
    X_train, _, y_train, _ = train_test_split(
        df[blind_features], y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    blind_model.fit(X_train, y_train)

    blind_proba = blind_model.predict_proba(df[blind_features])[:, 1]
    blind_df = df.assign(y_pred=(blind_proba >= DECISION_THRESHOLD).astype(int))

    rows = []
    for col in PROTECTED_COLUMNS:
        full = group_metrics(df, col).set_index("group")
        blind = group_metrics(blind_df, col).set_index("group")
        rows.append({
            "protected_column": col,
            "flag_ratio_with": disparate_impact_ratio(full["positive_rate"]),
            "flag_ratio_blind": disparate_impact_ratio(blind["positive_rate"]),
            "catch_ratio_with": disparate_impact_ratio(full["tpr"]),
            "catch_ratio_blind": disparate_impact_ratio(blind["tpr"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    scored = df.assign(
        y_true=df["is_high_cost"].astype(int),
        y_proba=pipeline.predict_proba(df[features])[:, 1],
    )
    scored["y_pred"] = (scored["y_proba"] >= DECISION_THRESHOLD).astype(int)

    print(f"Auditing at threshold {DECISION_THRESHOLD} "
          f"(label: >${HIGH_COST_THRESHOLD_USD:,} plan-paid)")

    lines = [f"# Fairness audit\n", f"Decision threshold: {DECISION_THRESHOLD}\n"]
    for col in PROTECTED_COLUMNS:
        table = group_metrics(scored, col)
        ratio = disparate_impact_ratio(table.set_index("group")["positive_rate"])
        verdict = "PASS" if passes_80_percent_rule(ratio) else "FAIL"

        print(f"\n=== {col} ===")
        print(table.to_string(index=False))
        print(f"disparate impact ratio (positive rate): {ratio:.3f} → {verdict}")

        lines += [f"\n## {col}\n", table.to_markdown(index=False),
                  f"\n\nDisparate impact ratio: {ratio:.3f} ({verdict})\n"]

    print("\n=== Blind model comparison ===")
    comparison = blind_model_comparison(scored)
    print(comparison.to_string(index=False))
    lines += ["\n## Blind model comparison\n", comparison.to_markdown(index=False),
              f"\n\n## Assessment\n\n{ASSESSMENT}\n"]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
