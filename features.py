"""
WEEK 3 STARTER — build the member-level feature table, then audit it.

`data/claims.csv` has one row per claim. A model needs one row per *member*.
This file is where you do that roll-up yourself, and then decide which of the
resulting columns you are actually allowed to feed a model.

Fill in every section marked TODO. When you run this file it should:
  1. Load data/members.csv and data/claims.csv
  2. Keep only claims that were actually approved
  3. Aggregate claims to one row per member
  4. Attach the is_high_cost label
  5. Write data/member_features.csv
  6. Print your feature audit

Run:
    python features.py

Check your work:
    pytest tests/test_features.py -v
"""

import pandas as pd
from pathlib import Path

from config import HIGH_COST_THRESHOLD_CENTS, HIGH_COST_THRESHOLD_USD

MEMBERS_PATH = Path("data/members.csv")
CLAIMS_PATH = Path("data/claims.csv")
OUT_PATH = Path("data/member_features.csv")

# Every non-identifier column your feature table will contain once you finish
# build_member_features() and add_label(). You will classify each one below.
AUDITABLE_COLUMNS = [
    "age_bucket",
    "sex",
    "region",
    "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    "total_billed_cents",
    "total_allowed_cents",
    "total_plan_paid_cents",
    "total_oop_cents",
    "unique_diagnosis_codes",
    "inpatient_claims",
    "has_specialty_drug",
    "is_high_cost",
]

VALID_VERDICTS = ("safe", "leaky", "debatable", "not_a_feature")

# ---------------------------------------------------------------------------
# TODO 4 — the feature audit.
#
# For EVERY column in AUDITABLE_COLUMNS, replace the ("TODO", "") placeholder
# with your verdict and a one-line justification:
#
#   "safe"           observable at prediction time, safe to use as an input
#   "leaky"          a restatement of the outcome; using it is cheating
#   "debatable"      observable in principle, but arguably inside the outcome
#                    window — say what your position is and why
#   "not_a_feature"  real column, but not a model input (id, date, the label
#                    itself, ...)
#
# The test suite will not accept an empty justification, and it will not accept
# a column in your model feature list that can reproduce the label on its own.
# The question to ask for each column is always the same one:
#
#     "On the day I need this prediction, do I already know this number?"
#
# A prediction is made to decide who to reach out to BEFORE the plan year's
# spending happens. Keep that timeline in mind.
# ---------------------------------------------------------------------------
FEATURE_AUDIT: dict[str, tuple[str, str]] = {
    "age_bucket": ("safe", "enrollment fact, known on Jan 1"),
    "sex": ("safe", "enrollment fact, known on Jan 1"),
    "region": ("safe", "enrollment fact, known on Jan 1"),
    # Counts of care. Debatable: they come from the same year as the label, but
    # a real plan would use LAST year's counts, which are known on Jan 1. They
    # do not contain the dollars the label is made from (single-column AUC
    # 0.71-0.87, far below the 0.99 leak line), so I use them as a stand-in for
    # prior-year history. In production they must be rebuilt from prior-year claims.
    "total_claims": ("debatable", "same year as label; stand-in for last year's claim count"),
    "medical_claims": ("debatable", "same year as label; stand-in for last year's medical use"),
    "pharmacy_claims": ("debatable", "same year as label; stand-in for last year's pharmacy use"),
    "specialty_drug_fills": ("debatable", "same year as label; stand-in for last year's specialty fills"),
    "total_billed_cents": ("leaky", "this year's dollars; ranks the label at AUC 0.97 and is nearly the same number as plan-paid"),
    "total_allowed_cents": ("leaky", "plan-paid plus member share; nearly restates the label (AUC 0.998)"),
    "total_plan_paid_cents": ("leaky", "this is the label's source column: is_high_cost is cut from it"),
    "total_oop_cents": ("leaky", "the member's slice of the same claims whose plan-paid defines the label; only known once claims adjudicate"),
    "unique_diagnosis_codes": ("debatable", "same year as label; stand-in for last year's diagnosis history"),
    "inpatient_claims": ("debatable", "same year as label; stand-in for last year's inpatient stays"),
    "has_specialty_drug": ("debatable", "same year as label; stand-in for being on a specialty drug last year"),
    "is_high_cost": ("not_a_feature", "the label itself, known only after Dec 31"),
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two source tables. Given to you — nothing to do here."""
    if not CLAIMS_PATH.exists() or not MEMBERS_PATH.exists():
        print("Source CSVs not found — running the generator first.")
        from generate_claims_data import generate

        generate()
    return pd.read_csv(MEMBERS_PATH), pd.read_csv(CLAIMS_PATH)


def approved_claims(claims: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 1 — return only the claims that were approved.

    Denied claims were submitted but never paid. Look at the
    `adjudication_status` column and decide what belongs in a feature table
    that is supposed to describe care the plan actually paid for.

    Roughly 8% of claims are denied, so this line changes your numbers.
    """
    return claims[claims["adjudication_status"] == "approved"].copy()


def build_member_features(members: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 2 — aggregate claims to one row per member and merge onto `members`.

    Return a DataFrame with one row per member in `members` and these columns
    (names matter — the tests check them):

        member_id
        age_bucket, sex, region, plan_year_start, plan_year_end   (from members)
        total_claims             count of claims
        medical_claims           count where is_pharmacy is False
        pharmacy_claims          count where is_pharmacy is True
        specialty_drug_fills     count where is_specialty_drug is True
        total_billed_cents       sum of billed_cents
        total_allowed_cents      sum of allowed_cents
        total_plan_paid_cents    sum of plan_paid_cents
        total_oop_cents          sum of out_of_pocket_cents
        unique_diagnosis_codes   number of distinct non-empty diagnosis_code
        inpatient_claims         count where claim_type == "Institutional"
        has_specialty_drug       True if any claim is a specialty drug

    Hints:
      - df.groupby("member_id").agg(new_name=("source_col", "how"), ...)
      - "how" can be a string ("count", "sum") or a lambda for the trickier ones
        (nunique on non-empty strings, any, boolean counts)
      - .reset_index() after groupby, then members.merge(agg, on="member_id",
        how="left")
      - Think about members with zero approved claims. A left merge leaves NaN
        for them. Should their total_claims be NaN, or 0? Whatever you choose,
        the column dtypes should still be numeric.
    """
    agg = claims.groupby("member_id").agg(
        total_claims=("claim_id", "count"),
        medical_claims=("is_pharmacy", lambda s: (~s.astype(bool)).sum()),
        pharmacy_claims=("is_pharmacy", lambda s: s.astype(bool).sum()),
        specialty_drug_fills=("is_specialty_drug", lambda s: s.astype(bool).sum()),
        total_billed_cents=("billed_cents", "sum"),
        total_allowed_cents=("allowed_cents", "sum"),
        total_plan_paid_cents=("plan_paid_cents", "sum"),
        total_oop_cents=("out_of_pocket_cents", "sum"),
        # nunique() skips NaN, so pharmacy claims (no diagnosis) are not counted
        unique_diagnosis_codes=("diagnosis_code", "nunique"),
        inpatient_claims=("claim_type", lambda s: (s == "Institutional").sum()),
        has_specialty_drug=("is_specialty_drug", lambda s: s.astype(bool).any()),
    ).reset_index()

    # Left join keeps members with no approved claims; for them "no claims"
    # is a fact (0), not missing data, so fill and restore integer dtypes.
    table = members.merge(agg, on="member_id", how="left")
    fill_cols = [c for c in agg.columns if c != "member_id"]
    table[fill_cols] = table[fill_cols].fillna(0)
    int_cols = [c for c in fill_cols if c != "has_specialty_drug"]
    table[int_cols] = table[int_cols].astype(int)
    table["has_specialty_drug"] = table["has_specialty_drug"].astype(bool)
    return table


def add_label(features: pd.DataFrame) -> pd.DataFrame:
    """
    TODO 3 — add the `is_high_cost` column (int 0/1).

    A member is high-cost when their plan-paid total for the year exceeds the
    threshold. Use HIGH_COST_THRESHOLD_CENTS (not the dollar constant — the
    columns are in cents).

    Write the one line that creates the label, then look hard at which column
    you just used. You will need that thought in TODO 4.
    """
    features["is_high_cost"] = (
        features["total_plan_paid_cents"] > HIGH_COST_THRESHOLD_CENTS
    ).astype(int)
    return features


def model_features() -> list[str]:
    """
    TODO 5 — return the list of column names you will actually train on.

    Derive it from FEATURE_AUDIT rather than typing a list by hand, so that
    changing a verdict above changes what the model sees. Decide for yourself
    whether "debatable" columns are in or out, and be ready to defend it.
    """
    # Position on "debatable": IN. The counts are a fair stand-in for last
    # year's history (see the audit comments); "leaky" and "not_a_feature" are out.
    keep = ("safe", "debatable")
    return [col for col in AUDITABLE_COLUMNS if FEATURE_AUDIT[col][0] in keep]


def print_audit() -> None:
    print(f"\n=== Feature audit (threshold ${HIGH_COST_THRESHOLD_USD:,}) ===")
    for col in AUDITABLE_COLUMNS:
        verdict, why = FEATURE_AUDIT.get(col, ("MISSING", ""))
        print(f"  {col:24s} {verdict:14s} {why}")


def main() -> None:
    members, claims = load_raw()
    features = add_label(build_member_features(members, approved_claims(claims)))

    OUT_PATH.parent.mkdir(exist_ok=True)
    features.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(features)} member rows to {OUT_PATH}")
    print(f"High-cost prevalence: {100 * features['is_high_cost'].mean():.1f}%")
    print_audit()
    print(f"\nTraining on {len(model_features())} features: {model_features()}")


if __name__ == "__main__":
    main()
