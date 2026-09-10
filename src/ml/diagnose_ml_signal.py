from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/ml_employee_dataset.csv"
)

RESULTS_DIR = Path(
    "reports/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "ml_signal_diagnostics.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\n" + "-" * 55)
print("ML Signal Diagnostic Analysis")
print("-" * 55)

df = pd.read_csv(DATA_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ---------------------------------------------------------
# DEFINE TARGET
# ---------------------------------------------------------

TARGET = "future_phishing_failure"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

y = df[TARGET].astype(int)


# ---------------------------------------------------------
# EXCLUDE IDENTIFIERS / LEAKAGE
# ---------------------------------------------------------

exclude_columns = [
    TARGET,
    "future_failures",
    "employee_id",
]

candidate_features = [
    col
    for col in df.columns
    if col not in exclude_columns
]


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nTarget Distribution:")

target_counts = y.value_counts().sort_index()

for label, count in target_counts.items():

    percentage = (
        count
        / len(y)
        * 100
    )

    print(
        f"Class {label}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# NUMERIC FEATURE DIAGNOSTICS
# ---------------------------------------------------------

numeric_features = (
    df[candidate_features]
    .select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)

diagnostic_rows = []


for feature in numeric_features:

    feature_series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    class_0_values = feature_series[
        y == 0
    ]

    class_1_values = feature_series[
        y == 1
    ]

    mean_class_0 = class_0_values.mean()
    mean_class_1 = class_1_values.mean()

    mean_difference = (
        mean_class_1
        - mean_class_0
    )

    correlation = (
        feature_series
        .corr(y)
    )

    try:

        valid_mask = (
            feature_series
            .notna()
        )

        feature_auc = roc_auc_score(
            y[valid_mask],
            feature_series[valid_mask]
        )

        # AUC below 0.5 may simply mean
        # the feature is inversely associated.
        separability_auc = max(
            feature_auc,
            1 - feature_auc
        )

    except ValueError:

        feature_auc = float("nan")
        separability_auc = float("nan")

    diagnostic_rows.append(
        {
            "feature": feature,
            "mean_class_0": mean_class_0,
            "mean_class_1": mean_class_1,
            "mean_difference": mean_difference,
            "correlation_with_target": correlation,
            "raw_auc": feature_auc,
            "separability_auc": separability_auc,
        }
    )


diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


diagnostics_df = diagnostics_df.sort_values(
    by="separability_auc",
    ascending=False
)


# ---------------------------------------------------------
# DISPLAY STRONGEST FEATURES
# ---------------------------------------------------------

print(
    "\nTop Numeric Features by "
    "Single-Feature Separability:"
)

print(
    diagnostics_df
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# CHECK KEY SECURITY FEATURES
# ---------------------------------------------------------

key_features = [
    "previous_training_score",
    "training_risk",
    "historical_failures",
    "historical_failure_rate",
    "historical_clicks",
    "historical_click_rate",
    "historical_credentials_submitted",
    "historical_credential_rate",
    "historical_reports",
    "historical_reporting_rate",
    "historical_targeted_attacks",
    "average_historical_failure_probability",
    "privilege_risk_value",
]


print("\nKey Security Feature Comparison:")

for feature in key_features:

    if feature not in df.columns:
        continue

    feature_series = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    mean_0 = feature_series[
        y == 0
    ].mean()

    mean_1 = feature_series[
        y == 1
    ].mean()

    print(
        f"{feature:<42} "
        f"non-failure={mean_0:.4f}   "
        f"future-failure={mean_1:.4f}"
    )


# ---------------------------------------------------------
# CATEGORICAL ANALYSIS
# ---------------------------------------------------------

categorical_features = (
    df[candidate_features]
    .select_dtypes(
        exclude=["number", "bool"]
    )
    .columns
    .tolist()
)


for feature in categorical_features:

    print(
        f"\nFuture Failure Rate by "
        f"{feature}:"
    )

    grouped = (
        df.groupby(
            feature
        )[TARGET]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )

    grouped["mean"] = (
        grouped["mean"]
        * 100
    )

    grouped = grouped.rename(
        columns={
            "mean":
                "future_failure_rate_percent"
        }
    )

    print(
        grouped
        .round(2)
        .to_string()
    )


# ---------------------------------------------------------
# INTERPRET OVERALL SIGNAL
# ---------------------------------------------------------

best_auc = (
    diagnostics_df[
        "separability_auc"
    ]
    .max()
)

print("\n" + "-" * 55)
print("Diagnostic Interpretation")
print("-" * 55)

print(
    f"\nBest individual numeric "
    f"feature separability AUC: "
    f"{best_auc:.4f}"
)

if best_auc < 0.60:

    print(
        "Signal assessment: WEAK"
    )

    print(
        "Historical features provide "
        "limited separation between "
        "future failure classes."
    )

elif best_auc < 0.70:

    print(
        "Signal assessment: MODERATE"
    )

    print(
        "Some historical variables contain "
        "useful predictive information, but "
        "the signal remains noisy."
    )

else:

    print(
        "Signal assessment: STRONG"
    )

    print(
        "Historical behavior contains "
        "meaningful predictive separation."
    )


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

diagnostics_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nDiagnostic results saved to: "
    f"{OUTPUT_FILE}"
)

print(
    "\nFINAL STATUS: "
    "ML SIGNAL DIAGNOSTICS COMPLETED"
)