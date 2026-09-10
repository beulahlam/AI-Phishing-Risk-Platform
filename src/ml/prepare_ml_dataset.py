import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

events_file = Path(
    "data/raw/phishing_events.csv"
)

employees_file = Path(
    "data/raw/employees.csv"
)

output_directory = Path(
    "data/processed"
)

output_file = (
    output_directory
    / "ml_employee_dataset.csv"
)


# ---------------------------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------------------------

if not events_file.exists():
    raise FileNotFoundError(
        "phishing_events.csv not found."
    )

if not employees_file.exists():
    raise FileNotFoundError(
        "employees.csv not found."
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

events = pd.read_csv(events_file)
employees = pd.read_csv(employees_file)


print("\n--------------------------------------------")
print("Machine Learning Dataset Preparation")
print("--------------------------------------------")

print(
    f"\nBehavioral events loaded: "
    f"{len(events)}"
)

print(
    f"Employees loaded: "
    f"{len(employees)}"
)


# ---------------------------------------------------------
# EXTRACT NUMERIC SCENARIO ORDER
# ---------------------------------------------------------

events["scenario_number"] = (
    events["scenario_id"]
    .str.replace(
        "PHISH",
        "",
        regex=False
    )
    .astype(int)
)


# ---------------------------------------------------------
# DEFINE HISTORICAL AND FUTURE WINDOWS
# ---------------------------------------------------------

historical_events = events[
    events["scenario_number"] <= 7
].copy()

future_events = events[
    events["scenario_number"] >= 8
].copy()


print(
    f"\nHistorical events: "
    f"{len(historical_events)}"
)

print(
    f"Future events: "
    f"{len(future_events)}"
)


# ---------------------------------------------------------
# BUILD HISTORICAL BEHAVIOR FEATURES
# ---------------------------------------------------------

historical_features = (
    historical_events
    .groupby("employee_id")
    .agg(

        historical_simulations=(
            "event_id",
            "count"
        ),

        historical_failures=(
            "simulation_failed",
            "sum"
        ),

        historical_clicks=(
            "link_clicked",
            "sum"
        ),

        historical_attachments_opened=(
            "attachment_opened",
            "sum"
        ),

        historical_credentials_submitted=(
            "credentials_submitted",
            "sum"
        ),

        historical_reports=(
            "email_reported",
            "sum"
        ),

        historical_targeted_attacks=(
            "targeted_department",
            "sum"
        ),

        average_historical_failure_probability=(
            "failure_probability",
            "mean"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# HISTORICAL RATES
# ---------------------------------------------------------

historical_features[
    "historical_failure_rate"
] = (

    historical_features[
        "historical_failures"
    ]
    /
    historical_features[
        "historical_simulations"
    ]
)


historical_features[
    "historical_click_rate"
] = (

    historical_features[
        "historical_clicks"
    ]
    /
    historical_features[
        "historical_simulations"
    ]
)


historical_features[
    "historical_attachment_rate"
] = (

    historical_features[
        "historical_attachments_opened"
    ]
    /
    historical_features[
        "historical_simulations"
    ]
)


historical_features[
    "historical_credential_rate"
] = (

    historical_features[
        "historical_credentials_submitted"
    ]
    /
    historical_features[
        "historical_simulations"
    ]
)


historical_features[
    "historical_reporting_rate"
] = (

    historical_features[
        "historical_reports"
    ]
    /
    historical_features[
        "historical_simulations"
    ]
)


# ---------------------------------------------------------
# FUTURE TARGET
# ---------------------------------------------------------

future_target = (
    future_events
    .groupby("employee_id")
    .agg(
        future_failures=(
            "simulation_failed",
            "sum"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# BINARY ML TARGET
# ---------------------------------------------------------

future_target[
    "future_phishing_failure"
] = (

    future_target[
        "future_failures"
    ] > 0

).astype(int)


# ---------------------------------------------------------
# EMPLOYEE CONTEXT
# ---------------------------------------------------------

employee_context = employees[
    [
        "employee_id",
        "department",
        "role",
        "privilege_level",
        "security_training_completed",
        "previous_training_score"
    ]
].copy()


# ---------------------------------------------------------
# MERGE DATASETS
# ---------------------------------------------------------

ml_df = employee_context.merge(
    historical_features,
    on="employee_id",
    how="inner"
)

ml_df = ml_df.merge(
    future_target,
    on="employee_id",
    how="inner"
)


# ---------------------------------------------------------
# PRIVILEGE ENCODING
# ---------------------------------------------------------

privilege_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Privileged": 4
}

ml_df["privilege_risk_value"] = (
    ml_df["privilege_level"]
    .map(privilege_mapping)
)


# ---------------------------------------------------------
# TRAINING RISK
# ---------------------------------------------------------

ml_df["training_risk"] = (
    1 -
    (
        ml_df["previous_training_score"]
        / 100
    )
)


# ---------------------------------------------------------
# ROUND DECIMAL VALUES
# ---------------------------------------------------------

decimal_columns = [
    "average_historical_failure_probability",
    "historical_failure_rate",
    "historical_click_rate",
    "historical_attachment_rate",
    "historical_credential_rate",
    "historical_reporting_rate",
    "training_risk"
]

ml_df[decimal_columns] = (
    ml_df[decimal_columns]
    .round(4)
)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

if len(ml_df) != 500:
    raise ValueError(
        f"Expected 500 employees, "
        f"but found {len(ml_df)}."
    )

if ml_df["employee_id"].duplicated().any():
    raise ValueError(
        "Duplicate employee IDs detected."
    )

if ml_df.isnull().any().any():
    raise ValueError(
        "Missing values detected "
        "in ML dataset."
    )


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("ML Dataset Results")
print("--------------------------------------------")

print(
    f"\nEmployees in ML dataset: "
    f"{len(ml_df)}"
)

print(
    f"Dataset shape: "
    f"{ml_df.shape}"
)


print("\nFuture Target Distribution:")

print(
    ml_df[
        "future_phishing_failure"
    ]
    .value_counts()
)


future_failure_rate = (
    ml_df[
        "future_phishing_failure"
    ]
    .mean()
    * 100
)

print(
    f"\nEmployees failing at least one "
    f"future simulation: "
    f"{future_failure_rate:.2f}%"
)


print("\nHistorical Feature Averages:")

print(
    ml_df[
        [
            "historical_failure_rate",
            "historical_click_rate",
            "historical_credential_rate",
            "historical_reporting_rate",
            "training_risk"
        ]
    ]
    .mean()
    .round(4)
)


print(
    f"\nML dataset saved successfully to: "
    f"{output_file}"
)