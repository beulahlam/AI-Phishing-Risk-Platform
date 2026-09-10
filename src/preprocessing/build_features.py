import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

events_file = Path("data/raw/phishing_events.csv")
employees_file = Path("data/raw/employees.csv")

output_directory = Path("data/processed")
output_file = output_directory / "employee_features.csv"


# ---------------------------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------------------------

if not events_file.exists():
    raise FileNotFoundError(
        "phishing_events.csv not found. "
        "Run the event generator first."
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
print("Employee Behavioral Feature Engineering")
print("--------------------------------------------")

print(f"\nBehavioral events loaded: {len(events)}")
print(f"Employees loaded: {len(employees)}")


# ---------------------------------------------------------
# AGGREGATE EMPLOYEE BEHAVIOR
# ---------------------------------------------------------

features = (
    events
    .groupby("employee_id")
    .agg(

        total_simulations=(
            "event_id",
            "count"
        ),

        emails_opened=(
            "email_opened",
            "sum"
        ),

        links_clicked=(
            "link_clicked",
            "sum"
        ),

        attachments_opened=(
            "attachment_opened",
            "sum"
        ),

        credentials_submitted=(
            "credentials_submitted",
            "sum"
        ),

        phishing_emails_reported=(
            "email_reported",
            "sum"
        ),

        simulations_failed=(
            "simulation_failed",
            "sum"
        ),

        remediation_trainings=(
            "remediation_training_assigned",
            "sum"
        ),

        targeted_scenarios=(
            "targeted_department",
            "sum"
        ),

        average_failure_probability=(
            "failure_probability",
            "mean"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# REPORTING RESPONSE TIME
# ---------------------------------------------------------

reported_events = events[
    events["email_reported"] == 1
]

response_features = (
    reported_events
    .groupby("employee_id")
    ["response_time_seconds"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "response_time_seconds":
            "average_reporting_time_seconds"
        }
    )
)

features = features.merge(
    response_features,
    on="employee_id",
    how="left"
)

# Employees who never reported a phishing message
# receive 0 instead of NaN.
features[
    "average_reporting_time_seconds"
] = features[
    "average_reporting_time_seconds"
].fillna(0)


# ---------------------------------------------------------
# TARGETED ATTACK PERFORMANCE
# ---------------------------------------------------------

targeted_events = events[
    events["targeted_department"] == 1
]

targeted_failure = (
    targeted_events
    .groupby("employee_id")
    ["simulation_failed"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "simulation_failed":
            "targeted_attack_failure_rate"
        }
    )
)

features = features.merge(
    targeted_failure,
    on="employee_id",
    how="left"
)

features[
    "targeted_attack_failure_rate"
] = features[
    "targeted_attack_failure_rate"
].fillna(0)


# ---------------------------------------------------------
# CALCULATE BEHAVIORAL RATES
# ---------------------------------------------------------

features["email_open_rate"] = (
    features["emails_opened"]
    /
    features["total_simulations"]
)

features["click_rate"] = (
    features["links_clicked"]
    /
    features["total_simulations"]
)

features["attachment_open_rate"] = (
    features["attachments_opened"]
    /
    features["total_simulations"]
)

features["credential_submission_rate"] = (
    features["credentials_submitted"]
    /
    features["total_simulations"]
)

features["reporting_rate"] = (
    features["phishing_emails_reported"]
    /
    features["total_simulations"]
)

features["failure_rate"] = (
    features["simulations_failed"]
    /
    features["total_simulations"]
)

features["remediation_rate"] = (
    features["remediation_trainings"]
    /
    features["total_simulations"]
)


# ---------------------------------------------------------
# SAFE-BEHAVIOR FEATURE
# ---------------------------------------------------------

features["non_failure_rate"] = (
    1 - features["failure_rate"]
)


# ---------------------------------------------------------
# REPORT-TO-FAILURE RELATIONSHIP
# ---------------------------------------------------------

features["report_to_failure_ratio"] = np.where(
    features["simulations_failed"] > 0,

    features["phishing_emails_reported"]
    /
    features["simulations_failed"],

    features["phishing_emails_reported"]
)


# Cap extreme ratios for easier downstream analysis.
features["report_to_failure_ratio"] = (
    features["report_to_failure_ratio"]
    .clip(upper=3)
)


# ---------------------------------------------------------
# ADD EMPLOYEE CONTEXT
# ---------------------------------------------------------

employee_context_columns = [
    "employee_id",
    "department",
    "role",
    "privilege_level",
    "security_training_completed",
    "previous_training_score"
]

employee_context = employees[
    employee_context_columns
].copy()

features = employee_context.merge(
    features,
    on="employee_id",
    how="left"
)


# ---------------------------------------------------------
# PRIVILEGE RISK ENCODING
# ---------------------------------------------------------

privilege_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Privileged": 4
}

features["privilege_risk_value"] = (
    features["privilege_level"]
    .map(privilege_mapping)
)


# ---------------------------------------------------------
# TRAINING RISK FEATURE
# ---------------------------------------------------------

features["training_risk"] = (
    1 -
    (
        features["previous_training_score"]
        / 100
    )
)


# ---------------------------------------------------------
# REPORTING QUALITY FEATURE
# ---------------------------------------------------------

features["reporting_quality"] = (
    features["reporting_rate"]
    *
    (
        features[
            "security_training_completed"
        ] + 1
    )
    / 2
)


# ---------------------------------------------------------
# ROUND DECIMAL FEATURES
# ---------------------------------------------------------

decimal_columns = [
    "average_failure_probability",
    "average_reporting_time_seconds",
    "targeted_attack_failure_rate",
    "email_open_rate",
    "click_rate",
    "attachment_open_rate",
    "credential_submission_rate",
    "reporting_rate",
    "failure_rate",
    "remediation_rate",
    "non_failure_rate",
    "report_to_failure_ratio",
    "training_risk",
    "reporting_quality"
]

features[decimal_columns] = (
    features[decimal_columns]
    .round(4)
)


# ---------------------------------------------------------
# DATA QUALITY CHECKS
# ---------------------------------------------------------

if len(features) != len(employees):
    raise ValueError(
        "Feature dataset does not contain "
        "one row per employee."
    )

if features["employee_id"].duplicated().any():
    raise ValueError(
        "Duplicate employee profiles detected."
    )

if features.isnull().any().any():
    raise ValueError(
        "Missing values detected in feature dataset."
    )


# ---------------------------------------------------------
# SAVE PROCESSED DATA
# ---------------------------------------------------------

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Feature Engineering Results")
print("--------------------------------------------")

print(
    f"\nEmployee profiles created: "
    f"{len(features)}"
)

print(
    f"Features per employee: "
    f"{len(features.columns)}"
)

print(
    f"\nDataset shape: "
    f"{features.shape}"
)


print("\nAverage Behavioral Rates:")

print(
    f"Email Open Rate: "
    f"{features['email_open_rate'].mean() * 100:.2f}%"
)

print(
    f"Click Rate: "
    f"{features['click_rate'].mean() * 100:.2f}%"
)

print(
    f"Credential Submission Rate: "
    f"{features['credential_submission_rate'].mean() * 100:.2f}%"
)

print(
    f"Reporting Rate: "
    f"{features['reporting_rate'].mean() * 100:.2f}%"
)

print(
    f"Failure Rate: "
    f"{features['failure_rate'].mean() * 100:.2f}%"
)


# ---------------------------------------------------------
# DEPARTMENT ANALYSIS
# ---------------------------------------------------------

department_analysis = (
    features
    .groupby("department")
    .agg(
        average_failure_rate=(
            "failure_rate",
            "mean"
        ),
        average_reporting_rate=(
            "reporting_rate",
            "mean"
        ),
        average_training_score=(
            "previous_training_score",
            "mean"
        )
    )
)

department_analysis[
    "average_failure_rate"
] *= 100

department_analysis[
    "average_reporting_rate"
] *= 100


print("\nDepartment Behavioral Summary:")

print(
    department_analysis.round(2)
)


# ---------------------------------------------------------
# HIGH FAILURE EMPLOYEES
# ---------------------------------------------------------

highest_failure = (
    features[
        [
            "employee_id",
            "department",
            "privilege_level",
            "failure_rate",
            "credential_submission_rate",
            "reporting_rate"
        ]
    ]
    .sort_values(
        "failure_rate",
        ascending=False
    )
    .head(10)
)


print("\nTop 10 Employees by Failure Rate:")

print(
    highest_failure.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# FINAL CONFIRMATION
# ---------------------------------------------------------

print(
    f"\nEmployee feature dataset saved successfully to: "
    f"{output_file}"
)