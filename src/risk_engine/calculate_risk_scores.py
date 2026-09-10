import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

features_file = Path(
    "data/processed/employee_features.csv"
)

output_directory = Path(
    "data/processed"
)

output_file = output_directory / "employee_risk_scores.csv"


# ---------------------------------------------------------
# CHECK INPUT FILE
# ---------------------------------------------------------

if not features_file.exists():
    raise FileNotFoundError(
        "employee_features.csv not found. "
        "Run build_features.py first."
    )


# ---------------------------------------------------------
# LOAD FEATURE DATASET
# ---------------------------------------------------------

df = pd.read_csv(features_file)


print("\n--------------------------------------------")
print("Enterprise Human Cyber Risk Engine")
print("--------------------------------------------")

print(
    f"\nEmployee profiles loaded: {len(df)}"
)


# ---------------------------------------------------------
# NORMALIZE PRIVILEGE RISK
# ---------------------------------------------------------

df["privilege_risk_normalized"] = (
    (df["privilege_risk_value"] - 1)
    / 3
)


# ---------------------------------------------------------
# COMPONENT 1 — FAILURE BEHAVIOR
# ---------------------------------------------------------

df["failure_risk_component"] = (
    df["failure_rate"] * 30
)


# ---------------------------------------------------------
# COMPONENT 2 — CREDENTIAL SUBMISSION
# ---------------------------------------------------------

df["credential_risk_component"] = (
    df["credential_submission_rate"] * 20
)


# ---------------------------------------------------------
# COMPONENT 3 — LINK CLICKING
# ---------------------------------------------------------

df["click_risk_component"] = (
    df["click_rate"] * 15
)


# ---------------------------------------------------------
# COMPONENT 4 — TRAINING RISK
# ---------------------------------------------------------

df["training_risk_component"] = (
    df["training_risk"] * 10
)


# ---------------------------------------------------------
# COMPONENT 5 — TARGETED ATTACK FAILURE
# ---------------------------------------------------------

df["targeted_risk_component"] = (
    df["targeted_attack_failure_rate"] * 10
)


# ---------------------------------------------------------
# COMPONENT 6 — PRIVILEGE CONTEXT
# ---------------------------------------------------------

df["privilege_risk_component"] = (
    df["privilege_risk_normalized"] * 10
)


# ---------------------------------------------------------
# COMPONENT 7 — POSITIVE REPORTING BEHAVIOR
# ---------------------------------------------------------

df["reporting_reduction"] = (
    df["reporting_rate"] * 5
)


# ---------------------------------------------------------
# RAW RISK SCORE
# ---------------------------------------------------------

df["raw_risk_score"] = (
    df["failure_risk_component"]
    + df["credential_risk_component"]
    + df["click_risk_component"]
    + df["training_risk_component"]
    + df["targeted_risk_component"]
    + df["privilege_risk_component"]
    - df["reporting_reduction"]
)


# ---------------------------------------------------------
# RESCALE TO 0–100
# ---------------------------------------------------------

theoretical_max = 95

df["risk_score"] = (
    df["raw_risk_score"]
    / theoretical_max
    * 100
)


df["risk_score"] = (
    df["risk_score"]
    .clip(
        lower=0,
        upper=100
    )
    .round(2)
)


# ---------------------------------------------------------
# RISK CLASSIFICATION
# ---------------------------------------------------------

def classify_risk(score):

    if score <= 20:
        return "Low"

    elif score <= 35:
        return "Moderate"

    elif score <= 50:
        return "High"

    else:
        return "Critical"


df["risk_level"] = (
    df["risk_score"]
    .apply(classify_risk)
)


# ---------------------------------------------------------
# RECOMMENDED SECURITY ACTION
# ---------------------------------------------------------

def recommend_action(row):

    if row["risk_level"] == "Critical":

        return (
            "Immediate security review, targeted awareness "
            "training, and follow-up phishing simulation"
        )

    elif row["risk_level"] == "High":

        return (
            "Assign targeted phishing awareness training "
            "and reassess within 30 days"
        )

    elif row["risk_level"] == "Moderate":

        return (
            "Provide refresher awareness training and "
            "continue behavioral monitoring"
        )

    else:

        return (
            "Maintain standard awareness training "
            "and periodic phishing simulation"
        )


df["recommended_action"] = (
    df.apply(
        recommend_action,
        axis=1
    )
)


# ---------------------------------------------------------
# PRIMARY RISK DRIVER
# ---------------------------------------------------------

risk_component_columns = {

    "Phishing Failures":
        "failure_risk_component",

    "Credential Submission":
        "credential_risk_component",

    "Link Clicking":
        "click_risk_component",

    "Training Weakness":
        "training_risk_component",

    "Targeted Attack Failures":
        "targeted_risk_component",

    "Privilege Exposure":
        "privilege_risk_component"
}


def determine_primary_risk_driver(row):

    component_values = {

        label: row[column]

        for label, column
        in risk_component_columns.items()
    }

    return max(
        component_values,
        key=component_values.get
    )


df["primary_risk_driver"] = (
    df.apply(
        determine_primary_risk_driver,
        axis=1
    )
)


# ---------------------------------------------------------
# VALIDATE RISK SCORES
# ---------------------------------------------------------

if not df["risk_score"].between(0, 100).all():

    raise ValueError(
        "Risk scores outside 0–100 detected."
    )


if df["risk_level"].isnull().any():

    raise ValueError(
        "Missing risk classifications detected."
    )


# ---------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Risk Scoring Results")
print("--------------------------------------------")


print(
    f"\nEmployees Scored: {len(df)}"
)


print(
    f"\nAverage Enterprise Risk Score: "
    f"{df['risk_score'].mean():.2f}"
)


print("\nRisk Level Distribution:")

print(
    df["risk_level"]
    .value_counts()
)


# ---------------------------------------------------------
# TOP HIGH-RISK EMPLOYEES
# ---------------------------------------------------------

top_risk = (

    df[
        [
            "employee_id",
            "department",
            "privilege_level",
            "risk_score",
            "risk_level",
            "primary_risk_driver"
        ]
    ]

    .sort_values(
        "risk_score",
        ascending=False
    )

    .head(15)
)


print("\nTop 15 Highest-Risk Employees:")

print(
    top_risk.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# DEPARTMENT RISK
# ---------------------------------------------------------

department_risk = (

    df
    .groupby("department")
    ["risk_score"]

    .agg(
        [
            "mean",
            "max",
            "count"
        ]
    )

    .sort_values(
        "mean",
        ascending=False
    )
)


print("\nDepartment Risk Summary:")

print(
    department_risk.round(2)
)


# ---------------------------------------------------------
# PRIMARY RISK DRIVERS
# ---------------------------------------------------------

print("\nPrimary Risk Drivers:")

print(
    df["primary_risk_driver"]
    .value_counts()
)


# ---------------------------------------------------------
# FINAL CONFIRMATION
# ---------------------------------------------------------

print(
    f"\nEmployee risk scores saved successfully to: "
    f"{output_file}"
)