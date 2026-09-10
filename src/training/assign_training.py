from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path("data/processed/employee_risk_scores.csv")
OUTPUT_DIRECTORY = Path("reports/training")
OUTPUT_FILE = OUTPUT_DIRECTORY / "training_assignments.csv"


# ------------------------------------------------------------
# LOAD EMPLOYEE RISK DATA
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Employee risk score file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    "\n------------------------------------------------------------"
)
print(
    "Adaptive Security Awareness Training Engine"
)
print(
    "------------------------------------------------------------"
)

print(
    f"\nEmployee risk records loaded: {len(df)}"
)


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "employee_id",
    "department",
    "role",
    "privilege_level",
    "risk_score",
    "risk_level",
    "primary_risk_driver"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# TRAINING CATALOG
# ------------------------------------------------------------

TRAINING_CATALOG = {

    "Phishing Failures": {
        "training_module":
            "Advanced Phishing Recognition",
        "training_topic":
            "Phishing indicators, impersonation, urgency, spoofing",
        "duration_minutes":
            30
    },

    "Credential Submission": {
        "training_module":
            "Credential Protection and MFA Awareness",
        "training_topic":
            "Credential theft, password protection, MFA fatigue attacks",
        "duration_minutes":
            35
    },

    "Link Clicking": {
        "training_module":
            "Malicious Link and URL Recognition",
        "training_topic":
            "Suspicious URLs, redirects, shortened links, domain spoofing",
        "duration_minutes":
            25
    },

    "Training Weakness": {
        "training_module":
            "Security Awareness Fundamentals Refresher",
        "training_topic":
            "Core phishing awareness and safe email behavior",
        "duration_minutes":
            30
    },

    "Targeted Attack Failures": {
        "training_module":
            "Spear Phishing and Social Engineering Defense",
        "training_topic":
            "Targeted phishing, executive impersonation, social engineering",
        "duration_minutes":
            40
    },

    "Privilege Exposure": {
        "training_module":
            "Privileged Account Security Awareness",
        "training_topic":
            "Administrative accounts, least privilege, credential protection",
        "duration_minutes":
            40
    }

}


# ------------------------------------------------------------
# SELECT TRAINING MODULE
# ------------------------------------------------------------

def select_training_module(primary_risk_driver):

    training = TRAINING_CATALOG.get(
        primary_risk_driver
    )

    if training is None:

        return {
            "training_module":
                "General Cybersecurity Awareness",
            "training_topic":
                "General phishing and security awareness",
            "duration_minutes":
                25
        }

    return training


# ------------------------------------------------------------
# TRAINING PRIORITY
# ------------------------------------------------------------

def determine_training_priority(risk_level):

    if risk_level == "Critical":
        return "Immediate"

    elif risk_level == "High":
        return "High"

    elif risk_level == "Moderate":
        return "Standard"

    else:
        return "Routine"


# ------------------------------------------------------------
# TRAINING DEADLINE
# ------------------------------------------------------------

def determine_completion_deadline(risk_level):

    if risk_level == "Critical":
        return 3

    elif risk_level == "High":
        return 7

    elif risk_level == "Moderate":
        return 30

    else:
        return 90


# ------------------------------------------------------------
# FOLLOW-UP PHISHING SIMULATION
# ------------------------------------------------------------

def determine_follow_up_days(risk_level):

    if risk_level == "Critical":
        return 14

    elif risk_level == "High":
        return 30

    elif risk_level == "Moderate":
        return 60

    else:
        return 90


# ------------------------------------------------------------
# ESCALATION REQUIREMENT
# ------------------------------------------------------------

def determine_escalation(risk_level):

    if risk_level == "Critical":
        return "Security Team Review"

    elif risk_level == "High":
        return "Manager Notification"

    elif risk_level == "Moderate":
        return "Monitor"

    else:
        return "None"


# ------------------------------------------------------------
# ASSIGN TRAINING
# ------------------------------------------------------------

training_records = []

for _, row in df.iterrows():

    training = select_training_module(
        row["primary_risk_driver"]
    )

    training_records.append(
        {
            "employee_id":
                row["employee_id"],

            "department":
                row["department"],

            "role":
                row["role"],

            "privilege_level":
                row["privilege_level"],

            "risk_score":
                row["risk_score"],

            "risk_level":
                row["risk_level"],

            "primary_risk_driver":
                row["primary_risk_driver"],

            "training_module":
                training["training_module"],

            "training_topic":
                training["training_topic"],

            "training_duration_minutes":
                training["duration_minutes"],

            "training_priority":
                determine_training_priority(
                    row["risk_level"]
                ),

            "completion_deadline_days":
                determine_completion_deadline(
                    row["risk_level"]
                ),

            "follow_up_simulation_days":
                determine_follow_up_days(
                    row["risk_level"]
                ),

            "escalation_requirement":
                determine_escalation(
                    row["risk_level"]
                )
        }
    )


training_df = pd.DataFrame(
    training_records
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if training_df.empty:
    raise ValueError(
        "No training assignments were generated."
    )

if training_df["training_module"].isnull().any():
    raise ValueError(
        "Missing training modules detected."
    )

if training_df[
    "completion_deadline_days"
].isnull().any():
    raise ValueError(
        "Missing training deadlines detected."
    )


# ------------------------------------------------------------
# SAVE TRAINING ASSIGNMENTS
# ------------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# TRAINING SUMMARY
# ------------------------------------------------------------

print(
    "\n------------------------------------------------------------"
)
print(
    "Training Assignment Summary"
)
print(
    "------------------------------------------------------------"
)

print(
    f"\nTotal employees assigned training: "
    f"{len(training_df)}"
)

print(
    "\nTraining Priority Distribution:"
)

print(
    training_df[
        "training_priority"
    ].value_counts()
)


print(
    "\nTraining Modules Assigned:"
)

print(
    training_df[
        "training_module"
    ].value_counts()
)


# ------------------------------------------------------------
# HIGH-PRIORITY TRAINING
# ------------------------------------------------------------

priority_training = (
    training_df[
        training_df["risk_level"].isin(
            ["Critical", "High"]
        )
    ]
    [
        [
            "employee_id",
            "department",
            "risk_score",
            "risk_level",
            "primary_risk_driver",
            "training_module",
            "training_priority",
            "completion_deadline_days"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(15)
)

print(
    "\nTop Priority Training Assignments:"
)

print(
    priority_training.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print(
    f"\nTraining assignments saved successfully to: "
    f"{OUTPUT_FILE}"
)

print(
    "\nFINAL STATUS: "
    "ADAPTIVE SECURITY TRAINING ASSIGNMENT COMPLETED"
)