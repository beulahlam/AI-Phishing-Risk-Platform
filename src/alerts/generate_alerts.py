from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/employee_risk_scores.csv")
OUTPUT_DIRECTORY = Path("reports/alerts")
OUTPUT_FILE = OUTPUT_DIRECTORY / "security_alerts.csv"


# ---------------------------------------------------------
# LOAD EMPLOYEE RISK DATA
# ---------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Risk score dataset was not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\n" + "-" * 60)
print("Enterprise Human Cyber Risk Alert Engine")
print("-" * 60)

print(f"\nEmployee risk records loaded: {len(df)}")


# ---------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "employee_id",
    "department",
    "role",
    "privilege_level",
    "risk_score",
    "risk_level",
    "recommended_action",
    "primary_risk_driver",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ---------------------------------------------------------
# SELECT ALERT-WORTHY EMPLOYEES
# ---------------------------------------------------------

alerts = df[
    df["risk_level"].isin(
        ["High", "Critical"]
    )
].copy()

print(f"Employees requiring alerts: {len(alerts)}")


# ---------------------------------------------------------
# ASSIGN ALERT SEVERITY
# ---------------------------------------------------------

def determine_alert_severity(risk_level):
    if risk_level == "Critical":
        return "P1 - Critical"

    if risk_level == "High":
        return "P2 - High"

    return "P3 - Monitor"


alerts["alert_severity"] = (
    alerts["risk_level"]
    .apply(determine_alert_severity)
)


# ---------------------------------------------------------
# CREATE ALERT TYPE
# ---------------------------------------------------------

def determine_alert_type(row):

    driver = row["primary_risk_driver"]

    if driver == "Credential Submission":
        return "Credential Exposure Risk"

    if driver == "Phishing Failures":
        return "Repeated Phishing Failure"

    if driver == "Privilege Exposure":
        return "Privileged User Risk"

    if driver == "Targeted Attack Failures":
        return "Targeted Phishing Risk"

    if driver == "Training Weakness":
        return "Security Awareness Risk"

    if driver == "Link Clicking":
        return "Unsafe Link Interaction"

    return "Human Cyber Risk Alert"


alerts["alert_type"] = alerts.apply(
    determine_alert_type,
    axis=1
)


# ---------------------------------------------------------
# CREATE HUMAN-READABLE ALERT MESSAGE
# ---------------------------------------------------------

def create_alert_message(row):

    return (
        f"Employee {row['employee_id']} "
        f"in {row['department']} has been classified as "
        f"{row['risk_level']} human cyber risk "
        f"with a risk score of {row['risk_score']:.2f}. "
        f"Primary risk driver: {row['primary_risk_driver']}. "
        f"Recommended response: {row['recommended_action']}."
    )


alerts["alert_message"] = alerts.apply(
    create_alert_message,
    axis=1
)


# ---------------------------------------------------------
# ADD ALERT METADATA
# ---------------------------------------------------------

timestamp = datetime.now(timezone.utc)

alerts["alert_id"] = [
    f"ALERT-{i:04d}"
    for i in range(1, len(alerts) + 1)
]

alerts["alert_created_utc"] = (
    timestamp.isoformat()
)

alerts["alert_status"] = "Open"

alerts["assigned_team"] = "Security Operations"


# ---------------------------------------------------------
# PRIORITIZE ALERTS
# ---------------------------------------------------------

severity_order = {
    "P1 - Critical": 1,
    "P2 - High": 2,
    "P3 - Monitor": 3,
}

alerts["severity_rank"] = (
    alerts["alert_severity"]
    .map(severity_order)
)

alerts = alerts.sort_values(
    by=[
        "severity_rank",
        "risk_score"
    ],
    ascending=[
        True,
        False
    ]
)


# ---------------------------------------------------------
# FINAL ALERT DATASET
# ---------------------------------------------------------

alert_columns = [
    "alert_id",
    "alert_created_utc",
    "alert_severity",
    "alert_type",
    "alert_status",
    "assigned_team",
    "employee_id",
    "department",
    "role",
    "privilege_level",
    "risk_score",
    "risk_level",
    "primary_risk_driver",
    "recommended_action",
    "alert_message",
]

alerts = alerts[alert_columns]


# ---------------------------------------------------------
# SAVE ALERTS
# ---------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

alerts.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("Alert Generation Summary")
print("-" * 60)

print(f"\nTotal Alerts Generated: {len(alerts)}")

print("\nAlerts by Severity:")
print(
    alerts["alert_severity"]
    .value_counts()
)

print("\nAlerts by Type:")
print(
    alerts["alert_type"]
    .value_counts()
)

print("\nTop Priority Alerts:")

print(
    alerts[
        [
            "alert_id",
            "employee_id",
            "department",
            "risk_score",
            "alert_severity",
            "alert_type",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print(
    f"\nSecurity alerts saved successfully to: "
    f"{OUTPUT_FILE}"
)

print(
    "\nFINAL STATUS: "
    "SECURITY ALERT GENERATION COMPLETED"
)