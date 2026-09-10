import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

COMPANY_NAME = "NovaCore Financial Services"


# ---------------------------------------------------------
# PHISHING SCENARIO LIBRARY
# ---------------------------------------------------------

scenarios = [

    {
        "scenario_id": "PHISH001",
        "scenario_name": "Microsoft 365 Password Expiration",
        "attack_type": "Credential Phishing",
        "delivery_method": "Email Link",
        "difficulty": "Easy",
        "target_department": "All",
        "social_engineering_trigger": "Urgency",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.35
    },

    {
        "scenario_id": "PHISH002",
        "scenario_name": "HR Benefits Update",
        "attack_type": "Malicious Attachment Simulation",
        "delivery_method": "Email Attachment",
        "difficulty": "Medium",
        "target_department": "All",
        "social_engineering_trigger": "Authority",
        "mitre_technique": "T1566.001",
        "mitre_name": "Spearphishing Attachment",
        "base_failure_probability": 0.25
    },

    {
        "scenario_id": "PHISH003",
        "scenario_name": "CEO Urgent Payment Request",
        "attack_type": "Business Email Compromise",
        "delivery_method": "Email",
        "difficulty": "Hard",
        "target_department": "Finance",
        "social_engineering_trigger": "Authority and Urgency",
        "mitre_technique": "T1566",
        "mitre_name": "Phishing",
        "base_failure_probability": 0.20
    },

    {
        "scenario_id": "PHISH004",
        "scenario_name": "Package Delivery Notification",
        "attack_type": "Link Phishing",
        "delivery_method": "Email Link",
        "difficulty": "Easy",
        "target_department": "All",
        "social_engineering_trigger": "Curiosity",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.40
    },

    {
        "scenario_id": "PHISH005",
        "scenario_name": "Shared Cloud Document",
        "attack_type": "Cloud Credential Phishing",
        "delivery_method": "Cloud Service Link",
        "difficulty": "Medium",
        "target_department": "All",
        "social_engineering_trigger": "Familiarity",
        "mitre_technique": "T1566.003",
        "mitre_name": "Spearphishing via Service",
        "base_failure_probability": 0.30
    },

    {
        "scenario_id": "PHISH006",
        "scenario_name": "Fake MFA Verification Request",
        "attack_type": "Identity Phishing",
        "delivery_method": "Email Link",
        "difficulty": "Hard",
        "target_department": "Information Technology",
        "social_engineering_trigger": "Urgency and Fear",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.18
    },

    {
        "scenario_id": "PHISH007",
        "scenario_name": "Payroll Account Verification",
        "attack_type": "Credential Phishing",
        "delivery_method": "Email Link",
        "difficulty": "Medium",
        "target_department": "Human Resources",
        "social_engineering_trigger": "Financial Concern",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.28
    },

    {
        "scenario_id": "PHISH008",
        "scenario_name": "QR Code Security Verification",
        "attack_type": "QR Phishing",
        "delivery_method": "QR Code",
        "difficulty": "Hard",
        "target_department": "All",
        "social_engineering_trigger": "Security Concern",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.22
    },

    {
        "scenario_id": "PHISH009",
        "scenario_name": "Resume Attachment",
        "attack_type": "Malicious Attachment Simulation",
        "delivery_method": "Email Attachment",
        "difficulty": "Medium",
        "target_department": "Human Resources",
        "social_engineering_trigger": "Job Responsibility",
        "mitre_technique": "T1566.001",
        "mitre_name": "Spearphishing Attachment",
        "base_failure_probability": 0.32
    },

    {
        "scenario_id": "PHISH010",
        "scenario_name": "IT Helpdesk Account Verification",
        "attack_type": "Credential Phishing",
        "delivery_method": "Email Link",
        "difficulty": "Hard",
        "target_department": "All",
        "social_engineering_trigger": "Authority and Fear",
        "mitre_technique": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "base_failure_probability": 0.20
    }

]


# ---------------------------------------------------------
# CONVERT SCENARIOS TO DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(scenarios)


# ---------------------------------------------------------
# ADD COMPANY INFORMATION
# ---------------------------------------------------------

df.insert(
    1,
    "company",
    COMPANY_NAME
)


# ---------------------------------------------------------
# VALIDATE THE SCENARIO LIBRARY
# ---------------------------------------------------------

if df["scenario_id"].duplicated().any():
    raise ValueError("Duplicate scenario IDs detected.")

if not df["base_failure_probability"].between(0, 1).all():
    raise ValueError(
        "Failure probabilities must be between 0 and 1."
    )


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

output_directory = Path("data/raw")

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_directory / "phishing_scenarios.csv"

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Enterprise Phishing Scenario Library")
print("--------------------------------------------")

print(f"\nCompany: {COMPANY_NAME}")

print(f"\nTotal Scenarios: {len(df)}")

print("\nDataset Shape:")
print(df.shape)

print("\nScenario Library:")
print(
    df[
        [
            "scenario_id",
            "scenario_name",
            "difficulty",
            "target_department"
        ]
    ].to_string(index=False)
)

print("\nDifficulty Distribution:")
print(df["difficulty"].value_counts())

print("\nAttack Type Distribution:")
print(df["attack_type"].value_counts())

print("\nMITRE ATT&CK Distribution:")
print(df["mitre_technique"].value_counts())

print(
    f"\nScenario library saved successfully to: "
    f"{output_file}"
)