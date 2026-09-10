import pandas as pd
import numpy as np
import random
from pathlib import Path

# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------

random.seed(42)
np.random.seed(42)

COMPANY_NAME = "NovaCore Financial Services"
NUM_EMPLOYEES = 500

# ---------------------------------------------------------
# ENTERPRISE DEPARTMENTS
# ---------------------------------------------------------

departments = [
    "Finance",
    "Human Resources",
    "Information Technology",
    "Sales",
    "Operations",
    "Legal",
    "Marketing",
    "Executive"
]

# ---------------------------------------------------------
# JOB ROLES BY DEPARTMENT
# ---------------------------------------------------------

roles = {
    "Finance": [
        "Financial Analyst",
        "Accountant",
        "Finance Manager"
    ],

    "Human Resources": [
        "HR Specialist",
        "Recruiter",
        "HR Manager"
    ],

    "Information Technology": [
        "IT Support Analyst",
        "System Administrator",
        "Security Analyst"
    ],

    "Sales": [
        "Sales Representative",
        "Account Executive",
        "Sales Manager"
    ],

    "Operations": [
        "Operations Analyst",
        "Operations Specialist",
        "Operations Manager"
    ],

    "Legal": [
        "Legal Assistant",
        "Compliance Analyst",
        "Legal Manager"
    ],

    "Marketing": [
        "Marketing Specialist",
        "Digital Marketing Analyst",
        "Marketing Manager"
    ],

    "Executive": [
        "Director",
        "Vice President",
        "Chief Officer"
    ]
}

# ---------------------------------------------------------
# PRIVILEGE LEVELS BY DEPARTMENT
# ---------------------------------------------------------

privilege_by_department = {
    "Finance": ["Medium", "High"],
    "Human Resources": ["Medium", "High"],
    "Information Technology": ["Medium", "High", "Privileged"],
    "Sales": ["Low", "Medium"],
    "Operations": ["Low", "Medium"],
    "Legal": ["Medium", "High"],
    "Marketing": ["Low", "Medium"],
    "Executive": ["High", "Privileged"]
}

# ---------------------------------------------------------
# CREATE SYNTHETIC EMPLOYEES
# ---------------------------------------------------------

employees = []

for employee_number in range(1, NUM_EMPLOYEES + 1):

    department = random.choice(departments)

    role = random.choice(
        roles[department]
    )

    privilege_level = random.choice(
        privilege_by_department[department]
    )

    years_with_company = round(
        np.random.uniform(0.2, 15),
        1
    )

    security_training_completed = np.random.choice(
        [0, 1],
        p=[0.15, 0.85]
    )

    previous_training_score = int(
        np.random.normal(78, 12)
    )

    previous_training_score = max(
        40,
        min(previous_training_score, 100)
    )

    employee = {
        "employee_id": f"EMP{employee_number:04d}",
        "company": COMPANY_NAME,
        "department": department,
        "role": role,
        "privilege_level": privilege_level,
        "years_with_company": years_with_company,
        "security_training_completed": security_training_completed,
        "previous_training_score": previous_training_score
    }

    employees.append(employee)

# ---------------------------------------------------------
# CONVERT TO DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(employees)

# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

output_directory = Path("data/raw")

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_directory / "employees.csv"

df.to_csv(
    output_file,
    index=False
)

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Synthetic Enterprise Employee Dataset")
print("--------------------------------------------")

print(f"\nCompany: {COMPANY_NAME}")

print(f"\nTotal Employees: {len(df)}")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 10 Employees:")
print(df.head(10))

print("\nDepartment Distribution:")
print(df["department"].value_counts())

print("\nPrivilege Level Distribution:")
print(df["privilege_level"].value_counts())

print("\nSecurity Training Completion:")
print(df["security_training_completed"].value_counts())

print(f"\nDataset saved successfully to: {output_file}")