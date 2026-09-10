import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

employees_file = Path("data/raw/employees.csv")
scenarios_file = Path("data/raw/phishing_scenarios.csv")
events_file = Path("data/raw/phishing_events.csv")

report_directory = Path("reports/results")
report_file = report_directory / "data_validation_report.csv"


# ---------------------------------------------------------
# CHECK THAT REQUIRED FILES EXIST
# ---------------------------------------------------------

required_files = [
    employees_file,
    scenarios_file,
    events_file
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )


# ---------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------

employees = pd.read_csv(employees_file)
scenarios = pd.read_csv(scenarios_file)
events = pd.read_csv(events_file)


print("\n--------------------------------------------")
print("Enterprise Data Quality Validation")
print("--------------------------------------------")

print("\nDatasets successfully loaded.")

print(f"Employees: {len(employees)}")
print(f"Scenarios: {len(scenarios)}")
print(f"Events: {len(events)}")


# ---------------------------------------------------------
# VALIDATION RESULTS
# ---------------------------------------------------------

validation_results = []


def add_result(check_name, passed, details):
    """
    Stores the result of each validation test.
    """

    validation_results.append(
        {
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "details": details
        }
    )


# ---------------------------------------------------------
# CHECK 1 — EMPLOYEE COUNT
# ---------------------------------------------------------

employee_count_passed = len(employees) == 500

add_result(
    "Employee count equals 500",
    employee_count_passed,
    f"Actual employee count: {len(employees)}"
)


# ---------------------------------------------------------
# CHECK 2 — SCENARIO COUNT
# ---------------------------------------------------------

scenario_count_passed = len(scenarios) == 10

add_result(
    "Scenario count equals 10",
    scenario_count_passed,
    f"Actual scenario count: {len(scenarios)}"
)


# ---------------------------------------------------------
# CHECK 3 — EVENT COUNT
# ---------------------------------------------------------

expected_events = (
    len(employees)
    * len(scenarios)
)

event_count_passed = (
    len(events) == expected_events
)

add_result(
    "Behavioral event count is correct",
    event_count_passed,
    (
        f"Expected: {expected_events}, "
        f"Actual: {len(events)}"
    )
)


# ---------------------------------------------------------
# CHECK 4 — UNIQUE EMPLOYEE IDs
# ---------------------------------------------------------

employee_ids_unique = (
    not employees["employee_id"]
    .duplicated()
    .any()
)

add_result(
    "Employee IDs are unique",
    employee_ids_unique,
    (
        f"Duplicate employee IDs: "
        f"{employees['employee_id'].duplicated().sum()}"
    )
)


# ---------------------------------------------------------
# CHECK 5 — UNIQUE SCENARIO IDs
# ---------------------------------------------------------

scenario_ids_unique = (
    not scenarios["scenario_id"]
    .duplicated()
    .any()
)

add_result(
    "Scenario IDs are unique",
    scenario_ids_unique,
    (
        f"Duplicate scenario IDs: "
        f"{scenarios['scenario_id'].duplicated().sum()}"
    )
)


# ---------------------------------------------------------
# CHECK 6 — UNIQUE EVENT IDs
# ---------------------------------------------------------

event_ids_unique = (
    not events["event_id"]
    .duplicated()
    .any()
)

add_result(
    "Event IDs are unique",
    event_ids_unique,
    (
        f"Duplicate event IDs: "
        f"{events['event_id'].duplicated().sum()}"
    )
)


# ---------------------------------------------------------
# CHECK 7 — MISSING VALUES
# ---------------------------------------------------------

employee_missing = int(
    employees.isnull().sum().sum()
)

scenario_missing = int(
    scenarios.isnull().sum().sum()
)

event_missing = int(
    events.isnull().sum().sum()
)

total_missing = (
    employee_missing
    + scenario_missing
    + event_missing
)

add_result(
    "No missing values",
    total_missing == 0,
    (
        f"Employees: {employee_missing}, "
        f"Scenarios: {scenario_missing}, "
        f"Events: {event_missing}"
    )
)


# ---------------------------------------------------------
# CHECK 8 — EVERY EMPLOYEE HAS 10 EVENTS
# ---------------------------------------------------------

events_per_employee = (
    events.groupby("employee_id")
    .size()
)

employee_event_count_valid = (
    events_per_employee
    .eq(len(scenarios))
    .all()
)

invalid_employee_event_counts = int(
    (~events_per_employee.eq(len(scenarios))).sum()
)

add_result(
    "Every employee has all phishing scenarios",
    employee_event_count_valid,
    (
        f"Employees with incorrect event count: "
        f"{invalid_employee_event_counts}"
    )
)


# ---------------------------------------------------------
# CHECK 9 — ALL EMPLOYEES ARE REPRESENTED
# ---------------------------------------------------------

employee_ids = set(
    employees["employee_id"]
)

event_employee_ids = set(
    events["employee_id"]
)

employees_represented = (
    employee_ids == event_employee_ids
)

add_result(
    "All employees represented in event dataset",
    employees_represented,
    (
        f"Employee dataset IDs: {len(employee_ids)}, "
        f"Event dataset IDs: {len(event_employee_ids)}"
    )
)


# ---------------------------------------------------------
# CHECK 10 — ALL SCENARIOS ARE REPRESENTED
# ---------------------------------------------------------

scenario_ids = set(
    scenarios["scenario_id"]
)

event_scenario_ids = set(
    events["scenario_id"]
)

scenarios_represented = (
    scenario_ids == event_scenario_ids
)

add_result(
    "All scenarios represented in event dataset",
    scenarios_represented,
    (
        f"Scenario dataset IDs: {len(scenario_ids)}, "
        f"Event dataset IDs: {len(event_scenario_ids)}"
    )
)


# ---------------------------------------------------------
# CHECK 11 — BINARY COLUMNS
# ---------------------------------------------------------

binary_columns = [
    "security_training_completed",
    "targeted_department",
    "email_opened",
    "link_clicked",
    "attachment_opened",
    "credentials_submitted",
    "email_reported",
    "simulation_failed",
    "remediation_training_assigned"
]

invalid_binary_values = {}

for column in binary_columns:

    invalid_values = events[
        ~events[column].isin([0, 1])
    ]

    if len(invalid_values) > 0:
        invalid_binary_values[column] = len(
            invalid_values
        )

binary_values_valid = (
    len(invalid_binary_values) == 0
)

add_result(
    "Binary behavioral fields contain only 0 or 1",
    binary_values_valid,
    (
        "No invalid binary values"
        if binary_values_valid
        else str(invalid_binary_values)
    )
)


# ---------------------------------------------------------
# CHECK 12 — FAILURE PROBABILITY RANGE
# ---------------------------------------------------------

probability_valid = (
    events["failure_probability"]
    .between(0, 1)
    .all()
)

invalid_probability_count = int(
    (
        ~events["failure_probability"]
        .between(0, 1)
    ).sum()
)

add_result(
    "Failure probabilities are between 0 and 1",
    probability_valid,
    (
        f"Invalid probabilities: "
        f"{invalid_probability_count}"
    )
)


# ---------------------------------------------------------
# CHECK 13 — TRAINING SCORE RANGE
# ---------------------------------------------------------

training_score_valid = (
    events["previous_training_score"]
    .between(0, 100)
    .all()
)

invalid_training_scores = int(
    (
        ~events["previous_training_score"]
        .between(0, 100)
    ).sum()
)

add_result(
    "Training scores are between 0 and 100",
    training_score_valid,
    (
        f"Invalid training scores: "
        f"{invalid_training_scores}"
    )
)


# ---------------------------------------------------------
# CHECK 14 — CREDENTIAL LOGIC
# ---------------------------------------------------------

credential_logic_invalid = events[
    (events["credentials_submitted"] == 1)
    &
    (events["simulation_failed"] == 0)
]

credential_logic_valid = (
    len(credential_logic_invalid) == 0
)

add_result(
    "Credential submissions correspond to failures",
    credential_logic_valid,
    (
        f"Invalid credential events: "
        f"{len(credential_logic_invalid)}"
    )
)


# ---------------------------------------------------------
# CHECK 15 — LINK CLICK LOGIC
# ---------------------------------------------------------

link_logic_invalid = events[
    (events["link_clicked"] == 1)
    &
    (events["email_opened"] == 0)
]

link_logic_valid = (
    len(link_logic_invalid) == 0
)

add_result(
    "Link clicks require opened phishing messages",
    link_logic_valid,
    (
        f"Invalid link events: "
        f"{len(link_logic_invalid)}"
    )
)


# ---------------------------------------------------------
# CHECK 16 — ATTACHMENT LOGIC
# ---------------------------------------------------------

attachment_logic_invalid = events[
    (events["attachment_opened"] == 1)
    &
    (events["email_opened"] == 0)
]

attachment_logic_valid = (
    len(attachment_logic_invalid) == 0
)

add_result(
    "Attachment opens require opened messages",
    attachment_logic_valid,
    (
        f"Invalid attachment events: "
        f"{len(attachment_logic_invalid)}"
    )
)


# ---------------------------------------------------------
# CHECK 17 — REPORTING RESPONSE TIME
# ---------------------------------------------------------

reporting_time_invalid = events[
    (
        (events["email_reported"] == 1)
        &
        (events["response_time_seconds"] <= 0)
    )
    |
    (
        (events["email_reported"] == 0)
        &
        (events["response_time_seconds"] != 0)
    )
]

reporting_time_valid = (
    len(reporting_time_invalid) == 0
)

add_result(
    "Reporting response times are logically consistent",
    reporting_time_valid,
    (
        f"Invalid reporting-time events: "
        f"{len(reporting_time_invalid)}"
    )
)


# ---------------------------------------------------------
# CHECK 18 — REMEDIATION LOGIC
# ---------------------------------------------------------

remediation_invalid = events[
    events["remediation_training_assigned"]
    !=
    events["simulation_failed"]
]

remediation_valid = (
    len(remediation_invalid) == 0
)

add_result(
    "Failed simulations trigger remediation training",
    remediation_valid,
    (
        f"Invalid remediation events: "
        f"{len(remediation_invalid)}"
    )
)


# ---------------------------------------------------------
# CONVERT RESULTS TO DATAFRAME
# ---------------------------------------------------------

validation_df = pd.DataFrame(
    validation_results
)


# ---------------------------------------------------------
# DISPLAY VALIDATION REPORT
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Validation Results")
print("--------------------------------------------")

print(
    validation_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# CALCULATE SUMMARY
# ---------------------------------------------------------

passed_checks = int(
    (validation_df["status"] == "PASS").sum()
)

failed_checks = int(
    (validation_df["status"] == "FAIL").sum()
)

total_checks = len(validation_df)


print("\n--------------------------------------------")
print("Validation Summary")
print("--------------------------------------------")

print(f"\nTotal Checks: {total_checks}")
print(f"Passed: {passed_checks}")
print(f"Failed: {failed_checks}")


# ---------------------------------------------------------
# SAVE VALIDATION REPORT
# ---------------------------------------------------------

report_directory.mkdir(
    parents=True,
    exist_ok=True
)

validation_df.to_csv(
    report_file,
    index=False
)

print(
    f"\nValidation report saved successfully to: "
    f"{report_file}"
)


# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

if failed_checks == 0:

    print(
        "\nFINAL STATUS: DATA VALIDATION PASSED"
    )

else:

    print(
        "\nFINAL STATUS: DATA VALIDATION FAILED"
    )

    print(
        "Review failed checks before continuing."
    )