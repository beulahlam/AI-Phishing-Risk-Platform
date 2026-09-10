import pandas as pd
import numpy as np
import random
from pathlib import Path


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

employees_file = Path("data/raw/employees.csv")
scenarios_file = Path("data/raw/phishing_scenarios.csv")
output_file = Path("data/raw/phishing_events.csv")


# ---------------------------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------------------------

if not employees_file.exists():
    raise FileNotFoundError(
        "employees.csv was not found. Run generate_employees.py first."
    )

if not scenarios_file.exists():
    raise FileNotFoundError(
        "phishing_scenarios.csv was not found. "
        "Run generate_phishing_scenarios.py first."
    )


# ---------------------------------------------------------
# LOAD EMPLOYEES AND SCENARIOS
# ---------------------------------------------------------

employees = pd.read_csv(employees_file)
scenarios = pd.read_csv(scenarios_file)

print("\nEmployees loaded:", len(employees))
print("Phishing scenarios loaded:", len(scenarios))


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------

def clamp(value, minimum=0.02, maximum=0.95):
    """
    Keeps a probability between the allowed minimum
    and maximum values.
    """
    return max(minimum, min(value, maximum))


# ---------------------------------------------------------
# GENERATE PHISHING EVENTS
# ---------------------------------------------------------

events = []

event_number = 1


for _, employee in employees.iterrows():

    # -----------------------------------------------------
    # EMPLOYEE-SPECIFIC SUSCEPTIBILITY
    # -----------------------------------------------------

    training_score = employee[
        "previous_training_score"
    ]

    training_completed = employee[
        "security_training_completed"
    ]

    # Convert training score into a risk value.
    # Example:
    # training score 90 -> training risk 0.10
    # training score 60 -> training risk 0.40
    training_risk = (
        100 - training_score
    ) / 100


    # -----------------------------------------------------
    # PERSISTENT EMPLOYEE SUSCEPTIBILITY
    # -----------------------------------------------------

    # Every employee receives a hidden susceptibility
    # level that remains consistent across campaigns.
    #
    # IMPORTANT:
    # This value is used only to generate synthetic
    # behavior. It is NOT stored as an ML feature.

    base_employee_susceptibility = (
        0.10
        + (training_risk * 0.55)
    )


    # Employees who did not complete awareness training
    # receive additional simulated susceptibility.

    if training_completed == 0:
        base_employee_susceptibility += 0.12


    # Add individual variation so employees with the
    # same training score do not behave identically.

    employee_random_factor = np.random.normal(
        loc=0,
        scale=0.08
    )


    employee_susceptibility = (
        base_employee_susceptibility
        + employee_random_factor
    )


    employee_susceptibility = clamp(
        employee_susceptibility,
        minimum=0.05,
        maximum=0.80
    )


    for _, scenario in scenarios.iterrows():

        # -------------------------------------------------
        # START WITH SCENARIO BASELINE
        # -------------------------------------------------

        scenario_baseline = scenario[
            "base_failure_probability"
        ]


        # -------------------------------------------------
        # COMBINE EMPLOYEE + SCENARIO RISK
        # -------------------------------------------------

        failure_probability = (
            employee_susceptibility * 0.75
            +
            scenario_baseline * 0.25
        )


        # -------------------------------------------------
        # DEPARTMENT RELEVANCE
        # -------------------------------------------------

        targeted_department = (
            scenario["target_department"]
            == employee["department"]
        )

        if targeted_department:
            failure_probability += 0.07


        # -------------------------------------------------
        # SCENARIO DIFFICULTY
        # -------------------------------------------------

        if scenario["difficulty"] == "Medium":
            failure_probability += 0.03

        elif scenario["difficulty"] == "Hard":
            failure_probability += 0.06


        # -------------------------------------------------
        # CAMPAIGN-SPECIFIC RANDOMNESS
        # -------------------------------------------------

        campaign_noise = np.random.normal(
            loc=0,
            scale=0.06
        )

        failure_probability += campaign_noise


        # -------------------------------------------------
        # KEEP PROBABILITY VALID
        # -------------------------------------------------

        failure_probability = clamp(
            failure_probability,
            minimum=0.03,
            maximum=0.90
        )
        

        # -------------------------------------------------
        # DETERMINE WHETHER EMPLOYEE FAILS
        # -------------------------------------------------

        simulation_failed = int(
            np.random.random()
            < failure_probability
        )


        # -------------------------------------------------
        # EMAIL OPEN BEHAVIOR
        # -------------------------------------------------

        email_open_probability = 0.82

        email_opened = int(
            np.random.random()
            < email_open_probability
        )


        # -------------------------------------------------
        # INITIALIZE BEHAVIOR VALUES
        # -------------------------------------------------

        link_clicked = 0
        attachment_opened = 0
        credentials_submitted = 0


        # -------------------------------------------------
        # GENERATE FAILURE BEHAVIOR
        # -------------------------------------------------

        if email_opened and simulation_failed:

            delivery_method = scenario[
                "delivery_method"
            ]

            attack_type = scenario[
                "attack_type"
            ]


            # Link-based phishing
            if (
                "Link" in delivery_method
                or "Cloud Service" in delivery_method
                or "QR Code" in delivery_method
            ):
                link_clicked = 1


            # Attachment-based phishing
            if "Attachment" in delivery_method:
                attachment_opened = 1


            # Credential submission only applies to
            # credential/identity-focused phishing.
            if (
                "Credential" in attack_type
                or "Identity" in attack_type
            ):

                credentials_submitted = int(
                    np.random.random() < 0.65
                )


        # -------------------------------------------------
        # REPORTING BEHAVIOR
        # -------------------------------------------------

        reporting_probability = (
            0.20
            + (training_score / 100) * 0.45
        )

        if training_completed == 0:
            reporting_probability -= 0.10

        if simulation_failed:
            reporting_probability -= 0.15

        reporting_probability = clamp(
            reporting_probability,
            minimum=0.02,
            maximum=0.90
        )

        email_reported = int(
            np.random.random()
            < reporting_probability
        )


        # -------------------------------------------------
        # RESPONSE TIME
        # -------------------------------------------------

        if email_reported:

            response_time_seconds = int(
                np.random.lognormal(
                    mean=5.4,
                    sigma=0.6
                )
            )

            response_time_seconds = min(
                response_time_seconds,
                3600
            )

        else:

            response_time_seconds = 0


        # -------------------------------------------------
        # TRAINING ASSIGNED AFTER FAILURE
        # -------------------------------------------------

        remediation_training_assigned = int(
            simulation_failed == 1
        )


        # -------------------------------------------------
        # CREATE EVENT RECORD
        # -------------------------------------------------

        event = {

            "event_id":
                f"EVT{event_number:06d}",

            "employee_id":
                employee["employee_id"],

            "department":
                employee["department"],

            "role":
                employee["role"],

            "privilege_level":
                employee["privilege_level"],

            "security_training_completed":
                employee[
                    "security_training_completed"
                ],

            "previous_training_score":
                employee[
                    "previous_training_score"
                ],

            "scenario_id":
                scenario["scenario_id"],

            "scenario_name":
                scenario["scenario_name"],

            "attack_type":
                scenario["attack_type"],

            "difficulty":
                scenario["difficulty"],

            "target_department":
                scenario["target_department"],

            "mitre_technique":
                scenario["mitre_technique"],

            "targeted_department":
                int(targeted_department),

            "failure_probability":
                round(
                    failure_probability,
                    4
                ),

            "email_opened":
                email_opened,

            "link_clicked":
                link_clicked,

            "attachment_opened":
                attachment_opened,

            "credentials_submitted":
                credentials_submitted,

            "email_reported":
                email_reported,

            "response_time_seconds":
                response_time_seconds,

            "simulation_failed":
                simulation_failed,

            "remediation_training_assigned":
                remediation_training_assigned
        }

        events.append(event)

        event_number += 1


# ---------------------------------------------------------
# CONVERT TO DATAFRAME
# ---------------------------------------------------------

events_df = pd.DataFrame(events)


# ---------------------------------------------------------
# BASIC DATA VALIDATION
# ---------------------------------------------------------

expected_events = (
    len(employees)
    * len(scenarios)
)

if len(events_df) != expected_events:
    raise ValueError(
        "Unexpected number of phishing events."
    )

if events_df["event_id"].duplicated().any():
    raise ValueError(
        "Duplicate event IDs detected."
    )


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

events_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Enterprise Phishing Behavioral Dataset")
print("--------------------------------------------")

print(
    f"\nTotal Employees: "
    f"{events_df['employee_id'].nunique()}"
)

print(
    f"Total Scenarios: "
    f"{events_df['scenario_id'].nunique()}"
)

print(
    f"Total Behavioral Events: "
    f"{len(events_df)}"
)

print("\nDataset Shape:")
print(events_df.shape)


# ---------------------------------------------------------
# BEHAVIORAL STATISTICS
# ---------------------------------------------------------

print("\nSimulation Results:")
print(
    events_df[
        "simulation_failed"
    ].value_counts()
)

print("\nLink Clicks:")
print(
    events_df[
        "link_clicked"
    ].value_counts()
)

print("\nCredential Submissions:")
print(
    events_df[
        "credentials_submitted"
    ].value_counts()
)

print("\nPhishing Reports:")
print(
    events_df[
        "email_reported"
    ].value_counts()
)


# ---------------------------------------------------------
# FAILURE RATE
# ---------------------------------------------------------

failure_rate = (
    events_df["simulation_failed"].mean()
    * 100
)

reporting_rate = (
    events_df["email_reported"].mean()
    * 100
)

print(
    f"\nOverall Simulation Failure Rate: "
    f"{failure_rate:.2f}%"
)

print(
    f"Overall Reporting Rate: "
    f"{reporting_rate:.2f}%"
)


# ---------------------------------------------------------
# FAILURE RATE BY DEPARTMENT
# ---------------------------------------------------------

department_failure = (
    events_df
    .groupby("department")[
        "simulation_failed"
    ]
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

print("\nFailure Rate by Department:")
print(
    department_failure.round(2)
)


# ---------------------------------------------------------
# FAILURE RATE BY DIFFICULTY
# ---------------------------------------------------------

difficulty_failure = (
    events_df
    .groupby("difficulty")[
        "simulation_failed"
    ]
    .mean()
    .mul(100)
)

print("\nFailure Rate by Scenario Difficulty:")
print(
    difficulty_failure.round(2)
)


# ---------------------------------------------------------
# FINAL CONFIRMATION
# ---------------------------------------------------------

print(
    f"\nBehavioral dataset saved successfully to: "
    f"{output_file}"
)