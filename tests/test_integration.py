from pathlib import Path
import sqlite3

import pandas as pd


# ============================================================
# PROJECT OUTPUT FILES
# ============================================================

RISK_FILE = Path("data/processed/employee_risk_scores.csv")
ALERT_FILE = Path("reports/alerts/security_alerts.csv")
TRAINING_FILE = Path("reports/training/training_assignments.csv")
DATABASE_FILE = Path("data/database/human_cyber_risk.db")


def load_data():
    """Load outputs produced by the major platform components."""

    assert RISK_FILE.exists(), f"Missing risk file: {RISK_FILE}"
    assert ALERT_FILE.exists(), f"Missing alert file: {ALERT_FILE}"
    assert TRAINING_FILE.exists(), f"Missing training file: {TRAINING_FILE}"
    assert DATABASE_FILE.exists(), f"Missing database: {DATABASE_FILE}"

    risk = pd.read_csv(RISK_FILE)
    alerts = pd.read_csv(ALERT_FILE)
    training = pd.read_csv(TRAINING_FILE)

    return risk, alerts, training


# ============================================================
# TEST 1 — ALL PIPELINE OUTPUTS EXIST
# ============================================================

def test_pipeline_outputs_exist():
    """Every major pipeline component must produce its output."""

    assert RISK_FILE.exists()
    assert ALERT_FILE.exists()
    assert TRAINING_FILE.exists()
    assert DATABASE_FILE.exists()


# ============================================================
# TEST 2 — RISK ENGINE → ALERT ENGINE
# ============================================================

def test_high_risk_employees_generate_alerts():
    """
    Every employee classified as Critical should appear
    in the security alert output.
    """

    risk, alerts, _ = load_data()

    critical_employees = set(
        risk.loc[risk["risk_level"] == "Critical", "employee_id"]
    )

    alerted_employees = set(alerts["employee_id"])

    assert critical_employees.issubset(alerted_employees)


# ============================================================
# TEST 3 — RISK ENGINE → TRAINING ENGINE
# ============================================================

def test_critical_employees_receive_training():
    """
    Every Critical employee should receive an adaptive
    security-awareness training assignment.
    """

    risk, _, training = load_data()

    critical_employees = set(
        risk.loc[risk["risk_level"] == "Critical", "employee_id"]
    )

    trained_employees = set(training["employee_id"])

    assert critical_employees.issubset(trained_employees)


# ============================================================
# TEST 4 — CRITICAL RISK → P1 ALERT
# ============================================================

def test_critical_employees_receive_p1_alert():
    """Critical employees must receive P1 security alerts."""

    risk, alerts, _ = load_data()

    critical_employees = set(
        risk.loc[risk["risk_level"] == "Critical", "employee_id"]
    )

    p1_employees = set(
        alerts.loc[
            alerts["alert_severity"] == "P1 - Critical",
            "employee_id"
        ]
    )

    assert critical_employees.issubset(p1_employees)


# ============================================================
# TEST 5 — CRITICAL RISK → IMMEDIATE TRAINING
# ============================================================

def test_critical_employees_receive_immediate_training():
    """Critical employees must receive Immediate training priority."""

    risk, _, training = load_data()

    critical_employees = set(
        risk.loc[risk["risk_level"] == "Critical", "employee_id"]
    )

    immediate_training = set(
        training.loc[
            training["training_priority"] == "Immediate",
            "employee_id"
        ]
    )

    assert critical_employees.issubset(immediate_training)


# ============================================================
# TEST 6 — EMP0277 END-TO-END TRACE
# ============================================================

def test_emp0277_end_to_end():
    """
    Trace one known Critical employee through the entire
    risk → alert → training workflow.
    """

    risk, alerts, training = load_data()

    employee_risk = risk[risk["employee_id"] == "EMP0277"]
    employee_alert = alerts[alerts["employee_id"] == "EMP0277"]
    employee_training = training[
        training["employee_id"] == "EMP0277"
    ]

    assert len(employee_risk) == 1
    assert len(employee_alert) == 1
    assert len(employee_training) == 1

    assert employee_risk.iloc[0]["risk_level"] == "Critical"
    assert round(employee_risk.iloc[0]["risk_score"], 2) == 51.79

    assert (
        employee_alert.iloc[0]["alert_severity"]
        == "P1 - Critical"
    )

    assert (
        employee_training.iloc[0]["training_priority"]
        == "Immediate"
    )

    assert (
        employee_training.iloc[0]["training_module"]
        == "Advanced Phishing Recognition"
    )


# ============================================================
# TEST 7 — CROSS-SYSTEM RISK SCORE CONSISTENCY
# ============================================================

def test_risk_scores_consistent_across_components():
    """
    Risk scores must remain consistent when records move
    from the risk engine into alerts and training.
    """

    risk, alerts, training = load_data()

    risk_scores = risk.set_index("employee_id")["risk_score"]

    for _, row in alerts.iterrows():
        employee_id = row["employee_id"]

        assert employee_id in risk_scores.index
        assert round(row["risk_score"], 2) == round(
            risk_scores.loc[employee_id], 2
        )

    for _, row in training.iterrows():
        employee_id = row["employee_id"]

        assert employee_id in risk_scores.index
        assert round(row["risk_score"], 2) == round(
            risk_scores.loc[employee_id], 2
        )


# ============================================================
# TEST 8 — DATABASE EXISTS AND IS ACCESSIBLE
# ============================================================

def test_database_connection():
    """Verify that the enterprise SQLite database is accessible."""

    assert DATABASE_FILE.exists()

    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )

    tables = cursor.fetchall()

    connection.close()

    assert len(tables) > 0