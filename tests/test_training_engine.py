from pathlib import Path

import pandas as pd


TRAINING_FILE = Path("reports/training/training_assignments.csv")


def load_training():
    """Load the generated adaptive security training assignments."""
    assert TRAINING_FILE.exists(), (
        f"Training assignment file does not exist: {TRAINING_FILE}"
    )
    return pd.read_csv(TRAINING_FILE)


def test_training_file_exists():
    """Verify that the training engine created its output file."""
    assert TRAINING_FILE.exists()


def test_training_assignments_generated():
    """Verify that at least one employee received a training assignment."""
    df = load_training()

    assert not df.empty
    assert len(df) > 0


def test_expected_training_columns():
    """Verify that required training assignment fields exist."""
    df = load_training()

    required_columns = {
        "employee_id",
        "department",
        "risk_score",
        "risk_level",
        "training_module",
        "training_priority",
    }

    assert required_columns.issubset(df.columns)


def test_no_missing_employee_ids():
    """Every training assignment must belong to an employee."""
    df = load_training()

    assert df["employee_id"].notna().all()
    assert (df["employee_id"].astype(str).str.strip() != "").all()


def test_training_risk_scores_are_valid():
    """Training records must contain valid 0-100 risk scores."""
    df = load_training()

    assert df["risk_score"].notna().all()
    assert df["risk_score"].between(0, 100).all()


def test_training_risk_levels_are_valid():
    """Training assignments must use approved risk classifications."""
    df = load_training()

    valid_levels = {
        "Low",
        "Moderate",
        "High",
        "Critical",
    }

    assert set(df["risk_level"].unique()).issubset(valid_levels)


def test_training_priorities_are_valid():
    """Verify that only approved training priorities are generated."""
    df = load_training()

    valid_priorities = {
        "Routine",
        "Standard",
        "High",
        "Immediate",
    }

    assert set(df["training_priority"].unique()).issubset(valid_priorities)


def test_critical_employees_receive_immediate_training():
    """Critical-risk employees must receive immediate training."""
    df = load_training()

    critical = df[df["risk_level"] == "Critical"]

    assert not critical.empty
    assert (critical["training_priority"] == "Immediate").all()


def test_emp0277_training_assignment():
    """
    Integration check:
    EMP0277 should receive the expected adaptive training response.
    """
    df = load_training()

    employee = df[df["employee_id"] == "EMP0277"]

    assert len(employee) == 1
    assert employee.iloc[0]["risk_level"] == "Critical"
    assert employee.iloc[0]["training_priority"] == "Immediate"
    assert (
        employee.iloc[0]["training_module"]
        == "Advanced Phishing Recognition"
    )
    assert round(employee.iloc[0]["risk_score"], 2) == 51.79