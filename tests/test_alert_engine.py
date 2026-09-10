import pandas as pd
from pathlib import Path


ALERT_FILE = Path("reports/alerts/security_alerts.csv")


def load_alerts():
    """Load generated security alerts."""
    assert ALERT_FILE.exists(), (
        "security_alerts.csv does not exist. "
        "Run src/alerts/generate_alerts.py first."
    )
    return pd.read_csv(ALERT_FILE)


def test_alert_file_exists():
    assert ALERT_FILE.exists()


def test_alerts_generated():
    df = load_alerts()

    assert len(df) > 0


def test_expected_alert_columns():
    df = load_alerts()

    required_columns = {
        "alert_id",
        "employee_id",
        "department",
        "risk_score",
        "alert_severity",
        "alert_type",
    }

    assert required_columns.issubset(df.columns)


def test_alert_ids_are_unique():
    df = load_alerts()

    assert df["alert_id"].is_unique


def test_no_missing_employee_ids():
    df = load_alerts()

    assert df["employee_id"].notna().all()


def test_alert_risk_scores_are_valid():
    df = load_alerts()

    assert df["risk_score"].between(0, 100).all()


def test_alert_severity_is_valid():
    df = load_alerts()

    valid_severities = {
        "P1 - Critical",
        "P2 - High",
    }

    assert set(df["alert_severity"].unique()).issubset(
        valid_severities
    )


def test_critical_employees_receive_p1_alerts():
    df = load_alerts()

    critical = df[df["risk_score"] > 50]

    assert len(critical) > 0
    assert (critical["alert_severity"] == "P1 - Critical").all()


def test_emp0277_alert():
    df = load_alerts()

    employee = df[df["employee_id"] == "EMP0277"]

    assert len(employee) == 1
    assert employee.iloc[0]["alert_severity"] == "P1 - Critical"
    assert round(employee.iloc[0]["risk_score"], 2) == 51.79