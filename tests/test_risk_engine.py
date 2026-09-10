import pandas as pd

from src.risk_engine.calculate_risk_scores import classify_risk


def test_classify_risk_low():
    assert classify_risk(10) == "Low"
    assert classify_risk(20) == "Low"


def test_classify_risk_moderate():
    assert classify_risk(21) == "Moderate"
    assert classify_risk(35) == "Moderate"


def test_classify_risk_high():
    assert classify_risk(36) == "High"
    assert classify_risk(50) == "High"


def test_classify_risk_critical():
    assert classify_risk(51) == "Critical"
    assert classify_risk(100) == "Critical"


def test_employee_risk_output_exists():
    df = pd.read_csv("data/processed/employee_risk_scores.csv")

    assert len(df) == 500
    assert "risk_score" in df.columns
    assert "risk_level" in df.columns
    assert "primary_risk_driver" in df.columns


def test_risk_scores_are_valid():
    df = pd.read_csv("data/processed/employee_risk_scores.csv")

    assert df["risk_score"].between(0, 100).all()


def test_no_missing_risk_levels():
    df = pd.read_csv("data/processed/employee_risk_scores.csv")

    assert df["risk_level"].notna().all()


def test_expected_risk_levels():
    df = pd.read_csv("data/processed/employee_risk_scores.csv")

    valid_levels = {
        "Low",
        "Moderate",
        "High",
        "Critical"
    }

    assert set(df["risk_level"].unique()).issubset(valid_levels)