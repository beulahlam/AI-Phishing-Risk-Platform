from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Phishing Risk Platform",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "database"
    / "human_cyber_risk.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    return sqlite3.connect(DATABASE_FILE)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    connection = get_connection()

    employees = pd.read_sql_query(
        "SELECT * FROM employees",
        connection
    )

    risk_scores = pd.read_sql_query(
        "SELECT * FROM risk_scores",
        connection
    )

    alerts = pd.read_sql_query(
        "SELECT * FROM security_alerts",
        connection
    )

    training = pd.read_sql_query(
        "SELECT * FROM training_assignments",
        connection
    )

    connection.close()

    return (
        employees,
        risk_scores,
        alerts,
        training
    )


employees, risk_scores, alerts, training = load_data()


# =========================================================
# COMBINE EMPLOYEE + RISK DATA
# =========================================================

risk_data = employees.merge(
    risk_scores,
    on="employee_id",
    how="inner"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ Security Control Center")

st.sidebar.markdown(
    """
    **AI-Powered Phishing Risk Detection**

    Human cyber-risk monitoring,
    alerting and adaptive security
    awareness platform.
    """
)

st.sidebar.divider()


# Department filter

departments = sorted(
    risk_data["department"].dropna().unique()
)

selected_departments = st.sidebar.multiselect(
    "Department",
    departments,
    default=departments
)


# Risk-level filter

risk_order = [
    "Critical",
    "High",
    "Moderate",
    "Low"
]

available_risk_levels = [
    level
    for level in risk_order
    if level in risk_data["risk_level"].unique()
]

selected_risk_levels = st.sidebar.multiselect(
    "Risk Level",
    available_risk_levels,
    default=available_risk_levels
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_data = risk_data[
    risk_data["department"].isin(
        selected_departments
    )
    &
    risk_data["risk_level"].isin(
        selected_risk_levels
    )
].copy()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛡️ Enterprise Human Cyber Risk Command Center"
)

st.caption(
    "AI-Powered Phishing Risk Detection & "
    "Adaptive Security Awareness Platform"
)

st.divider()


# =========================================================
# KPI CARDS
# =========================================================

total_employees = len(filtered_data)

average_risk = (
    filtered_data["risk_score"].mean()
    if not filtered_data.empty
    else 0
)

critical_count = (
    filtered_data["risk_level"]
    .eq("Critical")
    .sum()
)

high_count = (
    filtered_data["risk_level"]
    .eq("High")
    .sum()
)

alert_employee_ids = set(
    alerts["employee_id"].astype(str)
)

visible_employee_ids = set(
    filtered_data["employee_id"].astype(str)
)

active_alerts = len(
    alert_employee_ids.intersection(
        visible_employee_ids
    )
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Employees Monitored",
    f"{total_employees:,}"
)

col2.metric(
    "Average Risk Score",
    f"{average_risk:.2f}"
)

col3.metric(
    "Critical Risk",
    int(critical_count)
)

col4.metric(
    "High Risk",
    int(high_count)
)

col5.metric(
    "Active Alerts",
    active_alerts
)


st.divider()


# =========================================================
# RISK DISTRIBUTION + DEPARTMENT RISK
# =========================================================

left, right = st.columns(2)


with left:

    st.subheader(
        "Risk Level Distribution"
    )

    risk_distribution = (
        filtered_data[
            "risk_level"
        ]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0
        )
    )

    st.bar_chart(
        risk_distribution
    )


with right:

    st.subheader(
        "Average Risk by Department"
    )

    department_risk = (
        filtered_data
        .groupby(
            "department"
        )["risk_score"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        department_risk
    )


st.divider()


# =========================================================
# PRIMARY RISK DRIVERS
# =========================================================

st.subheader(
    "Primary Human Cyber Risk Drivers"
)

risk_drivers = (
    filtered_data[
        "primary_risk_driver"
    ]
    .value_counts()
)

st.bar_chart(
    risk_drivers
)


st.divider()


# =========================================================
# HIGH-RISK EMPLOYEE TABLE
# =========================================================

st.subheader(
    "Priority Employee Risk Investigation"
)

priority_employees = (
    filtered_data[
        [
            "employee_id",
            "department",
            "role",
            "privilege_level",
            "risk_score",
            "risk_level",
            "primary_risk_driver",
            "recommended_action"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
)


st.dataframe(
    priority_employees,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# SECURITY ALERT CENTER
# =========================================================

st.subheader(
    "🚨 Security Alert Center"
)

if alerts.empty:

    st.success(
        "No active security alerts."
    )

else:

    visible_alerts = alerts[
        alerts["employee_id"]
        .astype(str)
        .isin(
            visible_employee_ids
        )
    ].copy()

    visible_alerts = visible_alerts.sort_values(
        "risk_score",
        ascending=False
    )

    st.dataframe(
        visible_alerts,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# =========================================================
# ADAPTIVE TRAINING
# =========================================================

st.subheader(
    "🎓 Adaptive Security Awareness Training"
)

visible_training = training[
    training["employee_id"]
    .astype(str)
    .isin(
        visible_employee_ids
    )
].copy()

visible_training = visible_training.sort_values(
    "risk_score",
    ascending=False
)


training_col1, training_col2 = st.columns(2)


with training_col1:

    st.markdown(
        "**Training Module Distribution**"
    )

    training_distribution = (
        visible_training[
            "training_module"
        ]
        .value_counts()
    )

    st.bar_chart(
        training_distribution
    )


with training_col2:

    st.markdown(
        "**Training Priority Distribution**"
    )

    priority_distribution = (
        visible_training[
            "training_priority"
        ]
        .value_counts()
    )

    st.bar_chart(
        priority_distribution
    )


st.dataframe(
    visible_training[
        [
            "employee_id",
            "department",
            "risk_score",
            "risk_level",
            "training_module",
            "training_priority",
            "completion_deadline_days"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# EMPLOYEE INVESTIGATION
# =========================================================

st.subheader(
    "🔎 Employee Risk Investigation"
)

employee_options = sorted(
    filtered_data[
        "employee_id"
    ].astype(str)
)


if employee_options:

    selected_employee = st.selectbox(
        "Select Employee",
        employee_options
    )

    employee_record = filtered_data[
        filtered_data[
            "employee_id"
        ].astype(str)
        == selected_employee
    ].iloc[0]


    investigation_col1, investigation_col2 = st.columns(2)


    with investigation_col1:

        st.markdown(
            "### Employee Context"
        )

        st.write(
            "**Employee ID:**",
            employee_record[
                "employee_id"
            ]
        )

        st.write(
            "**Department:**",
            employee_record[
                "department"
            ]
        )

        st.write(
            "**Role:**",
            employee_record[
                "role"
            ]
        )

        st.write(
            "**Privilege Level:**",
            employee_record[
                "privilege_level"
            ]
        )


    with investigation_col2:

        st.markdown(
            "### Risk Assessment"
        )

        st.metric(
            "Risk Score",
            f"{employee_record['risk_score']:.2f}"
        )

        st.write(
            "**Risk Level:**",
            employee_record[
                "risk_level"
            ]
        )

        st.write(
            "**Primary Risk Driver:**",
            employee_record[
                "primary_risk_driver"
            ]
        )

        st.write(
            "**Recommended Action:**",
            employee_record[
                "recommended_action"
            ]
        )


    st.markdown(
        "### Behavioral Risk Indicators"
    )


    indicator1, indicator2, indicator3, indicator4 = st.columns(4)


    indicator1.metric(
        "Phishing Failure Rate",
        f"{employee_record['failure_rate']:.2%}"
    )

    indicator2.metric(
        "Credential Submission",
        f"{employee_record['credential_submission_rate']:.2%}"
    )

    indicator3.metric(
        "Link Click Rate",
        f"{employee_record['click_rate']:.2%}"
    )

    indicator4.metric(
        "Reporting Rate",
        f"{employee_record['reporting_rate']:.2%}"
    )


else:

    st.warning(
        "No employees match the selected filters."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Enterprise Human Cyber Risk Analytics | "
    "Phishing Behavioral Analytics | "
    "Security Awareness Automation"
)