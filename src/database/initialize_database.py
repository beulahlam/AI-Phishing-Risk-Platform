from pathlib import Path
import sqlite3
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_FILE = DATABASE_DIR / "human_cyber_risk.db"

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_risk_scores.csv"
)

ALERT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "alerts"
    / "security_alerts.csv"
)

TRAINING_FILE = (
    PROJECT_ROOT
    / "reports"
    / "training"
    / "training_assignments.csv"
)


print("-" * 70)
print("Enterprise Human Cyber Risk Database Initialization")
print("-" * 70)


# ---------------------------------------------------------
# CREATE DATABASE DIRECTORY
# ---------------------------------------------------------

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# VERIFY REQUIRED FILES
# ---------------------------------------------------------

required_files = {
    "Employee Risk Scores": RISK_FILE,
    "Security Alerts": ALERT_FILE,
    "Training Assignments": TRAINING_FILE
}


print("\nChecking source files:")

for name, file_path in required_files.items():

    if not file_path.exists():

        raise FileNotFoundError(
            f"{name} file not found:\n{file_path}"
        )

    print(f"PASS - {name}")


# ---------------------------------------------------------
# LOAD SOURCE DATA
# ---------------------------------------------------------

risk_df = pd.read_csv(RISK_FILE)
alerts_df = pd.read_csv(ALERT_FILE)
training_df = pd.read_csv(TRAINING_FILE)


print("\nSource datasets loaded:")

print(
    f"Employee risk records: {len(risk_df)}"
)

print(
    f"Security alerts: {len(alerts_df)}"
)

print(
    f"Training assignments: {len(training_df)}"
)


# ---------------------------------------------------------
# CONNECT TO SQLITE DATABASE
# ---------------------------------------------------------

connection = sqlite3.connect(
    DATABASE_FILE
)

cursor = connection.cursor()


# ---------------------------------------------------------
# ENABLE FOREIGN KEY SUPPORT
# ---------------------------------------------------------

cursor.execute(
    "PRAGMA foreign_keys = ON;"
)


# ---------------------------------------------------------
# EMPLOYEES TABLE
# ---------------------------------------------------------

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS employees (

        employee_id TEXT PRIMARY KEY,

        department TEXT NOT NULL,

        role TEXT,

        privilege_level TEXT,

        security_training_completed INTEGER,

        previous_training_score REAL

    );
    """
)


# ---------------------------------------------------------
# RISK SCORES TABLE
# ---------------------------------------------------------

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS risk_scores (

        risk_id INTEGER PRIMARY KEY AUTOINCREMENT,

        employee_id TEXT NOT NULL UNIQUE,

        risk_score REAL NOT NULL,

        risk_level TEXT NOT NULL,

        primary_risk_driver TEXT,

        recommended_action TEXT,

        failure_rate REAL,

        credential_submission_rate REAL,

        click_rate REAL,

        reporting_rate REAL,

        training_risk REAL,

        privilege_risk_value REAL,

        FOREIGN KEY(employee_id)
        REFERENCES employees(employee_id)

    );
    """
)


# ---------------------------------------------------------
# SECURITY ALERTS TABLE
# ---------------------------------------------------------

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS security_alerts (

        alert_id TEXT PRIMARY KEY,

        employee_id TEXT NOT NULL,

        department TEXT,

        risk_score REAL,

        alert_severity TEXT,

        alert_type TEXT,

        FOREIGN KEY(employee_id)
        REFERENCES employees(employee_id)

    );
    """
)


# ---------------------------------------------------------
# TRAINING ASSIGNMENTS TABLE
# ---------------------------------------------------------

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS training_assignments (

        assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,

        employee_id TEXT NOT NULL,

        department TEXT,

        risk_score REAL,

        risk_level TEXT,

        primary_risk_driver TEXT,

        training_module TEXT,

        training_priority TEXT,

        completion_deadline_days INTEGER,

        FOREIGN KEY(employee_id)
        REFERENCES employees(employee_id)

    );
    """
)


# ---------------------------------------------------------
# CREATE INDEXES
# ---------------------------------------------------------

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_risk_level
    ON risk_scores(risk_level);
    """
)

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_employee_department
    ON employees(department);
    """
)

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_alert_severity
    ON security_alerts(alert_severity);
    """
)

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_training_priority
    ON training_assignments(training_priority);
    """
)


connection.commit()


# ---------------------------------------------------------
# PREPARE EMPLOYEE DATA
# ---------------------------------------------------------

employee_columns = [
    "employee_id",
    "department",
    "role",
    "privilege_level",
    "security_training_completed",
    "previous_training_score"
]


employees_df = (
    risk_df[employee_columns]
    .drop_duplicates(
        subset=["employee_id"]
    )
)


# ---------------------------------------------------------
# PREPARE RISK DATA
# ---------------------------------------------------------

risk_columns = [
    "employee_id",
    "risk_score",
    "risk_level",
    "primary_risk_driver",
    "recommended_action",
    "failure_rate",
    "credential_submission_rate",
    "click_rate",
    "reporting_rate",
    "training_risk",
    "privilege_risk_value"
]


risk_table_df = risk_df[
    risk_columns
].copy()


# ---------------------------------------------------------
# CLEAR OLD TABLE DATA
# ---------------------------------------------------------

cursor.execute(
    "DELETE FROM security_alerts;"
)

cursor.execute(
    "DELETE FROM training_assignments;"
)

cursor.execute(
    "DELETE FROM risk_scores;"
)

cursor.execute(
    "DELETE FROM employees;"
)

connection.commit()


# ---------------------------------------------------------
# LOAD EMPLOYEES
# ---------------------------------------------------------

employees_df.to_sql(
    "employees",
    connection,
    if_exists="append",
    index=False
)


# ---------------------------------------------------------
# LOAD RISK SCORES
# ---------------------------------------------------------

risk_table_df.to_sql(
    "risk_scores",
    connection,
    if_exists="append",
    index=False
)


# ---------------------------------------------------------
# LOAD ALERT DATA
# ---------------------------------------------------------

alert_columns = [
    "alert_id",
    "employee_id",
    "department",
    "risk_score",
    "alert_severity",
    "alert_type"
]


alerts_table_df = alerts_df[
    alert_columns
].copy()


alerts_table_df.to_sql(
    "security_alerts",
    connection,
    if_exists="append",
    index=False
)


# ---------------------------------------------------------
# LOAD TRAINING ASSIGNMENTS
# ---------------------------------------------------------

training_columns = [
    "employee_id",
    "department",
    "risk_score",
    "risk_level",
    "primary_risk_driver",
    "training_module",
    "training_priority",
    "completion_deadline_days"
]


training_table_df = training_df[
    training_columns
].copy()


training_table_df.to_sql(
    "training_assignments",
    connection,
    if_exists="append",
    index=False
)


connection.commit()


# ---------------------------------------------------------
# DATABASE VALIDATION
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("Database Validation")
print("-" * 70)


tables = [
    "employees",
    "risk_scores",
    "security_alerts",
    "training_assignments"
]


for table in tables:

    result = cursor.execute(
        f"SELECT COUNT(*) FROM {table};"
    ).fetchone()

    print(
        f"{table}: {result[0]} records"
    )


# ---------------------------------------------------------
# RISK LEVEL QUERY
# ---------------------------------------------------------

print("\nRisk Level Distribution:")

risk_distribution = pd.read_sql_query(
    """
    SELECT
        risk_level,
        COUNT(*) AS employee_count
    FROM risk_scores
    GROUP BY risk_level
    ORDER BY employee_count DESC;
    """,
    connection
)

print(
    risk_distribution.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# HIGH-RISK DEPARTMENT QUERY
# ---------------------------------------------------------

print("\nDepartment Risk Summary:")

department_risk = pd.read_sql_query(
    """
    SELECT
        e.department,
        ROUND(
            AVG(r.risk_score),
            2
        ) AS average_risk_score,
        MAX(r.risk_score)
            AS maximum_risk_score,
        COUNT(*)
            AS employee_count
    FROM employees e
    JOIN risk_scores r
        ON e.employee_id =
           r.employee_id
    GROUP BY e.department
    ORDER BY average_risk_score DESC;
    """,
    connection
)

print(
    department_risk.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# CRITICAL EMPLOYEE QUERY
# ---------------------------------------------------------

print("\nCritical Risk Employees:")

critical_employees = pd.read_sql_query(
    """
    SELECT
        e.employee_id,
        e.department,
        e.role,
        e.privilege_level,
        r.risk_score,
        r.primary_risk_driver
    FROM employees e
    JOIN risk_scores r
        ON e.employee_id =
           r.employee_id
    WHERE r.risk_level = 'Critical'
    ORDER BY r.risk_score DESC;
    """,
    connection
)

if critical_employees.empty:

    print(
        "No critical-risk employees found."
    )

else:

    print(
        critical_employees.to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# ALERT SUMMARY
# ---------------------------------------------------------

print("\nAlert Severity Summary:")

alert_summary = pd.read_sql_query(
    """
    SELECT
        alert_severity,
        COUNT(*) AS alert_count
    FROM security_alerts
    GROUP BY alert_severity
    ORDER BY alert_count DESC;
    """,
    connection
)

print(
    alert_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# TRAINING SUMMARY
# ---------------------------------------------------------

print("\nTraining Module Summary:")

training_summary = pd.read_sql_query(
    """
    SELECT
        training_module,
        COUNT(*) AS assigned_employees
    FROM training_assignments
    GROUP BY training_module
    ORDER BY assigned_employees DESC;
    """,
    connection
)

print(
    training_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# CLOSE DATABASE
# ---------------------------------------------------------

connection.close()


print(
    "\nDatabase saved successfully to:"
)

print(
    DATABASE_FILE
)

print(
    "\nFINAL STATUS: "
    "ENTERPRISE DATABASE INITIALIZATION COMPLETED"
)