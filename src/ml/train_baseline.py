import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

input_file = Path(
    "data/processed/ml_employee_dataset.csv"
)

results_directory = Path(
    "reports/results"
)

results_file = (
    results_directory
    / "logistic_regression_results.csv"
)


# ---------------------------------------------------------
# CHECK INPUT FILE
# ---------------------------------------------------------

if not input_file.exists():
    raise FileNotFoundError(
        "ml_employee_dataset.csv not found. "
        "Run prepare_ml_dataset.py first."
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(input_file)


print("\n--------------------------------------------")
print("Logistic Regression Baseline Model")
print("--------------------------------------------")

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ---------------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------------

feature_columns = [

    "security_training_completed",
    "previous_training_score",
    "privilege_risk_value",
    "historical_failure_rate",
    "historical_click_rate",
    "historical_attachment_rate",
    "historical_credential_rate",
    "historical_reporting_rate",
    "average_historical_failure_probability",
    "training_risk"
]


X = df[feature_columns].copy()

y = df[
    "future_phishing_failure"
].copy()


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\nTrain/Test Split:")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


print("\nTraining Target Distribution:")

print(
    y_train.value_counts()
)


print("\nTesting Target Distribution:")

print(
    y_test.value_counts()
)


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ---------------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------

model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# MAKE PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(
    X_test_scaled
)

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]


# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ---------------------------------------------------------
# NAIVE BASELINE
# ---------------------------------------------------------

majority_prediction = [1] * len(
    y_test
)

baseline_accuracy = accuracy_score(
    y_test,
    majority_prediction
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n--------------------------------------------")
print("Model Performance")
print("--------------------------------------------")

print(
    f"\nNaive Majority-Class Accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Logistic Regression Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


print("\nConfusion Matrix:")

print(
    f"True Negatives: {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives: {tp}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# ---------------------------------------------------------
# MODEL COEFFICIENTS
# ---------------------------------------------------------

coefficient_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "coefficient": model.coef_[0]
    }
)

coefficient_df[
    "absolute_importance"
] = (
    coefficient_df[
        "coefficient"
    ]
    .abs()
)

coefficient_df = (
    coefficient_df
    .sort_values(
        "absolute_importance",
        ascending=False
    )
)


print("\nFeature Influence:")

print(
    coefficient_df[
        [
            "feature",
            "coefficient"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE METRICS
# ---------------------------------------------------------

results = pd.DataFrame(
    [
        {
            "model":
                "Logistic Regression",

            "naive_baseline_accuracy":
                round(
                    baseline_accuracy,
                    4
                ),

            "accuracy":
                round(
                    accuracy,
                    4
                ),

            "precision":
                round(
                    precision,
                    4
                ),

            "recall":
                round(
                    recall,
                    4
                ),

            "f1_score":
                round(
                    f1,
                    4
                ),

            "roc_auc":
                round(
                    roc_auc,
                    4
                ),

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp)
        }
    ]
)


results_directory.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    results_file,
    index=False
)


print(
    f"\nModel results saved successfully to: "
    f"{results_file}"
)