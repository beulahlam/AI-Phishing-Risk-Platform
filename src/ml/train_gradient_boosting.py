from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/ml_employee_dataset.csv"
)

MODEL_DIR = Path("models")

RESULTS_DIR = Path(
    "reports/results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "gradient_boosting_model.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "gradient_boosting_metadata.json"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "gradient_boosting_results.csv"
)

IMPORTANCE_FILE = (
    RESULTS_DIR
    / "gradient_boosting_feature_importance.csv"
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("\n" + "-" * 50)
print("Gradient Boosting Phishing Risk Model")
print("-" * 50)

df = pd.read_csv(
    DATA_FILE
)

print(
    f"\nRows loaded: "
    f"{len(df)}"
)

print(
    f"Columns loaded: "
    f"{len(df.columns)}"
)


# ---------------------------------------------------------
# DEFINE TARGET
# ---------------------------------------------------------

TARGET = "future_phishing_failure"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' "
        f"was not found."
    )

y = df[TARGET].astype(int)


# ---------------------------------------------------------
# REMOVE IDENTIFIERS AND LEAKAGE
# ---------------------------------------------------------

exclude_columns = [
    TARGET,

    # Future outcome information
    "future_failures",

    # Identifiers
    "employee_id",
    "employee_name",
    "name",
    "email",
]


feature_columns = [

    column

    for column in df.columns

    if column not in exclude_columns
]


X = df[
    feature_columns
].copy()


print(
    f"\nFeatures used: "
    f"{len(feature_columns)}"
)


# ---------------------------------------------------------
# IDENTIFY FEATURE TYPES
# ---------------------------------------------------------

numeric_features = (

    X.select_dtypes(
        include=[
            "number",
            "bool"
        ]
    )
    .columns
    .tolist()
)


categorical_features = (

    X.select_dtypes(
        exclude=[
            "number",
            "bool"
        ]
    )
    .columns
    .tolist()
)


print(
    f"Numeric features: "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]
)


categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ]
)


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
    f"Training examples: "
    f"{len(X_train)}"
)

print(
    f"Testing examples: "
    f"{len(X_test)}"
)


# ---------------------------------------------------------
# CREATE GRADIENT BOOSTING MODEL
# ---------------------------------------------------------

gb_model = GradientBoostingClassifier(

    n_estimators=150,

    learning_rate=0.05,

    max_depth=3,

    min_samples_split=8,

    min_samples_leaf=4,

    random_state=42
)


pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            gb_model
        )

    ]
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

print(
    "\nTraining Gradient Boosting..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ---------------------------------------------------------
# MAKE PREDICTIONS
# ---------------------------------------------------------

y_pred = pipeline.predict(
    X_test
)

y_probability = (

    pipeline
    .predict_proba(
        X_test
    )[:, 1]
)


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

tn, fp, fn, tp = (

    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
    .ravel()
)


# ---------------------------------------------------------
# NAIVE BASELINE
# ---------------------------------------------------------

majority_accuracy = (

    y_test
    .value_counts(
        normalize=True
    )
    .max()
)


# ---------------------------------------------------------
# DISPLAY PERFORMANCE
# ---------------------------------------------------------

print("\n" + "-" * 50)

print(
    "Gradient Boosting Model Performance"
)

print("-" * 50)


print(
    f"\nNaive Majority-Class Accuracy: "
    f"{majority_accuracy:.4f}"
)

print(
    f"Gradient Boosting Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1 Score: "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)


print("\nConfusion Matrix:")

print(
    f"True Negatives: "
    f"{tn}"
)

print(
    f"False Positives: "
    f"{fp}"
)

print(
    f"False Negatives: "
    f"{fn}"
)

print(
    f"True Positives: "
    f"{tp}"
)


print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0
    )
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

fitted_preprocessor = (

    pipeline.named_steps[
        "preprocessor"
    ]
)


feature_names = (

    fitted_preprocessor
    .get_feature_names_out()
)


importances = (

    pipeline
    .named_steps[
        "classifier"
    ]
    .feature_importances_
)


importance_df = pd.DataFrame(

    {
        "feature":
            feature_names,

        "importance":
            importances
    }
)


importance_df = (

    importance_df

    .sort_values(

        "importance",

        ascending=False
    )
)


print(
    "\nTop 15 Most Important Features:"
)

print(

    importance_df

    .head(15)

    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE PREDICTION RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(

    {
        "actual":
            y_test.values,

        "predicted":
            y_pred,

        "failure_probability":
            y_probability
    }
)


results_df.to_csv(

    RESULTS_FILE,

    index=False
)


importance_df.to_csv(

    IMPORTANCE_FILE,

    index=False
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(

    pipeline,

    MODEL_FILE
)


# ---------------------------------------------------------
# SAVE METADATA
# ---------------------------------------------------------

metadata = {

    "model":
        "GradientBoostingClassifier",

    "target":
        TARGET,

    "training_examples":
        len(X_train),

    "testing_examples":
        len(X_test),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "true_negatives":
        int(tn),

    "false_positives":
        int(fp),

    "false_negatives":
        int(fn),

    "true_positives":
        int(tp)
}


with open(

    METADATA_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metadata,

        file,

        indent=4
    )


# ---------------------------------------------------------
# FINAL CONFIRMATION
# ---------------------------------------------------------

print(
    f"\nModel saved successfully to: "
    f"{MODEL_FILE}"
)

print(
    f"Model metadata saved to: "
    f"{METADATA_FILE}"
)

print(
    f"Prediction results saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"Feature importance saved to: "
    f"{IMPORTANCE_FILE}"
)


print(
    "\nFINAL STATUS: "
    "GRADIENT BOOSTING TRAINING COMPLETED"
)