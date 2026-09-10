from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
# 1. Project paths
# ---------------------------------------------------------

DATA_FILE = Path("data/processed/ml_employee_dataset.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"
METADATA_FILE = MODEL_DIR / "random_forest_metadata.json"
RESULTS_FILE = RESULTS_DIR / "random_forest_results.csv"
IMPORTANCE_FILE = RESULTS_DIR / "random_forest_feature_importance.csv"


# ---------------------------------------------------------
# 2. Load ML dataset
# ---------------------------------------------------------

print("\n" + "-" * 50)
print("Random Forest Phishing Risk Model")
print("-" * 50)

df = pd.read_csv(DATA_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ---------------------------------------------------------
# 3. Define target
# ---------------------------------------------------------

TARGET = "future_phishing_failure"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

y = df[TARGET].astype(int)


# ---------------------------------------------------------
# 4. Remove identifiers and leakage-prone fields
# ---------------------------------------------------------

exclude_columns = [
    TARGET,
    "future_failures",   # Prevent target leakage
    "employee_id",
    "employee_name",
    "name",
    "email",
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

X = df[feature_columns].copy()

print(f"\nFeatures used: {len(feature_columns)}")


# ---------------------------------------------------------
# 5. Detect numeric and categorical columns
# ---------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ---------------------------------------------------------
# 6. Preprocessing pipelines
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 7. Create train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain/Test Split:")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# ---------------------------------------------------------
# 8. Build Random Forest
# ---------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model),
    ]
)


# ---------------------------------------------------------
# 9. Train model
# ---------------------------------------------------------

print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ---------------------------------------------------------
# 10. Generate predictions
# ---------------------------------------------------------

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 11. Evaluate model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)
roc_auc = roc_auc_score(y_test, y_probability)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

majority_accuracy = y_test.value_counts(
    normalize=True
).max()


print("\n" + "-" * 50)
print("Random Forest Model Performance")
print("-" * 50)

print(
    f"\nNaive Majority-Class Accuracy: "
    f"{majority_accuracy:.4f}"
)
print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# 12. Feature importance
# ---------------------------------------------------------

fitted_preprocessor = pipeline.named_steps[
    "preprocessor"
]

feature_names = fitted_preprocessor.get_feature_names_out()

importances = pipeline.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\nTop 15 Most Important Features:")
print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 13. Save test predictions/results
# ---------------------------------------------------------

results_df = pd.DataFrame(
    {
        "actual": y_test.values,
        "predicted": y_pred,
        "failure_probability": y_probability,
    }
)

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


# ---------------------------------------------------------
# 14. Save trained model
# ---------------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_FILE,
)


# ---------------------------------------------------------
# 15. Save model metadata
# ---------------------------------------------------------

metadata = {
    "model": "RandomForestClassifier",
    "target": TARGET,
    "training_examples": len(X_train),
    "testing_examples": len(X_test),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# 16. Completion
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
    "\nFINAL STATUS: RANDOM FOREST TRAINING COMPLETED"
)