from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/ml_employee_dataset.csv"
)

RESULTS_DIR = Path(
    "reports/results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "model_comparison.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\n" + "-" * 55)
print("Enterprise ML Model Evaluation")
print("-" * 55)

df = pd.read_csv(DATA_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ---------------------------------------------------------
# DEFINE TARGET
# ---------------------------------------------------------

TARGET = "future_phishing_failure"

y = df[TARGET].astype(int)


# ---------------------------------------------------------
# REMOVE IDENTIFIERS AND FUTURE LEAKAGE
# ---------------------------------------------------------

exclude_columns = [
    TARGET,
    "future_failures",
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

X = df[feature_columns].copy()

print(
    f"\nFeatures used: "
    f"{len(feature_columns)}"
)


# ---------------------------------------------------------
# IDENTIFY FEATURE TYPES
# ---------------------------------------------------------

numeric_features = (
    X.select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(
        exclude=["number", "bool"]
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
# PREPROCESSING FOR TREE MODELS
# ---------------------------------------------------------

numeric_tree_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
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

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_tree_pipeline,
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
# PREPROCESSING FOR LOGISTIC REGRESSION
# ---------------------------------------------------------

numeric_logistic_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_logistic_pipeline,
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
# DEFINE MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression":

        Pipeline(
            steps=[
                (
                    "preprocessor",
                    logistic_preprocessor,
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),

    "Random Forest":

        Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

    "Gradient Boosting":

        Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
}


# ---------------------------------------------------------
# STRATIFIED CROSS-VALIDATION
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# SCORING METRICS
# ---------------------------------------------------------

scoring = {

    "accuracy":
        "accuracy",

    "balanced_accuracy":
        "balanced_accuracy",

    "precision":
        make_scorer(
            precision_score,
            zero_division=0,
        ),

    "recall":
        make_scorer(
            recall_score,
            zero_division=0,
        ),

    "f1":
        make_scorer(
            f1_score,
            zero_division=0,
        ),

    "roc_auc":
        "roc_auc",

    "average_precision":
        "average_precision",
}


# ---------------------------------------------------------
# EVALUATE MODELS
# ---------------------------------------------------------

comparison_results = []

for model_name, model in models.items():

    print(
        f"\nEvaluating: "
        f"{model_name}"
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    result = {
        "model":
            model_name,

        "accuracy_mean":
            scores[
                "test_accuracy"
            ].mean(),

        "balanced_accuracy_mean":
            scores[
                "test_balanced_accuracy"
            ].mean(),

        "precision_mean":
            scores[
                "test_precision"
            ].mean(),

        "recall_mean":
            scores[
                "test_recall"
            ].mean(),

        "f1_mean":
            scores[
                "test_f1"
            ].mean(),

        "roc_auc_mean":
            scores[
                "test_roc_auc"
            ].mean(),

        "pr_auc_mean":
            scores[
                "test_average_precision"
            ].mean(),

        "accuracy_std":
            scores[
                "test_accuracy"
            ].std(),

        "roc_auc_std":
            scores[
                "test_roc_auc"
            ].std(),
    }

    comparison_results.append(
        result
    )


# ---------------------------------------------------------
# CREATE RESULTS DATAFRAME
# ---------------------------------------------------------

results_df = pd.DataFrame(
    comparison_results
)

results_df = results_df.sort_values(
    by=[
        "balanced_accuracy_mean",
        "roc_auc_mean",
        "f1_mean",
    ],
    ascending=False,
)


# ---------------------------------------------------------
# ROUND RESULTS
# ---------------------------------------------------------

numeric_columns = (
    results_df
    .select_dtypes(
        include="number"
    )
    .columns
)

results_df[
    numeric_columns
] = (
    results_df[
        numeric_columns
    ]
    .round(4)
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "-" * 55)
print("Cross-Validated Model Comparison")
print("-" * 55)

print(
    "\n"
    + results_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# MAJORITY CLASS BASELINE
# ---------------------------------------------------------

majority_accuracy = (
    y.value_counts(
        normalize=True
    )
    .max()
)

print(
    f"\nMajority-Class Accuracy Baseline: "
    f"{majority_accuracy:.4f}"
)


# ---------------------------------------------------------
# SELECT BEST MODEL
# ---------------------------------------------------------

best_model = results_df.iloc[0]

print("\nRecommended Model:")

print(
    best_model[
        "model"
    ]
)

print(
    f"Balanced Accuracy: "
    f"{best_model['balanced_accuracy_mean']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_model['roc_auc_mean']:.4f}"
)

print(
    f"F1 Score: "
    f"{best_model['f1_mean']:.4f}"
)

print(
    f"PR-AUC: "
    f"{best_model['pr_auc_mean']:.4f}"
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nModel comparison saved to: "
    f"{OUTPUT_FILE}"
)

print(
    "\nFINAL STATUS: "
    "MODEL EVALUATION COMPLETED"
)