from __future__ import annotations

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "severity",
    "category",
    "service",
    "assigned_team",
    "region",
    "business_unit",
    "alerts_count",
    "backlog_at_creation",
    "repeat_incident",
    "affected_users",
    "assignment_delay_hours",
    "num_comments",
    "created_hour",
    "is_weekend",
]


def build_pipeline() -> Pipeline:
    categorical = ["severity", "category", "service", "assigned_team", "region", "business_unit"]
    numeric = [
        "alerts_count",
        "backlog_at_creation",
        "repeat_incident",
        "affected_users",
        "assignment_delay_hours",
        "num_comments",
        "created_hour",
        "is_weekend",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(data_path: str | Path = "data/incidents.csv", model_dir: str | Path = "models", report_dir: str | Path = "reports") -> dict:
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    train_df = df[df["status"] == "Closed"].copy()

    X = train_df[FEATURE_COLUMNS]
    y = train_df["breached_sla"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "confusion_matrix": matrix.tolist(),
    }

    joblib.dump(pipeline, model_dir / "sla_breach_model.joblib")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importances = pipeline.named_steps["model"].feature_importances_

    feature_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values("importance", ascending=False)
    feature_df.to_csv(model_dir / "top_features.csv", index=False)

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, interpolation="nearest")
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, int(value), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(report_dir / "figures" / "model_confusion_matrix.png", dpi=160)
    plt.close()

    report_markdown = f"""# Model Performance Report

## Task
Binary classification to predict whether a **closed incident** will breach SLA.

## Evaluation split
- Training rows: **{metrics['train_rows']:,}**
- Test rows: **{metrics['test_rows']:,}**
- Positive rate (train): **{metrics['positive_rate_train']:.1%}**
- Positive rate (test): **{metrics['positive_rate_test']:.1%}**

## Metrics
- Accuracy: **{metrics['accuracy']:.2%}**
- Precision: **{metrics['precision']:.2%}**
- Recall: **{metrics['recall']:.2%}**
- F1 Score: **{metrics['f1']:.2%}**
- ROC-AUC: **{metrics['roc_auc']:.2%}**

## Confusion matrix
- True negatives: **{matrix[0, 0]}**
- False positives: **{matrix[0, 1]}**
- False negatives: **{matrix[1, 0]}**
- True positives: **{matrix[1, 1]}**

See: `figures/model_confusion_matrix.png`

## Top feature drivers
The saved file `models/top_features.csv` contains the full ranked feature list. In this synthetic setup, the strongest signals typically include:
- assignment delay hours
- affected users
- created hour
- alert count
- backlog at creation
- repeat incident flag

## Notes
This model is trained on synthetic data and is intended for portfolio demonstration rather than production deployment.
"""
    (report_dir / "model_report.md").write_text(report_markdown, encoding="utf-8")

    return metrics


if __name__ == "__main__":
    output = train_model()
    print(json.dumps(output, indent=2))
