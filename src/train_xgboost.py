import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MLFLOW_EXPERIMENT_NAME = "doorbell-detector"

DATA_QUALITY_DIR = Path("./data/data_quality")
# (json file, metric prefix) pairs produced by extract_data_quality.py (raw,
# pre-chunking annotations) and draw_data.py (post-chunking/balancing) —
# logged to MLflow so quality of the data feeding a run is tied to its
# model/loss metrics.
DATA_QUALITY_METRIC_FILES = [
    ("sample_based_quality.json", "dq_raw"),
    ("chunk_balanced_quality.json", "dq_balanced"),
]
DATA_QUALITY_ARTIFACT_FILES = [
    "sample_based_quality.json",
    "chunk_balanced_quality.json",
    "samples_per_label.csv",
    "chunks_per_label.csv",
]


def log_data_quality():
    for filename, prefix in DATA_QUALITY_METRIC_FILES:
        path = DATA_QUALITY_DIR / filename
        with open(path) as f:
            metrics = json.load(f)
        mlflow.log_metrics(
            {
                f"{prefix}_{key}": value
                for key, value in metrics.items()
                if value is not None
            }
        )

    for filename in DATA_QUALITY_ARTIFACT_FILES:
        mlflow.log_artifact(DATA_QUALITY_DIR / filename, artifact_path="data_quality")


def compute_metrics(y_true, y_pred, prefix):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        f"{prefix}_f1_score": report["weighted avg"]["f1-score"],
        f"{prefix}_recall": report["weighted avg"]["recall"],
        f"{prefix}_precision": report["weighted avg"]["precision"],
    }


def prepare_data(df):
    # Convert MFCC features to 1D arrays
    X = np.vstack([x.flatten() for x in df["mfcc_features"]])
    # Convert labels to binary (background=0, non-background=1)
    le = LabelEncoder()
    # Convert to inary problem
    df["label"] = df["label"].apply(
        lambda x: "background" if x == "background" else "bell"
    )
    y = le.fit_transform(df["label"])
    return X, y, le


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]

    # Load data
    df = pd.read_hdf("./data/balanced_data.h5", key="data")
    X, y, le = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["training"]["test_size"], random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_datasets=False)

    with mlflow.start_run():
        mlflow.log_param("test_size", params["training"]["test_size"])
        # eval_set order below drives XGBoost's auto-generated eval names,
        # which is what the per-round loss curve in MLflow gets logged under.
        mlflow.log_param("eval_set_names", "validation_0=train, validation_1=test")
        log_data_quality()

        # Train model. Passing both train and test as eval_set makes XGBoost
        # report train/test eval_metric every boosting round, which
        # mlflow.xgboost.autolog logs as stepped metrics -> the loss curve.
        model = xgb.XGBClassifier(objective="binary:logistic", **model_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True,
        )

        # Evaluate and save metrics
        y_pred = model.predict(X_test)
        y_pred_decoded = le.inverse_transform(y_pred)
        y_test_decoded = le.inverse_transform(y_test)
        y_train_pred = model.predict(X_train)

        mlflow.log_metrics(compute_metrics(y_train, y_train_pred, "train"))
        mlflow.log_metrics(compute_metrics(y_test, y_pred, "val"))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test_decoded, y_pred_decoded, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Save model
        model_path = Path("./models/xgboost_model.json")
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(model_path)


if __name__ == "__main__":
    main()
