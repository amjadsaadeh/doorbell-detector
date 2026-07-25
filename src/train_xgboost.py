import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

from paths import DATA_FILE, DATA_QUALITY_DIR, MODEL_DIR
from splits import aggregate_fold_metrics, iter_folds

MLFLOW_EXPERIMENT_NAME = "doorbell-detector"
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
        # Logged explicitly because Keras' own val_accuracy comes from the
        # *last* epoch while these come from the restored best weights, so
        # the two disagree in MLflow (see EpochLogger in train_cnn.py).
        f"{prefix}_accuracy": report["accuracy"],
    }


def get_git_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def prepare_data(df):
    # The feature column is named after the representation that produced it
    # (mfcc_features, stft_features, ...) — detect it so this script works
    # unchanged across feature-extraction variants/branches, and surface the
    # type for run naming.
    feature_col = next(c for c in df.columns if c.endswith("_features"))
    # Convert features to 1D arrays
    X = np.vstack([x.flatten() for x in df[feature_col]])
    # Convert labels to binary (background=0, non-background=1)
    le = LabelEncoder()
    # Convert to inary problem
    df["label"] = df["label"].apply(
        lambda x: "background" if x == "background" else "bell"
    )
    y = le.fit_transform(df["label"])
    return X, y, le, feature_col.removesuffix("_features")


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]["xgboost"]
    training = params["training"]

    # Load data
    df = pd.read_hdf(DATA_FILE, key="data")
    X, y, le, feature_type = prepare_data(df)
    # Group-aware CV; iter_folds owns the leakage argument and the
    # why-not-one-fold argument (see train_cnn.py).
    groups = df["split_group"].to_numpy()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # log_models=False: one model per fold would be logged otherwise, and
    # the deliverable is fold 0's, saved explicitly below.
    mlflow.xgboost.autolog(log_datasets=False, log_models=False)

    git_branch = get_git_branch()

    with mlflow.start_run(run_name=f"xgboost-{feature_type}-{git_branch}"):
        mlflow.set_tag("git_branch", git_branch)
        mlflow.log_param("feature_type", feature_type)
        # Data lineage: md5 of the training dataset matches the entry DVC
        # records for it in dvc.lock/its cache, so any MLflow run can be
        # traced back to the exact dataset version and restored via dvc.
        mlflow.log_param("balanced_data_md5", hashlib.md5(DATA_FILE.read_bytes()).hexdigest())
        mlflow.log_param("n_chunks", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("test_size", training["test_size"])
        # eval_set order below drives XGBoost's auto-generated eval names,
        # which is what the per-round loss curve in MLflow gets logged under.
        mlflow.log_param("eval_set_names", "validation_0=train, validation_1=test")
        log_data_quality()

        val_per_fold, train_per_fold, test_fractions = [], [], []
        artifact = None

        for fold, train_idx, test_idx, n_splits in iter_folds(
            X, y, groups, training["test_size"], training["n_eval_folds"]
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Passing both train and test as eval_set makes XGBoost report
            # train/test eval_metric every boosting round, which
            # mlflow.xgboost.autolog logs as stepped metrics -> the loss curve.
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                random_state=training["random_state"] + fold,
                **model_params,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            fold_val = compute_metrics(y_test, y_pred, "val")
            fold_train = compute_metrics(y_train, y_train_pred, "train")
            mlflow.log_metrics(
                {f"fold{fold}_{k}": v for k, v in {**fold_val, **fold_train}.items()}
            )
            val_per_fold.append(fold_val)
            train_per_fold.append(fold_train)
            test_fractions.append(len(test_idx) / len(y))

            print(
                f"fold {fold}: val_f1={fold_val['val_f1_score']:.4f} "
                f"({len(test_idx)} chunks)"
            )

            if fold == 0:
                artifact = (model, y_test, y_pred)

        n_evaluated = len(val_per_fold)
        mlflow.log_param("n_eval_folds", n_evaluated)
        mlflow.log_param(
            "split_strategy",
            f"StratifiedGroupKFold by split_group, {n_evaluated}/{n_splits} folds evaluated",
        )
        # Group sizes vary, so the realized fold fraction can deviate from
        # the nominal test_size — log it for honest comparison across runs.
        mlflow.log_param(
            "realized_test_fraction", round(float(np.mean(test_fractions)), 4)
        )

        val_summary = aggregate_fold_metrics(val_per_fold)
        mlflow.log_metrics(val_summary)
        mlflow.log_metrics(aggregate_fold_metrics(train_per_fold))

        print(
            f"val_f1 across {n_evaluated} folds: "
            f"{val_summary['val_f1_score']:.4f} "
            f"+/- {val_summary['val_f1_score_std']:.4f} "
            f"(worst {val_summary['val_f1_score_min']:.4f})"
        )

        model, y_test, y_pred = artifact

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            le.inverse_transform(y_test), le.inverse_transform(y_pred), ax=ax
        )
        mlflow.log_figure(fig, "confusion_matrix_fold0.png")
        plt.close(fig)

        # Save model
        model_path = MODEL_DIR / "xgboost_model.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
