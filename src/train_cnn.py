"""
Small keyword-spotting-style CNN trained on the 2D time-frequency chunks in
balanced_data.h5 (proposal 1). Feature-representation agnostic like
train_xgboost.py: it auto-detects the *_features column, so the same script
trains on MFCC (cnn-mfcc branch) or STFT spectrogram (cnn-spectrogram
branch) chunks — the runs are named cnn-<feature_type>-<git_branch>.

Unlike the XGBoost path the chunks are NOT flattened: the depthwise-
separable CNN sees the (n_bins, n_frames) structure directly, which is the
point of this experiment. Split, MLflow logging and dataset lineage mirror
train_xgboost.py so runs stay comparable across model families.
"""

import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from train_xgboost import (
    DATA_FILE,
    MLFLOW_EXPERIMENT_NAME,
    compute_metrics,
    get_git_branch,
    log_data_quality,
)

MODEL_PATH = Path("./models/cnn_model.keras")


def prepare_data(df: pd.DataFrame):
    """Like train_xgboost.prepare_data, but keeps the chunks 2D (adding a
    trailing channel axis) instead of flattening them."""
    feature_col = next(c for c in df.columns if c.endswith("_features"))
    X = np.stack([np.asarray(x, dtype=np.float32) for x in df[feature_col]])
    X = X[..., np.newaxis]
    le = LabelEncoder()
    df["label"] = df["label"].apply(
        lambda x: "background" if x == "background" else "bell"
    )
    y = le.fit_transform(df["label"])
    return X, y, le, feature_col.removesuffix("_features")


def normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency-bin mean/std over the training set (samples x time),
    shaped for broadcasting onto (n, bins, frames, 1) batches."""
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)[0]
    std = X_train.std(axis=(0, 2, 3), keepdims=True)[0]
    return mean, np.maximum(std, 1e-6)


def build_model(input_shape: tuple[int, int, int], dropout: float):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def main():
    from tensorflow import keras

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    model_params = params["model"]

    keras.utils.set_random_seed(model_params["random_state"])

    df = pd.read_hdf(DATA_FILE, key="data")
    X, y, le, feature_type = prepare_data(df)

    # Raw STFT magnitudes span orders of magnitude; log-compression makes
    # them tractable for a CNN. MFCCs are already log-domain (branch config).
    if model_params["log_compress"]:
        X = np.log(X + 1e-6)

    # Same leakage-safe group split as train_xgboost.py (see comment there)
    groups = df["split_group"].to_numpy()
    n_splits = max(2, round(1 / params["training"]["test_size"]))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # CNNs (unlike trees) need scaled inputs; stats come from train only
    mean, std = normalization_stats(X_train)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    git_branch = get_git_branch()

    with mlflow.start_run(run_name=f"cnn-{feature_type}-{git_branch}"):
        mlflow.set_tag("git_branch", git_branch)
        mlflow.log_param("feature_type", feature_type)
        mlflow.log_param("balanced_data_md5", hashlib.md5(DATA_FILE.read_bytes()).hexdigest())
        mlflow.log_param("n_chunks", X.shape[0])
        mlflow.log_param("input_shape", str(X.shape[1:]))
        mlflow.log_param("test_size", params["training"]["test_size"])
        mlflow.log_param(
            "split_strategy", f"StratifiedGroupKFold by split_group, 1/{n_splits} test fold"
        )
        mlflow.log_param("realized_test_fraction", round(len(test_idx) / len(y), 4))
        mlflow.log_params(model_params)
        log_data_quality()

        model = build_model(X_train.shape[1:], model_params["dropout"])
        mlflow.log_param("n_model_params", model.count_params())
        model.compile(
            optimizer=keras.optimizers.Adam(model_params["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Per-epoch train/val curves, the CNN counterpart of the per-round
        # eval_set metrics mlflow autologs for XGBoost
        class EpochLogger(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics(
                    {k: float(v) for k, v in (logs or {}).items()}, step=epoch
                )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=model_params["early_stopping_patience"],
                    restore_best_weights=True,
                ),
                EpochLogger(),
            ],
            verbose=2,
        )

        y_pred = (model.predict(X_test, verbose=0)[:, 0] > 0.5).astype(int)
        y_train_pred = (model.predict(X_train, verbose=0)[:, 0] > 0.5).astype(int)

        mlflow.log_metrics(compute_metrics(y_train, y_train_pred, "train"))
        mlflow.log_metrics(compute_metrics(y_test, y_pred, "val"))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            le.inverse_transform(y_test), le.inverse_transform(y_pred), ax=ax
        )
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        MODEL_PATH.parent.mkdir(exist_ok=True)
        model.save(MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        # Inference needs the same normalization; keep the stats next to the
        # model weights
        stats_path = MODEL_PATH.with_name("cnn_normalization.npz")
        np.savez(stats_path, mean=mean, std=std)
        mlflow.log_artifact(stats_path)


if __name__ == "__main__":
    main()
