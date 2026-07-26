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
from sklearn.preprocessing import LabelEncoder

from features import get_spec
from paths import DATA_FILE, MODEL_DIR
from splits import aggregate_fold_metrics, iter_folds
from train_xgboost import (
    MLFLOW_EXPERIMENT_NAME,
    compute_metrics,
    get_git_branch,
    log_data_quality,
)

MODEL_PATH = MODEL_DIR / "cnn_model.keras"


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


def load_dataset(params: dict):
    """Load balanced_data.h5 and apply the branch's log-compression setting.

    Shared with export_tflite.py so the exported model is calibrated and
    scored on exactly the tensors it was trained on.
    """
    df = pd.read_hdf(DATA_FILE, key="data")
    X, y, le, feature_type = prepare_data(df)

    # Whether this is needed is a property of the representation, declared
    # in features.py, not a flag someone has to remember to flip.
    if get_spec(params["feature_extraction"]["type"]).needs_log_compression:
        X = np.log(X + 1e-6)

    groups = df["split_group"].to_numpy()
    return X, y, le, feature_type, groups


def normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frequency-bin mean/std over the training set (samples x time),
    shaped for broadcasting onto (n, bins, frames, 1) batches."""
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)[0]
    std = X_train.std(axis=(0, 2, 3), keepdims=True)[0]
    return mean, np.maximum(std, 1e-6)


def build_and_compile(input_shape, model_params: dict):
    from tensorflow import keras

    model = build_model(input_shape, model_params["dropout"])
    model.compile(
        optimizer=keras.optimizers.Adam(model_params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
    model_params = params["model"]["cnn"]
    training = params["training"]

    X, y, le, feature_type, groups = load_dataset(params)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    git_branch = get_git_branch()

    with mlflow.start_run(run_name=f"cnn-{feature_type}-{git_branch}"):
        mlflow.set_tag("git_branch", git_branch)
        mlflow.log_param("feature_type", feature_type)
        mlflow.log_param("balanced_data_md5", hashlib.md5(DATA_FILE.read_bytes()).hexdigest())
        mlflow.log_param("n_chunks", X.shape[0])
        mlflow.log_param("input_shape", str(X.shape[1:]))
        mlflow.log_param("test_size", training["test_size"])
        mlflow.log_params(model_params)
        log_data_quality()

        # Per-epoch train/val curves, the CNN counterpart of the per-round
        # eval_set metrics mlflow autologs for XGBoost. Only fold 0 gets one:
        # five overlapping series on the same step axis is unreadable.
        class EpochLogger(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics(
                    {k: float(v) for k, v in (logs or {}).items()}, step=epoch
                )

        val_per_fold, train_per_fold, test_fractions, best_epochs = [], [], [], []
        fold0_eval = None

        for fold, train_idx, test_idx, n_splits in iter_folds(
            X, y, groups, training["test_size"], training["n_eval_folds"]
        ):
            # Offset per fold so each fold is independently reproducible
            # rather than every fold sharing one initialization.
            keras.utils.set_random_seed(training["random_state"] + fold)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # CNNs (unlike trees) need scaled inputs; stats come from this
            # fold's training portion only, never the whole dataset
            mean, std = normalization_stats(X_train)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

            model = build_and_compile(X_train.shape[1:], model_params)

            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=model_params["early_stopping_patience"],
                restore_best_weights=True,
            )
            callbacks = [early_stopping]
            if fold == 0:
                callbacks.append(EpochLogger())

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=model_params["epochs"],
                batch_size=model_params["batch_size"],
                callbacks=callbacks,
                verbose=2,
            )

            y_pred = (model.predict(X_test, verbose=0)[:, 0] > 0.5).astype(int)
            y_train_pred = (model.predict(X_train, verbose=0)[:, 0] > 0.5).astype(int)

            fold_val = compute_metrics(y_test, y_pred, "val")
            fold_train = compute_metrics(y_train, y_train_pred, "train")
            mlflow.log_metrics(
                {f"fold{fold}_{k}": v for k, v in {**fold_val, **fold_train}.items()}
            )
            val_per_fold.append(fold_val)
            train_per_fold.append(fold_train)
            test_fractions.append(len(test_idx) / len(y))
            # Where val_loss actually bottomed out. The final refit has no
            # validation split to early-stop against, so it borrows the
            # epoch count these folds converged at.
            best_epochs.append(getattr(early_stopping, "best_epoch", len(history.epoch) - 1) + 1)

            print(
                f"fold {fold}: val_f1={fold_val['val_f1_score']:.4f} "
                f"({len(test_idx)} chunks)"
            )

            # Fold 0's held-out predictions are kept only for the confusion
            # matrix; the shipped model is the full-data refit below.
            if fold == 0:
                fold0_eval = (y_test, y_pred)

        n_evaluated = len(val_per_fold)
        mlflow.log_param("n_eval_folds", n_evaluated)
        mlflow.log_param(
            "split_strategy",
            f"StratifiedGroupKFold by split_group, {n_evaluated}/{n_splits} folds evaluated",
        )
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

        y_test, y_pred = fold0_eval
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            le.inverse_transform(y_test), le.inverse_transform(y_pred), ax=ax
        )
        mlflow.log_figure(fig, "confusion_matrix_fold0.png")
        plt.close(fig)

        # ---- Final model: refit on 100% of the chunks ----
        #
        # The CV block above is the generalization estimate and is the only
        # thing entitled to make one; this refit is the deliverable, and it
        # gets the ~20% of chunks that sat in the validation fold of every
        # single CV model.
        #
        # The consequence is deliberate and has to be understood downstream:
        # this model has NO held-out data, so nothing after this point can
        # measure its accuracy honestly. evaluate_quantized.py therefore
        # measures a float-vs-int8 delta rather than a quality score.
        final_epochs = max(1, round(float(np.mean(best_epochs))))
        mlflow.log_param("final_fit_epochs", final_epochs)
        mlflow.log_param("final_fit_chunks", int(len(y)))
        print(
            f"refitting on all {len(y)} chunks for {final_epochs} epochs "
            f"(mean best epoch across folds: {np.mean(best_epochs):.1f})"
        )

        keras.utils.set_random_seed(training["random_state"])
        mean, std = normalization_stats(X)
        X_full = (X - mean) / std

        model = build_and_compile(X_full.shape[1:], model_params)
        mlflow.log_param("n_model_params", model.count_params())
        # No EarlyStopping: there is no validation split to monitor. The epoch
        # count comes from where the CV folds actually converged, which is the
        # only unbiased estimate available here.
        model.fit(
            X_full,
            y,
            epochs=final_epochs,
            batch_size=model_params["batch_size"],
            verbose=2,
        )

        y_full_pred = (model.predict(X_full, verbose=0)[:, 0] > 0.5).astype(int)
        # Named to keep it unmistakable that this is in-sample: it is a
        # did-it-fit check, not a performance claim.
        mlflow.log_metrics(compute_metrics(y, y_full_pred, "insample_full"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        # Inference needs the same normalization; keep the stats next to the
        # model weights
        stats_path = MODEL_PATH.with_name("cnn_normalization.npz")
        np.savez(stats_path, mean=mean, std=std)
        mlflow.log_artifact(stats_path)


if __name__ == "__main__":
    main()
