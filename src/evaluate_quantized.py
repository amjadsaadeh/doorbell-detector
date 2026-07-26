"""Score the int8 model against its float32 self and gate on the difference.

Full-integer post-training quantization can quietly destroy a model that
looks fine in float, so this re-runs the same leakage-safe fold the trained
model was validated on, scores the int8 interpreter, and fails the stage if
F1 drops by more than quantization.max_f1_drop.

Both models are scored through the *same wrapped graph* (normalization baked
in), so the delta measures quantization and nothing else.

Results land in their own MLflow run, named after the training run with a
`-quantized` suffix, carrying the calibration subset that produced the
quantized graph as an artifact.
"""

import hashlib
import json
import os

import mlflow
import numpy as np
import yaml
from tensorflow import keras

from paths import (
    CALIBRATION_CHUNKS,
    CALIBRATION_MANIFEST,
    DATA_FILE,
    QUANTIZED_METRICS,
)
from quantize_model import NORMALIZATION_PATH, TFLITE_PATH
from splits import prepare_split
from tflite_utils import build_export_model, largest_activation_bytes, tflite_predict
from train_cnn import MODEL_PATH, load_dataset
from train_xgboost import MLFLOW_EXPERIMENT_NAME, compute_metrics, get_git_branch


def score_pair(export_model, tflite_model: bytes, X_test, y_test) -> dict:
    """Float and int8 metrics on the same fold, plus how far apart they are."""
    float_scores = export_model.predict(X_test, verbose=0)[:, 0]
    float_pred = (float_scores > 0.5).astype(int)
    int8_scores = tflite_predict(tflite_model, X_test)
    int8_pred = (int8_scores > 0.5).astype(int)

    float_metrics = compute_metrics(y_test, float_pred, "float_val")
    int8_metrics = compute_metrics(y_test, int8_pred, "int8_val")

    # F1 and label agreement both saturate at 1.0 on a fold this small, which
    # makes them blind to quantization damage that has not yet flipped a
    # label. The raw sigmoid deviation does not saturate, so it is the metric
    # that moves first if a future retrain degrades.
    deviation = np.abs(float_scores - int8_scores)

    return {
        **float_metrics,
        **int8_metrics,
        "f1_drop": float_metrics["float_val_f1_score"] - int8_metrics["int8_val_f1_score"],
        "float_int8_agreement": float((float_pred == int8_pred).mean()),
        "max_score_deviation": float(deviation.max()),
        "mean_score_deviation": float(deviation.mean()),
        "val_chunks": int(len(y_test)),
    }


def skip(reason: str) -> None:
    QUANTIZED_METRICS.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZED_METRICS.write_text(json.dumps({"skipped": 1}, indent=4))
    print(f"quantized evaluation skipped: {reason}")


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    head = params["training"]["head"]
    if head != "cnn":
        skip(f"training.head is {head!r}; nothing was quantized")
        return

    X, y, _, feature_type, groups = load_dataset(params)
    # fold 0: the only split the saved model never trained on
    _, test_idx, _ = prepare_split(X, y, groups, params["training"]["test_size"])
    X_test, y_test = X[test_idx], y[test_idx]

    trained = keras.models.load_model(MODEL_PATH)
    stats = np.load(NORMALIZATION_PATH)
    export_model = build_export_model(trained, stats["mean"], stats["std"])
    tflite_model = TFLITE_PATH.read_bytes()

    metrics = score_pair(export_model, tflite_model, X_test, y_test)
    calibration = np.load(CALIBRATION_CHUNKS)
    metrics.update(
        {
            "tflite_bytes": len(tflite_model),
            "peak_activation_bytes": largest_activation_bytes(tflite_model),
            "calibration_chunks": int(len(calibration["row_index"])),
        }
    )
    QUANTIZED_METRICS.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZED_METRICS.write_text(json.dumps(metrics, indent=4))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_branch = get_git_branch()

    with mlflow.start_run(run_name=f"{head}-{feature_type}-{git_branch}-quantized"):
        mlflow.set_tag("git_branch", git_branch)
        mlflow.set_tag("stage", "quantized")
        mlflow.log_param("feature_type", feature_type)
        mlflow.log_param("head", head)
        mlflow.log_param(
            "balanced_data_md5", hashlib.md5(DATA_FILE.read_bytes()).hexdigest()
        )
        mlflow.log_param("input_shape", str(X.shape[1:]))
        mlflow.log_params(params["quantization"])
        # The calibration subset is part of the quantized model's identity:
        # same weights + same chunks reproduce the same graph. Log the md5 so
        # a run can be matched against a DVC-tracked calibration set, and the
        # files themselves so it is recoverable without the DVC cache.
        mlflow.log_param(
            "calibration_md5", hashlib.md5(CALIBRATION_CHUNKS.read_bytes()).hexdigest()
        )
        mlflow.log_artifact(CALIBRATION_CHUNKS, artifact_path="calibration")
        mlflow.log_artifact(CALIBRATION_MANIFEST, artifact_path="calibration")
        mlflow.log_artifact(TFLITE_PATH)
        mlflow.log_artifact(QUANTIZED_METRICS)
        mlflow.log_metrics(metrics)

    print(f"val chunks    : {metrics['val_chunks']}")
    print(f"float val F1  : {metrics['float_val_f1_score']:.4f}")
    print(
        f"int8  val F1  : {metrics['int8_val_f1_score']:.4f}  "
        f"(drop {metrics['f1_drop']:+.4f})"
    )
    print(f"label agree   : {metrics['float_int8_agreement']:.4f}")
    print(
        f"score dev     : max {metrics['max_score_deviation']:.4f}  "
        f"mean {metrics['mean_score_deviation']:.4f}"
    )
    print(f"calibrated on : {metrics['calibration_chunks']} chunks")
    print(f"tflite size   : {metrics['tflite_bytes'] / 1024:.1f} KB")
    print(f"peak tensor   : {metrics['peak_activation_bytes'] / 1024:.1f} KB")

    max_drop = params["quantization"]["max_f1_drop"]
    if metrics["f1_drop"] > max_drop:
        raise SystemExit(
            f"int8 quantization cost {metrics['f1_drop']:.4f} val F1, limit is "
            f"{max_drop:.4f}. Model not fit for deployment."
        )


if __name__ == "__main__":
    main()
