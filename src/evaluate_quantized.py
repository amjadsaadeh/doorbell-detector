"""Score the int8 model against its float32 self and gate on the difference.

This measures **quantization damage, not model quality**, and the distinction
matters. The shipped model is refit on 100% of the chunks, so it has no
held-out data and no honest accuracy can be computed here -- the
generalization estimate is the cross-validated val_f1_score on the training
run, and nothing in this stage supersedes it.

What *can* be measured without held-out data is the difference between two
versions of the same model. So both are scored across the entire dataset
through the same wrapped graph (normalization baked in), which makes every
number here a float-vs-int8 delta. Using all chunks rather than one fold is
the point: on a 183-chunk fold, F1 saturated at 1.0000 on both sides and the
gate was blind by construction. Nine times the chunks gives label agreement
and score deviation enough resolution to actually catch a regression.

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
from tflite_utils import build_export_model, largest_activation_bytes, tflite_predict
from train_cnn import MODEL_PATH, load_dataset
from train_xgboost import MLFLOW_EXPERIMENT_NAME, compute_metrics, get_git_branch


def score_pair(export_model, tflite_model: bytes, X_test, y_test) -> dict:
    """Float and int8 metrics over the same chunks, plus how far apart.

    The f1 numbers are in-sample (see module docstring) and prefixed to say
    so; f1_drop between them is still a valid measure of what quantization
    cost, because both models see identical data.
    """
    float_scores = export_model.predict(X_test, verbose=0)[:, 0]
    float_pred = (float_scores > 0.5).astype(int)
    int8_scores = tflite_predict(tflite_model, X_test)
    int8_pred = (int8_scores > 0.5).astype(int)

    float_metrics = compute_metrics(y_test, float_pred, "float_insample")
    int8_metrics = compute_metrics(y_test, int8_pred, "int8_insample")

    # F1 saturates at 1.0 in-sample, which makes it blind to quantization
    # damage that has not yet flipped a label. The raw sigmoid deviation does
    # not saturate, so it is the metric that moves first if a future retrain
    # degrades under int8.
    deviation = np.abs(float_scores - int8_scores)

    return {
        **float_metrics,
        **int8_metrics,
        "f1_drop": float_metrics["float_insample_f1_score"]
        - int8_metrics["int8_insample_f1_score"],
        "float_int8_agreement": float((float_pred == int8_pred).mean()),
        "max_score_deviation": float(deviation.max()),
        "mean_score_deviation": float(deviation.mean()),
        "scored_chunks": int(len(y_test)),
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

    # Every chunk: this is a float-vs-int8 comparison, so there is no
    # held-out requirement, and more chunks means more sensitivity.
    X, y, _, feature_type, _ = load_dataset(params)

    trained = keras.models.load_model(MODEL_PATH)
    stats = np.load(NORMALIZATION_PATH)
    export_model = build_export_model(trained, stats["mean"], stats["std"])
    tflite_model = TFLITE_PATH.read_bytes()

    metrics = score_pair(export_model, tflite_model, X, y)
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
        # So nobody mistakes these f1 numbers for a generalization estimate.
        mlflow.set_tag("metric_scope", "in-sample float-vs-int8 delta")
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

    print(f"scored chunks : {metrics['scored_chunks']} (in-sample)")
    print(f"float F1      : {metrics['float_insample_f1_score']:.4f}")
    print(
        f"int8  F1      : {metrics['int8_insample_f1_score']:.4f}  "
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
            f"int8 quantization cost {metrics['f1_drop']:.4f} in-sample F1, limit is "
            f"{max_drop:.4f}. Model not fit for deployment."
        )


if __name__ == "__main__":
    main()
