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
quantized graph plus the per-chunk diagnostics needed to find out why chunks
changed. A gate breach fails the stage *and* leaves that run marked FAILED,
so the evidence outlives the dead pipeline.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

from paths import (
    CALIBRATION_CHUNKS,
    CALIBRATION_MANIFEST,
    DATA_FILE,
    MANIFEST_PATH,
    QUANTIZED_METRICS,
)
from quantize_model import NORMALIZATION_PATH, TFLITE_PATH
from tflite_utils import build_export_model, largest_activation_bytes, tflite_predict
from train_cnn import MODEL_PATH, load_dataset
from train_xgboost import MLFLOW_EXPERIMENT_NAME, compute_metrics, get_git_branch

WORST_DEVIATIONS_LOGGED = 20
ON_FAILURE_MODES = ("fail", "warn")


def gate_passed(f1_drop: float, max_f1_drop: float) -> bool:
    """Inclusive: a drop exactly at the limit passes.

    Only a *drop* can fail. A negative f1_drop means int8 scored higher than
    float, which happens when quantization nudges a borderline chunk the
    right way -- noise, but not a reason to block.
    """
    return f1_drop <= max_f1_drop


def resolve_on_failure(quantization: dict) -> str:
    """Reject typos loudly. A stray value would otherwise fall through to
    the non-'fail' branch and silently turn the gate advisory."""
    mode = quantization["on_failure"]
    if mode not in ON_FAILURE_MODES:
        raise SystemExit(
            f"quantization.on_failure is {mode!r}; expected one of "
            f"{', '.join(ON_FAILURE_MODES)}"
        )
    return mode


def score_pair(export_model, tflite_model: bytes, X, y) -> tuple[dict, dict]:
    """Float and int8 metrics over the same chunks, plus how far apart.

    The f1 numbers are in-sample (see module docstring) and prefixed to say
    so; f1_drop between them is still a valid measure of what quantization
    cost, because both models see identical data.

    Returns the aggregates and the per-chunk arrays behind them -- the
    aggregates say a regression happened, the arrays say where.
    """
    float_scores = export_model.predict(X, verbose=0)[:, 0]
    float_pred = (float_scores > 0.5).astype(int)
    int8_scores = tflite_predict(tflite_model, X)
    int8_pred = (int8_scores > 0.5).astype(int)

    float_metrics = compute_metrics(y, float_pred, "float_insample")
    int8_metrics = compute_metrics(y, int8_pred, "int8_insample")

    # F1 saturates at 1.0 in-sample, which makes it blind to quantization
    # damage that has not yet flipped a label. The raw sigmoid deviation does
    # not saturate, so it is the metric that moves first if a future retrain
    # degrades under int8.
    deviation = np.abs(float_scores - int8_scores)

    metrics = {
        **float_metrics,
        **int8_metrics,
        "f1_drop": float_metrics["float_insample_f1_score"]
        - int8_metrics["int8_insample_f1_score"],
        "float_int8_agreement": float((float_pred == int8_pred).mean()),
        "max_score_deviation": float(deviation.max()),
        "mean_score_deviation": float(deviation.mean()),
        "scored_chunks": int(len(y)),
    }
    detail = {
        "float_score": float_scores,
        "int8_score": int8_scores,
        "float_pred": float_pred,
        "int8_pred": int8_pred,
        "score_deviation": deviation,
    }
    return metrics, detail


def chunk_diagnostics(y: np.ndarray, detail: dict) -> pd.DataFrame:
    """Per-chunk float-vs-int8 table joined to the chunks' provenance.

    Aggregates cannot answer "why did the gate fire". This can: it names the
    recording, the offset and the SNR variant behind every chunk where the
    two models diverge, which is what distinguishes damage concentrated in
    the deeply-buried augmented samples from damage spread evenly.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    columns = [
        c
        for c in ["audio_file_name", "chunk_start", "chunk_end", "label", "split_group"]
        if c in manifest.columns
    ]
    table = manifest[columns].copy()
    table.insert(0, "row_index", np.arange(len(table)))
    table["true_label"] = y
    for name, values in detail.items():
        table[name] = values
    return table


def write_diagnostics(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Two focused files: where the labels flipped, and where int8 moved most.

    Kept separate because they answer different questions -- a flip is a
    behaviour change that already happened, a large deviation is a chunk
    sitting close enough to the threshold to flip on the next retrain.
    """
    directory.mkdir(parents=True, exist_ok=True)

    disagreements = table[table["float_pred"] != table["int8_pred"]]
    disagreement_path = directory / "disagreements.csv"
    disagreements.to_csv(disagreement_path, index=False)

    worst = table.nlargest(WORST_DEVIATIONS_LOGGED, "score_deviation")
    worst_path = directory / "worst_score_deviations.csv"
    worst.to_csv(worst_path, index=False)

    return [disagreement_path, worst_path]


def skip(reason: str) -> None:
    QUANTIZED_METRICS.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZED_METRICS.write_text(json.dumps({"skipped": 1}, indent=4))
    print(f"quantized evaluation skipped: {reason}")


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    quantization = params["quantization"]

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

    metrics, detail = score_pair(export_model, tflite_model, X, y)
    calibration = np.load(CALIBRATION_CHUNKS)
    metrics.update(
        {
            "tflite_bytes": len(tflite_model),
            "peak_activation_bytes": largest_activation_bytes(tflite_model),
            "calibration_chunks": int(len(calibration["row_index"])),
        }
    )

    on_failure = resolve_on_failure(quantization)
    max_drop = quantization["max_f1_drop"]
    passed = gate_passed(metrics["f1_drop"], max_drop)
    # A metric, not just a raised exception: the verdict is then visible in
    # the run list and in any comparison view, without re-deriving it.
    metrics["gate_passed"] = int(passed)

    QUANTIZED_METRICS.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZED_METRICS.write_text(json.dumps(metrics, indent=4))

    table = chunk_diagnostics(y, detail)
    n_disagree = int((table["float_pred"] != table["int8_pred"]).sum())

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_branch = get_git_branch()

    with tempfile.TemporaryDirectory() as tmp_dir:
        diagnostics = write_diagnostics(table, Path(tmp_dir))

        # Raising *inside* the run context is what marks the run FAILED:
        # mlflow's context manager derives the status from the exception. So
        # a gate breach leaves a permanent, findable record instead of only a
        # dead pipeline.
        with mlflow.start_run(run_name=f"{head}-{feature_type}-{git_branch}-quantized"):
            mlflow.set_tag("git_branch", git_branch)
            mlflow.set_tag("stage", "quantized")
            # So nobody mistakes these f1 numbers for a generalization estimate.
            mlflow.set_tag("metric_scope", "in-sample float-vs-int8 delta")
            mlflow.set_tag("gate", "passed" if passed else "failed")
            mlflow.log_param("feature_type", feature_type)
            mlflow.log_param("head", head)
            mlflow.log_param(
                "balanced_data_md5", hashlib.md5(DATA_FILE.read_bytes()).hexdigest()
            )
            mlflow.log_param("input_shape", str(X.shape[1:]))
            mlflow.log_params(quantization)
            # The calibration subset is part of the quantized model's identity:
            # same weights + same chunks reproduce the same graph. Log the md5 so
            # a run can be matched against a DVC-tracked calibration set, and the
            # files themselves so it is recoverable without the DVC cache.
            mlflow.log_param(
                "calibration_md5",
                hashlib.md5(CALIBRATION_CHUNKS.read_bytes()).hexdigest(),
            )
            mlflow.log_artifact(CALIBRATION_CHUNKS, artifact_path="calibration")
            mlflow.log_artifact(CALIBRATION_MANIFEST, artifact_path="calibration")
            for path in diagnostics:
                mlflow.log_artifact(path, artifact_path="diagnostics")
            mlflow.log_artifact(TFLITE_PATH)
            mlflow.log_artifact(QUANTIZED_METRICS)
            mlflow.log_metrics(metrics)

            print(f"scored chunks : {metrics['scored_chunks']} (in-sample)")
            print(f"float F1      : {metrics['float_insample_f1_score']:.4f}")
            print(
                f"int8  F1      : {metrics['int8_insample_f1_score']:.4f}  "
                f"(drop {metrics['f1_drop']:+.4f}, limit {max_drop:.4f})"
            )
            print(
                f"label agree   : {metrics['float_int8_agreement']:.4f}  "
                f"({n_disagree} disagreeing chunks)"
            )
            print(
                f"score dev     : max {metrics['max_score_deviation']:.4f}  "
                f"mean {metrics['mean_score_deviation']:.4f}"
            )
            print(f"calibrated on : {metrics['calibration_chunks']} chunks")
            print(f"tflite size   : {metrics['tflite_bytes'] / 1024:.1f} KB")
            print(f"peak tensor   : {metrics['peak_activation_bytes'] / 1024:.1f} KB")
            print(f"gate          : {'PASSED' if passed else 'FAILED'}")

            if not passed and on_failure == "fail":
                raise SystemExit(
                    f"int8 quantization cost {metrics['f1_drop']:.4f} in-sample F1, "
                    f"limit is {max_drop:.4f}. Model not fit for deployment. The "
                    f"MLflow run is marked FAILED; its diagnostics/ artifacts list "
                    f"the {n_disagree} chunks that changed."
                )

        if not passed:
            print(
                "WARNING: the quantization gate failed but quantization.on_failure "
                "is 'warn', so the pipeline continued. The int8 model in "
                "models/export is NOT fit for deployment."
            )


if __name__ == "__main__":
    main()
