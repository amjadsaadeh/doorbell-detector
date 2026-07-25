"""Dispatch to the configured training head.

The DAG needs one static `cmd` for the train_model stage so that a variant
is `dvc exp run -S training.head=...` rather than a branch with a different
dvc.yaml. This picks the head and validates it against the feature
representation before either trainer starts loading TensorFlow.
"""

import shutil

import yaml

from features import get_spec
from paths import MODEL_DIR

HEADS = ("cnn", "xgboost")


def check_compatible(head: str, feature_type: str) -> None:
    if head not in HEADS:
        raise SystemExit(
            f"unknown training.head {head!r}; known heads: {', '.join(HEADS)}"
        )
    spec = get_spec(feature_type)
    if head == "cnn" and not spec.keeps_time_axis:
        raise SystemExit(
            f"training.head 'cnn' needs a 2D (bins, frames) chunk, but feature "
            f"type {feature_type!r} pools the time axis away into a single "
            f"embedding vector. Use training.head 'xgboost' for {feature_type}."
        )


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    head = params["training"]["head"]
    check_compatible(head, params["feature_extraction"]["type"])

    # A head switch must not leave the previous head's artifact sitting in
    # the output directory; dvc repro clears outs for us, a direct run does not.
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if head == "cnn":
        from train_cnn import main as train
    else:
        from train_xgboost import main as train

    train()


if __name__ == "__main__":
    main()
