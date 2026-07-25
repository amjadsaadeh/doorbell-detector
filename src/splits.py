"""Cross-validation splitting and fold aggregation, shared by both heads.

Lives in its own module rather than in one of the trainers because both
import it and train_cnn.py already imports helpers from train_xgboost.py --
putting it in either would be a cycle.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def iter_folds(X, y, groups, test_size: float, n_eval_folds: int | None = None):
    """Yield (fold, train_idx, test_idx, n_splits) for every evaluated fold.

    Leakage-safe by construction: chunks are cut from heavily overlapping
    sliding windows and augmented samples are SNR/gain variants of real
    chunks, so a random chunk-level split leaks near-duplicates. Grouping by
    split_group (source recording) keeps every window and synthetic variant
    of one recording on one side. test_size becomes the fold fraction
    (1/n_splits), stratified at group level.

    Scoring a single fold is how every feature variant came to report val F1
    1.0000: one fold of this dataset is ~183 chunks, where a single flipped
    label moves F1 by ~0.003, so it cannot separate a genuinely perfect
    model from a lucky split. Averaging all folds costs n_splits x the
    training time and buys a spread that discriminates.

    n_eval_folds=None evaluates all folds; an int caps it, and 1 reproduces
    the old single-fold behaviour for a fast iteration loop.
    """
    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    limit = n_splits if n_eval_folds is None else max(1, min(n_eval_folds, n_splits))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        if fold >= limit:
            return
        yield fold, train_idx, test_idx, n_splits


def prepare_split(X, y, groups, test_size: float, fold: int = 0):
    """Indices for one fold.

    export_tflite.py scores the saved model, and the saved model is fold 0's
    (see the trainers), so the default here has to stay 0 -- otherwise the
    export would be validated against data the model was trained on.
    """
    for current, train_idx, test_idx, n_splits in iter_folds(X, y, groups, test_size):
        if current == fold:
            return train_idx, test_idx, n_splits
    raise ValueError(f"fold {fold} is out of range for test_size {test_size}")


def aggregate_fold_metrics(per_fold: list[dict]) -> dict:
    """Mean, spread and worst case across CV folds.

    The mean keeps the plain metric name (val_f1_score, ...) so it stays the
    headline number and existing run comparisons keep working. The _std and
    _min companions are the point of doing this at all: they are what say
    whether a 1.0000 is a real result or one lucky fold.
    """
    if not per_fold:
        return {}
    aggregated = {}
    for key in per_fold[0]:
        values = [metrics[key] for metrics in per_fold]
        aggregated[key] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_min"] = float(np.min(values))
    return aggregated
