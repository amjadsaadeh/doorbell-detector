"""Held-out predictions for every chunk the cross-validation loop scored.

The fold metrics say how well a model built this way generalizes. They cannot
say *which* chunks it got wrong, and by the time the folds are aggregated the
per-chunk predictions have been thrown away -- only fold 0's survived, and
only far enough to draw a confusion matrix. This keeps all of them: each
fold's held-out scores, joined to the chunk that produced them.

Held-out is the entire point. The shipped model is a refit on 100% of the
chunks (see train_cnn.py), so its own predictions are in-sample, and a chunk
it gets wrong there is a chunk it failed to memorize -- a much weaker signal
than one a model got wrong having never seen it. The CV folds are the only
place in this pipeline where a chunk is scored by a model that did not train
on it, which makes this table the honest one to go hunting for label errors
and hard samples in. Nothing downstream of train_model can reproduce it.

Coverage is complete only when every fold runs: with `training.n_eval_folds`
set to 1 this covers just that fold's validation chunks, roughly a fifth of
the dataset. `oof_coverage` records which of the two happened.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from paths import MANIFEST_PATH, OOF_PREDICTIONS
from provenance import chunk_provenance

# Both heads emit a probability and cut it here; kept as a named constant so
# the table's `margin` column has a documented reference point rather than a
# magic 0.5 repeated in three files.
DECISION_THRESHOLD = 0.5

# background=0, bell=1 -- the ordering LabelEncoder produces in
# dataset.binarize_labels, spelled out so the CSV is readable without it.
CLASS_NAMES = ("background", "bell")


class OutOfFoldPredictions:
    """Accumulates each fold's held-out scores, then writes them out once.

    Fed from inside the fold loop, which is the only place the mapping from
    a validation prediction back to a dataset row still exists: `test_idx`
    indexes the full dataset, so the chunk ids come free.
    """

    def __init__(self, n_chunks: int):
        self.n_chunks = n_chunks
        self._folds: list[np.ndarray] = []
        self._row_index: list[np.ndarray] = []
        self._y_true: list[np.ndarray] = []
        self._y_score: list[np.ndarray] = []

    def add(self, fold: int, row_index, y_true, y_score) -> None:
        """One fold's validation chunks: their dataset rows, labels, scores.

        `y_score` is the positive-class probability, not a thresholded label
        -- how close a chunk sat to the boundary is most of what makes this
        table worth having.
        """
        row_index = np.asarray(row_index)
        y_score = np.asarray(y_score, dtype=float)
        if not (len(row_index) == len(y_true) == len(y_score)):
            raise ValueError(
                f"fold {fold}: {len(row_index)} rows, {len(y_true)} labels, "
                f"{len(y_score)} scores -- these must line up row for row"
            )
        self._folds.append(np.full(len(row_index), fold))
        self._row_index.append(row_index)
        self._y_true.append(np.asarray(y_true))
        self._y_score.append(y_score)

    def to_frame(self, manifest_path=MANIFEST_PATH) -> pd.DataFrame:
        """The predictions joined to where their chunks came from.

        Sorted by row_index rather than by fold, so the file is byte-stable
        for a given set of scores and diffs sensibly between runs.
        """
        if not self._folds:
            raise ValueError("no folds were added; nothing to write")

        row_index = np.concatenate(self._row_index)
        duplicated = len(row_index) != len(np.unique(row_index))
        if duplicated:
            raise ValueError(
                "a chunk was held out by more than one fold -- these are "
                "supposed to be disjoint, so the split is not what it claims"
            )

        y_true = np.concatenate(self._y_true).astype(int)
        y_score = np.concatenate(self._y_score)
        y_pred = (y_score > DECISION_THRESHOLD).astype(int)

        predictions = pd.DataFrame(
            {
                "row_index": row_index,
                "fold": np.concatenate(self._folds),
                "y_true": y_true,
                "y_pred": y_pred,
                "true_class": [CLASS_NAMES[i] for i in y_true],
                "predicted_class": [CLASS_NAMES[i] for i in y_pred],
                "y_score": y_score,
                # Distance from the decision boundary: small means the model
                # was guessing, and a *correct* prediction with a tiny margin
                # is as interesting as an outright error.
                "margin": np.abs(y_score - DECISION_THRESHOLD),
                "correct": y_true == y_pred,
                "outcome": [outcome(t, p) for t, p in zip(y_true, y_pred)],
            }
        )

        provenance = chunk_provenance(manifest_path)
        if len(provenance) != self.n_chunks:
            raise SystemExit(
                f"the manifest has {len(provenance)} rows but the dataset has "
                f"{self.n_chunks} chunks -- one of them is stale, so the "
                f"predictions cannot be attributed to a chunk"
            )
        merged = provenance.merge(predictions, on="row_index", how="inner")
        return merged.sort_values("row_index", ignore_index=True)

    def write(self, path: Path = OOF_PREDICTIONS, manifest_path=MANIFEST_PATH) -> Path:
        table = self.to_frame(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        return path

    def summary(self, manifest_path=MANIFEST_PATH) -> dict:
        """Counts worth having in MLflow next to the fold metrics.

        Deliberately not another F1: `val_f1_score` already reports that, and
        two subtly different accuracy numbers on one run is how a run gets
        misread. These are the counts that tell you how much there is to look
        at -- and `oof_coverage` below 1.0 is the flag that n_eval_folds was
        capped and most of the dataset has no held-out prediction at all.
        """
        table = self.to_frame(manifest_path)
        outcomes = table["outcome"].value_counts()
        return {
            "oof_coverage": len(table) / self.n_chunks,
            "oof_chunks": float(len(table)),
            "oof_errors": float((~table["correct"]).sum()),
            "oof_false_negatives": float(outcomes.get("false_negative", 0)),
            "oof_false_positives": float(outcomes.get("false_positive", 0)),
        }


def outcome(y_true: int, y_pred: int) -> str:
    """The confusion-matrix cell a chunk lands in, as a filterable string.

    A false negative is a missed doorbell and a false positive is a phantom
    ring; they are not equally bad and should not share one `correct=False`
    bucket when you are picking which chunks to listen to.
    """
    if y_true == y_pred:
        return "true_positive" if y_true == 1 else "true_negative"
    return "false_negative" if y_true == 1 else "false_positive"
