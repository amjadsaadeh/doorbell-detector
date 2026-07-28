"""Listen to the chunks the model got wrong, in a browser.

Renumics Spotlight renders a pandas DataFrame as a filterable table with an
audio player and a spectrogram per row, so "show me every held-out chunk
labeled front_doorbell that scored below 0.5" becomes a filter instead of a
notebook. The predictions come from src/oof.py -- held-out, so a wrong one is
a real error rather than a memorization failure -- and every row carries the
Label Studio annotation id, the recording, and the chunk's offset inside it,
which is what you need to go fix a label at the source.

Two subcommands, split because the first is slow and cacheable:

  prepare  cuts one wav per chunk (minutes on a cold cache, instant after)
           and writes data/spotlight/inspection.parquet.
  show     serves that parquet and the wavs to the browser.

src/inspect_dataset.sh runs both in order; that is the intended entry point.

Spotlight requires librosa>=0.11 and pyarrow>=21, which is why this project
runs those versions. That was checked rather than assumed before the bump:
log-mel, MFCC and STFT features are bit-identical between librosa 0.10.2 and
0.11.0 on this configuration, so the viewer's dependencies cost the dataset
nothing. Re-check that if librosa ever moves again -- a feature-extraction
dependency changing under a cached dataset is a silent-corruption risk, not a
routine upgrade.

Nothing here is a DVC stage. It reads pipeline outputs and produces a cache
that can be deleted at any time.
"""

import argparse
from pathlib import Path

import pandas as pd

from paths import (
    OOF_PREDICTIONS,
    SPOTLIGHT_AUDIO_DIR,
    SPOTLIGHT_DIR,
    SPOTLIGHT_TABLE,
)

# Column order in the table view: the verdict first, then how confident the
# model was, then enough provenance to find the audio again. Anything not
# listed here still exists in the data and can be switched on in the UI.
VISIBLE_COLUMNS = [
    "audio",
    "outcome",
    "y_score",
    "margin",
    "label",
    "predicted_class",
    "annotation_id",
    "audio_file_name",
    "chunk_start",
    "chunk_end",
    "snr_db",
    "fold",
]

CATEGORICAL_COLUMNS = [
    "outcome",
    "correct",
    "label",
    "true_class",
    "predicted_class",
    "split_group",
    "noise_pool",
]

# Spotlight builds a Category's value list by sorting the column's uniques,
# which raises TypeError the moment a None sits next to the strings. Two of
# these columns legitimately have gaps -- noise_pool is empty for anything
# that did not come from an external pool, and under --all-chunks every
# prediction column is empty for chunks no fold held out -- so the gap gets
# an explicit name instead. It is then filterable, which "missing" is not.
NOT_APPLICABLE = "n/a"


def load_table(all_chunks: bool) -> pd.DataFrame:
    """The out-of-fold predictions, optionally padded with the chunks that
    have none.

    With `training.n_eval_folds` capped, most chunks were never held out by
    any fold; --all-chunks includes them anyway (empty prediction columns) so
    the dataset can still be browsed as a dataset. The default is the
    predictions alone, because the errors are the point.
    """
    if not OOF_PREDICTIONS.exists():
        raise SystemExit(
            f"{OOF_PREDICTIONS} does not exist. It is written by the "
            f"train_model stage -- run `uv run dvc repro train_model` (or "
            f"`dvc pull`) first."
        )
    predictions = pd.read_csv(OOF_PREDICTIONS)
    if not all_chunks:
        return predictions

    from provenance import chunk_provenance

    prediction_columns = [
        "row_index",
        "fold",
        "y_true",
        "y_pred",
        "true_class",
        "predicted_class",
        "y_score",
        "margin",
        "correct",
        "outcome",
    ]
    provenance = chunk_provenance()
    return provenance.merge(
        predictions[[c for c in prediction_columns if c in predictions.columns]],
        on="row_index",
        how="left",
    )


def chunk_filename(row) -> str:
    """Names the wav after what it is, so the export directory is browsable
    on its own and a stale file is obvious."""
    stem = Path(str(row["audio_file_name"])).stem
    return f"{int(row['row_index']):05d}_{stem}_{int(row['chunk_start'])}ms.wav"


def export_chunks(table: pd.DataFrame, directory: Path, refresh: bool) -> list[str]:
    """Cut one wav per chunk, exactly the window the manifest describes.

    Sliced with pydub at millisecond resolution rather than re-derived from
    sample counts: chunk_start/chunk_end are milliseconds, and going through
    the same units the manifest uses removes any chance of the audio you
    listen to being off by a frame from the audio the model scored.

    Rows are processed grouped by source file so each recording is decoded
    once; a single recording contributes dozens of overlapping chunks.
    """
    from pydub import AudioSegment

    from audio_sources import resolve_audio_path

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[int, str] = {}
    exported = 0

    for audio_file_name, rows in table.groupby("audio_file_name", sort=False):
        source = None
        for _, row in rows.iterrows():
            target = directory / chunk_filename(row)
            paths[row["row_index"]] = str(target.resolve())
            if target.exists() and not refresh:
                continue
            if source is None:
                source = AudioSegment.from_file(
                    resolve_audio_path(str(audio_file_name))
                ).set_channels(1)
            clip = source[int(row["chunk_start"]) : int(row["chunk_end"])]
            clip.export(target, format="wav")
            exported += 1

    print(f"chunk audio: {exported} written, {len(paths) - exported} reused")
    return [paths[row_index] for row_index in table["row_index"]]


def compute_embeddings(row_index) -> list:
    """Penultimate-layer activations of the shipped CNN, for the similarity map.

    This is what turns Spotlight from a table with sound into an error
    analysis tool: chunks the model considers alike land together, so a
    cluster of mislabeled samples shows up as a cluster rather than as
    scattered rows.

    Note whose embeddings these are. The shipped model is refit on 100% of
    the chunks, so it has seen every row here -- fine for a projection, which
    is a picture and not a measurement, but it is *not* an out-of-fold view
    the way the prediction columns are.
    """
    import numpy as np
    import yaml
    from tensorflow import keras

    from dataset import load_dataset
    from quantize_model import NORMALIZATION_PATH
    from train_cnn import MODEL_PATH

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"--embeddings needs the trained CNN at {MODEL_PATH}; with "
            f"training.head set to xgboost there is none. Drop the flag."
        )

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    X, *_ = load_dataset(params)
    stats = np.load(NORMALIZATION_PATH)
    X = (X - stats["mean"]) / stats["std"]

    model = keras.models.load_model(MODEL_PATH)
    pooled = next(
        layer
        for layer in reversed(model.layers)
        if isinstance(layer, keras.layers.GlobalAveragePooling2D)
    )
    embedder = keras.Model(model.inputs, pooled.output)
    embeddings = embedder.predict(X[np.asarray(row_index)], verbose=0)
    return [row.astype(float) for row in embeddings]


def name_the_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Give every categorical blank a name Spotlight can sort (NOT_APPLICABLE).

    `correct` gets the same treatment for a different reason: it is the
    primary sort key of the table view, and a bool column holding None sorts
    no better than a Category does.
    """
    for column in CATEGORICAL_COLUMNS:
        if column in table.columns:
            table[column] = table[column].fillna(NOT_APPLICABLE)
    if "correct" in table.columns:
        table["correct"] = (
            table["correct"]
            .map({True: "correct", False: "wrong"})
            .fillna(NOT_APPLICABLE)
        )
    return table


def prepare(args) -> None:
    table = name_the_gaps(load_table(args.all_chunks))
    table["audio"] = export_chunks(table, SPOTLIGHT_AUDIO_DIR, args.refresh)

    if args.embeddings:
        table["embedding"] = compute_embeddings(table["row_index"])

    SPOTLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(SPOTLIGHT_TABLE, index=False)

    print(f"{len(table)} rows -> {SPOTLIGHT_TABLE}")
    if "outcome" in table:
        print(table["outcome"].value_counts().to_string())


def build_layout(has_embedding: bool):
    """Open on the errors, sorted worst-first, with the audio one click away.

    The default Spotlight layout is a generic table; this one is shaped
    around the question the tool exists to answer, so the first screen is
    already the interesting one rather than row 0 of the manifest.
    """
    from renumics.spotlight import layout as sl
    from renumics.spotlight.layout import lenses

    table = sl.table(
        visible_columns=VISIBLE_COLUMNS,
        # Wrong first, and within that the confident mistakes first: a chunk
        # the model was sure about and still got wrong is either a genuinely
        # hard sample or a bad label, and both are worth a listen. Descending
        # on the labels name_the_gaps() writes puts "wrong" above "correct".
        sort_by_columns=[("correct", "descending"), ("margin", "descending")],
    )
    side = (
        sl.similaritymap(columns=["embedding"], color_by_column="outcome")
        if has_embedding
        else sl.histogram(column="y_score", stack_by_column="outcome")
    )
    inspector = sl.inspector(
        lenses=[
            lenses.audio("audio"),
            lenses.spectrogram("audio", frequency_scale="logarithmic"),
            lenses.scalar("y_score"),
            lenses.scalar("label"),
            lenses.scalar("annotation_id"),
            lenses.scalar("audio_file_name"),
            lenses.scalar("chunk_start"),
            lenses.scalar("chunk_end"),
        ],
        num_columns=2,
    )
    return sl.layout(
        sl.split(
            sl.tab(table, weight=3),
            sl.tab(
                side,
                sl.confusion_matrix(x_column="true_class", y_column="predicted_class"),
                weight=2,
            ),
            orientation="horizontal",
            weight=3,
        ),
        sl.tab(inspector, weight=2),
        orientation="vertical",
    )


def show(args) -> None:
    from renumics import spotlight

    if not SPOTLIGHT_TABLE.exists():
        raise SystemExit(
            f"{SPOTLIGHT_TABLE} does not exist -- run "
            f"`uv run python src/inspect_dataset.py prepare` first, or use "
            f"./src/inspect_dataset.sh which does both."
        )

    table = pd.read_parquet(SPOTLIGHT_TABLE)
    dtype = {"audio": spotlight.Audio}
    dtype.update(
        {c: spotlight.Category for c in CATEGORICAL_COLUMNS if c in table.columns}
    )
    has_embedding = "embedding" in table.columns
    if has_embedding:
        dtype["embedding"] = spotlight.Embedding

    print(f"{len(table)} rows from {SPOTLIGHT_TABLE}")
    spotlight.show(
        table,
        dtype=dtype,
        layout=build_layout(has_embedding),
        host=args.host,
        port=args.port,
        no_browser=args.no_browser,
        # Otherwise the process exits the moment the server is up and the
        # browser tab points at nothing.
        wait="forever",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subcommands.add_parser(
        "prepare", help="cut chunk audio and build the inspection table"
    )
    prepare_parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="include chunks no fold held out (empty prediction columns)",
    )
    prepare_parser.add_argument(
        "--embeddings",
        action="store_true",
        help="add CNN penultimate-layer embeddings, enabling the similarity map",
    )
    prepare_parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-cut chunk wavs that already exist",
    )
    prepare_parser.set_defaults(func=prepare)

    show_parser = subcommands.add_parser("show", help="open the Spotlight viewer")
    show_parser.add_argument("--host", default="127.0.0.1")
    show_parser.add_argument("--port", default="auto")
    show_parser.add_argument("--no-browser", action="store_true")
    show_parser.set_defaults(func=show)

    args = parser.parse_args()
    if args.command == "show" and args.port != "auto":
        args.port = int(args.port)
    args.func(args)


if __name__ == "__main__":
    main()
