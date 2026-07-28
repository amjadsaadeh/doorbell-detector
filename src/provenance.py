"""Where a chunk came from, in the columns needed to go find it again.

Aggregate metrics say that a model is wrong; they never say *on what*. Every
per-chunk diagnostic in this pipeline therefore joins its numbers back to the
manifest row that produced the chunk -- the Label Studio annotation, the
recording, and the offset inside it. This module is that join, written once,
so the quantization diagnostics and the training run's out-of-fold
predictions describe a chunk in exactly the same terms.

Everything keys on `row_index`. chunk_manifest.csv, balanced_data.h5 and
every array derived from them are row-aligned by construction (src/dataset.py
enforces the lengths agree), so the positional index is a stable,
dataset-global chunk id -- the same one quantize_model.py already stores in
the calibration set.
"""

import numpy as np
import pandas as pd

from paths import MANIFEST_PATH

# Ordered from "which annotation" outward to "which slice of which file",
# because that is the order someone tracking down a suspicious chunk needs
# them in. Filtered against the manifest's actual columns on read: snr_db and
# noise_pool only exist once augmentation/external pools have contributed
# rows, and a manifest built without them is still valid.
PROVENANCE_COLUMNS = [
    # The Label Studio annotation this chunk descends from. Augmented rows
    # carry a synthetic id (aug_snr-15_15_0) and external noise rows a
    # noise_<pool>_<stem> one, so the id also says which source produced it.
    "annotation_id",
    "file_id",
    # Resolves to a path on disk via audio_sources.resolve_audio_path.
    "audio_file_name",
    # s3:// original, so a chunk stays traceable when data/audio is gone.
    "remote_audio_path",
    # The annotated span, in seconds -- the label's extent in the recording.
    "start",
    "end",
    # The chunk window inside that recording, in milliseconds. This pair plus
    # audio_file_name is what uniquely identifies the audio behind a row.
    "chunk_start",
    "chunk_end",
    "label",
    # Source recording; the CV grouping key, so it also says which chunks
    # were forced onto the same side of a split.
    "split_group",
    # Augmented rows only: the SNR the mix was built at.
    "snr_db",
    # External background rows only: esc50 / demand.
    "noise_pool",
]


def chunk_provenance(manifest_path=MANIFEST_PATH) -> pd.DataFrame:
    """The manifest's identifying columns, indexed by dataset-global row.

    Returned in manifest order with `row_index` prepended, so it merges onto
    any per-chunk array by position without the caller having to re-derive
    what "position" means.
    """
    manifest = pd.read_csv(manifest_path)
    columns = [column for column in PROVENANCE_COLUMNS if column in manifest.columns]
    table = manifest[columns].copy()
    table.insert(0, "row_index", np.arange(len(table)))
    return table
