"""
This script is to draw the data from the bigger dataset. It also takes care about data balancing.
"""

import json
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from pydub.utils import mediainfo
from dataqualityutils import get_data_quality_metrics
from tqdm import tqdm
from pandarallel import pandarallel


YAMNET_FEATURES_FILE_BASE = Path('./data/yamnet_data')
AUDIO_FILE_BASE = Path('./data/audio')
AUGMENTED_AUDIO_FILE_BASE = Path('./data/augmented_audio')
NOISE_POOL_FILE_BASE = Path('./data/noise')
# YAMNet emits one 1024-dim embedding per 0.48s hop (0.96s window), must
# match extract_yamnet_features.py
YAMNET_HOP_MS = 480


def resolve_audio_path(audio_file_name: str) -> Path:
    """Real annotations point at data/audio; augmented (mixed) samples live
    under data/augmented_audio; external noise pool files under
    data/noise/<pool>."""
    bases = [AUDIO_FILE_BASE, AUGMENTED_AUDIO_FILE_BASE]
    bases += sorted(d for d in NOISE_POOL_FILE_BASE.glob('*') if d.is_dir())
    for base in bases:
        candidate = base / audio_file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(audio_file_name)


def external_background_rows(pool_dirs: list[str]) -> pd.DataFrame:
    """One full-file background annotation row per external noise pool file,
    mirroring the tag-only convention of real background rows (no end time;
    the real file duration is filled in later). The pool name is kept in
    noise_pool so balancing can draw real and external background separately.
    """
    rows = []
    for pool_path in pool_dirs:
        pool_path = Path(pool_path)
        for wav_path in sorted(pool_path.glob('*.wav')):
            rows.append(
                {
                    "annotation_id": f"noise_{pool_path.name}_{wav_path.stem}",
                    "file_id": f"noise_{pool_path.name}_{wav_path.stem}",
                    "start": 0.0,
                    "end": np.nan,
                    "label": "background",
                    "audio_file_name": wav_path.name,
                    "remote_audio_path": "",
                    "noise_pool": pool_path.name,
                }
            )
    return pd.DataFrame(rows)


def split_background_draw(
    n_target: int, external_ratio: float, n_real_available: int, n_external_available: int
) -> tuple[int, int]:
    """How many background chunks to draw from real recordings vs the
    external noise pools. external_ratio is the desired external share; if
    one source can't fill its share (e.g. inbalance_ratio > 1 exhausts the
    real background chunks), the other tops it up.
    """
    n_external = int(round(n_target * external_ratio))
    n_real = n_target - n_external

    if n_real > n_real_available:
        n_real = n_real_available
        n_external = min(n_target - n_real, n_external_available)
    elif n_external > n_external_available:
        n_external = n_external_available
        n_real = min(n_target - n_external, n_real_available)

    return n_real, n_external

def get_yamnet_features(start: int, end: int, audio_file_name: str) -> np.ndarray:
    """Reads the per-file YAMNet frame embeddings, cuts the frames covering
    [start, end) and mean-pools them into a single fixed-size vector.

    Slicing uses the fixed YAMNet hop, not a per-file average frame rate —
    same principle as the old MFCC path (librosa's constant +1 frame offset
    skewed per-file rates for exactly chunk_size-long augmented clips).
    Pooling additionally makes the result shape-independent of the frame
    count: a 2s augmented clip yields 3 frames (no final partial window)
    while a 2s slice of a long file yields 4, and both must produce the
    same feature length for np.vstack downstream.

    Args:
        start (int): start in ms
        end (int): end in ms
        audio_file_name (str): file to read the embeddings from

    Returns:
        np.ndarray: mean-pooled embedding vector for the chunk
    """

    features_file_path = YAMNET_FEATURES_FILE_BASE / audio_file_name.replace('.wav', '.npy')
    embeddings = np.load(features_file_path)

    start_frame = int(start // YAMNET_HOP_MS)
    n_frames = max(1, int((end - start) // YAMNET_HOP_MS))
    chunk_frames = embeddings[:, start_frame:start_frame + n_frames]

    if chunk_frames.shape[1] == 0:
        # chunk starts past the last emitted frame (file-end edge case)
        chunk_frames = embeddings[:, -1:]

    return chunk_frames.mean(axis=1)

if __name__ == "__main__":
    
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")
    # Synthetic front_doorbell samples from the augmentation stage (mixed
    # signal+noise at target SNRs) are appended as regular rows so they flow
    # through the same chunking/balancing logic as real annotations.
    augmented_data = pd.read_csv("./data/augmented_annotations.csv")
    annotated_data = pd.concat([annotated_data, augmented_data], ignore_index=True)

    # External noise pool files (ESC-50, DEMAND) join as background rows so
    # the negative class also sees diverse non-Pi-mic noise, not just the
    # labeled recordings.
    external_data = external_background_rows(params["augmentation"]["external_noise_pools"])
    annotated_data = pd.concat([annotated_data, external_data], ignore_index=True)

    # Group key for leakage-safe train/val splitting downstream: augmented rows
    # already carry the source file of their signal chunk; real rows (no
    # split_group column in their CSV) group by their own file. Overlapping
    # sliding windows and SNR variants of one recording thus share a group.
    annotated_data["split_group"] = annotated_data["split_group"].fillna(
        annotated_data["audio_file_name"]
    )

    # Preload duration, so we don't have to read the file for each chunk
    annotated_data["audio_file_duration"] = annotated_data["audio_file_name"].parallel_apply(lambda x: mediainfo(resolve_audio_path(x))["duration"])

    # Full-file background annotations (tag-only in Label Studio) carry no end
    # time; use the real file duration
    annotated_data["end"] = annotated_data["end"].fillna(annotated_data["audio_file_duration"].astype(float))

    # Cut in chunks
    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]

    # Create chunks using list comprehension instead of iterative append
    chunks = [
        row.to_dict() | {"chunk_start": chunk_start, "chunk_end": (chunk_start + chunk_size)}
        for _, row in annotated_data.iterrows()
        for chunk_start in range(
            int(row["start"] * 1000),  # convert to ms
            int(row["end"] * 1000) - chunk_size,
            chunk_overlap,
        )
    ]
    chunks = pd.DataFrame(chunks)
    
    # Split into background and non-background samples
    background_samples = chunks[chunks["label"] == "background"]

    # TODO try imputations
    non_background_samples = chunks[chunks["label"] != "background"]

    # Randomly sample from background class relative to the minority class
    # size (inbalance_ratio > 1 means more background than positives). Real
    # and external background are drawn separately so the huge external
    # pools can't crowd out the deployed-mic recordings.
    n_background_target = int(len(non_background_samples) * params['inbalance_ratio'])
    real_background = background_samples[background_samples['noise_pool'].isna()]
    external_background = background_samples[background_samples['noise_pool'].notna()]
    n_real, n_external = split_background_draw(
        n_background_target,
        params['external_background_ratio'],
        len(real_background),
        len(external_background),
    )
    if n_real + n_external < n_background_target:
        print(
            f"Warning: only {n_real + n_external} background chunks available, "
            f"target was {n_background_target}"
        )
    balanced_background = pd.concat(
        [
            real_background.sample(n=n_real, random_state=42),
            external_background.sample(n=n_external, random_state=42),
        ]
    )

    # Combine balanced datasets
    balanced_df = pd.concat([balanced_background, non_background_samples])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add mean-pooled YAMNet embedding column
    balanced_df['yamnet_features'] = balanced_df.parallel_apply(
        lambda row: get_yamnet_features(
            row['chunk_start'],
            row['chunk_end'],
            row['audio_file_name'],
        ),
        axis=1
    )

    chunks_per_label = balanced_df.groupby("label").size()
    chunks_per_label.to_csv("./data/data_quality/chunks_per_label.csv")

    chunk_data_quality = get_data_quality_metrics(balanced_df)
    with open("./data/data_quality/chunk_balanced_quality.json", "w") as f:
        json.dump(chunk_data_quality, f, indent=4)

    # Save balanced dataset
    balanced_df.to_hdf("./data/balanced_data.h5", key='data', index=False)
