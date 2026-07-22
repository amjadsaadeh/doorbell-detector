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


MFCC_FEATURES_FILE_BASE = Path('./data/mfcc_data')
AUDIO_FILE_BASE = Path('./data/audio')
AUGMENTED_AUDIO_FILE_BASE = Path('./data/augmented_audio')
SAMPLING_RATE = 16000
HOP_LENGTH = 512  # librosa.feature.mfcc default, must match extract_mfcc_features.py
FRAMES_PER_MS = SAMPLING_RATE / HOP_LENGTH / 1000


def resolve_audio_path(audio_file_name: str) -> Path:
    """Real annotations point at data/audio; augmented (mixed) samples live
    under data/augmented_audio instead."""
    for base in (AUDIO_FILE_BASE, AUGMENTED_AUDIO_FILE_BASE):
        candidate = base / audio_file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(audio_file_name)

def get_mfcc_features(start: int,  end: int, audio_file_name: str) -> np.ndarray:
    """Reads out the MFCC features from the file and cut them according to start and end.

    Args:
        start (int): start in ms
        end (int): end in ms
        audio_file_name (str): file to read the mfcc features from

    Returns:
        np.ndarray: mfcc features interval according to start and end
    """

    mfcc_file_path = MFCC_FEATURES_FILE_BASE / audio_file_name.replace('.wav', '.npy')
    mfccs = np.load(mfcc_file_path)

    # Use the fixed sample_rate/hop_length rate librosa used to build the
    # array, not a per-file average (mfccs.shape[1] / duration): librosa's
    # frame count has a constant +1 offset, which is negligible for long
    # real files (avg rate ~= true rate) but dominant for exactly
    # chunk_size-long augmented clips (avg rate skews high), so per-file
    # rates produced inconsistent chunk widths and broke np.vstack downstream.
    start_sample = int(FRAMES_PER_MS * start)
    chunk_size = int(FRAMES_PER_MS * (end - start))
    end_sample = start_sample + chunk_size # This way the shape of the slice it more reliably the same every time

    res = mfccs[:, start_sample:end_sample]

    return res

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

    # Randomly sample from background class to match minority class size
    balanced_background = background_samples.sample(
        n=int(len(non_background_samples) * params['inbalance_ratio']), random_state=42
    )

    # Combine balanced datasets
    balanced_df = pd.concat([balanced_background, non_background_samples])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    sample_rate = 16000
    n_fft = params["feature_extraction"]["n_fft"]
    # Add MFCC features column
    balanced_df['mfcc_features'] = balanced_df.parallel_apply(
        lambda row: get_mfcc_features(
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
