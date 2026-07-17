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
SAMPLING_RATE = 16000

def get_mfcc_features(start: int,  end: int, audio_file_name: str, audio_file_duration: int) -> np.ndarray:
    """Reads out the MFCC features from the file and cut them according to start and end.

    Args:
        start (int): start in ms
        end (int): end in ms
        duration (int): duration of the orifinal sound file in s (otherwise I didn't find a proper way to deduce the number of mel samples)
        audio_file_name (str): file to read the mfcc features from

    Returns:
        np.ndarray: mfcc features interval according to start and end
    """

    mfcc_file_path = MFCC_FEATURES_FILE_BASE / audio_file_name.replace('.wav', '.npy')
    mfccs = np.load(mfcc_file_path)
    
    # Calculate the start and end sample
    mel_coeffs_per_second = mfccs.shape[1] / float(audio_file_duration)

    start_sample = int((mel_coeffs_per_second * start) / 1000)
    chunk_size = int(mel_coeffs_per_second * (end - start) / 1000)
    end_sample = start_sample + chunk_size # This way the shape of the slice it more reliably the same every time

    res = mfccs[:, start_sample:end_sample]

    return res

if __name__ == "__main__":
    
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")

    # Preload duration, so we don't have to read the file for each chunk
    annotated_data["audio_file_duration"] = annotated_data["audio_file_name"].parallel_apply(lambda x: mediainfo(AUDIO_FILE_BASE / x)["duration"])

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
            row['audio_file_duration']
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
