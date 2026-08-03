"""Decide which audio chunks make up the training set -- before any features exist.

This is the first half of what draw_data.py used to do. Splitting it out buys
two things:

1. **Feature extraction stops being speculative.** It used to run over every
   file in every source, including all ~1900 external noise-pool files, and
   draw_data then sampled a few hundred background chunks out of that. Most
   of the extracted arrays were discarded -- cheap for log-mel, 769 MB for
   STFT. extract_features.py now reads this manifest and touches only the
   files a chunk actually references.

2. **Sampling is shared across feature variants.** Nothing here depends on
   `feature_extraction`, so switching representation (or sweeping n_mels)
   reuses this stage's cache instead of re-deciding the split. Every variant
   therefore trains on exactly the same chunks, which is what makes the
   MLflow comparison between them mean anything.

Output is data/chunk_manifest.csv, in final (shuffled) row order -- draw_data
just attaches features to it, so the ordering fed to the trainer is decided
here and only here.
"""

import json
import pandas as pd
import yaml
from pandarallel import pandarallel
from pydub.utils import mediainfo
from tqdm import tqdm

from audio_sources import external_background_rows, resolve_audio_path
from dataqualityutils import get_data_quality_metrics
from paths import DATA_QUALITY_DIR, MANIFEST_PATH


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


def build_annotations(params: dict) -> pd.DataFrame:
    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")
    # Synthetic front_doorbell samples from the augmentation stage (mixed
    # signal+noise at target SNRs) are appended as regular rows so they flow
    # through the same chunking/balancing logic as real annotations.
    augmented_data = pd.read_csv("./data/augmented_annotations.csv")
    annotated_data = pd.concat([annotated_data, augmented_data], ignore_index=True)

    # External noise pool files (ESC-50, DEMAND) join as background rows so
    # the negative class also sees diverse non-Pi-mic noise, not just the
    # labeled recordings.
    external_data = external_background_rows(
        params["augmentation"]["external_noise_pools"]
    )
    annotated_data = pd.concat([annotated_data, external_data], ignore_index=True)

    # Group key for leakage-safe train/val splitting downstream: augmented rows
    # already carry the source file of their signal chunk; real rows (no
    # split_group column in their CSV) group by their own file. Overlapping
    # sliding windows and SNR variants of one recording thus share a group.
    annotated_data["split_group"] = annotated_data["split_group"].fillna(
        annotated_data["audio_file_name"]
    )
    return annotated_data


def cut_into_chunks(annotated_data: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    return pd.DataFrame(
        [
            row.to_dict()
            | {"chunk_start": chunk_start, "chunk_end": (chunk_start + chunk_size)}
            for _, row in annotated_data.iterrows()
            for chunk_start in range(
                int(row["start"] * 1000),  # convert to ms
                int(row["end"] * 1000) - chunk_size,
                chunk_overlap,
            )
        ]
    )


def balance_chunks(chunks: pd.DataFrame, params: dict) -> pd.DataFrame:
    background_samples = chunks[chunks["label"] == "background"]
    # TODO try imputations
    non_background_samples = chunks[chunks["label"] != "background"]

    # Randomly sample from background class relative to the minority class
    # size (inbalance_ratio > 1 means more background than positives). Real
    # and external background are drawn separately so the huge external
    # pools can't crowd out the deployed-mic recordings.
    n_background_target = int(len(non_background_samples) * params["inbalance_ratio"])
    real_background = background_samples[background_samples["noise_pool"].isna()]
    external_background = background_samples[background_samples["noise_pool"].notna()]
    n_real, n_external = split_background_draw(
        n_background_target,
        params["external_background_ratio"],
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

    balanced_df = pd.concat([balanced_background, non_background_samples])
    # Shuffle here rather than downstream: this is the row order the trainer
    # ultimately sees, and it must not depend on the feature variant.
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    annotated_data = build_annotations(params)

    # Preload duration, so we don't have to read the file for each chunk
    annotated_data["audio_file_duration"] = annotated_data[
        "audio_file_name"
    ].parallel_apply(lambda x: mediainfo(resolve_audio_path(x))["duration"])

    # Full-file background annotations (tag-only in Label Studio) carry no end
    # time; use the real file duration
    annotated_data["end"] = annotated_data["end"].fillna(
        annotated_data["audio_file_duration"].astype(float)
    )

    chunks = cut_into_chunks(
        annotated_data, params["chunk_size"], params["chunk_overlap"]
    )
    balanced_df = balance_chunks(chunks, params)

    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    balanced_df.groupby("label").size().to_csv(DATA_QUALITY_DIR / "chunks_per_label.csv")
    with open(DATA_QUALITY_DIR / "chunk_balanced_quality.json", "w") as f:
        json.dump(get_data_quality_metrics(balanced_df), f, indent=4)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(MANIFEST_PATH, index=False)

    n_files = balanced_df["audio_file_name"].nunique()
    print(f"{len(balanced_df)} chunks from {n_files} distinct files -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
