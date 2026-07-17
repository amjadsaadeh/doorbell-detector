import os
import argparse
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
import tqdm


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Splits a canonical s3://bucket/key URI into bucket and key.

    Args:
        uri (str): s3://bucket/key URI

    Returns:
        tuple[str, str]: bucket name and object key

    Raises:
        ValueError: if the URI is not a complete s3 URI
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Not a valid s3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def parse_args():
    parser = argparse.ArgumentParser(description="Download audio files.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Target directory for downloaded audio files",
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="data/annotations_per_line.csv",
        help="CSV file with annotations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir = Path(args.target_dir)
    annotations_file = Path(args.annotations_file)

    # Load the data (one row per annotation, so multiple rows per audio file)
    data = pd.read_csv(annotations_file)
    files = data.drop_duplicates(subset=["audio_file_name"])

    target_dir.mkdir(parents=True, exist_ok=True)

    # AWS_ENDPOINT_URL allows pointing at S3-compatible storage (e.g. MinIO);
    # credentials come from the default boto3 chain (AWS_ACCESS_KEY_ID etc.)
    s3_client = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))

    # Download the audio files
    for i, row in tqdm.tqdm(
        files.iterrows(), desc="Downloading audio files", total=len(files.index)
    ):
        audio_file_target_path = target_dir / row["audio_file_name"]

        if audio_file_target_path.exists():
            continue

        bucket, key = parse_s3_uri(row["remote_audio_path"])
        # download to a temp name first so an interrupted transfer never
        # leaves a truncated file that the exists-check above would skip
        partial_path = audio_file_target_path.with_suffix(".part")
        s3_client.download_file(bucket, key, str(partial_path))
        partial_path.rename(audio_file_target_path)

    # The DVC output is persisted across stage runs (persist: true), so files
    # whose tasks were removed or renamed in Label Studio must be pruned here
    expected_files = set(files["audio_file_name"])
    for existing_file in target_dir.iterdir():
        if existing_file.name not in expected_files:
            existing_file.unlink()
