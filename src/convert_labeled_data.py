"""
This script is for preparation of meta data received by label studio.
It consumes the CSV from label studio (each line is a file/task) and converts it into a CSV whith each sample as line (sample = labeled interval).
"""

import base64
import binascii
import json
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlparse

import pandas as pd


def normalize_audio_uri(uri: str) -> str:
    """Normalizes a Label Studio audio reference to a canonical s3://bucket/key URI.

    Label Studio with S3 source storage emits one of several shapes depending on
    the storage settings (presigned URLs on/off). Presigned URLs change on every
    export, so they must be normalized or the DVC pipeline would consider the
    annotation data changed on every fetch.

    Supported shapes:
    - s3://bucket/key (returned as-is)
    - presigned/plain https URL, path-style (https://endpoint/bucket/key?X-Amz-...)
      or virtual-hosted AWS style (https://bucket.s3.region.amazonaws.com/key)
    - Label Studio resolver path with a fileuri query param (plain or base64)

    Args:
        uri (str): audio reference from the Label Studio export

    Returns:
        str: canonical s3://bucket/key URI

    Raises:
        ValueError: if the reference has none of the known shapes
    """
    parsed = urlparse(uri)

    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Incomplete s3 URI: {uri}")
        return f"s3://{parsed.netloc}{parsed.path}"

    if parsed.scheme in ("http", "https"):
        host_labels = parsed.netloc.split(".")
        path = PurePosixPath(parsed.path)
        if len(host_labels) > 2 and host_labels[1].startswith("s3"):
            # virtual-hosted style: bucket is the first host label
            bucket = host_labels[0]
            key = str(path).lstrip("/")
        else:
            # path-style (MinIO and friends): bucket is the first path segment
            parts = path.parts
            bucket = parts[1] if len(parts) > 1 else ""
            key = "/".join(parts[2:])
        if not bucket or not key:
            raise ValueError(f"Cannot extract bucket/key from URL: {uri}")
        return f"s3://{bucket}/{key}"

    # Label Studio storage resolver path, e.g. /data/s3/?fileuri=...
    fileuri = parse_qs(parsed.query).get("fileuri", [None])[0]
    if fileuri is not None:
        if not fileuri.startswith("s3://"):
            try:
                fileuri = base64.b64decode(fileuri, validate=True).decode()
            except (binascii.Error, UnicodeDecodeError):
                raise ValueError(f"Cannot decode fileuri in: {uri}")
        return normalize_audio_uri(fileuri)

    raise ValueError(f"Unrecognized audio reference: {uri}")


def derive_audio_file_name(s3_uri: str) -> str:
    """Derives the plain file name from a normalized s3://bucket/key URI.

    Args:
        s3_uri (str): canonical s3://bucket/key URI

    Returns:
        str: file name (last path component)
    """
    return PurePosixPath(urlparse(s3_uri).path).name


def annotation_to_sample_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Coverts a DataFrame with soundfile per row into an annotation per row format.

    Args:
        df (pd.DataFrame): DataFrame with soundfile per row format

    Returns:
        pd.DataFrame: Dataframe with annotation per row format
    """

    result = {
        "annotation_id": list(),
        "file_id": list(),
        "start": list(),
        "end": list(),
        "label": list(),
        "audio_file_name": list(),
        "remote_audio_path": list()
    }

    for row in df.iterrows():
        row = row[1]
        annotation_data = json.loads(row.label)
        audio_uri = normalize_audio_uri(row.audio)

        for annotation in annotation_data:
            for label in annotation["labels"]:
                result["label"].append(label)
                result["start"].append(annotation["start"])
                result["end"].append(annotation["end"])
                result["file_id"].append(row.id)
                result["remote_audio_path"].append(audio_uri)
                result["audio_file_name"].append(derive_audio_file_name(audio_uri))
                result["annotation_id"].append(row.annotation_id)

    return pd.DataFrame(result)


if __name__ == "__main__":

    data_file_path = "./data/labeled_data.csv"
    target_file_path = "./data/annotation_per_row_data.csv"

    df = pd.read_csv(data_file_path)
    converted_df = annotation_to_sample_per_row(df)
    converted_df.to_csv(target_file_path)
