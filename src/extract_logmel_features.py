"""Log-mel filterbank energies — the MFCC pipeline minus the final DCT.

MFCCs are log-mel energies run through a DCT, which decorrelates the bins so
a tree/GMM sees compact, roughly independent features. A CNN doesn't need
that: it wants locally correlated structure along the frequency axis, which
the DCT destroys. This branch swaps the representation and keeps everything
else (geometry, chunking, model, split) identical to cnn-mfcc so the two
MLflow runs are directly comparable.

The log is applied *here* rather than in train_cnn.py — as on the MFCC
branch, params.yaml therefore sets model.log_compress: false. Audio is kept
at int16 scale (not normalized to [-1, 1]) to match the other extractors.
"""

from pathlib import Path
import functools
from multiprocessing import Pool

import librosa
import numpy as np
import tqdm
import yaml
from pydub import AudioSegment
from pydub.utils import mediainfo

AUDIO_DATA_PATH = Path("./data/audio")
AUGMENTED_AUDIO_DATA_PATH = Path("./data/augmented_audio")


def extract_logmel_features(
    file_path: Path | str,
    n_mels: int = 13,
    n_fft: int = 512,
    hop_length: int = 512,
    fmin: int = 0,
    fmax: int | None = None,
    log_offset: float = 1e-6,
) -> np.ndarray:
    """Return log-mel energies of shape (n_mels, n_frames) as float32."""
    audio = AudioSegment.from_wav(file_path)
    info = mediainfo(file_path)
    # downmix to mono; stereo files would double the samples (interleaved)
    # and break the frames-per-second assumption in draw_data.py
    audio = audio.set_channels(1)
    audio_segment = np.array(audio.get_array_of_samples(), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=int(info["sample_rate"]),
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    # log_offset keeps digital-silence frames finite; librosa's power_to_db
    # would additionally clip to top_db below the per-file peak, which is a
    # per-file operation we don't want ahead of a global normalization.
    return np.log(mel + log_offset).astype(np.float32)


def process_single_file(audio_file, output_path, params):
    feature_params = params["feature_extraction"]
    logmel = extract_logmel_features(
        audio_file,
        n_mels=feature_params["n_mels"],
        n_fft=feature_params["n_fft"],
        hop_length=feature_params["hop_length"],
        fmin=feature_params["fmin"],
        fmax=feature_params["fmax"],
        log_offset=feature_params["log_offset"],
    )
    # Save features with same name but .npy extension
    output_file = output_path / (audio_file.stem + ".npy")
    np.save(output_file, logmel)


def process_audio_data(params, input_path: Path, output_path: Path):
    audio_files = list(input_path.glob("*.wav"))

    output_path.mkdir(parents=True, exist_ok=True)

    process_single_file_partial = functools.partial(
        process_single_file, output_path=output_path, params=params
    )

    with Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(process_single_file_partial, audio_files),
                total=len(audio_files),
                desc=f"Extracting log-mel features ({input_path.name})",
            )
        )


if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    output_path = Path(params["feature_extraction"]["features_dir"])

    process_audio_data(params, AUDIO_DATA_PATH, output_path)
    # Augmented (mixed) chunks land in the same output dir; filenames are
    # unique across both sources so there's no collision.
    process_audio_data(params, AUGMENTED_AUDIO_DATA_PATH, output_path)
    # External noise pool files too (used as extra background chunks in
    # draw_data.py). Also flat: pool naming schemes (ESC-50 fold-id clips,
    # DEMAND <ENV>_ch01) don't collide with recordings or aug_* files.
    for pool_path in params["augmentation"]["external_noise_pools"]:
        process_audio_data(params, Path(pool_path), output_path)
