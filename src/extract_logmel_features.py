"""Log-mel filterbank energies, sized for microcontroller inference.

This is the ESP32-S3 branch's feature extractor. Unlike
extract_stft_features.py (129 bins x 250 frames per 2 s chunk) the geometry
here is deliberately tiny -- 40 mel bins x 100 frames -- because the first
conv layer's activation, not the parameter count, is what has to fit in the
MCU's SRAM:

    129 x 250 x 32  =  1.03 MB int8   -> PSRAM only, ~45 M MACs
     40 x 100 x 32  =  0.13 MB int8   -> internal SRAM, ~5.7 M MACs

The log is applied *here* rather than in train_cnn.py (params.yaml sets
model.log_compress: false on this branch). The .npy files are therefore the
exact tensor the firmware's C frontend has to reproduce, which turns
train/serve parity into a single comparable artifact -- see the golden-vector
test in the firmware phases.

Audio is kept at int16 scale (not normalized to [-1, 1]) to match what I2S
hands the firmware directly.
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
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 320,
    fmin: int = 50,
    fmax: int = 8000,
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
