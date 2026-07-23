import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
import yaml
import tqdm
import functools
from multiprocessing import Pool


AUDIO_DATA_PATH = Path("./data/audio")
AUGMENTED_AUDIO_DATA_PATH = Path("./data/augmented_audio")


def extract_mfcc_features(
    file_path: Path | str,
    n_mfcc: int = 25,
    n_fft=2048,
) -> np.ndarray:
    # Extract MFCC features
    audio = AudioSegment.from_wav(file_path)
    info = mediainfo(file_path)
    # downmix to mono; stereo files would double the samples (interleaved)
    # and break the frames-per-second assumption in draw_data.py
    audio = audio.set_channels(1)
    audio_segment = np.array(audio.get_array_of_samples(), dtype=np.float32)
    mfccs = librosa.feature.mfcc(
        y=audio_segment, sr=int(info["sample_rate"]), n_mfcc=n_mfcc, n_fft=n_fft
    )

    return mfccs

    
def process_single_file(audio_file, output_path, params):
    mfccs = extract_mfcc_features(
        audio_file,
        params["feature_extraction"]["n_mfcc"],
        params["feature_extraction"]["n_fft"]
    )
    # Save MFCC features with same name but .npy extension
    output_file = output_path / (audio_file.stem + '.npy')
    np.save(output_file, mfccs)


def process_audio_data(
    params, input_path: Path, output_path: Path, audio_base_path: Path = AUDIO_DATA_PATH
):
    audio_data = input_path.glob("*.wav")

    # Convert audio files to chunks for feature extraction
    # Get list of all audio files first
    audio_files = list(audio_data)

    output_path.mkdir(parents=True, exist_ok=True)

    process_single_file_partial = functools.partial(
        process_single_file, output_path=output_path, params=params
    )

    # Use multiprocessing with tqdm progress bar
    with Pool() as pool:
        list(tqdm.tqdm(
            pool.imap(process_single_file_partial, audio_files),
            total=len(audio_files),
            desc="Extracting MFCC features"
        ))

if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    process_audio_data(
        params, AUDIO_DATA_PATH, Path("./data/mfcc_data")
    )
    # Augmented (mixed) chunks land in the same output dir; filenames are
    # unique across both sources so there's no collision.
    process_audio_data(
        params, AUGMENTED_AUDIO_DATA_PATH, Path("./data/mfcc_data")
    )
    # External noise pool files too (used as extra background chunks in
    # draw_data.py). Also flat: pool naming schemes (ESC-50 fold-id clips,
    # DEMAND <ENV>_ch01) don't collide with recordings or aug_* files.
    for pool_path in params["augmentation"]["external_noise_pools"]:
        process_audio_data(params, Path(pool_path), Path("./data/mfcc_data"))
