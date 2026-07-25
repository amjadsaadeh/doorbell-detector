"""Where the pipeline's audio comes from, and how a bare filename resolves.

Three sources feed the same flat namespace: labeled recordings
(data/audio), synthetic SNR mixes from the augmentation stage
(data/augmented_audio), and the external noise pools (data/noise/<pool>).
Filenames are unique across all three -- ESC-50 fold-id clips, DEMAND
<ENV>_ch01 and aug_* names don't collide with uploaded recordings -- so
downstream stages carry only `audio_file_name` and resolve it here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_FILE_BASE = Path("./data/audio")
AUGMENTED_AUDIO_FILE_BASE = Path("./data/augmented_audio")
NOISE_POOL_FILE_BASE = Path("./data/noise")

SAMPLING_RATE = 16000  # project-wide convention, see CLAUDE.md


def audio_search_bases() -> list[Path]:
    bases = [AUDIO_FILE_BASE, AUGMENTED_AUDIO_FILE_BASE]
    bases += sorted(d for d in NOISE_POOL_FILE_BASE.glob("*") if d.is_dir())
    return bases


def resolve_audio_path(audio_file_name: str) -> Path:
    """Real annotations point at data/audio; augmented (mixed) samples live
    under data/augmented_audio; external noise pool files under
    data/noise/<pool>."""
    for base in audio_search_bases():
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
        for wav_path in sorted(pool_path.glob("*.wav")):
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
