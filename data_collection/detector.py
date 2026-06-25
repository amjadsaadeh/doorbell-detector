"""Pattern-matching doorbell detector — Phase 1 skeleton.

Opens the configured audio input device, loads a reference doorbell WAV
template (resampling to 16 kHz if needed), and enters a continuous capture
loop that reads chunks from the microphone.  Phase 1 is a runnable skeleton:
the capture loop reads audio and discards it.  Phase 2 will extend this with
cross-correlation detection, threshold/cooldown logic, and MQTT notification.

Usage:
    python3 detector.py --template path/to/doorbell.wav [--device-name NAME]
                        [--chunk-size-ms MS]
"""

import argparse
import datetime
import logging
import sys
import time
from math import gcd

import numpy as np
import pyaudio
import soundfile as sf
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Audio constants — project-wide contract; do NOT change without updating all
# scripts in this repository.
# ---------------------------------------------------------------------------
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 16000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_template(path: str) -> np.ndarray:
    """Load a reference doorbell WAV file and return it as a float32 array.

    Uses soundfile to read the WAV, then resamples to RATE with scipy if the
    source sample rate differs.  Returns values in float32 range [-1.0, 1.0].

    Raises:
        FileNotFoundError: if *path* does not exist.
        Exception: for any other soundfile load error.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != RATE:
        g = gcd(RATE, sr)
        audio = resample_poly(audio, RATE // g, sr // g).astype(np.float32)
    return audio


def find_device(p: pyaudio.PyAudio, device_name: str):
    """Search PyAudio devices for one whose name contains *device_name*.

    Args:
        p: An initialised PyAudio instance.
        device_name: Substring to search for in device info names.

    Returns:
        The integer device index if a matching input device is found,
        or None if no match exists.
    """
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if device_name in info["name"] and info["maxInputChannels"] > 0:
            return i
    return None


def compute_score(audio_bytes: bytes, template: np.ndarray) -> float:
    """Return the peak absolute normalized cross-correlation between the audio chunk and the template.

    The raw cross-correlation is normalized by the template energy so that a
    perfect amplitude match returns a score of 1.0 regardless of the template
    amplitude.  Silence (all-zero audio) always returns 0.0.

    Args:
        audio_bytes: Raw PCM bytes from PyAudio (16-bit signed integers, mono).
        template: Reference doorbell waveform as a float32 array at RATE Hz.

    Returns:
        A non-negative float.  A value > 0.5 indicates a strong match.
        Returns exactly 0.0 if the audio is silent or the template has zero energy.
    """
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    corr = np.correlate(audio, template, mode='full')
    template_energy = float(np.dot(template, template))
    if template_energy == 0.0:
        return 0.0
    return float(np.max(np.abs(corr)) / template_energy)


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the detector script."""
    parser = argparse.ArgumentParser(
        description="Pattern-matching doorbell detector using audio cross-correlation."
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to the reference doorbell WAV file.",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default="seeed-2mic-voicecard",
        help="Audio input device name substring to match (default: seeed-2mic-voicecard).",
    )
    parser.add_argument(
        "--chunk-size-ms",
        type=int,
        default=500,
        help="Audio capture chunk size in milliseconds (default: 500).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Cross-correlation score threshold for a detection to fire (default: 0.7).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=10.0,
        help="Minimum seconds between successive detections (default: 10.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, load template, open audio device, capture loop."""
    args = parse_args()

    # -- Template loading --------------------------------------------------
    try:
        template = load_template(args.template)
    except FileNotFoundError:
        log.error("Template file not found: %s", args.template)
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to load template '%s': %s", args.template, exc)
        sys.exit(1)

    log.info("Loaded template from %s (%d samples)", args.template, len(template))

    # -- Derived constants -------------------------------------------------
    chunk = int(RATE * args.chunk_size_ms / 1000)

    # -- PyAudio init + device discovery -----------------------------------
    p = pyaudio.PyAudio()
    device_idx = find_device(p, args.device_name)
    if device_idx is None:
        log.error("Audio device '%s' not found", args.device_name)
        p.terminate()
        sys.exit(1)

    log.info(
        "Using audio device %d: %s",
        device_idx,
        p.get_device_info_by_index(device_idx)["name"],
    )

    # -- Stream open -------------------------------------------------------
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=device_idx,
    )
    log.info(
        "Capture loop started — chunk=%d samples (%.0f ms), Ctrl-C to stop",
        chunk,
        args.chunk_size_ms,
    )

    # -- Detection state ---------------------------------------------------
    last_detection_time: float = 0.0

    # -- Capture loop ------------------------------------------------------
    try:
        while True:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                score = compute_score(data, template)
                now = time.monotonic()
                if score >= args.threshold and (now - last_detection_time) >= args.cooldown_seconds:
                    last_detection_time = now
                    log.info("Doorbell detected! score=%.4f", score)
            except OSError:
                log.warning("Buffer overflow — stream read error, continuing")
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
