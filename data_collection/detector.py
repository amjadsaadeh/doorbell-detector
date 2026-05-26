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
import logging
import sys
import wave

import librosa
import numpy as np
import pyaudio

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

    Uses librosa.load() with sr=RATE so the audio is automatically resampled
    to 16 kHz if the source sample rate differs.  Returns values normalised to
    the float32 range [-1.0, 1.0].

    Raises:
        FileNotFoundError: if *path* does not exist.
        Exception: for any other librosa / soundfile load error.
    """
    audio, _sr = librosa.load(path, sr=RATE, mono=True)
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

    # -- Capture loop (Phase 1 stub: reads and discards) -------------------
    try:
        while True:
            try:
                _data = stream.read(chunk, exception_on_overflow=False)
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
