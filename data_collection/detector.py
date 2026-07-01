"""Pattern-matching doorbell detector.

Opens the configured audio input device, loads a reference doorbell WAV
template (resampling to 16 kHz if needed), and enters a continuous capture
loop.  On each chunk, peak absolute normalized cross-correlation is computed
against the template; when the score exceeds --threshold and the cooldown has
elapsed, a detection event fires.  Detections are published to an MQTT broker
(optional) and, when --save is active, a timestamped WAV clip is written to
disk in a background daemon thread.  A ring buffer of pre-trigger audio is
combined with post-trigger chunks to produce complete doorbell clips suitable
for growing the training dataset.

Usage:
    python3 detector.py --template path/to/doorbell.wav [--device-name NAME]
                        [--chunk-size-ms MS] [--threshold FLOAT]
                        [--cooldown-seconds FLOAT]
                        [--save] [--save-dir DIR]
                        [--buffer-minutes FLOAT] [--post-trigger-seconds FLOAT]
"""

import argparse
import collections
import datetime
import logging
import os
import sys
import threading
import time
import wave

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from math import gcd
from pathlib import Path

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


def save_clip(frames: list, filepath: str) -> None:
    """Write PCM byte buffers to a WAV file. Runs safely in a daemon thread.

    Args:
        frames: List of raw PCM byte buffers (int16, mono, RATE Hz).
        filepath: Destination file path (must be writable).
    """
    try:
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)   # int16 = 2 bytes per sample
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        total_bytes = len(b"".join(frames))
        total_samples = total_bytes // 2  # int16 = 2 bytes per sample
        total_seconds = total_samples / RATE
        log.info("Saved WAV clip to %s (%d chunks, %.2f s)", filepath, len(frames), total_seconds)
    except Exception as exc:
        log.error("Failed to write clip to %s: %s", filepath, exc)


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

    mqtt_group = parser.add_argument_group("MQTT notification")
    mqtt_group.add_argument(
        "--mqtt-host",
        type=str,
        default=None,
        help="MQTT broker hostname/IP. If omitted, MQTT notification is disabled.",
    )
    mqtt_group.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883).",
    )
    mqtt_group.add_argument(
        "--mqtt-detect-topic",
        type=str,
        default="doorbell/detected",
        help="MQTT topic to publish detections to (default: doorbell/detected).",
    )
    mqtt_group.add_argument(
        "--mqtt-username",
        type=str,
        default=None,
        help="MQTT broker username for password-based authentication.",
    )
    mqtt_group.add_argument(
        "--mqtt-password",
        type=str,
        default=None,
        help="MQTT broker password. Use with --mqtt-username.",
    )
    mqtt_group.add_argument(
        "--mqtt-tls",
        action="store_true",
        help="Enable TLS/SSL for the MQTT connection.",
    )
    mqtt_group.add_argument(
        "--mqtt-tls-ca",
        type=str,
        default=None,
        help="Path to CA certificate file for TLS verification.",
    )
    mqtt_group.add_argument(
        "--mqtt-tls-certfile",
        type=str,
        default=None,
        help="Path to client certificate file for mutual TLS auth.",
    )
    mqtt_group.add_argument(
        "--mqtt-tls-keyfile",
        type=str,
        default=None,
        help="Path to client private key file for mutual TLS auth.",
    )
    mqtt_group.add_argument(
        "--mqtt-tls-insecure",
        action="store_true",
        help="Disable TLS server certificate verification (insecure, for testing only).",
    )
    mqtt_group.add_argument(
        "--mqtt-trigger-topic",
        type=str,
        default=None,
        help="MQTT topic to subscribe to for manual recording trigger. Requires --save.",
    )
    mqtt_group.add_argument(
        "--mqtt-trigger-value",
        type=str,
        default=None,
        help="Expected payload to trigger recording. If omitted, any message on the topic triggers.",
    )

    save_group = parser.add_argument_group("Save clips")
    save_group.add_argument(
        "--save",
        action="store_true",
        help="Save a WAV clip to --save-dir on each detection.",
    )
    save_group.add_argument(
        "--save-dir",
        type=str,
        default="recordings",
        help="Directory for saved clips (default: recordings).",
    )
    save_group.add_argument(
        "--buffer-minutes",
        type=float,
        default=float(os.environ.get("BUFFER_SECONDS", 3)) / 60,
        help="Pre-trigger ring buffer size in minutes (default: from BUFFER_SECONDS env var in seconds / 60, or 3s = 0.05min).",
    )
    save_group.add_argument(
        "--post-trigger-seconds",
        type=float,
        default=float(os.environ.get("POST_TRIGGER_MINUTES", 0.1)) * 60,
        help="Post-trigger audio duration in seconds (default: from POST_TRIGGER_MINUTES env var in minutes * 60, or 0.1min = 6s).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# MQTT publisher
# ---------------------------------------------------------------------------

def setup_mqtt_publisher(args, trigger_event=None):
    """Connect to an MQTT broker for publishing detection events.

    Returns an mqtt.Client instance with loop_start() already running, ready
    for publish() calls.  Returns None if paho-mqtt is not installed or if the
    connection attempt fails (error is logged in both cases).

    If trigger_event is provided and args.mqtt_trigger_topic is set, also
    subscribes to that topic and sets trigger_event on each matching message.
    """
    if not MQTT_AVAILABLE:
        log.warning(
            "paho-mqtt not installed — MQTT notification disabled. "
            "Install with: pip install paho-mqtt"
        )
        return None

    client = mqtt.Client()

    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password or None)
        log.info("MQTT auth enabled for user '%s'", args.mqtt_username)

    if args.mqtt_tls:
        import ssl
        client.tls_set(
            ca_certs=args.mqtt_tls_ca,
            certfile=args.mqtt_tls_certfile,
            keyfile=args.mqtt_tls_keyfile,
            tls_version=ssl.PROTOCOL_TLS,
        )
        if args.mqtt_tls_insecure:
            client.tls_insecure_set(True)
            log.warning("MQTT TLS: server certificate verification disabled")
        log.info("MQTT TLS enabled")

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            log.info("MQTT connected to %s:%d", args.mqtt_host, args.mqtt_port)
            if args.mqtt_trigger_topic:
                client.subscribe(args.mqtt_trigger_topic)
                log.info("MQTT subscribed to trigger topic '%s'", args.mqtt_trigger_topic)
        else:
            log.error("MQTT connection failed (rc=%d)", rc)

    def on_message(client, userdata, msg):
        payload = msg.payload.decode(errors="replace").strip()
        if args.mqtt_trigger_value is None or payload == args.mqtt_trigger_value:
            log.info("MQTT trigger received on '%s': '%s'", msg.topic, payload)
            if trigger_event is not None:
                trigger_event.set()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
        client.loop_start()
        return client
    except Exception as exc:
        log.error(
            "Failed to connect to MQTT broker at %s:%d — %s",
            args.mqtt_host, args.mqtt_port, exc,
        )
        return None


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

    log.info("Loaded template from %s (%d samples, %.2f s)", args.template, len(template), len(template) / RATE)

    # -- Derived constants -------------------------------------------------
    chunk_size_s = args.chunk_size_ms / 1000.0
    chunk = int(RATE * chunk_size_s)

    # -- Save mode setup ---------------------------------------------------
    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        buffer_chunks = max(1, int(args.buffer_minutes * 60 / chunk_size_s))
        ring_buf: collections.deque = collections.deque(maxlen=buffer_chunks)
        log.info(
            "Save mode enabled — ring buffer %.1f min (%d chunks), post-trigger %.1f s",
            args.buffer_minutes, buffer_chunks, args.post_trigger_seconds,
        )
    else:
        ring_buf = None

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

    # -- MQTT publisher + optional trigger subscriber ----------------------
    trigger_event = threading.Event()
    mqtt_client = setup_mqtt_publisher(args, trigger_event) if args.mqtt_host else None

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
        "Capture loop started — chunk=%d samples (%.0f ms, %.2f s), Ctrl-C to stop",
        chunk,
        args.chunk_size_ms,
        chunk / RATE,
    )

    # -- Detection state ---------------------------------------------------
    last_detection_time: float = 0.0

    # -- Capture loop ------------------------------------------------------
    try:
        while True:
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                if ring_buf is not None:
                    ring_buf.append(data)

                if trigger_event.is_set():
                    trigger_event.clear()
                    if args.save and ring_buf is not None:
                        ts = datetime.datetime.now().strftime("doorbell_%Y%m%d_%H%M%S.wav")
                        filepath = str(save_dir / ts)
                        pre_frames = list(ring_buf)
                        post_chunks = max(1, int(args.post_trigger_seconds / chunk_size_s))
                        post_frames = []
                        for _ in range(post_chunks):
                            try:
                                post_frames.append(stream.read(chunk, exception_on_overflow=False))
                            except OSError:
                                log.warning("Buffer overflow during post-trigger collection")
                                break
                        threading.Thread(
                            target=save_clip,
                            args=(pre_frames + post_frames, filepath),
                            daemon=True,
                        ).start()
                        total_seconds = (len(pre_frames) + len(post_frames)) * chunk_size_s
                        log.info(
                            "Saving clip %s (%d pre + %d post chunks, %.2f s)",
                            filepath, len(pre_frames), len(post_frames), total_seconds,
                        )
                    else:
                        log.warning("MQTT trigger received but --save is not active; ignoring")
                    continue

                score = compute_score(data, template)
                now = time.monotonic()
                if score >= args.threshold and (now - last_detection_time) >= args.cooldown_seconds:
                    last_detection_time = now
                    log.info("Doorbell detected! score=%.4f", score)
                    if mqtt_client is not None:
                        mqtt_client.publish(args.mqtt_detect_topic, datetime.datetime.now().isoformat())
                    if args.save and ring_buf is not None:
                        ts = datetime.datetime.now().strftime("doorbell_%Y%m%d_%H%M%S.wav")
                        filepath = str(save_dir / ts)
                        pre_frames = list(ring_buf)
                        post_chunks = max(1, int(args.post_trigger_seconds / chunk_size_s))
                        post_frames = []
                        for _ in range(post_chunks):
                            try:
                                post_frames.append(stream.read(chunk, exception_on_overflow=False))
                            except OSError:
                                log.warning("Buffer overflow during post-trigger collection")
                                break
                        threading.Thread(
                            target=save_clip,
                            args=(pre_frames + post_frames, filepath),
                            daemon=True,
                        ).start()
                        total_seconds = (len(pre_frames) + len(post_frames)) * chunk_size_s
                        log.info(
                            "Saving clip %s (%d pre + %d post chunks, %.2f s)",
                            filepath, len(pre_frames), len(post_frames), total_seconds,
                        )
            except OSError:
                log.warning("Buffer overflow — stream read error, continuing")
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
