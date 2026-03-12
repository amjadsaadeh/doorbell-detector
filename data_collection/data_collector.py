"""Continuous audio collector with ring buffer and multiple trigger sources.

Runs indefinitely, keeping a rolling ring buffer of the last N minutes of
audio.  A recording is saved to disk whenever any of the following triggers
fires:

  1. XGBoost model probability exceeds --threshold  (ML detection)
  2. An MQTT message arrives on --mqtt-topic         (smart-home / phone)
  3. A physical GPIO button is pressed on --gpio-pin (Pi button)

Pass --manual-trigger-only to disable ML inference entirely and rely solely
on MQTT / GPIO triggers.  In this mode --model-path is not required.

This script is self-contained and does *not* depend on the parent repo's
params.yaml.  All tunable values are exposed as CLI flags so the systemd
unit can configure them via an EnvironmentFile.
"""

import argparse
import collections
import datetime
import logging
import os
import threading
import wave
from pathlib import Path

import librosa
import numpy as np
import pyaudio
import xgboost as xgb

# ---------------------------------------------------------------------------
# Optional dependencies – absent on non-Pi dev machines
# ---------------------------------------------------------------------------
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

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
# Audio helpers
# ---------------------------------------------------------------------------

def save_audio(buffer_snapshot: list, stream, filename: Path,
               chunk: int, post_trigger_seconds: float) -> None:
    """Write ring-buffer contents + post-trigger audio to a WAV file."""
    frames = list(buffer_snapshot)
    extra_chunks = int(RATE * post_trigger_seconds / chunk)
    for _ in range(extra_chunks):
        frames.append(stream.read(chunk, exception_on_overflow=False))

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for Int16
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    window_size_s = chunk / RATE
    log.info("Saved %d chunks (%.1f s) to %s",
             len(frames), len(frames) * window_size_s, filename)


# ---------------------------------------------------------------------------
# Trigger sources
# ---------------------------------------------------------------------------

def setup_mqtt(args, trigger_event: threading.Event):
    """Connect to the MQTT broker and fire trigger_event on matching messages."""
    if not MQTT_AVAILABLE:
        log.warning("paho-mqtt not installed – MQTT trigger disabled. "
                    "Install with: pip install paho-mqtt")
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
            client.subscribe(args.mqtt_topic)
            log.info("MQTT connected to %s:%d, subscribed to '%s'",
                     args.mqtt_host, args.mqtt_port, args.mqtt_topic)
        else:
            log.error("MQTT connection failed (rc=%d)", rc)

    def on_message(client, userdata, msg):
        payload = msg.payload.decode(errors="replace").strip()
        if args.mqtt_trigger_value is None or payload == args.mqtt_trigger_value:
            log.info("MQTT trigger received on '%s': '%s'", msg.topic, payload)
            trigger_event.set()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
        client.loop_start()
        return client
    except Exception as exc:
        log.error("Failed to connect to MQTT broker at %s:%d – %s",
                  args.mqtt_host, args.mqtt_port, exc)
        return None


def setup_gpio(pin: int, trigger_event: threading.Event) -> None:
    """Configure a GPIO input pin to fire trigger_event on button press."""
    if not GPIO_AVAILABLE:
        log.warning("RPi.GPIO not installed – GPIO trigger disabled.")
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _callback(channel):
        log.info("GPIO trigger on pin %d", channel)
        trigger_event.set()

    GPIO.add_event_detect(pin, GPIO.FALLING, callback=_callback, bouncetime=300)
    log.info("GPIO trigger configured on BCM pin %d", pin)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous audio monitor with ring buffer and multiple trigger sources"
    )

    # Manual-only mode
    parser.add_argument(
        "--manual-trigger-only", action="store_true",
        help="Disable ML inference and rely solely on MQTT / GPIO triggers. "
             "When set, --model-path is not required."
    )

    # Paths
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to the XGBoost model file (.json). "
             "Required unless --manual-trigger-only is set."
    )
    parser.add_argument(
        "--save-dir", type=str, default="recordings",
        help="Directory to save triggered recordings (default: recordings)"
    )

    # Ring buffer & recording
    parser.add_argument(
        "--buffer-minutes", type=float, default=5.0,
        help="Ring buffer size in minutes (default: 5.0)"
    )
    parser.add_argument(
        "--post-trigger-seconds", type=float, default=7.0,
        help="Seconds of audio to record after a trigger fires (default: 7.0)"
    )

    # Audio / feature-extraction params – must match how the model was trained
    parser.add_argument(
        "--chunk-size-ms", type=int, default=500,
        help="Audio window size in milliseconds (default: 500). "
             "Must match the value used during model training."
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=13,
        help="Number of MFCC coefficients (default: 13). "
             "Must match the value used during model training."
    )
    parser.add_argument(
        "--n-fft", type=int, default=512,
        help="FFT window size for MFCC extraction (default: 512). "
             "Must match the value used during model training."
    )

    # ML trigger
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="XGBoost prediction probability threshold (default: 0.5)"
    )
    parser.add_argument(
        "--skip-chunks", type=int, default=10,
        help="Chunks to skip after startup or stream reset (default: 10)"
    )

    # MQTT trigger
    parser.add_argument(
        "--mqtt-host", type=str, default=None,
        help="MQTT broker hostname/IP. If omitted, MQTT trigger is disabled."
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--mqtt-topic", type=str, default="doorbell/trigger",
        help="MQTT topic to subscribe to (default: doorbell/trigger)"
    )
    parser.add_argument(
        "--mqtt-trigger-value", type=str, default=None,
        help="Expected MQTT payload to trigger a save. "
             "If omitted, any message on the topic triggers."
    )

    # MQTT authentication
    parser.add_argument(
        "--mqtt-username", type=str, default=None,
        help="MQTT broker username for password-based authentication."
    )
    parser.add_argument(
        "--mqtt-password", type=str, default=None,
        help="MQTT broker password. Use together with --mqtt-username."
    )

    # MQTT TLS
    parser.add_argument(
        "--mqtt-tls", action="store_true",
        help="Enable TLS/SSL for the MQTT connection."
    )
    parser.add_argument(
        "--mqtt-tls-ca", type=str, default=None,
        help="Path to the CA certificate file for TLS verification."
    )
    parser.add_argument(
        "--mqtt-tls-certfile", type=str, default=None,
        help="Path to the client certificate file for mutual TLS auth."
    )
    parser.add_argument(
        "--mqtt-tls-keyfile", type=str, default=None,
        help="Path to the client private key file for mutual TLS auth."
    )
    parser.add_argument(
        "--mqtt-tls-insecure", action="store_true",
        help="Disable TLS server certificate verification (insecure, for testing only)."
    )

    # GPIO trigger
    parser.add_argument(
        "--gpio-pin", type=int, default=None,
        help="BCM GPIO pin number for a physical button trigger. "
             "If omitted, GPIO trigger is disabled."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.manual_trigger_only and args.model_path is None:
        import sys
        print("error: --model-path is required unless --manual-trigger-only is set", file=sys.stderr)
        sys.exit(2)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Derived audio constants
    window_size_s = args.chunk_size_ms / 1000.0
    chunk = int(RATE * window_size_s)

    # Ring buffer sized for --buffer-minutes of audio
    buffer_chunks = max(1, int(args.buffer_minutes * 60 / window_size_s))
    audio_buffer: collections.deque = collections.deque(maxlen=buffer_chunks)
    log.info("Ring buffer: %.1f min / %d chunks (%.0f s)",
             args.buffer_minutes, buffer_chunks, buffer_chunks * window_size_s)

    # Shared event for out-of-band triggers
    trigger_event = threading.Event()

    # Optional trigger sources
    mqtt_client = None
    if args.mqtt_host:
        mqtt_client = setup_mqtt(args, trigger_event)

    if args.gpio_pin is not None:
        setup_gpio(args.gpio_pin, trigger_event)

    # Audio device
    p = pyaudio.PyAudio()

    device_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "seeed-2mic-voicecard" in info["name"] and info["maxInputChannels"] > 0:
            device_idx = i
            break

    if device_idx is None:
        log.error("Microphone device 'seeed-2mic-voicecard' not found")
        p.terminate()
        return

    def open_stream():
        return p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=device_idx,
        )

    stream = open_stream()

    # XGBoost model (not needed in manual-trigger-only mode)
    model = None
    if args.manual_trigger_only:
        log.info("Manual-trigger-only mode – ML inference disabled")
    else:
        if not os.path.exists(args.model_path):
            log.error("Model file not found: %s", args.model_path)
            stream.close()
            p.terminate()
            return
        model = xgb.Booster()
        model.load_model(args.model_path)
        log.info("Loaded XGBoost model from %s", args.model_path)
    log.info("Monitoring audio – saving to %s/", save_dir)

    skip_remaining = args.skip_chunks
    inference_counter = 0  # run ML only every 3rd chunk to reduce CPU load
    is_saving = False       # prevent overlapping saves

    try:
        while True:
            try:
                data = stream.read(chunk, exception_on_overflow=False)

                # Warm-up: skip initial chunks so the mic settles
                if skip_remaining > 0:
                    skip_remaining -= 1
                    continue

                audio_buffer.append(data)

                # ---- Out-of-band trigger (MQTT / GPIO) ----
                if trigger_event.is_set() and not is_saving:
                    trigger_event.clear()
                    is_saving = True
                    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = save_dir / f"recording_{ts}.wav"
                    log.info("[%s] Manual trigger – saving to %s", ts, filename)
                    save_audio(list(audio_buffer), stream, filename,
                               chunk, args.post_trigger_seconds)
                    is_saving = False
                    continue

                # ---- ML inference (throttled to every 3rd chunk) ----
                if args.manual_trigger_only:
                    continue

                inference_counter += 1
                if inference_counter % 3 != 0:
                    continue

                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                mfccs = librosa.feature.mfcc(
                    y=audio_np, sr=RATE, n_mfcc=args.n_mfcc, n_fft=args.n_fft
                )
                X = xgb.DMatrix(mfccs.flatten().reshape(1, -1))
                prob = model.predict(X)[0]

                if prob > args.threshold and not is_saving:
                    is_saving = True
                    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = save_dir / f"recording_{ts}.wav"
                    log.info("[%s] ML trigger (p=%.3f > %.2f) – saving to %s",
                             ts, prob, args.threshold, filename)
                    save_audio(list(audio_buffer), stream, filename,
                               chunk, args.post_trigger_seconds)
                    is_saving = False

            except OSError:
                log.warning("Buffer overflow – reopening stream")
                try:
                    stream.close()
                except Exception:
                    pass
                stream = open_stream()
                skip_remaining = args.skip_chunks
                inference_counter = 0

    except KeyboardInterrupt:
        log.info("Interrupted – shutting down")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if GPIO_AVAILABLE and args.gpio_pin is not None:
            GPIO.cleanup()


if __name__ == "__main__":
    main()
