"""Ship a built firmware image to the board: MinIO upload + MQTT request.

Run through ../ota.sh, which builds the image and loads the credentials. The
board fetches "<firmware prefix>/<device>/<version>.bin" itself (see
src/ota.cpp); this script only stages it, asks for it, and waits on the
board's retained status topic for the outcome.
"""

import argparse
import hashlib
import os
import sys
import threading

import boto3
import paho.mqtt.client as mqtt

DEFAULT_FIRMWARE_PREFIX = "doorbell-detector/firmware"  # = kDefaultFirmwarePrefix
TOPIC_PREFIX = "doorbell/"  # = kMqttDeviceTopicPrefix


def upload(image: str, version: str, device: str) -> str:
    prefix = os.environ.get("ESP32_MINIO_FIRMWARE_PREFIX") or DEFAULT_FIRMWARE_PREFIX
    bucket, _, key_prefix = prefix.partition("/")
    key = "/".join(p for p in (key_prefix, device, f"{version}.bin") if p)
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{os.environ['ESP32_MINIO_ENDPOINT']}",  # host:port, as the board
        aws_access_key_id=os.environ["ESP32_MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["ESP32_MINIO_SECRET_KEY"],
        region_name=os.environ.get("ESP32_MINIO_REGION") or "us-east-1",
    )
    s3.upload_file(image, bucket, key)
    return f"s3://{bucket}/{key}"


def request_and_wait(version: str, md5: str, device: str, timeout: float) -> bool:
    ota_topic = f"{TOPIC_PREFIX}{device}/ota"
    status_topic = f"{TOPIC_PREFIX}{device}/status"
    done = threading.Event()
    outcome = {"ok": False}

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"mqtt: connect failed: {reason_code}", file=sys.stderr)
            done.set()
            return
        client.subscribe(status_topic)

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        # Subscribed before publishing, so no status about this request is missed.
        client.publish(ota_topic, f"{version} {md5}")
        print(f"requested {version} on {ota_topic}, waiting for {status_topic} ...")

    def on_message(client, userdata, msg):
        if msg.retain:
            return  # the status from before this request
        status = msg.payload.decode(errors="replace")
        print(f"board: {status}")
        fields = dict(f.split("=", 1) for f in status.split(" ") if "=" in f)
        state = fields.get("state")
        if fields.get("fw") == version and state == "running":
            outcome["ok"] = True
            done.set()
        elif state == "failed" or (state == "rolled-back" and fields.get("from") == version):
            done.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if os.environ.get("ESP32_MQTT_USERNAME"):
        client.username_pw_set(
            os.environ["ESP32_MQTT_USERNAME"], os.environ.get("ESP32_MQTT_PASSWORD")
        )
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect(os.environ["ESP32_MQTT_HOST"], int(os.environ.get("ESP32_MQTT_PORT") or 1883))
    client.loop_start()
    try:
        if not done.wait(timeout):
            print(f"no outcome within {timeout:.0f}s — is the board online?", file=sys.stderr)
    finally:
        client.loop_stop()
        client.disconnect()
    return outcome["ok"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="built firmware.bin")
    # A rollback only shows up after the board's 5 min confirmation timeout.
    parser.add_argument("--timeout", type=float, default=420.0)
    args = parser.parse_args()

    version = os.environ["ESP32_FW_VERSION"]
    device = os.environ["ESP32_DEVICE_ID"]
    with open(args.image, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    print(f"uploaded {upload(args.image, version, device)} (md5 {md5})")
    ok = request_and_wait(version, md5, device, args.timeout)
    print(f"{device} is running {version}" if ok else "update did not land")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
