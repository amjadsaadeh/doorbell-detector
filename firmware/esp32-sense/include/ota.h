// Over-the-air updates, pulled from MinIO on an MQTT request.
//
// A request on "doorbell/<DEVICE_ID>/ota" carries "<version> <md5hex>"; the
// image is fetched from "<firmware prefix>/<DEVICE_ID>/<version>.bin" with the
// same SigV4 signing the recording upload uses, written to the inactive OTA
// slot, MD5-checked, and booted.
//
// Rollback: a freshly OTA'd image boots as PENDING_VERIFY (see
// verifyRollbackLater() in ota.cpp) and is only marked valid once it reaches
// the MQTT broker — i.e. once it has proven it can receive the next update.
// If it doesn't within kOtaConfirmTimeoutMs, or resets before then, the
// bootloader boots the previous image, which reports "state=rolled-back".
//
// Every state change is published retained to "doorbell/<DEVICE_ID>/status"
// as "fw=<FW_VERSION> state=<state> [key=value ...]" — once the board is off
// USB, that topic is the only place to see what it is doing.
#pragma once

#include <cstddef>

namespace ota {

// Call once from setup(), before networking starts.
void begin();

// MQTT callback for the OTA topic. Only records the request; the download
// runs from tick().
void request(const char *payload, size_t length);

// Call from networkTask. Confirms a pending image once MQTT is up, rolls it
// back on timeout, and runs a requested update when no clip is recording.
void tick();

// For fatal init paths that halt instead of resetting: a halted new image
// would never reach its confirmation timeout, so give the slot back now.
// No-op for a confirmed image.
void roll_back_if_unconfirmed();

// Called by mqtt_client after each (re)connect: publishes the current status.
void on_mqtt_connected();

} // namespace ota
