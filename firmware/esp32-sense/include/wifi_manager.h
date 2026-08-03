#pragma once

namespace wifi_manager {

// Starts the (async) connection attempt.
void begin();

bool is_connected();

// Kicks off a reconnect if not currently connected. Call periodically from
// networkTask; cheap no-op when already connected.
void maybe_reconnect();

} // namespace wifi_manager
