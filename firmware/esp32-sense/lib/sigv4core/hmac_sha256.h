// See sha256.h for why this uses plain (mangled) C++ linkage instead of
// extern "C" — an unmangled `hmac_sha256` collides with ESP-IDF's own
// libwpa_supplicant.a symbol of the same name.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace sigv4core {

// Writes the 32-byte HMAC-SHA256 digest of `data` under `key` into `out`.
void hmac_sha256(const uint8_t *key, size_t keylen, const uint8_t *data,
                  size_t datalen, uint8_t out[32]);

} // namespace sigv4core
