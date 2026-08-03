// Portable SHA-256 (public-domain-style implementation), vendored so sigv4.cpp
// builds identically on the ESP32 Arduino target and the native test target
// without depending on mbedtls being present on the host.
//
// Deliberately namespaced with ordinary C++ (mangled) linkage rather than
// extern "C": ESP-IDF's bundled libwpa_supplicant.a defines its own C-linkage
// `hmac_sha256` symbol, and giving ours the same unmangled name caused a
// "multiple definition" link error against the real hardware target. Mangled
// names sidestep that collision (and any other clash with a bundled static
// lib) regardless of naming coincidences.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace sigv4core {

typedef struct {
  uint8_t data[64];
  uint32_t datalen;
  unsigned long long bitlen;
  uint32_t state[8];
} SHA256_CTX;

void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len);
void sha256_final(SHA256_CTX *ctx, uint8_t hash[32]);

} // namespace sigv4core
