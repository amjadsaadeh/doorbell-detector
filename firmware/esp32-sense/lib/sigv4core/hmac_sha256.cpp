#include "hmac_sha256.h"

#include <string.h>

#include "sha256.h"

namespace sigv4core {

void hmac_sha256(const uint8_t *key, size_t keylen, const uint8_t *data,
                  size_t datalen, uint8_t out[32]) {
  uint8_t key_block[64] = {0};

  if (keylen > 64) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, key, keylen);
    sha256_final(&ctx, key_block); // digest is 32 bytes; rest stays zero-padded
  } else {
    memcpy(key_block, key, keylen);
  }

  uint8_t k_ipad[64];
  uint8_t k_opad[64];
  for (int i = 0; i < 64; ++i) {
    k_ipad[i] = key_block[i] ^ 0x36;
    k_opad[i] = key_block[i] ^ 0x5c;
  }

  uint8_t inner_hash[32];
  SHA256_CTX ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, k_ipad, 64);
  sha256_update(&ctx, data, datalen);
  sha256_final(&ctx, inner_hash);

  sha256_init(&ctx);
  sha256_update(&ctx, k_opad, 64);
  sha256_update(&ctx, inner_hash, 32);
  sha256_final(&ctx, out);
}

} // namespace sigv4core
