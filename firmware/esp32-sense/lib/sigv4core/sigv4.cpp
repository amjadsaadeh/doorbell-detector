#include "sigv4.h"

#include <algorithm>

#include "hmac_sha256.h"
#include "sha256.h"

namespace sigv4 {

namespace {

std::string to_hex(const uint8_t *bytes, size_t len) {
  static const char *digits = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out[i * 2] = digits[bytes[i] >> 4];
    out[i * 2 + 1] = digits[bytes[i] & 0x0f];
  }
  return out;
}

void hmac(const uint8_t *key, size_t keylen, const std::string &data,
          uint8_t out[32]) {
  sigv4core::hmac_sha256(key, keylen,
                          reinterpret_cast<const uint8_t *>(data.data()),
                          data.size(), out);
}

} // namespace

std::string sha256_hex(const std::string &data) {
  sigv4core::SHA256_CTX ctx;
  uint8_t hash[32];
  sigv4core::sha256_init(&ctx);
  sigv4core::sha256_update(&ctx, reinterpret_cast<const uint8_t *>(data.data()),
                            data.size());
  sigv4core::sha256_final(&ctx, hash);
  return to_hex(hash, 32);
}

std::string canonical_request(const std::string &method,
                               const std::string &canonical_uri,
                               const std::string &canonical_query_string,
                               std::vector<Header> headers,
                               const std::string &payload_hash,
                               std::string &signed_headers_out) {
  std::sort(headers.begin(), headers.end(),
            [](const Header &a, const Header &b) { return a.name < b.name; });

  std::string canonical_headers;
  signed_headers_out.clear();
  for (size_t i = 0; i < headers.size(); ++i) {
    canonical_headers += headers[i].name + ":" + headers[i].value + "\n";
    if (i > 0)
      signed_headers_out += ";";
    signed_headers_out += headers[i].name;
  }

  return method + "\n" + canonical_uri + "\n" + canonical_query_string + "\n" +
         canonical_headers + "\n" + signed_headers_out + "\n" + payload_hash;
}

std::string string_to_sign(const std::string &amz_date,
                            const std::string &date_stamp,
                            const std::string &region,
                            const std::string &service,
                            const std::string &canonical_request_str) {
  const std::string scope =
      date_stamp + "/" + region + "/" + service + "/aws4_request";
  return "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" +
         sha256_hex(canonical_request_str);
}

void derive_signing_key(const std::string &secret_key,
                         const std::string &date_stamp,
                         const std::string &region, const std::string &service,
                         uint8_t out_key[32]) {
  uint8_t k_date[32];
  uint8_t k_region[32];
  uint8_t k_service[32];

  const std::string seed = "AWS4" + secret_key;
  hmac(reinterpret_cast<const uint8_t *>(seed.data()), seed.size(), date_stamp,
       k_date);
  hmac(k_date, 32, region, k_region);
  hmac(k_region, 32, service, k_service);
  hmac(k_service, 32, "aws4_request", out_key);
}

std::string sign_hex(const uint8_t signing_key[32],
                      const std::string &string_to_sign_str) {
  uint8_t signature[32];
  hmac(signing_key, 32, string_to_sign_str, signature);
  return to_hex(signature, 32);
}

std::string authorization_header(const std::string &access_key,
                                  const std::string &date_stamp,
                                  const std::string &region,
                                  const std::string &service,
                                  const std::string &signed_headers,
                                  const std::string &signature_hex) {
  const std::string scope =
      date_stamp + "/" + region + "/" + service + "/aws4_request";
  return "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + scope +
         ", SignedHeaders=" + signed_headers + ", Signature=" + signature_hex;
}

} // namespace sigv4
