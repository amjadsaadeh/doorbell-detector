// Pure AWS Signature Version 4 request signing.
//
// Deliberately free of any networking/Arduino dependency so it can be
// unit-tested on the host (`pio test -e native`) against the published AWS
// SigV4 test vectors, independent of hardware or a live MinIO instance.
// s3_uploader.cpp is the only caller and owns all networking/SD I/O.
#pragma once

#include <stdint.h>

#include <string>
#include <vector>

namespace sigv4 {

struct Header {
  std::string name; // must already be lowercase, no leading/trailing whitespace
  std::string value; // must already be trimmed
};

// Hex-encoded SHA-256 of `data`.
std::string sha256_hex(const std::string &data);

// Builds the canonical request (method\ncanonical_uri\ncanonical_query_string\n
// canonical_headers\nsigned_headers\npayload_hash) per the SigV4 spec.
// `headers` is sorted by name internally; `signed_headers_out` receives the
// semicolon-joined sorted header names used in both the canonical request and
// the final Authorization header.
std::string canonical_request(const std::string &method,
                               const std::string &canonical_uri,
                               const std::string &canonical_query_string,
                               std::vector<Header> headers,
                               const std::string &payload_hash,
                               std::string &signed_headers_out);

// "AWS4-HMAC-SHA256\n<amz_date>\n<date_stamp>/<region>/<service>/aws4_request\n
// <hex sha256 of canonical_request_str>"
std::string string_to_sign(const std::string &amz_date,
                            const std::string &date_stamp,
                            const std::string &region,
                            const std::string &service,
                            const std::string &canonical_request_str);

// Derives the final signing key via the AWS4 HMAC chain:
// secret -> date -> region -> service -> "aws4_request".
void derive_signing_key(const std::string &secret_key,
                         const std::string &date_stamp,
                         const std::string &region, const std::string &service,
                         uint8_t out_key[32]);

// Hex-encoded HMAC-SHA256(signing_key, string_to_sign_str).
std::string sign_hex(const uint8_t signing_key[32],
                      const std::string &string_to_sign_str);

// Assembles the final `Authorization` header value.
std::string authorization_header(const std::string &access_key,
                                  const std::string &date_stamp,
                                  const std::string &region,
                                  const std::string &service,
                                  const std::string &signed_headers,
                                  const std::string &signature_hex);

} // namespace sigv4
