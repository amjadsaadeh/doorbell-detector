#include "s3_uploader.h"

#include <FS.h>
#include <HTTPClient.h>
#include <SD.h>
#include <WiFi.h>

#include <ctime>

#include "config.h"
#include "sigv4.h"

namespace s3_uploader {

namespace {

std::string basename_of(const std::string &path) {
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Percent-encodes everything except unreserved characters (RFC 3986:
// A-Z a-z 0-9 - _ . ~); '/' is preserved as a path separator when
// `preserve_slash` is set, matching SigV4's canonical-URI encoding rules.
std::string uri_encode(const std::string &s, bool preserve_slash) {
  static const char *hex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : s) {
    bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                       (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                       c == '.' || c == '~';
    if (unreserved || (preserve_slash && c == '/')) {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 0x0f];
    }
  }
  return out;
}

// e.g. "20260803T030000Z" / "20260803" — both in UTC, per the SigV4 spec
// (NTP-synced system time is UTC internally regardless of TZ_STRING).
void utc_amz_date(std::string &amz_date_out, std::string &date_stamp_out) {
  time_t now = time(nullptr);
  tm utc{};
  gmtime_r(&now, &utc);
  char amz[17];
  char stamp[9];
  std::strftime(amz, sizeof(amz), "%Y%m%dT%H%M%SZ", &utc);
  std::strftime(stamp, sizeof(stamp), "%Y%m%d", &utc);
  amz_date_out = amz;
  date_stamp_out = stamp;
}

} // namespace

bool upload_file(const std::string &local_path) {
  File f = SD.open(local_path.c_str(), FILE_READ);
  if (!f) {
    return false;
  }
  const size_t file_size = f.size();

  const std::string key = std::string(kS3KeyPrefix) + "/" + DEVICE_ID + "/" +
                           basename_of(local_path);
  const std::string canonical_uri = "/" + uri_encode(MINIO_BUCKET, false) +
                                     "/" + uri_encode(key, true);

  std::string amz_date, date_stamp;
  utc_amz_date(amz_date, date_stamp);

  const std::string payload_hash = "UNSIGNED-PAYLOAD";
  std::string signed_headers;
  const std::string creq = sigv4::canonical_request(
      "PUT", canonical_uri, "",
      {
          {"host", MINIO_ENDPOINT},
          {"x-amz-content-sha256", payload_hash},
          {"x-amz-date", amz_date},
      },
      payload_hash, signed_headers);

  const std::string sts = sigv4::string_to_sign(
      amz_date, date_stamp, MINIO_REGION, kS3Service, creq);

  uint8_t signing_key[32];
  sigv4::derive_signing_key(MINIO_SECRET_KEY, date_stamp, MINIO_REGION,
                             kS3Service, signing_key);
  const std::string signature = sigv4::sign_hex(signing_key, sts);

  const std::string authorization = sigv4::authorization_header(
      MINIO_ACCESS_KEY, date_stamp, MINIO_REGION, kS3Service, signed_headers,
      signature);

  const std::string url =
      std::string("http://") + MINIO_ENDPOINT + canonical_uri;

  WiFiClient client;
  HTTPClient http;
  http.begin(client, url.c_str());
  http.addHeader("x-amz-date", amz_date.c_str());
  http.addHeader("x-amz-content-sha256", payload_hash.c_str());
  http.addHeader("Authorization", authorization.c_str());

  int status = http.sendRequest("PUT", &f, file_size);
  http.end();
  f.close();

  return status >= 200 && status < 300;
}

} // namespace s3_uploader
