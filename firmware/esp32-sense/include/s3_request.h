// SigV4-signed plain-HTTP requests against MinIO, shared by the recording
// uploader (PUT) and the OTA download (GET).
#pragma once

#include <HTTPClient.h>
#include <WiFiClient.h>

#include <string>

namespace s3_request {

// Points `http` at <MINIO_ENDPOINT>/<bucket_and_prefix>/<key> and adds the
// SigV4 headers for `method` (payload unsigned). `bucket_and_prefix` may carry
// a key prefix ("doorbell-detector/raw"); its '/' stays a path separator.
// Returns false if http.begin() fails. The caller sends the request and calls
// http.end().
bool prepare(const char *method, const std::string &bucket_and_prefix,
             const std::string &key, HTTPClient &http, WiFiClient &client);

} // namespace s3_request
