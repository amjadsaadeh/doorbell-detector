#include "s3_uploader.h"

#include <FS.h>
#include <HTTPClient.h>
#include <SD.h>
#include <WiFi.h>

#include "config.h"
#include "s3_request.h"

namespace s3_uploader {

namespace {

std::string basename_of(const std::string &path) {
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

} // namespace

bool upload_file(const std::string &local_path) {
  File f = SD.open(local_path.c_str(), FILE_READ);
  if (!f) {
    Serial.printf("upload: cannot open %s\n", local_path.c_str());
    return false;
  }
  const size_t file_size = f.size();

  const std::string key = std::string(kS3KeyPrefix) + "/" + DEVICE_ID + "/" +
                           basename_of(local_path);

  WiFiClient client;
  HTTPClient http;
  if (!s3_request::prepare("PUT", MINIO_BUCKET, key, http, client)) {
    Serial.printf("upload: %s -> cannot start request\n", key.c_str());
    f.close();
    return false;
  }

  int status = http.sendRequest("PUT", &f, file_size);
  if (status >= 200 && status < 300) {
    Serial.printf("upload: %s -> HTTP %d\n", key.c_str(), status);
  } else if (status > 0) {
    Serial.printf("upload: %s -> HTTP %d: %s\n", key.c_str(), status,
                  http.getString().c_str());
  } else {
    Serial.printf("upload: %s -> failed (%d %s)\n", key.c_str(), status,
                  HTTPClient::errorToString(status).c_str());
  }
  http.end();
  f.close();

  return status >= 200 && status < 300;
}

} // namespace s3_uploader
