// SigV4-signed single PUT of one local file to the MinIO bucket. Streams the
// file from SD rather than loading it fully into RAM; uses
// x-amz-content-sha256: UNSIGNED-PAYLOAD so the ~288KB clip isn't hashed
// separately from being streamed (avoids reading it from SD twice).
#pragma once

#include <string>

namespace s3_uploader {

// `local_path` is an SD path (e.g. "/recordings/doorbell_manual_...wav").
// Uploads to MINIO_BUCKET at key "<kS3KeyPrefix>/<DEVICE_ID>/<basename>".
// Caller must hold the SD mutex for the duration of this call (it opens and
// reads local_path).
bool upload_file(const std::string &local_path);

} // namespace s3_uploader
