#include "sd_storage.h"

#include <SPI.h>

#include "config.h"

namespace sd_storage {

namespace {
SemaphoreHandle_t g_mutex = nullptr;
}

bool begin() {
  g_mutex = xSemaphoreCreateMutex();
  if (g_mutex == nullptr) {
    return false;
  }

  if (!SD.begin(kSdCsPin)) {
    return false;
  }

  if (!SD.exists(kRecordingsDir)) {
    SD.mkdir(kRecordingsDir);
  }

  return true;
}

bool lock(uint32_t timeout_ms) {
  return xSemaphoreTake(g_mutex, timeout_ms == portMAX_DELAY
                                     ? portMAX_DELAY
                                     : pdMS_TO_TICKS(timeout_ms)) == pdTRUE;
}

void unlock() { xSemaphoreGive(g_mutex); }

std::vector<std::string> list_recordings() {
  std::vector<std::string> out;

  File dir = SD.open(kRecordingsDir);
  if (!dir || !dir.isDirectory()) {
    return out;
  }

  File entry = dir.openNextFile();
  while (entry) {
    if (!entry.isDirectory()) {
      std::string name = entry.name();
      const bool is_wav = name.size() > 4 &&
                           name.compare(name.size() - 4, 4, ".wav") == 0;
      if (is_wav) {
        // Depending on the core version, File::name() from a directory
        // iterator may return a bare filename or an already-absolute path —
        // handle both rather than assume one.
        std::string path = (!name.empty() && name[0] == '/')
                                ? name
                                : std::string(kRecordingsDir) + "/" + name;
        out.push_back(path);
      }
    }
    entry = dir.openNextFile();
  }

  return out;
}

bool remove_file(const std::string &path) { return SD.remove(path.c_str()); }

size_t file_size(const std::string &path) {
  File f = SD.open(path.c_str(), FILE_READ);
  if (!f) {
    return 0;
  }
  size_t size = f.size();
  f.close();
  return size;
}

std::string temp_path_for(const std::string &final_path) {
  return final_path + kTempSuffix;
}

bool commit_temp_file(const std::string &final_path) {
  return SD.rename(temp_path_for(final_path).c_str(), final_path.c_str());
}

} // namespace sd_storage
