#include <dmlc/io.h>

#ifndef SUPPORT_FILE_IO_H_
#define SUPPORT_FILE_IO_H_

namespace tvm {
namespace support {

/*!
 * \brief A dmlc stream which wraps standard file operations.
 */
struct SimpleBinaryFileStream : public dmlc::Stream {
 public:
  SimpleBinaryFileStream(const std::string& path, bool read) {
    const char* fname = path.c_str();
    if (read) {
      fp_ = std::fopen(fname, "rb");
    } else {
      fp_ = std::fopen(fname, "wb");
    }
    CHECK(fp_) << "Unable to open file " << path;
    read_ = read;
  }
  virtual ~SimpleBinaryFileStream(void) { this->Close(); }
  virtual size_t Read(void* ptr, size_t size) {
    CHECK(read_) << "File opened in write-mode, cannot read.";
    return std::fread(ptr, 1, size, fp_);
  }
  virtual void Write(const void* ptr, size_t size) {
    CHECK(!read_) << "File opened in read-mode, cannot write.";
    CHECK(std::fwrite(ptr, 1, size, fp_) == size) << "SimpleBinaryFileStream.Write incomplete";
  }
  inline void Close(void) {
    if (fp_ != NULL) {
      std::fclose(fp_);
      fp_ = NULL;
    }
  }

 private:
  std::FILE* fp_ = nullptr;
  bool read_;  // if false, then in write mode.
};             // class SimpleBinaryFileStream
}  // namespace support
}  // namespace tvm
#endif  // SUPPORT_FILE_IO_H_
