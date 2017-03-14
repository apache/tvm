/*!
 *  Copyright (c) 2017 by Contributors
 * \file pipe.h
 * \brief Platform independent pipe, used for IPC.
 */
#ifndef TVM_COMMON_PIPE_H_
#define TVM_COMMON_PIPE_H_

#include <dmlc/logging.h>
#include <dmlc/io.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <cstdlib>
#endif

namespace tvm {
namespace common {

/*! \brief Platform independent pipe */
class Pipe : public dmlc::Stream {
 public:
#ifdef _WIN32
  using PipeHandle = HANDLE;
#else
  using PipeHandle = int;
#endif
  /*! \brief Construct a pipe from system handle. */
  explicit Pipe(int64_t handle)
      : handle_(static_cast<PipeHandle>(handle)) {}
  /*! \brief destructor */
  ~Pipe() {
    Flush();
  }
  using Stream::Read;
  using Stream::Write;
  /*!
   * \brief reads data from a file descriptor
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  size_t Read(void *ptr, size_t size) final {
    if (size == 0) return 0;
#ifdef _WIN32
    DWORD nread;
    CHECK(ReadFile(handle_, static_cast<TCHAR*>(ptr),
                   &nread, nullptr))
        << "Read Error: " << GetLastError();
#else
    ssize_t nread;
    nread = read(handle_, ptr, size);
    CHECK_GE(nread, 0)
        << "Write Error: " << strerror(errno);
#endif
    return static_cast<size_t>(nread);
  }
  /*!
   * \brief write data to a file descriptor
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  void Write(const void *ptr, size_t size) final {
    if (size == 0) return;
#ifdef _WIN32
    DWORD nwrite;
    CHECK(WriteFile(handle_, static_cast<const TCHAR*>(ptr),
                    &nwrite, nullptr) &&
          static_cast<size_t>(nwrite) == size)
        << "Write Error: " << GetLastError();
#else
    ssize_t nwrite;
    nwrite = write(handle_, ptr, size);
    CHECK_EQ(static_cast<size_t>(nwrite), size)
        << "Write Error: " << strerror(errno);
#endif
  }
  /*!
   * \brief Flush the pipe;
   */
  void Flush() {
#ifdef _WIN32
    FlushFileBuffers(handle_);
#endif
  }
  /*! \brief close the pipe */
  void Close() {
#ifdef _WIN32
    CloseHandle(handle_);
#else
    close(handle_);
#endif
  }

 private:
  PipeHandle handle_;
};
}  // namespace common
}  // namespace tvm

#endif  // TVM_COMMON_PIPE_H_
