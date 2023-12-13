/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file pipe.h
 * \brief Platform independent pipe, used for IPC.
 */
#ifndef TVM_SUPPORT_PIPE_H_
#define TVM_SUPPORT_PIPE_H_

#include <dmlc/io.h>
#include <tvm/runtime/logging.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#endif

namespace tvm {
namespace support {

static int GetLastErrorCode() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
  }

/*! \brief Platform independent pipe */
class Pipe : public dmlc::Stream {
 public:
#ifdef _WIN32
  using PipeHandle = HANDLE;
  explicit Pipe(int64_t handle) : handle_(reinterpret_cast<PipeHandle>(handle)) {}
#else
  using PipeHandle = int;
  explicit Pipe(int64_t handle) : handle_(static_cast<PipeHandle>(handle)) {}
#endif
  /*! \brief destructor */
  ~Pipe() { Flush(); }

  /*!
   * \brief Call a function and retry if an EINTR error is encountered.
   *
   *  Socket operations can return EINTR when the interrupt handler
   *  is registered by the execution environment(e.g. python).
   *  We should retry if there is no KeyboardInterrupt recorded in
   *  the environment.
   *
   * \note This function is needed to avoid rare interrupt event
   *       in long running server code.
   *
   * \param func The function to retry.
   * \return The return code returned by function f or error_value on retry failure.
   */
  template <typename FuncType>
  ssize_t RetryCallOnEINTR(FuncType func) {
    ssize_t ret = func();
    // common path
    if (ret != -1) return ret;
    // less common path
    do {
      if (GetLastErrorCode() == EINTR) {
        // Call into env check signals to see if there are
        // environment specific(e.g. python) signal exceptions.
        // This function will throw an exception if there is
        // if the process received a signal that requires TVM to return immediately (e.g. SIGINT).
        runtime::EnvCheckSignals();
      } else {
        // other errors
        return ret;
      }
      ret = func();
    } while (ret == -1);
    return ret;
  }

  using Stream::Read;
  using Stream::Write;
  /*!
   * \brief reads data from a file descriptor
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  size_t Read(void* ptr, size_t size) final {
    if (size == 0) return 0;
#ifdef _WIN32
    DWORD nread;
    ICHECK(ReadFile(handle_, static_cast<TCHAR*>(ptr), size, &nread, nullptr))
        << "Read Error: " << GetLastError();
#else
    ssize_t nread;
    //nread = read(handle_, ptr, size);

    nread = RetryCallOnEINTR(
          [&]() { return read(handle_, ptr, size); });
    ICHECK_GE(nread, 0) << "Write Error: " << strerror(errno);
#endif
    return static_cast<size_t>(nread);
  }
  /*!
   * \brief write data to a file descriptor
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  void Write(const void* ptr, size_t size) final {
    if (size == 0) return;
#ifdef _WIN32
    DWORD nwrite;
    ICHECK(WriteFile(handle_, static_cast<const TCHAR*>(ptr), size, &nwrite, nullptr) &&
           static_cast<size_t>(nwrite) == size)
        << "Write Error: " << GetLastError();
#else
    ssize_t nwrite;
    nwrite = RetryCallOnEINTR(
          [&]() { return write(handle_, ptr, size); });
    ICHECK_EQ(static_cast<size_t>(nwrite), size) << "Write Error: " << strerror(errno);
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
}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_PIPE_H_
