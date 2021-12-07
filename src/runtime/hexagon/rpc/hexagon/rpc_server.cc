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

extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_error.h>
#include <qurt_hvx.h>
}

#include <dlfcn.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <memory>
#include <string>

#include "../../../library_module.h"
#include "../../../minrpc/minrpc_server.h"
#include "../../hexagon/hexagon_common.h"
#include "hexagon_rpc.h"

#define TVM_HEXAGON_RPC_BUFF_SIZE_BYTES 1024 * 1024

#define TVM_LOG_CUSTOMIZE 1

namespace tvm {
namespace runtime {
namespace hexagon {

/*!
 * \brief Hexagon IO Handler used in HexagonRPCServer(MinRPCServer).
 *
 * \param read_buffer The pointer to read buffer.
 * \param read_buffer_size_bytes The read buffer size in bytes.
 */
class HexagonIOHandler {
 public:
  explicit HexagonIOHandler(uint8_t* read_buffer, size_t read_buffer_size_bytes)
      : read_buffer_{read_buffer}, read_buffer_size_bytes_{read_buffer_size_bytes}, read_buffer_index_{0} {}

  void MessageStart(size_t message_size_bytes) {}

  ssize_t PosixWrite(const uint8_t* buf, size_t write_len_bytes) {
    HEXAGON_PRINT(ALWAYS, "HexagonIOHandler PosixWrite called, write_len_bytes: %d",
                  write_len_bytes);
    size_t written_size = static_cast<size_t>(
        write_buffer_.sputn(reinterpret_cast<const char*>(buf), write_len_bytes));
    if (written_size != write_len_bytes) {
      HEXAGON_PRINT(ALWAYS, "HexagonIOHandler written_size failed");
    }
    return (ssize_t)written_size;
  }

  void MessageDone() {}

  ssize_t PosixRead(uint8_t* buf, size_t read_len_bytes) {
    HEXAGON_PRINT(ALWAYS, "HexagonIOHandler PosixRead called, %d, %d", read_len_bytes,
                  read_buffer_index_);

    uint32_t bytes_to_read = 0;
    if ((read_buffer_index_ - read_len_bytes) < 0) {
      bytes_to_read = read_buffer_index_;
    } else {
      bytes_to_read = read_len_bytes;
    }

    std::memcpy(buf, read_buffer_, bytes_to_read);
    read_buffer_ += bytes_to_read;
    read_buffer_index_ -= bytes_to_read;
    if (bytes_to_read != read_len_bytes) {
      HEXAGON_PRINT(ERROR, "Error bytes_to_read (%d) < read_len_bytes (%d).", bytes_to_read,
                    read_len_bytes);
    }
    return (ssize_t)bytes_to_read;
  }

  /*!
   * \brief Set read buffer in IOHandler to data pointer.
   * \param data The data pointer.
   * \param data_size_bytes The size of data in bytes.
   * 
   * \return The status
   */
  AEEResult SetReadBuffer(const uint8_t* data, size_t data_size_bytes) {
    HEXAGON_PRINT(ALWAYS,
                  "HexagonIOHandler SetReadBuffer called: %d, prev read_buffer_index_: ",
                  data_size_bytes, read_buffer_index_);
    if (data_size_bytes > read_buffer_size_bytes_) {
      return AEE_EFAILED;
    }
    read_buffer_ = data;
    read_buffer_index_ = data_size_bytes;
    return AEE_SUCCESS;
  }

  /*!
   * \brief Get pointer to the buffer that a packet has been written to.
   * \param buf The data pointer.
   * \param read_size_bytes The size of read in bytes.
   *
   * \return The size of data that is read in bytes.
   */
  int64_t GetWriteBuffer(uint8_t* buf, size_t read_size_bytes) {
    HEXAGON_PRINT(ALWAYS, "HexagonIOHandler GetWriteBuffer called, read_len_bytes: %d",
                  read_size_bytes);
    return write_buffer_.sgetn(reinterpret_cast<char*>(buf), read_size_bytes);
  }

  void Close() { HEXAGON_PRINT(ALWAYS, "HexagonIOHandler Close called"); }

  void Exit(int code) { exit(code); }

 private:
  const uint8_t* read_buffer_;
  uint32_t read_buffer_index_;
  size_t read_buffer_size_bytes_;

  std::stringbuf write_buffer_;
};

class HexagonRPCServer {
 public:
  explicit HexagonRPCServer(uint8_t* receive_buffer, size_t receive_buffer_size_bytes) : io_{receive_buffer, receive_buffer_size_bytes}, rpc_server_{&io_} {};

  /*!
   * \brief Wrtie to IOHandler.
   * \param data The data pointer
   * \param data_size_bytes The data size in bytes.
   *
   * \return The size of data written to IOHandler.
   */
  int64_t Write(const uint8_t* data, size_t data_size_bytes) {
    if (io_.SetReadBuffer(data, data_size_bytes) != AEE_SUCCESS) {
      return -1;
    }
    rpc_server_.ProcessOnePacket();
    return (int64_t)data_size_bytes;
  }

  /*!
   * \brief Read from IOHandler.
   * \param buf The buffer pointer
   * \param read_size_bytes Read request size in bytes.
   *
   * \return The size of data that is read in bytes.
   */
  int64_t Read(uint8_t* buf, size_t read_size_bytes) {
    return io_.GetWriteBuffer(buf, read_size_bytes);
  }

 private:
  HexagonIOHandler io_;
  MinRPCServer<HexagonIOHandler> rpc_server_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

namespace {
  tvm::runtime::hexagon::HexagonRPCServer* get_hexagon_rpc_server() {
    static tvm::runtime::hexagon::HexagonRPCServer g_hexagon_rpc_server(new uint8_t[TVM_HEXAGON_RPC_BUFF_SIZE_BYTES], TVM_HEXAGON_RPC_BUFF_SIZE_BYTES);
    return &g_hexagon_rpc_server;
  }
}

const tvm::runtime::PackedFunc get_runtime_func(const std::string& name) {
  if (const tvm::runtime::PackedFunc* pf = tvm::runtime::Registry::Get(name)) {
    return *pf;
  }
  return tvm::runtime::PackedFunc();
}

void reset_device_api() {
  const tvm::runtime::PackedFunc api = get_runtime_func("device_api.hexagon.v2");
  tvm::runtime::Registry::Register("device_api.hexagon", true).set_body(api);
}

int __QAIC_HEADER(hexagon_rpc_open)(const char* uri, remote_handle64* handle) {
  *handle = static_cast<remote_handle64>(reinterpret_cast<uintptr_t>(malloc(1)));
  if (!*handle) {
    HEXAGON_PRINT(ERROR, "%s: cannot allocate memory", __func__);
    return AEE_ENOMEMORY;
  }
  reset_device_api();
  get_hexagon_rpc_server();
  return AEE_SUCCESS;
}

int __QAIC_HEADER(hexagon_rpc_close)(remote_handle64 handle) {
  HEXAGON_PRINT(ALWAYS, "%s", __func__);
  if (handle) {
    free(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)));
  }
  return AEE_SUCCESS;
}

/*!
 * \brief Send data from Host to Hexagon over RPCSession.
 * \param _handle The remote handle
 * \param data The data sent to host.
 * \param dataLen The size of the data.
 *
 * \return The status.
 */
AEEResult __QAIC_HEADER(hexagon_rpc_send)(remote_handle64 _handle, const unsigned char* data,
                                          int dataLen) {
  int64_t written_size = get_hexagon_rpc_server()->Write(reinterpret_cast<const uint8_t*>(data),
                                                     static_cast<size_t>(dataLen));
  if (written_size != dataLen) {
    HEXAGON_PRINT(ERROR, "RPC Server Write failed, written_size (%d) != dataLen (%d)", written_size,
                  dataLen);
    return AEE_EFAILED;
  }
  return AEE_SUCCESS;
}

/*!
 * \brief Receive data from Hexagon adn send to host over RPCSession.
 * \param _handle The remote handle
 * \param data The buffer for receiving data
 * \param dataLen The size of the data that is requested to read in bytes.
 * \param buf_written_size The size of the data that is actually read in bytes.
 *
 * \return The status.
 */
AEEResult __QAIC_HEADER(hexagon_rpc_receive)(remote_handle64 _handle, unsigned char* buf,
                                             int bufLen, int64_t* buf_written_size) {
  int64_t read_size =
      get_hexagon_rpc_server()->Read(reinterpret_cast<uint8_t*>(buf), static_cast<size_t>(bufLen));
  *buf_written_size = read_size;
  if (read_size == bufLen) {
    return AEE_SUCCESS;
  } else {
    HEXAGON_PRINT(ALWAYS, "RPC Server Read failed, read_size (%d) != dataLen (%d)", read_size,
                  bufLen);
    return AEE_EFAILED;
  }
}

TVM_REGISTER_GLOBAL("tvm.hexagon.load_module")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::string soname = args[0];
      tvm::ObjectPtr<tvm::runtime::Library> n = tvm::runtime::CreateDSOLibraryObject(soname);
      *rv = CreateModuleFromLibrary(n, tvm::runtime::hexagon::WrapPackedFunc);
    });
