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
}

#include <dlfcn.h>
#include <stdlib.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <memory>
#include <string>

#include "../../../library_module.h"
#include "../../../minrpc/minrpc_server.h"
#include "../../hexagon/hexagon_common.h"
#include "../../hexagon/hexagon_device_api.h"
#include "hexagon_rpc.h"

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
      : read_buffer_{read_buffer},
        read_buffer_index_{0},
        read_buffer_size_bytes_{read_buffer_size_bytes},
        write_buffer_available_length_{0} {}

  void MessageStart(size_t message_size_bytes) {}

  ssize_t PosixWrite(const uint8_t* buf, size_t write_len_bytes) {
    LOG(INFO) << "HexagonIOHandler PosixWrite called, write_len_bytes(" << write_len_bytes << ")";
    int32_t written_size = write_buffer_.sputn(reinterpret_cast<const char*>(buf), write_len_bytes);
    if (written_size != write_len_bytes) {
      LOG(ERROR) << "written_size(" << written_size << ") != write_len_bytes(" << write_len_bytes
                 << ")";
    }
    write_buffer_available_length_ += written_size;
    return (ssize_t)written_size;
  }

  void MessageDone() { LOG(INFO) << "Message Done."; }

  ssize_t PosixRead(uint8_t* buf, size_t read_len_bytes) {
    LOG(INFO) << "HexagonIOHandler PosixRead called, read_len_bytes(" << read_len_bytes
              << "), read_buffer_index_(" << read_buffer_index_ << ")";

    uint32_t bytes_to_read = 0;
    if (read_buffer_index_ < read_len_bytes) {
      bytes_to_read = read_buffer_index_;
    } else {
      bytes_to_read = read_len_bytes;
    }

    std::memcpy(buf, read_buffer_, bytes_to_read);
    read_buffer_ += bytes_to_read;
    read_buffer_index_ -= bytes_to_read;
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
    LOG(INFO) << "HexagonIOHandler SetReadBuffer: data_size_bytes(" << data_size_bytes
              << "), read_buffer_index_(" << read_buffer_index_ << "), read_buffer_size_bytes_("
              << read_buffer_size_bytes_ << ")";
    if (data_size_bytes > read_buffer_size_bytes_) {
      LOG(ERROR) << "ERROR: data_size_bytes(" << data_size_bytes << ") > read_buffer_size_bytes_("
                 << read_buffer_size_bytes_ << ")";
      return AEE_EFAILED;
    }
    std::memcpy(reinterpret_cast<void*>(read_buffer_), reinterpret_cast<const void*>(data),
                data_size_bytes);
    read_buffer_index_ = data_size_bytes;
    return AEE_SUCCESS;
  }

  /*!
   * \brief Read from the write buffer that a packet has been written to.
   * \param buf The data pointer.
   * \param read_size_bytes The size of read in bytes.
   *
   * \return The size of data that is read in bytes.
   */
  int64_t ReadFromWriteBuffer(uint8_t* buf, size_t read_size_bytes) {
    LOG(INFO) << "HexagonIOHandler ReadFromWriteBuffer called, read_size_bytes: "
              << read_size_bytes;
    int64_t size = (int64_t)write_buffer_.sgetn(reinterpret_cast<char*>(buf), read_size_bytes);
    write_buffer_available_length_ -= size;

    // Clear buffer
    if (write_buffer_available_length_ == 0) {
      write_buffer_.str("");
    }
    return size;
  }

  void Close() { LOG(INFO) << "HexagonIOHandler Close called"; }

  void Exit(int code) { exit(code); }

 private:
  uint8_t* read_buffer_;
  uint32_t read_buffer_index_;
  size_t read_buffer_size_bytes_;

  std::stringbuf write_buffer_;
  uint32_t write_buffer_available_length_;
};

// Internal allocator that redirects alloc to TVM's C API.
template <typename TIOHandler>
class HexagonPageAllocator {
 public:
  using ArenaPageHeader = tvm::support::ArenaPageHeader;

  explicit HexagonPageAllocator(TIOHandler* io) : io_(io) {}

  ArenaPageHeader* allocate(size_t min_size) {
    size_t npages = ((min_size + kPageSize - 1) / kPageSize);
    void* data;

    data = malloc(npages * kPageSize);

    ArenaPageHeader* header = static_cast<ArenaPageHeader*>(data);
    header->size = npages * kPageSize;
    header->offset = sizeof(ArenaPageHeader);
    return header;
  }

  void deallocate(ArenaPageHeader* page) { free(page); }

  static const constexpr int kPageSize = 2 << 10;
  static const constexpr int kPageAlign = 8;

 private:
  TIOHandler* io_;
};

class HexagonRPCServer {
 public:
  explicit HexagonRPCServer(uint8_t* receive_buffer, size_t receive_buffer_size_bytes)
      : io_{receive_buffer, receive_buffer_size_bytes}, rpc_server_{&io_} {};

  /*!
   * \brief Wrtie to IOHandler.
   * \param data The data pointer
   * \param data_size_bytes The data size in bytes.
   *
   * \return The size of data written to IOHandler if no error.
   * Otherwise, returns -1;
   */
  int64_t Write(const uint8_t* data, size_t data_size_bytes) {
    AEEResult rc = io_.SetReadBuffer(data, data_size_bytes);
    if (rc != AEE_SUCCESS) {
      LOG(ERROR) << "ERROR: SetReadBuffer failed: " << rc;
      return -1;
    }

    if (!rpc_server_.ProcessOnePacket()) {
      LOG(ERROR) << "ERROR: ProcessOnePacket failed";
      return -1;
    }
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
    return io_.ReadFromWriteBuffer(buf, read_size_bytes);
  }

 private:
  HexagonIOHandler io_;
  MinRPCServer<HexagonIOHandler, HexagonPageAllocator> rpc_server_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

namespace {
static tvm::runtime::hexagon::HexagonRPCServer* g_hexagon_rpc_server;
tvm::runtime::hexagon::HexagonRPCServer* get_hexagon_rpc_server(
    uint32_t rpc_receive_buff_size_bytes = 0) {
  if (g_hexagon_rpc_server) {
    return g_hexagon_rpc_server;
  }
  CHECK_GT(rpc_receive_buff_size_bytes, 0) << "RPC receive buffer size is not valid.";
  static tvm::runtime::hexagon::HexagonRPCServer hexagon_rpc_server(
      new uint8_t[rpc_receive_buff_size_bytes], rpc_receive_buff_size_bytes);
  g_hexagon_rpc_server = &hexagon_rpc_server;
  return g_hexagon_rpc_server;
}
}  // namespace

const tvm::runtime::PackedFunc get_runtime_func(const std::string& name) {
  if (const tvm::runtime::PackedFunc* pf = tvm::runtime::Registry::Get(name)) {
    return *pf;
  }
  return tvm::runtime::PackedFunc();
}

void reset_device_api() {
  const tvm::runtime::PackedFunc api = get_runtime_func("device_api.hexagon");
  // Registering device_api.cpu as device_api.hexagon since we use hexagon as sub-target of LLVM.
  tvm::runtime::Registry::Register("device_api.cpu", true).set_body(api);
}

int __QAIC_HEADER(hexagon_rpc_open)(const char* uri, remote_handle64* handle) {
  *handle = static_cast<remote_handle64>(reinterpret_cast<uintptr_t>(malloc(1)));
  if (!*handle) {
    LOG(ERROR) << __func__ << ": cannot allocate memory";
    return AEE_ENOMEMORY;
  }
  reset_device_api();

  return AEE_SUCCESS;
}

int __QAIC_HEADER(hexagon_rpc_close)(remote_handle64 handle) {
  LOG(INFO) << __func__;
  if (handle) {
    free(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)));
  }
  return AEE_SUCCESS;
}

int __QAIC_HEADER(hexagon_rpc_init)(remote_handle64 _h, uint32_t buff_size_bytes) {
  get_hexagon_rpc_server(buff_size_bytes);
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
    LOG(ERROR) << "ERROR: hexagon_rpc_send failed, written_size (" << written_size
               << ") != dataLen (" << dataLen << ")";
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
  if (read_size == static_cast<int64_t>(bufLen)) {
    return AEE_SUCCESS;
  } else {
    LOG(ERROR) << "ERROR: RPC Server Read failed, read_size (" << read_size << ") != bufLen ("
               << static_cast<int64_t>(bufLen) << ")";
    return AEE_EFAILED;
  }
}

// Workaround for missing functions in 8.5.08
extern "C" {
__attribute__((weak)) void _Get_eh_data() {}
__attribute__((weak)) void _Parse_fde_instr() {}
}

TVM_REGISTER_GLOBAL("tvm.hexagon.load_module")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::string soname = args[0];
      tvm::ObjectPtr<tvm::runtime::Library> n = tvm::runtime::CreateDSOLibraryObject(soname);
      *rv = CreateModuleFromLibrary(n);
    });
