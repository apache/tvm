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
 * \file minrpc_server.h
 * \brief Minimum RPC server implementation,
 *        redirects all the calls to C runtime API.
 *
 * \note This file do not depend on c++ std or c std,
 *       and only depends on TVM's C runtime API.
 */
#ifndef TVM_RUNTIME_RPC_MINRPC_MINRPC_SERVER_H_
#define TVM_RUNTIME_RPC_MINRPC_MINRPC_SERVER_H_

#include <dmlc/endian.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../rpc_protocol.h"
#include "../../../support/arena.h"

/*! \brief Whether or not to enable glog style DLOG */
#ifndef TVM_MINRPC_ENABLE_LOGGING
#define TVM_MINRPC_ENABLE_LOGGING 0
#endif

#ifndef MINRPC_CHECK
#define MINRPC_CHECK(cond)                                      \
  if (!(cond)) this->ThrowError(RPCServerStatus::kCheckError);
#endif

#if TVM_MINRPC_ENABLE_LOGGING
#include <dmlc/logging.h>
#endif


namespace tvm {
namespace runtime {

/*!
 * \brief A minimum RPC server that only depends on the tvm C runtime..
 *
 *  All the dependencies are provided by the io arguments.
 *
 * \tparam TIOHandler IO provider to provide io handling.
 *         An IOHandler needs to provide the following functions:
 *         - PosixWrite, PosixRead, Close: posix style, read, write, close API.
 *         - Exit: exit with status code.
 */
template<typename TIOHandler>
class MinRPCServer {
 public:
  /*!
   * \brief Constructor.
   * \param io The IO handler.
   */
  explicit MinRPCServer(TIOHandler io)
      : io_(io), arena_(PageAllocator(io)) {}

  /*! \brief Run the server loop until shutdown signal is received. */
  void ServerLoop() {
    RPCCode code;
    uint64_t packet_len;

    while (true) {
      arena_.RecycleAll();
      allow_clean_shutdown_ = true;

      this->Read(&packet_len);
      if (packet_len == 0) continue;
      this->Read(&code);

      allow_clean_shutdown_ = false;

      if (code >= RPCCode::kSyscallCodeStart) {
        this->HandleSyscallFunc(code);
      } else {
        switch (code) {
          case RPCCode::kCallFunc: {
            HandleNormalCallFunc();
            break;
          }
          case RPCCode::kInitServer: {
            HandleInitServer();
            break;
          }
          case RPCCode::kCopyFromRemote: {
            HandleCopyFromRemote();
            break;
          }
          case RPCCode::kCopyToRemote: {
            HandleCopyToRemote();
            break;
          }
          case RPCCode::kShutdown: {
            this->Shutdown();
            return;
          }
          default: {
            this->ThrowError(RPCServerStatus::kUnknownRPCCode);
            break;
          }
        }
      }
    }
  }

  void Shutdown() {
    arena_.FreeAll();
    io_.Close();
  }

  void HandleNormalCallFunc() {
    uint64_t call_handle;
    TVMValue* values;
    int* tcodes;
    int num_args;
    TVMValue ret_value[3];
    int ret_tcode[3];

    this->Read(&call_handle);
    RecvPackedSeq(&values, &tcodes, &num_args);

    int call_ecode = TVMFuncCall(
        reinterpret_cast<void*>(call_handle),
        values, tcodes, num_args,
        &(ret_value[1]), &(ret_tcode[1]));

    if (call_ecode == 0) {
      // Return value encoding as in LocalSession
      int rv_tcode = ret_tcode[1];
      ret_tcode[0] = kDLInt;
      ret_value[0].v_int64 = rv_tcode;
      if (rv_tcode == kTVMNDArrayHandle) {
        ret_tcode[1] = kTVMDLTensorHandle;
        ret_value[2].v_handle = ret_value[1].v_handle;
        ret_tcode[2] = kTVMOpaqueHandle;
        this->ReturnPackedSeq(ret_value, ret_tcode, 3);
      } else if (rv_tcode == kTVMPackedFuncHandle ||
                 rv_tcode == kTVMModuleHandle) {
        ret_tcode[1] = kTVMOpaqueHandle;
        this->ReturnPackedSeq(ret_value, ret_tcode, 2);
      } else {
        this->ReturnPackedSeq(ret_value, ret_tcode, 2);
      }
    } else {
      this->ReturnLastTVMError();
    }
  }

  void HandleCopyFromRemote() {
    uint64_t handle, offset, num_bytes;
    TVMContext ctx;
    DLDataType type_hint;

    this->Read(&handle);
    this->Read(&offset);
    this->Read(&num_bytes);
    this->Read(&ctx);
    this->Read(&type_hint);

    uint8_t* data_ptr;
    int call_ecode = 0;
    if (ctx.device_type == kDLCPU) {
      data_ptr = reinterpret_cast<uint8_t*>(handle) + offset;
    } else {
      data_ptr = this->ArenaAlloc<uint8_t>(num_bytes);
      call_ecode = TVMDeviceCopyDataFromTo(
              reinterpret_cast<void*>(handle), offset,
              data_ptr, 0, num_bytes,
              ctx, DLContext{kDLCPU, 0},
              type_hint, nullptr);
    }

    if (call_ecode == 0) {
      RPCCode code = RPCCode::kCopyAck;
      uint64_t packet_nbytes = sizeof(code) + num_bytes;

      this->Write(packet_nbytes);
      this->Write(code);
      this->WriteArray(data_ptr, num_bytes);
    } else {
      this->ReturnLastTVMError();
    }
  }

  void HandleCopyToRemote() {
    uint64_t handle, offset, num_bytes;
    TVMContext ctx;
    DLDataType type_hint;

    this->Read(&handle);
    this->Read(&offset);
    this->Read(&num_bytes);
    this->Read(&ctx);
    this->Read(&type_hint);
    int call_ecode = 0;

    if (ctx.device_type == kDLCPU) {
      uint8_t* dptr = reinterpret_cast<uint8_t*>(handle) + offset;
      this->ReadArray(dptr, num_bytes);
    } else {
      uint8_t* temp_data = this->ArenaAlloc<uint8_t>(num_bytes);
      this->ReadArray(temp_data, num_bytes);

      call_ecode = TVMDeviceCopyDataFromTo(
              temp_data, 0,
              reinterpret_cast<void*>(handle), offset,
              num_bytes,
              DLContext{kDLCPU, 0}, ctx,
              type_hint, nullptr);
    }

    if (call_ecode == 0) {
      this->ReturnVoid();
    } else {
      this->ReturnLastTVMError();
    }
  }

  void HandleSyscallFunc(RPCCode code) {
    TVMValue* values;
    int* tcodes;
    int num_args;
    RecvPackedSeq(&values, &tcodes, &num_args);
    switch (code) {
      case RPCCode::kFreeHandle: {
        this->SyscallFreeHandle(values, tcodes, num_args);
        break;
      }
      case RPCCode::kGetGlobalFunc: {
        this->SyscallGetGlobalFunc(values, tcodes, num_args);
        break;
      }
      case RPCCode::kDevSetDevice: {
        this->ReturnException("SetDevice not supported");
        break;
      }
      case RPCCode::kDevGetAttr: {
        this->ReturnException("GetAttr not supported");
        break;
      }
      case RPCCode::kDevAllocData: {
        this->SyscallDevAllocData(values, tcodes, num_args);
        break;
      }
      case RPCCode::kDevFreeData: {
        this->SyscallDevFreeData(values, tcodes, num_args);
        break;
      }
      case RPCCode::kDevStreamSync: {
        this->SyscallDevStreamSync(values, tcodes, num_args);
        break;
      }
      case RPCCode::kCopyAmongRemote: {
        this->SyscallCopyAmongRemote(values, tcodes, num_args);
        break;
      }
      default: {
        this->ReturnException("Syscall not recognized");
        break;
      }
    }
  }

  void HandleInitServer() {
    uint64_t len;
    this->Read(&len);
    char* proto_ver = this->ArenaAlloc<char>(len + 1);
    this->ReadArray(proto_ver, len);

    TVMValue* values;
    int* tcodes;
    int num_args;
    RecvPackedSeq(&values, &tcodes, &num_args);
    MINRPC_CHECK(num_args == 0);
    this->ReturnVoid();
  }

  void SyscallFreeHandle(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 2);
    MINRPC_CHECK(tcodes[0] == kTVMOpaqueHandle);
    MINRPC_CHECK(tcodes[1] == kDLInt);

    void* handle = values[0].v_handle;
    int64_t type_code = values[1].v_int64;
    int call_ecode;

    if (type_code == kTVMNDArrayHandle) {
      call_ecode = TVMArrayFree(static_cast<TVMArrayHandle>(handle));
    } else if (type_code == kTVMPackedFuncHandle) {
      call_ecode = TVMFuncFree(handle);
    } else {
      MINRPC_CHECK(type_code == kTVMModuleHandle);
      call_ecode = TVMModFree(handle);
    }

    if (call_ecode == 0) {
      this->ReturnVoid();
    } else {
      this->ReturnLastTVMError();
    }
  }

  void SyscallGetGlobalFunc(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 1);
    MINRPC_CHECK(tcodes[0] == kTVMStr);

    void* handle;
    int call_ecode = TVMFuncGetGlobal(values[0].v_str, &handle);

    if (call_ecode == 0) {
      this->ReturnHandle(handle);
    } else {
      this->ReturnLastTVMError();
    }
  }

  void SyscallCopyAmongRemote(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 9);
    // from, from_offset
    MINRPC_CHECK(tcodes[0] == kTVMOpaqueHandle);
    MINRPC_CHECK(tcodes[1] == kDLInt);
    // to, to_offset
    MINRPC_CHECK(tcodes[2] == kTVMOpaqueHandle);
    MINRPC_CHECK(tcodes[3] == kDLInt);
    // size
    MINRPC_CHECK(tcodes[4] == kDLInt);
    // ctx_from, ctx_to
    MINRPC_CHECK(tcodes[5] == kTVMContext);
    MINRPC_CHECK(tcodes[6] == kTVMContext);
    // type_hint, stream
    MINRPC_CHECK(tcodes[7] == kTVMDataType);
    MINRPC_CHECK(tcodes[8] == kTVMOpaqueHandle);

    void* from = values[0].v_handle;
    int64_t from_offset = values[1].v_int64;
    void* to = values[2].v_handle;
    int64_t to_offset = values[3].v_int64;
    int64_t size = values[4].v_int64;
    TVMContext ctx_from = values[5].v_ctx;
    TVMContext ctx_to = values[6].v_ctx;
    DLDataType type_hint = values[7].v_type;
    TVMStreamHandle stream = values[8].v_handle;

    int call_ecode = TVMDeviceCopyDataFromTo(
        from, from_offset,
        to, to_offset, size,
        ctx_from, ctx_to, type_hint, stream);

    if (call_ecode == 0) {
      this->ReturnVoid();
    } else {
      this->ReturnLastTVMError();
    }
  }

  void SyscallDevAllocData(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 4);
    MINRPC_CHECK(tcodes[0] == kTVMContext);
    MINRPC_CHECK(tcodes[1] == kDLInt);
    MINRPC_CHECK(tcodes[2] == kDLInt);
    MINRPC_CHECK(tcodes[3] == kTVMDataType);

    TVMContext ctx = values[0].v_ctx;
    int64_t nbytes = values[1].v_int64;
    int64_t alignment = values[2].v_int64;
    DLDataType type_hint = values[3].v_type;

    void* handle;
    int call_ecode = TVMDeviceAllocDataSpace(
        ctx, nbytes, alignment, type_hint, &handle);

    if (call_ecode == 0) {
      this->ReturnHandle(handle);
    } else {
      this->ReturnLastTVMError();
    }
  }

  void SyscallDevFreeData(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 2);
    MINRPC_CHECK(tcodes[0] == kTVMContext);
    MINRPC_CHECK(tcodes[1] == kTVMOpaqueHandle);

    TVMContext ctx = values[0].v_ctx;
    void* handle = values[1].v_handle;

    int call_ecode = TVMDeviceFreeDataSpace(ctx, handle);

    if (call_ecode == 0) {
      this->ReturnVoid();
    } else {
      this->ReturnLastTVMError();
    }
  }

  void SyscallDevStreamSync(TVMValue* values, int* tcodes, int num_args) {
    MINRPC_CHECK(num_args == 2);
    MINRPC_CHECK(tcodes[0] == kTVMContext);
    MINRPC_CHECK(tcodes[1] == kTVMOpaqueHandle);

    TVMContext ctx = values[0].v_ctx;
    void* handle = values[1].v_handle;

    int call_ecode = TVMSynchronize(ctx.device_type, ctx.device_id, handle);

    if (call_ecode == 0) {
      this->ReturnVoid();
    } else {
      this->ReturnLastTVMError();
    }
  }

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) {
    io_.Exit(static_cast<int>(code));
  }

  template<typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  template<typename T>
  void Read(T* data) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    this->ReadRawBytes(data, sizeof(T));
  }

  template<typename T>
  void ReadArray(T* data, size_t count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return this->ReadRawBytes(data, sizeof(T) * count);
  }

  template<typename T>
  void Write(const T& data) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return this->WriteRawBytes(&data, sizeof(T));
  }

  template<typename T>
  void WriteArray(T* data, size_t count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return this->WriteRawBytes(data, sizeof(T) * count);
  }

 private:
  // Internal allocator that redirects alloc to TVM's C API.
  class PageAllocator {
   public:
    using ArenaPageHeader = tvm::support::ArenaPageHeader;

    explicit PageAllocator(TIOHandler io)
        : io_(io) {}

    ArenaPageHeader* allocate(size_t min_size) {
      size_t npages = ((min_size + kPageSize - 1) / kPageSize);
      void* data;

      if (TVMDeviceAllocDataSpace(
              DLContext{kDLCPU, 0}, npages * kPageSize, kPageAlign,
              DLDataType{kDLInt, 1, 1}, &data) != 0) {
        io_.Exit(static_cast<int>(RPCServerStatus::kAllocError));
      }

      ArenaPageHeader* header = static_cast<ArenaPageHeader*>(data);
      header->size = npages * kPageSize;
      header->offset = sizeof(ArenaPageHeader);
      return header;
    }

    void deallocate(ArenaPageHeader* page) {
      if (TVMDeviceFreeDataSpace(DLContext{kDLCPU, 0}, page) != 0) {
        io_.Exit(static_cast<int>(RPCServerStatus::kAllocError));
      }
    }

    static const constexpr int kPageSize = 2 << 10;
    static const constexpr int kPageAlign = 8;

   private:
    TIOHandler io_;
  };

  void RecvPackedSeq(TVMValue** out_values,
                     int** out_tcodes,
                     int* out_num_args) {
    RPCReference::RecvPackedSeq(
        out_values, out_tcodes, out_num_args, this);
  }

  void ReturnVoid() {
    int32_t num_args = 1;
    int32_t tcode = kTVMNullptr;
    RPCCode code = RPCCode::kReturn;

    uint64_t packet_nbytes =
        sizeof(code) + sizeof(num_args) + sizeof(tcode);

    this->Write(packet_nbytes);
    this->Write(code);
    this->Write(num_args);
    this->Write(tcode);
  }

  void ReturnHandle(void* handle) {
    int32_t num_args = 1;
    int32_t tcode = kTVMOpaqueHandle;
    RPCCode code = RPCCode::kReturn;
    uint64_t encode_handle = reinterpret_cast<uint64_t>(handle);

    uint64_t packet_nbytes =
        sizeof(code) + sizeof(num_args) +
        sizeof(tcode) + sizeof(encode_handle);

    this->Write(packet_nbytes);
    this->Write(code);
    this->Write(num_args);
    this->Write(tcode);
    this->Write(encode_handle);
  }

  void ReturnException(const char* msg) {
    RPCReference::ReturnException(msg, this);
  }

  void ReturnPackedSeq(const TVMValue* arg_values,
                       const int* type_codes,
                       int num_args) {
    RPCReference::ReturnPackedSeq(arg_values, type_codes, num_args, this);
  }

  void ReturnLastTVMError() {
    this->ReturnException(TVMGetLastError());
  }

  void ReadRawBytes(void* data, size_t size) {
    uint8_t* buf = reinterpret_cast<uint8_t*>(data);
    size_t ndone = 0;
    while (ndone <  size) {
      ssize_t ret = io_.PosixRead(buf, size - ndone);
      if (ret == 0) {
        if (allow_clean_shutdown_) {
          this->Shutdown();
          io_.Exit(0);
        } else {
          this->ThrowError(RPCServerStatus::kReadError);
        }
      }
      if (ret == -1) {
        this->ThrowError(RPCServerStatus::kReadError);
      }
      ndone += ret;
      buf += ret;
    }
  }

  void WriteRawBytes(const void* data, size_t size) {
    const uint8_t *buf = reinterpret_cast<const uint8_t*>(data);
    size_t ndone = 0;
    while (ndone <  size) {
      ssize_t ret = io_.PosixWrite(buf, size - ndone);
      if (ret == 0 || ret == -1) {
        this->ThrowError(RPCServerStatus::kWriteError);
      }
      buf += ret;
      ndone += ret;
    }
  }

  /*! \brief IO handler. */
  TIOHandler io_;
  /*! \brief internal arena. */
  support::GenericArena<PageAllocator> arena_;
  /*! \brief Whether we are in a state that allows clean shutdown. */
  bool allow_clean_shutdown_{true};
  static_assert(DMLC_LITTLE_ENDIAN, "MinRPC only works on little endian.");
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_MINRPC_MINRPC_SERVER_H_
