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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_DSPRPCAPI_H_
#define TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_DSPRPCAPI_H_

#ifdef __ANDROID__
#include <dmlc/logging.h>
#include <stdint.h>

#include "remote.h"
#include "remote64.h"
#include "rpcmem.h"

namespace tvm {
namespace runtime {

namespace hexagon {

/*!
 * Encapsulation of the API of lib(a|c)dsprpc.so (loaded via dlopen), allowing
 * for having versions of the library that do not implement all of the
 * functions.
 *
 * Functions defined in the DSP RPC library:
 *   remote_handle_close
 *   remote_handle_control
 *   remote_handle_invoke
 *   remote_handle_open
 *   remote_mmap
 *   remote_munmap
 *
 *   remote_handle64_close
 *   remote_handle64_control
 *   remote_handle64_invoke
 *   remote_handle64_open
 *   remote_mmap64
 *   remote_munmap64
 *
 *   remote_register_buf
 *   remote_register_buf_attr
 *   remote_register_dma_handle
 *   remote_register_dma_handle_attr
 *   remote_register_fd
 *
 *   remote_session_control
 *   remote_set_mode
 *
 *   rpcmem_init
 *   rpcmem_deinit
 *   rpcmem_alloc
 *   rpcmem_free
 *   rpcmem_to_fd
 */
class DspRpcAPI {
 public:
  DspRpcAPI();
  ~DspRpcAPI();

  using remote_handle = ::remote_handle;
  using remote_handle64 = ::remote_handle64;

#define DECLTYPE(ty) using ty##_t = decltype(::ty);
  DECLTYPE(remote_handle_close)
  DECLTYPE(remote_handle_control)
  DECLTYPE(remote_handle_invoke)
  DECLTYPE(remote_handle_open)
  DECLTYPE(remote_mmap)
  DECLTYPE(remote_munmap)

  DECLTYPE(remote_handle64_close)
  DECLTYPE(remote_handle64_control)
  DECLTYPE(remote_handle64_invoke)
  DECLTYPE(remote_handle64_open)
  DECLTYPE(remote_mmap64)
  DECLTYPE(remote_munmap64)

  DECLTYPE(remote_register_buf)
  DECLTYPE(remote_register_buf_attr)
  DECLTYPE(remote_register_dma_handle)
  DECLTYPE(remote_register_dma_handle_attr)
  DECLTYPE(remote_register_fd)

  DECLTYPE(remote_session_control)
  DECLTYPE(remote_set_mode)

  DECLTYPE(rpcmem_init)
  DECLTYPE(rpcmem_deinit)
  DECLTYPE(rpcmem_alloc)
  DECLTYPE(rpcmem_free)
  DECLTYPE(rpcmem_to_fd)
#undef DECLTYPE

#define DECLFUNC(fn)                                   \
  fn##_t* fn##_ptr(bool allow_nullptr = false) const { \
    if (!allow_nullptr) CHECK(fn##_ != nullptr);       \
    return fn##_;                                      \
  }
  DECLFUNC(remote_handle_close)
  DECLFUNC(remote_handle_control)
  DECLFUNC(remote_handle_invoke)
  DECLFUNC(remote_handle_open)
  DECLFUNC(remote_mmap)
  DECLFUNC(remote_munmap)

  DECLFUNC(remote_handle64_close)
  DECLFUNC(remote_handle64_control)
  DECLFUNC(remote_handle64_invoke)
  DECLFUNC(remote_handle64_open)
  DECLFUNC(remote_mmap64)
  DECLFUNC(remote_munmap64)

  DECLFUNC(remote_register_buf)
  DECLFUNC(remote_register_buf_attr)
  DECLFUNC(remote_register_dma_handle)
  DECLFUNC(remote_register_dma_handle_attr)
  DECLFUNC(remote_register_fd)

  DECLFUNC(remote_session_control)
  DECLFUNC(remote_set_mode)

  DECLFUNC(rpcmem_init)
  DECLFUNC(rpcmem_deinit)
  DECLFUNC(rpcmem_alloc)
  DECLFUNC(rpcmem_free)
  DECLFUNC(rpcmem_to_fd)
#undef DECLFUNC

  static const DspRpcAPI* Global();

 private:
  static constexpr const char* rpc_lib_name_ = "libadsprpc.so";
  void* lib_handle_ = nullptr;

#define DECLPTR(p) p##_t* p##_ = nullptr;
  DECLPTR(remote_handle_close)
  DECLPTR(remote_handle_control)
  DECLPTR(remote_handle_invoke)
  DECLPTR(remote_handle_open)
  DECLPTR(remote_mmap)
  DECLPTR(remote_munmap)

  DECLPTR(remote_handle64_close)
  DECLPTR(remote_handle64_control)
  DECLPTR(remote_handle64_invoke)
  DECLPTR(remote_handle64_open)
  DECLPTR(remote_mmap64)
  DECLPTR(remote_munmap64)

  DECLPTR(remote_register_buf)
  DECLPTR(remote_register_buf_attr)
  DECLPTR(remote_register_dma_handle)
  DECLPTR(remote_register_dma_handle_attr)
  DECLPTR(remote_register_fd)

  DECLPTR(remote_session_control)
  DECLPTR(remote_set_mode)

  DECLPTR(rpcmem_init)
  DECLPTR(rpcmem_deinit)
  DECLPTR(rpcmem_alloc)
  DECLPTR(rpcmem_free)
  DECLPTR(rpcmem_to_fd)
#undef DECLPTR

  template <typename T>
  T GetSymbol(const char* sym);
};

}  // namespace hexagon

}  // namespace runtime
}  // namespace tvm

#endif  // __ANDROID__
#endif  // TVM_RUNTIME_HEXAGON_TARGET_HEXAGON_DSPRPCAPI_H_
