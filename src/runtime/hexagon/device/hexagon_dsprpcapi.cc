/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */
#ifdef __ANDROID__
#include "hexagon_dsprpcapi.h"

#include <dlfcn.h>
#include <dmlc/logging.h>
#include <stdint.h>

#include "hexagon_device_log.h"

namespace tvm {
namespace runtime {

namespace hexagon {

DspRpcAPI::DspRpcAPI() {
  CHECK(lib_handle_ = dlopen(rpc_lib_name_, RTLD_LAZY | RTLD_LOCAL));

#define RESOLVE(n) n##_ = GetSymbol<n##_t*>(#n)
  RESOLVE(remote_handle_close);
  RESOLVE(remote_handle_control);
  RESOLVE(remote_handle_invoke);
  RESOLVE(remote_handle_open);
  RESOLVE(remote_mmap);
  RESOLVE(remote_munmap);

  RESOLVE(remote_handle64_close);
  RESOLVE(remote_handle64_control);
  RESOLVE(remote_handle64_invoke);
  RESOLVE(remote_handle64_open);
  RESOLVE(remote_mmap64);
  RESOLVE(remote_munmap64);

  RESOLVE(remote_register_buf);
  RESOLVE(remote_register_buf_attr);
  RESOLVE(remote_register_dma_handle);
  RESOLVE(remote_register_dma_handle_attr);
  RESOLVE(remote_register_fd);

  RESOLVE(remote_session_control);
  RESOLVE(remote_set_mode);

  RESOLVE(rpcmem_init);
  RESOLVE(rpcmem_deinit);
  RESOLVE(rpcmem_alloc);
  RESOLVE(rpcmem_free);
  RESOLVE(rpcmem_to_fd);
#undef RESOLVE
}

DspRpcAPI::~DspRpcAPI() {
  if (lib_handle_) dlclose(lib_handle_);
}

template <typename T>
T DspRpcAPI::GetSymbol(const char* sym) {
  if (!lib_handle_) {
    TVM_LOGE("error looking up symbol \"%s\": library not loaded", sym);
    return nullptr;
  }
  dlerror();  // Clear any previous errror conditions.
  if (T ret = reinterpret_cast<T>(dlsym(lib_handle_, sym))) return ret;

  const char* err = dlerror();
  const char* err_txt = err ? err : "symbol not found";
  TVM_LOGD("error looking up symbol \"%s\": %s", sym, err_txt);
  return nullptr;
}

const DspRpcAPI* DspRpcAPI::Global() {
  static const DspRpcAPI dsp_api;
  return &dsp_api;
}

}  // namespace hexagon

}  // namespace runtime
}  // namespace tvm

#endif  // __ANDROID__
