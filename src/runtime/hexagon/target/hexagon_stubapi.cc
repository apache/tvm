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

#ifdef __ANDROID__
#include "hexagon_stubapi.h"

#include <dlfcn.h>
#include <dmlc/logging.h>
#include <stdint.h>
#include <sys/stat.h>

#include "hexagon_target_log.h"

namespace tvm {
namespace runtime {
namespace hexagon {

StubAPI::StubAPI() {
  struct stat sb;
  if (!stat("/dev/subsys_cdsp", &sb)) {
    enable_domains_ = true;
    TVM_LOGD("CDSP subsystem present");
  } else if (!stat("/dev/subsys_adsp", &sb)) {
    enable_domains_ = false;
    TVM_LOGD("ADSP subsystem present");
  }

  constexpr auto domain_lib_name = "libtvm_hexagon_remote_stub.so";
  constexpr auto nondomain_lib_name = "libtvm_hexagon_remote_nd_stub.so";

  const char* lib_name =
      enable_domains_ ? domain_lib_name : nondomain_lib_name;
  CHECK(lib_handle_ = dlopen(lib_name, RTLD_LAZY | RTLD_LOCAL));

#define RESOLVE(fn) p##fn##_ = GetSymbol<fn##_t*>(#fn)
  if (enable_domains_) {
    RESOLVE(tvm_hexagon_remote_load_library);
    RESOLVE(tvm_hexagon_remote_release_library);
    RESOLVE(tvm_hexagon_remote_get_symbol);
    RESOLVE(tvm_hexagon_remote_kernel);
    RESOLVE(tvm_hexagon_remote_open);
    RESOLVE(tvm_hexagon_remote_close);
    RESOLVE(tvm_hexagon_remote_alloc_vtcm);
    RESOLVE(tvm_hexagon_remote_free_vtcm);
    RESOLVE(tvm_hexagon_remote_call_mmap64);
  } else {
    RESOLVE(tvm_hexagon_remote_nd_load_library);
    RESOLVE(tvm_hexagon_remote_nd_release_library);
    RESOLVE(tvm_hexagon_remote_nd_get_symbol);
    RESOLVE(tvm_hexagon_remote_nd_kernel);
    RESOLVE(tvm_hexagon_remote_nd_open);
    RESOLVE(tvm_hexagon_remote_nd_call_mmap64);
  }

  RESOLVE(rpcmem_init);
  RESOLVE(rpcmem_deinit);
  RESOLVE(rpcmem_alloc);
  RESOLVE(rpcmem_free);
  RESOLVE(rpcmem_to_fd);
#undef RESOLVE
}

StubAPI::~StubAPI() {
  if (lib_handle_) dlclose(lib_handle_);
}

template <typename T>
T StubAPI::GetSymbol(const char* sym) {
  if (!lib_handle_) {
    TVM_LOGE("error looking up symbol \"%s\": library not loaded", sym);
    return nullptr;
  }
  dlerror();  // Clear any previous errror conditions.
  if (T ret = reinterpret_cast<T>(dlsym(lib_handle_, sym))) {
    return ret;
  }

  const char* err = dlerror();
  const char* err_txt = err ? err : "symbol not found";
  TVM_LOGE("error looking up symbol \"%s\": %s", sym, err_txt);
  return nullptr;
}

const StubAPI* StubAPI::Global() {
  static const StubAPI stub_api;
  return &stub_api;
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // __ANDROID__
