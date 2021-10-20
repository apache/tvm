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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

#include <memory>

#define TVM_BUNDLE_FUNCTION __attribute__((visibility("default")))

extern "C" {

TVM_BUNDLE_FUNCTION void* tvm_runtime_create(const char* build_graph_json,
                                             const char* build_params_bin,
                                             const uint64_t build_params_bin_len) {
  const int build_graph_json_len = strlen(build_graph_json);
  const std::string json_data(&build_graph_json[0], &build_graph_json[0] + build_graph_json_len);
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  int device_type = kDLCPU;
  int device_id = 0;

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
      json_data, mod_syslib, device_type, device_id);
  TVMByteArray params;
  params.data = reinterpret_cast<const char*>(&build_params_bin[0]);
  params.size = build_params_bin_len;
  mod.GetFunction("load_params")(params);
  return new tvm::runtime::Module(mod);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_destroy(void* handle) {
  delete reinterpret_cast<tvm::runtime::Module*>(handle);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_set_input(void* handle, const char* name, void* tensor) {
  reinterpret_cast<tvm::runtime::Module*>(handle)->GetFunction("set_input")(
      name, reinterpret_cast<DLTensor*>(tensor));
}

TVM_BUNDLE_FUNCTION void tvm_runtime_run(void* handle) {
  reinterpret_cast<tvm::runtime::Module*>(handle)->GetFunction("run")();
}

TVM_BUNDLE_FUNCTION void tvm_runtime_get_output(void* handle, int index, void* tensor) {
  reinterpret_cast<tvm::runtime::Module*>(handle)->GetFunction("get_output")(
      index, reinterpret_cast<DLTensor*>(tensor));
}
}
