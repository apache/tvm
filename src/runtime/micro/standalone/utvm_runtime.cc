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
#include <cassert>

#include "tvm/runtime/micro/standalone/utvm_runtime.h"
#include "utvm_graph_runtime.h"

void* UTVMRuntimeCreate(const char* json, size_t json_len, void* module) {
  return new tvm::micro::MicroGraphRuntime(
      std::string(json, json + json_len),
      reinterpret_cast<tvm::micro::DSOModule*>(module));
}

void UTVMRuntimeDestroy(void* handle) {
  delete reinterpret_cast<tvm::micro::MicroGraphRuntime*>(handle);
}

void UTVMRuntimeSetInput(void* handle, int index, void* tensor) {
  reinterpret_cast<tvm::micro::MicroGraphRuntime*>(handle)->SetInput(
      index, reinterpret_cast<DLTensor*>(tensor));
}

void UTVMRuntimeRun(void* handle) {
  reinterpret_cast<tvm::micro::MicroGraphRuntime*>(handle)->Run();
}

void UTVMRuntimeGetOutput(void* handle, int index, void* tensor) {
  reinterpret_cast<tvm::micro::MicroGraphRuntime*>(handle)->CopyOutputTo(
      index, reinterpret_cast<DLTensor*>(tensor));
}
void* UTVMRuntimeDSOModuleCreate(const char* so, size_t so_len) {
  return new tvm::micro::DSOModule(std::string(so, so + so_len));
}

void UTVMRuntimeDSOModuleDestroy(void* module) {
  delete reinterpret_cast<tvm::micro::DSOModule*>(module);
}
