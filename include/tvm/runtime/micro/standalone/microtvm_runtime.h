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

#ifndef TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_H_
#define TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#define TVM_MICRO_RUNTIME_API_API extern "C" __attribute__((visibility("default")))

TVM_MICRO_RUNTIME_API_API void* MicroTVMRuntimeCreate(const char* json, size_t json_len,
                                                      void* module);

TVM_MICRO_RUNTIME_API_API void MicroTVMRuntimeDestroy(void* handle);

TVM_MICRO_RUNTIME_API_API void MicroTVMRuntimeSetInput(void* handle, int index, void* tensor);

TVM_MICRO_RUNTIME_API_API void MicroTVMRuntimeRun(void* handle);

TVM_MICRO_RUNTIME_API_API void MicroTVMRuntimeGetOutput(void* handle, int index, void* tensor);

TVM_MICRO_RUNTIME_API_API void* MicroTVMRuntimeDSOModuleCreate(const char* so, size_t so_len);

TVM_MICRO_RUNTIME_API_API void MicroTVMRuntimeDSOModuleDestroy(void* module);

#undef TVM_MICRO_RUNTIME_API_API

#endif  // TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_H_
