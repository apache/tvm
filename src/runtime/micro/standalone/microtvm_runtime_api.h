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
#ifndef TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_API_H_
#define TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_API_H_

#include <stdint.h>
#include <stdlib.h>

#include <cassert>

// The subset of the TVM runtime API that is implemented by the minimal runtime API.

#define TVM_MICRO_RUNTIME_API_BACKEND_API extern "C" __attribute__((weak, visibility("default")))

TVM_MICRO_RUNTIME_API_BACKEND_API int TVMBackendFreeWorkspace(int device_type, int device_id,
                                                              void* ptr);

TVM_MICRO_RUNTIME_API_BACKEND_API void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                                                 uint64_t nbytes,
                                                                 int dtype_code_hint,
                                                                 int dtype_bits_hint);

typedef struct {
  void* sync_handle;
  int32_t num_task;
} TVMParallelGroupEnv;

typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv, void* cdata);

TVM_MICRO_RUNTIME_API_BACKEND_API int TVMBackendParallelLaunch(FTVMParallelLambda flambda,
                                                               void* cdata, int num_task);

TVM_MICRO_RUNTIME_API_BACKEND_API void TVMAPISetLastError(const char* msg);
TVM_MICRO_RUNTIME_API_BACKEND_API const char* TVMGetLastError(void);

#undef TVM_MICRO_RUNTIME_API_BACKEND_API

#endif  // TVM_RUNTIME_MICRO_STANDALONE_MICROTVM_RUNTIME_API_H_
