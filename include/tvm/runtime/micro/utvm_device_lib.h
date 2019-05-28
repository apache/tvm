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
 *  Copyright (c) 2019 by Contributors
 * \file utvm_device_lib.h
 * \brief utvm device library definitions
 */
#ifndef TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_
#define TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

void *(*TVMBackendAllocWorkspace_)(int, int, uint64_t, int, int) =
    (void *(*)(int, int, uint64_t, int, int)) 1;
int (*TVMBackendFreeWorkspace_)(int, int, void*) = (int (*)(int, int, void*)) 1;
void (*TVMAPISetLastError_)(const char*) = (void (*)(const char*)) 1;

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
    int dtype_code_hint, int dtype_bits_hint) {
  return (*TVMBackendAllocWorkspace_)(device_type, device_id, size, dtype_code_hint,
                                      dtype_bits_hint);
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  return (*TVMBackendFreeWorkspace_)(device_type, device_id, ptr);
}

void TVMAPISetLastError(const char* msg) {
  (*TVMAPISetLastError_)(msg);
}

float min(float a, float b) {
  if (a < b) {
    return a;
  } else {
    return b;
  }
}

float max(float a, float b) {  // NOLINT(*)
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_MICRO_UTVM_DEVICE_LIB_H_
