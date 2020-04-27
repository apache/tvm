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
 * \file utvm_device_dylib_redirect.cc
 * \brief uTVM dynamic linking stubs
 *
 * This is a library that gets included in each uTVM library.  We redirect
 * each library call into a pre-defined global function pointer, and we patch
 * the correct addresses of each function into the pointers when we load the
 * library.
 */
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stddef.h>

// TODO(weberlo, areusch): compiler errors say volatile qualifier is discarded.
// should we just get rid of em?
void* (* volatile TVMBackendAllocWorkspace_)(int, int, uint64_t, int, int) = NULL;
int (* volatile TVMBackendFreeWorkspace_)(int, int, void*) = NULL;
void (* volatile TVMAPISetLastError_)(const char*) = NULL;

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

void *memset(void *s, int c, size_t n) {
  char *p = (char*) s;  // NOLINT(readability/casting): linter is configured for c++
  while (n > 0) {
    *p = (char) c;  // NOLINT(readability/casting): linter is configured for c++
    p++;
    n--;
  }
  return s;
}

void *memmove(void *to, const void *from, size_t n) {
  // TODO(weberlo, areusch): will need to factor memmove calls into workspace size calculation
  // NOLINTNEXTLINE(readability/casting): linter is configured for c++
  char *temp = (char*) TVMBackendAllocWorkspace(1, 1, (uint64_t) n, 2, 8);
  if (temp == NULL) {
    return NULL;
  }

  const char *from_pp = (char*) from;  // NOLINT(readability/casting): linter is configured for c++
  for (size_t i = 0; i < n; i++) {
    temp[i] = from_pp[i];
  }
  char *to_pp = (char*) to;  // NOLINT(readability/casting): linter is configured for c++
  for (size_t i = 0; i < n; i++) {
    to_pp[i] = temp[i];
  }

  // NOLINTNEXTLINE(readability/casting): linter is configured for c++
  if (TVMBackendFreeWorkspace(1, (uint64_t) 1, (void*) temp) != 0) {
    return NULL;
  }

  return to;
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
