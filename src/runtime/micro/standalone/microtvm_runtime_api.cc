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

#include "microtvm_runtime_api.h"

#include <stdlib.h>

#include <cassert>
#include <string>

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  void* ptr = nullptr;
  assert(nbytes > 0);
#ifdef __ANDROID__
  ptr = memalign(64, nbytes);
#else
  const int ret = posix_memalign(&ptr, 64, nbytes);
  (void)ret;
  assert(ret == 0);
#endif
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  free(ptr);
  return 0;
}

static thread_local std::string g_last_error;
void TVMAPISetLastError(const char* msg) { g_last_error = msg; }
const char* TVMGetLastError(void) { return g_last_error.c_str(); }

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}
