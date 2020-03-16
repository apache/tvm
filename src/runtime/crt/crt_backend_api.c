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

#include <tvm/runtime/c_backend_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  void* ptr = 0;
  assert(nbytes > 0);
  unsigned int dtype_bytes = dtype_bits_hint / 8;
#ifdef __ANDROID__
  ptr = memalign(64, nbytes * dtype_bytes);
#else
  const int ret = posix_memalign(&ptr, 64, nbytes * dtype_bytes);
  (void)ret;
  assert(ret == 0);
#endif
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  free(ptr);
  return 0;
}

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}

int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  snprintf(g_fexecs[g_fexecs_count].name, sizeof(g_fexecs[g_fexecs_count].name), name);
  g_fexecs[g_fexecs_count].fexec = ptr;
  g_fexecs_count++;
  return 0;
}
