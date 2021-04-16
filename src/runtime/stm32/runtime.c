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
 * \file runtime.c
 * \brief A minimal "C" runtime support required by the TVM 
 *        generated C code. Declared in "runtime/c_backend_api.h" 
 *        and "runtime/c_runtime_api.h"
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>

#include "tvm/runtime/c_runtime_api.h"

static char * g_last_error = NULL;

// ====================================================
//   TVMBackendAllocWorkspace
// ====================================================
void *
TVMBackendAllocWorkspace(
  int device_type,
  int device_id,
  uint64_t nbytes,
  int dtype_code_hint,
  int dtype_bits_hint
) {

  void * ptr = NULL;
  assert (nbytes > 0);

#ifdef __arm__
  ptr = malloc (nbytes);
#else //_x86_
  ptr = malloc (nbytes);
#endif

  return ptr;
}

// ====================================================
//   TVMBackendFreeWorkspace
// ====================================================
int
TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  free(ptr);
  return 0;
}

// ====================================================
//   TVMAPISetLastError
// ====================================================
void TVMAPISetLastError(const char * msg) {
  if (g_last_error) {
    free (g_last_error);
  }

  g_last_error = malloc (strlen(msg)+1);
  strcpy (g_last_error, msg);
}

// ====================================================
//   TVMGetLastError
// ====================================================
const char * TVMGetLastError(void) {
  assert (g_last_error);
  return g_last_error;
}
