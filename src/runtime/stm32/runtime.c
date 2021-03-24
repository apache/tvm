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
TVMBackendAllocWorkspace (
  int device_type,
  int device_id,
  uint64_t nbytes,
  int dtype_code_hint,
  int dtype_bits_hint
) {
  
  void * ptr = NULL;
  assert (nbytes > 0);

#ifdef __arm__
  //ptr = memalign (64, nbytes);
  ptr = malloc (nbytes);
#else //_x86_
  const int ret = posix_memalign (&ptr, 64, nbytes);
  assert(ret == 0);
#endif
  
  return ptr;
}

// ====================================================
//   TVMBackendFreeWorkspace
// ====================================================
int
TVMBackendFreeWorkspace (int device_type, int device_id, void* ptr)
{
  free(ptr);
  return 0;
}

// ====================================================
//   TVMAPISetLastError
// ====================================================
void TVMAPISetLastError (const char * msg)
{
  if (g_last_error) {
    free (g_last_error);
  }
  
  g_last_error = malloc (strlen(msg)+1);
  strcpy (g_last_error, msg);
}

// ====================================================
//   TVMGetLastError
// ====================================================
const char * TVMGetLastError (void)
{
  assert (g_last_error);
  return g_last_error;
}

#if 0
// =========================================================================
//   _print_indent
// =========================================================================
static void 
_print_indent (int n, FILE * file_p)
{
  int i;
  for (int i = 0; i < n; i++) {
    fprintf (file_p, " ");
  }
}
#endif // 0

// ====================================================
//   _dump_val
// ====================================================
static uint8_t *
_dump_val (
  uint8_t * ptr,
  DLDataType * dltype,
  FILE * file_p
)
{
  double val;

  if (dltype->bits == 8) {  // byte
    if (dltype->code == kDLInt) {
      val = (double)(*(int8_t*)ptr);
      //fprintf (file_p, "%.3g ", val);
    }
    else if (dltype->code == kDLUInt) {
      val = (double)(*(uint8_t*)ptr);
    }
    ptr += 1;
  }
  else if (dltype->bits == 16) {  // 2-byte
    if (dltype->code == kDLInt) {
      val = (double)(*(int16_t*)ptr);
      //fprintf (file_p, "%.3g ", val);
    }
    else if (dltype->code == kDLUInt) {
      val = (double)(*(uint16_t*)ptr);
      //fprintf (file_p, "%.3g ", val);
    }
    ptr += 2;
  }
  else if (dltype->bits == 32) {  // 4-byte
    if (dltype->code == kDLInt) {
      val = (double)(*(int32_t*)ptr);
      //fprintf (file_p, "%.3g ", val);
    }
    else if (dltype->code == kDLUInt) {
      val = (double)(*(uint32_t*)ptr);
      //fprintf (file_p, "%.3g ", val);
    }
    else if (dltype->code == kDLFloat) {
      val = (double)(*(float*)ptr);
      //fprintf (file_p, "%.3f ", val);
    }
    ptr += 4;
  }
  else if (dltype->bits == 64) {  // 8-byte
    if (dltype->code == kDLInt) {
      val = (double)(*(int64_t*)ptr);
    }
    else if (dltype->code == kDLUInt) {
      val = (double)(*(uint64_t*)ptr);
    }
    else if (dltype->code == kDLFloat) {
      val = (double)(*(double*)ptr);
    }
    ptr += 8;
  }

  fprintf (file_p, "%.3g ", val);

  return ptr;
}

// ====================================================
//   _dump_dim
// ====================================================
static uint8_t *
_dump_dim (
  uint8_t * ptr,
  DLDataType * dltype,
  int idx,
  int32_t * curr,
  int64_t * shape,
  int ndim,
  FILE * file_p
)
{
  int i;

  if (idx == ndim-1) {
    //for (i = 0; i < ndim-1; i++) {
    //  fprintf (file_p, "[%d]", curr[i]);
    //}
    //fprintf (file_p, "[]:\n");
    //_print_indent (idx+1, file_p);
    fprintf (file_p, "\t");

    for (i = 0; i < shape[idx]; i++) {
      //fprintf (file_p, "%.3f ", *ptr);
      //ptr++;
      ptr = _dump_val (ptr, dltype, file_p);
    }
    fprintf (file_p, "\n");
  }
  else {

    if (idx == ndim-2) {
      for (i = 0; i < ndim-2; i++) {
	fprintf (file_p, "[%ld]", curr[i]);
      }
      fprintf (file_p, "[][]:\n");
    }
    
    for (i = 0; i < shape[idx]; i++) {
      curr[idx] = i;
      ptr = _dump_dim (ptr, dltype, idx+1, curr, shape, ndim, file_p);
    }
  }
  
  //fprintf (file_p, "\n");

  return ptr;
}

// ====================================================
//   TVMDumpBuffer
//
//   Additional debugging facility.
// ====================================================
void TVMDumpBuffer (const char * name, TVMArrayHandle t, FILE * file_p)
{
  int i;

  int64_t * shape = t->shape;

  int32_t curr[t->ndim];
  //for (i = 0; i < t->ndim; i++) {
  //  curr[i] = 0;
  //}

  fprintf (file_p, "---------------------------------------\n");
  fprintf (file_p, "          %s: shape=(", name);
  for (i = 0; i < t->ndim; i++) {
    if (i < t->ndim-1) {
      fprintf (file_p, "%ld, ", (int32_t)shape[i]);
    }
    else {
      fprintf (file_p, "%ld", (int32_t)shape[i]);
    }
  }
  fprintf (file_p, ")\n");
  fprintf (file_p, "---------------------------------------\n");
    

  //float * data_p = (float*)t->data;
  uint8_t * data_p = (uint8_t*)t->data;

  //
  // Recursively print one level at a time
  //

  //fprintf (file_p, "dim %d: \n", 0);
  data_p = _dump_dim (data_p, &t->dtype, 0, curr, shape, t->ndim, file_p);
  
  return;
}
