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
 * \file tvm/runtime/packed_func.h
 * \brief Type-erased function used across TVM API.
 */
#ifndef TVM_RUNTIME_CRT_PACKED_FUNC_H_
#define TVM_RUNTIME_CRT_PACKED_FUNC_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>

#include "module.h"

static inline DLDataType String2DLDataType(const char* s) {
  DLDataType t;
  // handle None type
  if (strlen(s) == 0) {
    t.bits = 0;
    t.lanes = 0;
    t.code = kTVMOpaqueHandle;
    return t;
  }
  t.bits = 32;
  t.lanes = 1;
  const char* scan;
  if (!strncmp(s, "int", 3)) {
    t.code = kDLInt;
    scan = s + 3;
  } else if (!strncmp(s, "uint", 4)) {
    t.code = kDLUInt;
    scan = s + 4;
  } else if (!strncmp(s, "float", 5)) {
    t.code = kDLFloat;
    scan = s + 5;
  } else if (!strncmp(s, "handle", 6)) {
    t.code = kTVMOpaqueHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s + 6;
  } else if (!strcmp(s, "bool")) {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else {
    scan = s;
    fprintf(stderr, "unknown type %s\n", s);
  }
  char* xdelim;
  uint8_t bits = (uint8_t)(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = (uint16_t)(strtoul(xdelim + 1, &endpt, 10));
  }
  if (!(endpt == s + strlen(s))) {
    fprintf(stderr, "unknown type %s\n", s);
  }
  return t;
}

typedef struct TVMArgs {
  TVMValue values[TVM_CRT_MAX_ARGS];
  int tcodes[TVM_CRT_MAX_ARGS]; /* Data type should be identical to type_codes in TVMPackedCFunc */
  uint32_t values_count;
} TVMArgs;

static inline TVMArgs TVMArgs_Create(TVMValue* values, uint32_t* tcodes, uint32_t values_count) {
  uint32_t idx;
  TVMArgs args;
  memset(&args, 0, sizeof(args));
  for (idx = 0; idx < values_count; idx++) {
    memcpy(args.values + idx, values + idx, sizeof(TVMValue));
    args.tcodes[idx] = tcodes[idx];
  }
  args.values_count = values_count;
  return args;
}

static inline int TVMNoOperation(TVMValue* args, int* type_codes, int num_args,
                                 TVMRetValueHandle ret, void* res) {
  return 0;
}

typedef struct TVMPackedFunc {
  char name[200];
  TVMPackedCFunc fexec;
  TVMArgs args;
  void (*Call)(struct TVMPackedFunc* pf);
  void (*SetArgs)(struct TVMPackedFunc* pf, const struct TVMArgs* args);
} TVMPackedFunc;

static inline void TVMPackedFunc_Call(TVMPackedFunc* pf) {
  pf->fexec(pf->args.values, pf->args.tcodes, pf->args.values_count, 0, 0);
}

static inline void TVMPackedFunc_SetArgs(TVMPackedFunc* pf, const TVMArgs* args) {
  memcpy(&(pf->args), args, sizeof(TVMArgs));
}

TVMPackedFunc* g_fexecs = 0;
uint32_t g_fexecs_count = 0;

// Implement TVMModule::GetFunction
// Put implementation in this file so we have seen the TVMPackedFunc
static inline void TVMModule_GetFunction(TVMModule* mod, const char* name, TVMPackedFunc* pf) {
  int idx;
  memset(pf, 0, sizeof(TVMPackedFunc));
  assert(strlen(name) <= sizeof(pf->name));
  snprintf(pf->name, strlen(name), "%s", name);
  pf->Call = TVMPackedFunc_Call;
  pf->SetArgs = TVMPackedFunc_SetArgs;
  pf->fexec = &TVMNoOperation;
  for (idx = 0; idx < g_fexecs_count; idx++) {
    if (!strcmp(g_fexecs[idx].name, name)) {
      pf->fexec = g_fexecs[idx].fexec;
      break;
    }
  }
  if (idx == g_fexecs_count) {
    fprintf(stderr, "function handle for %s not found\n", name);
  }
}

#endif  // TVM_RUNTIME_CRT_PACKED_FUNC_H_
