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

// LINT_C_FILE

/*!
 * \file src/runtime/crt/common/packed_func.c
 * \brief PackedFunc implementation.
 */
#include <stdio.h>
#include <string.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/packed_func.h>

DLDataType String2DLDataType(const char* s) {
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
  } else if (!strncmp(s, "bfloat", 6)) {
    t.code = kDLBfloat;
    scan = s + 6;
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

int TVMPackedFunc_InitGlobalFunc(TVMPackedFunc* pf, const char* name, const TVMArgs* args) {
  int status = 0;

  pf->Call = &TVMPackedFunc_Call;
  pf->SetArgs = &TVMPackedFunc_SetArgs;

  status = TVMFuncGetGlobal(name, &pf->fexec);
  if (status != 0) {
    return status;
  }

  snprintf(pf->name, sizeof(pf->name), "%s", name);
  TVMPackedFunc_SetArgs(pf, args);
  return status;
}

int TVMPackedFunc_InitModuleFunc(TVMPackedFunc* pf, TVMModuleHandle module, const char* name,
                                 const TVMArgs* args) {
  int status = 0;

  pf->Call = &TVMPackedFunc_Call;
  pf->SetArgs = &TVMPackedFunc_SetArgs;

  status = TVMModGetFunction(module, name, 0, &pf->fexec);
  if (status != 0) {
    return status;
  }

  snprintf(pf->name, sizeof(pf->name), "%s", name);
  TVMPackedFunc_SetArgs(pf, args);
  return status;
}

TVMArgs TVMArgs_Create(TVMValue* values, uint32_t* tcodes, uint32_t values_count) {
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

int TVMPackedFunc_Call(TVMPackedFunc* pf) {
  pf->ret_value.values_count = 1;
  pf->ret_value.tcodes[0] = kTVMNullptr;
  return TVMFuncCall(pf->fexec, pf->args.values, pf->args.tcodes, pf->args.values_count,
                     pf->ret_value.values, pf->ret_value.tcodes);
}

void TVMPackedFunc_SetArgs(TVMPackedFunc* pf, const TVMArgs* args) {
  memcpy(&(pf->args), args, sizeof(TVMArgs));
}

TVMPackedFunc* g_fexecs;
uint32_t g_fexecs_count;
