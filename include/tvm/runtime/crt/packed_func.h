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
 * \file tvm/runtime/crt/packed_func.h
 * \brief Type-erased function used across TVM API.
 */
#ifndef TVM_RUNTIME_CRT_PACKED_FUNC_H_
#define TVM_RUNTIME_CRT_PACKED_FUNC_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/platform.h>

#include "crt_config.h"

DLDataType String2DLDataType(const char* s);

typedef struct TVMArgs {
  TVMValue values[TVM_CRT_MAX_ARGS];
  int tcodes[TVM_CRT_MAX_ARGS]; /* Data type should be identical to type_codes in TVMPackedCFunc */
  uint32_t values_count;
} TVMArgs;

TVMArgs TVMArgs_Create(TVMValue* values, uint32_t* tcodes, uint32_t values_count);

typedef struct TVMPackedFunc {
  char name[200];
  TVMFunctionHandle fexec;
  TVMArgs args;
  TVMArgs ret_value;
  int (*Call)(struct TVMPackedFunc* pf);
  void (*SetArgs)(struct TVMPackedFunc* pf, const struct TVMArgs* args);
} TVMPackedFunc;

int TVMPackedFunc_InitGlobalFunc(TVMPackedFunc* pf, const char* name, const TVMArgs* args);
int TVMPackedFunc_InitModuleFunc(TVMPackedFunc* pf, TVMModuleHandle module, const char* name,
                                 const TVMArgs* args);

int TVMPackedFunc_Call(TVMPackedFunc* pf);

void TVMPackedFunc_SetArgs(TVMPackedFunc* pf, const TVMArgs* args);

inline TVMModuleHandle TVMArgs_AsModuleHandle(const TVMArgs* args, size_t index) {
  if (index >= args->values_count) {
    TVMPlatformAbort((tvm_crt_error_t)-1);
  }

  if (args->tcodes[index] != kTVMModuleHandle) {
    TVMPlatformAbort((tvm_crt_error_t)-1);
  }

  return args->values[index].v_handle;
}

extern TVMPackedFunc* g_fexecs;
extern uint32_t g_fexecs_count;

#endif  // TVM_RUNTIME_CRT_PACKED_FUNC_H_
