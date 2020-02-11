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
#ifndef TVM_RUNTIME_PACKED_FUNC_H_
#define TVM_RUNTIME_PACKED_FUNC_H_

// #ifndef _LIBCPP_SGX_NO_IOSTREAMS
// #include <sstream>
// #endif
// #include <dmlc/logging.h>
// #include <functional>
// #include <tuple>
// #include <vector>
// #include <string>
// #include <limits>
// #include <memory>
// #include <utility>
// #include <type_traits>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/ndarray.h>
// #include "node_base.h"

// namespace HalideIR {
// // Forward declare type for extensions
// // The header works fine without depending on this.
// struct Type;
// struct Expr;
// }

// Whether use TVM runtime in header only mode.
#ifndef TVM_RUNTIME_HEADER_ONLY
#define TVM_RUNTIME_HEADER_ONLY 0
#endif

static inline DLDataType String2DLDataType(const char * s) {
  DLDataType t;
  // handle None type
  if (strlen(s) == 0) {
    t.bits = 0; t.lanes = 0; t.code = kTVMOpaqueHandle; // kHandle;
    return t;
  }
  t.bits = 32; t.lanes = 1;
  const char* scan;
  if (!strncmp(s, "int", 3)) {             // s.substr(0, 3) == "int"
    t.code = kDLInt;  scan = s + 3;
  } else if (!strncmp(s, "uint", 4)) {     // s.substr(0, 4) == "uint"
    t.code = kDLUInt; scan = s + 4;
  } else if (!strncmp(s, "float", 5)) { // s.substr(0, 5) == "float"
    t.code = kDLFloat; scan = s + 5;
  } else if (!strncmp(s, "handle", 6)) { // s.substr(0, 6) == "handle"
    t.code = kTVMOpaqueHandle; // kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s + 6;
  } else if (!strcmp(s, "bool")) {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else {
    scan = s;
    // LOG(FATAL) << "unknown type " << s;
    LOGE("unknown type %s\n", s);
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = (uint8_t)(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = (uint16_t)(strtoul(xdelim + 1, &endpt, 10));
  }
  if (!(endpt == s + strlen(s))){
    LOGE("unknown type %s\n", s);
  }
  return t;
}

typedef struct tvm_args_t {
  TVMValue values[TVM_CRT_MAX_ARGS];
  uint32_t tcodes[TVM_CRT_MAX_ARGS];
  uint32_t values_count;
} TVMArgs;

static inline TVMArgs TVMArgs_Create(TVMValue * values, uint32_t * tcodes, uint32_t values_count) {
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

// static inline int TVMNoOperation(TVMValue * args, uint32_t * type_codes, int num_args,
//                                  TVMRetValueHandle ret, void* resource_handle) {
//   return TVM_STATUS_SUCCESS;
// }

static inline int TVMNoOperation(void * args, void * type_codes, int num_args) {
  return TVM_STATUS_SUCCESS;
}

typedef struct packed_func_t {
  // Function (*GetFunction)();
  char name[200];
  TVMPackedCFunc fexec;
  TVMArgs args;
  void (*Call)(struct packed_func_t * pf);
  void (*SetArgs)(struct packed_func_t * pf, const struct tvm_args_t * args);
} PackedFunc;

static inline void PackedFunc_Call(PackedFunc * pf) {
  uint32_t idx;
  char args[200] = {0,};
  for (idx = 0; idx < pf->args.values_count; idx++) {
    char tmp[20];
    sprintf(tmp, "%s,", (pf->args.tcodes[idx]==kArrayHandle) ? "float *" : "unknown");
    strcat(args, tmp);
  }
  args[strlen(args)-1] = '\0';
#if TVM_CRT_DEBUG
  LOGI("calling %s(%s)", pf->name, args);
#endif // TVM_CRT_DEBUG
  pf->fexec(pf->args.values, pf->args.tcodes, pf->args.values_count);
}

static inline void PackedFunc_SetArgs(PackedFunc * pf, const TVMArgs * args) {
  memcpy(&(pf->args), args, sizeof(TVMArgs));
}

PackedFunc fexecs[GRAPH_RUNTIME_MAX_NODES];

void PackedFunc_SetupExecs();

// Implement Module::GetFunction
// Put implementation in this file so we have seen the PackedFunc
static inline void Module_GetFunction(const char * name, PackedFunc * pf) {
  int idx;
  // PackedFunc pf;
  memset(pf, 0, sizeof(PackedFunc));
  strcpy(pf->name, name);
  pf->Call = PackedFunc_Call;
  pf->SetArgs = PackedFunc_SetArgs;
  pf->fexec = &TVMNoOperation;
  for (idx = 0; idx < GRAPH_RUNTIME_MAX_NODES; idx++) {
    if (!strcmp(fexecs[idx].name, name)) {
      pf->fexec = fexecs[idx].fexec;
#if TVM_CRT_DEBUG
      LOGI("setup function %s", name);
#endif // TVM_CRT_DEBUG
      break;
    }
  }
  if (idx==GRAPH_RUNTIME_MAX_NODES) {
    LOGE("function handle for %s not found", name);
  }
  // return pf;
}

#endif  // TVM_RUNTIME_PACKED_FUNC_H_
