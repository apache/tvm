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
 * \file hexagon_utils.h
 */
#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_COMMON_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_COMMON_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#if defined(__hexagon__)
#include <HAP_farf.h>
#define HEXAGON_PRINT(level, ...) FARF(level, __VA_ARGS__)
#else
#include <cstdio>
#define HEXAGON_PRINT(level, ...) printf(__VA_ARGS__)
#endif

#define HEXAGON_SAFE_CALL(api_call)                                               \
  do {                                                                            \
    int result = api_call;                                                        \
    if (result != 0) {                                                            \
      HEXAGON_PRINT(ERROR, "ERROR: " #api_call " failed with error %d.", result); \
      abort();                                                                    \
    }                                                                             \
  } while (0)

inline bool IsHexagonDevice(DLDevice dev) { return dev.device_type == kDLHexagon; }

constexpr int kHexagonAllocAlignment = 2048;

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_COMMON_H_
