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

/*
 * \file tvm/runtime/base.h
 * \brief base macros
 */
#ifndef TVM_RUNTIME_BASE_H_
#define TVM_RUNTIME_BASE_H_

// TVM runtime fully relies on TVM FFI C API
// we will avoid defining extra C APIs here
#include <tvm/ffi/c_api.h>

// TVM version
#define TVM_VERSION "0.21.dev0"

// define extra macros for TVM DLL exprt
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define TVM_DLL EMSCRIPTEN_KEEPALIVE
#endif

// helper macro to suppress unused warning
#if defined(__GNUC__)
#define TVM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define TVM_ATTRIBUTE_UNUSED
#endif

#ifndef TVM_DLL
#ifdef _WIN32
#ifdef TVM_EXPORTS
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __declspec(dllimport)
#endif
#else
#define TVM_DLL __attribute__((visibility("default")))
#endif
#endif

#endif  // TVM_RUNTIME_BASE_H_
