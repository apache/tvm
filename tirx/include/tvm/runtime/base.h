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

// TIRx version. Overridable at build time via -DTVM_TIRX_VERSION="..." (scikit-build-core
// passes the setuptools_scm-resolved version through CMake). The literal below is the
// fallback for a bare build with no override.
#ifndef TVM_TIRX_VERSION
#define TVM_TIRX_VERSION "0.1.0a0"
#endif

// Core runtime visibility is independent of TIRx compiler visibility.
// TVM_RUNTIME_EXPORTS selects core-runtime exports on MSVC; downstream
// consumers import them. Embedders may override each library policy.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// Core IR, TIRx and shared compiler infrastructure live in libtvm_tirx_compiler.
#ifndef TVM_TIRX_DLL
#if defined(__EMSCRIPTEN__)
#define TVM_TIRX_DLL EMSCRIPTEN_KEEPALIVE
#elif defined(_MSC_VER)
#ifdef TVM_TIRX_EXPORTS
#define TVM_TIRX_DLL __declspec(dllexport)
#else
#define TVM_TIRX_DLL __declspec(dllimport)
#endif
#else
#define TVM_TIRX_DLL __attribute__((visibility("default")))
#endif
#endif

// --- TVM_RUNTIME_DLL family (libtvm_runtime) ---
#if !defined(TVM_RUNTIME_DLL) && defined(__EMSCRIPTEN__)
#define TVM_RUNTIME_DLL EMSCRIPTEN_KEEPALIVE
#define TVM_RUNTIME_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_RUNTIME_DLL) && defined(_MSC_VER)
#ifdef TVM_RUNTIME_EXPORTS
#define TVM_RUNTIME_DLL __declspec(dllexport)
#else
#define TVM_RUNTIME_DLL __declspec(dllimport)
#endif
#define TVM_RUNTIME_DLL_EXPORT __declspec(dllexport)
#endif
#ifndef TVM_RUNTIME_DLL
#define TVM_RUNTIME_DLL __attribute__((visibility("default")))
#define TVM_RUNTIME_DLL_EXPORT __attribute__((visibility("default")))
#endif

#endif  // TVM_RUNTIME_BASE_H_
