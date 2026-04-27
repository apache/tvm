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
#define TVM_VERSION "0.24.dev0"

// TVM ships two shared libraries: libtvm_compiler and libtvm_runtime.
// Each exposes its own DLL macro pair.  The two families are defined
// independently so that each can be overridden separately by downstream
// embedders who need custom visibility on only one of the two libraries.
//
// TVM_DLL / TVM_DLL_EXPORT: symbols in libtvm_compiler.
//   - TVM_DLL is dllexport when TVM_EXPORTS is defined (compiler build),
//     dllimport otherwise (downstream consumers, runtime TUs).
//   - TVM_DLL_EXPORT is always dllexport.
//
// TVM_RUNTIME_DLL / TVM_RUNTIME_DLL_EXPORT: symbols in libtvm_runtime.
//   - TVM_RUNTIME_DLL is dllexport when TVM_RUNTIME_EXPORTS is defined
//     (runtime build), dllimport otherwise.
//   - TVM_RUNTIME_DLL_EXPORT is always dllexport.
//
// On non-MSVC platforms the import/export decision is made by the dynamic
// loader, so all four macros expand to visibility("default"). Under
// Emscripten they expand to EMSCRIPTEN_KEEPALIVE.
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

// --- TVM_DLL family (libtvm_compiler) ---
#if !defined(TVM_DLL) && defined(__EMSCRIPTEN__)
#define TVM_DLL EMSCRIPTEN_KEEPALIVE
#define TVM_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_DLL) && defined(_MSC_VER)
#ifdef TVM_EXPORTS
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __declspec(dllimport)
#endif
#define TVM_DLL_EXPORT __declspec(dllexport)
#endif
#ifndef TVM_DLL
#define TVM_DLL __attribute__((visibility("default")))
#define TVM_DLL_EXPORT __attribute__((visibility("default")))
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
