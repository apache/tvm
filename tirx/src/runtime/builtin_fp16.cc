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
 * \file builtin_fp16.cc
 * \brief Functions for conversion between fp32 and fp16
 *
 * JIT-fallback rationale
 * ----------------------
 * This file exports TVM_RUNTIME_DLL symbols (__gnu_f2h_ieee, __gnu_h2f_ieee,
 * __truncdfhf2) so that loaded/JIT-compiled modules can resolve fp16 ABI calls
 * at load time.  In JIT and AOT-module-load scenarios the host process may not
 * have libgcc or compiler-rt fp16 builtins linked in; TVM codegen emits calls
 * to these symbols for fp16 ops, and without this fallback those calls would
 * fail symbol resolution at load time on such platforms.
 *
 * The TVM_FFI_WEAK attribute means the OS dynamic linker prefers the platform's
 * own compiler-rt symbols when they are present; TVM's copy only "wins" when no
 * other implementation is linked into the process.
 *
 * The inline implementations are provided by
 * 3rdparty/compiler-rt/builtin_fp16.h (included below as <builtin_fp16.h>,
 * which is on the CMake SYSTEM include path).  This file is NOT a duplicate of
 * that header -- it provides runtime symbol export, not inline-only usage.
 */
#include <builtin_fp16.h>
#include <tvm/runtime/base.h>

extern "C" {

// disable under msvc
#ifndef _MSC_VER

TVM_RUNTIME_DLL TVM_FFI_WEAK uint16_t __gnu_f2h_ieee(float a) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
}

TVM_RUNTIME_DLL TVM_FFI_WEAK float __gnu_h2f_ieee(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

TVM_RUNTIME_DLL TVM_FFI_WEAK uint16_t __truncdfhf2(double a) {
  return __truncXfYf2__<double, uint64_t, 52, uint16_t, uint16_t, 10>(a);
}

#else

TVM_RUNTIME_DLL uint16_t __gnu_f2h_ieee(float a) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
}

TVM_RUNTIME_DLL float __gnu_h2f_ieee(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

TVM_RUNTIME_DLL uint16_t __truncdfhf2(double a) {
  return __truncXfYf2__<double, uint64_t, 52, uint16_t, uint16_t, 10>(a);
}

#endif
}
