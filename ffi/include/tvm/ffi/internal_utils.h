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
 * \file tvm/ffi/internal_utils.h
 * \brief Utility functions and macros for internal use, not meant for
 */
#ifndef TVM_FFI_INTERNAL_UTILS_H_
#define TVM_FFI_INTERNAL_UTILS_H_

#include <tvm/ffi/c_ffi_abi.h>

#include <cstddef>

#if defined(_MSC_VER)
#define TVM_FFI_INLINE __forceinline
#else
#define TVM_FFI_INLINE inline __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define TVM_FFI_UNREACHABLE() __assume(false)
#else
#define TVM_FFI_UNREACHABLE() __builtin_unreachable()
#endif

namespace tvm {
namespace ffi {

namespace details {

/********** Atomic Operations *********/

TVM_FFI_INLINE int32_t AtomicIncrementRelaxed(int32_t* ptr) {
#ifdef _MSC_VER
  return _InterlockedIncrement(reinterpret_cast<volatile long*>(ptr)) - 1;
#else
  return __atomic_fetch_add(ptr, 1, __ATOMIC_RELAXED);
#endif
}

TVM_FFI_INLINE int32_t AtomicDecrementRelAcq(int32_t* ptr) {
#ifdef _MSC_VER
  return _InterlockedDecrement(reinterpret_cast<volatile long*>(ptr)) + 1;
#else
  return __atomic_fetch_sub(ptr, 1, __ATOMIC_ACQ_REL);
#endif
}

TVM_FFI_INLINE int32_t AtomicLoadRelaxed(const int32_t* ptr) {
  int32_t* raw_ptr = const_cast<int32_t*>(ptr);
#ifdef _MSC_VER
  // simply load the variable ptr out
  return (reinterpret_cast<const volatile long*>(raw_ptr))[0];
#else
  return __atomic_load_n(raw_ptr, __ATOMIC_RELAXED);
#endif
}
}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_INTERNAL_UTILS_H_
