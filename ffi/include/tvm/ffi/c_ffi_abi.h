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
 * \file tvm/ffi/c_ffi_abi.h
 * \brief This file defines
 */
#ifndef TVM_FFI_C_FFI_ABI_H_
#define TVM_FFI_C_FFI_ABI_H_

#include <dlpack/dlpack.h>
#include <stdint.h>

#if !defined(TVM_FFI_DLL) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_FFI_API EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_FFI_DLL) && defined(_MSC_VER)
#ifdef TVM_FFI_EXPORTS
#define TVM_FFI_DLL __declspec(dllexport)
#else
#define TVM_FFI_DLL __declspec(dllimport)
#endif
#endif
#ifndef TVM_FFI_DLL
#define TVM_FFI_DLL __attribute__((visibility("default")))
#endif

#ifndef TVM_FFI_ALLOW_DYN_TYPE
#define TVM_FFI_ALLOW_DYN_TYPE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
enum class TVMFFITypeIndex : int32_t {
#else
typedef enum {
#endif
  // [Section] On-stack POD Types: [0, kTVMFFIStaticObjectBegin)
  // N.B. `kTVMFFIRawStr` is a string backed by a `\0`-terminated char array,
  // which is not owned by TVMFFIAny. It is required that the following
  // invariant holds:
  // - `Any::type_index` is never `kTVMFFIRawStr`
  // - `AnyView::type_index` can be `kTVMFFIRawStr`
  kTVMFFINone = 0,
  kTVMFFIInt = 1,
  kTVMFFIFloat = 2,
  kTVMFFIOpaquePtr = 3,
  kTVMFFIDataType = 4,
  kTVMFFIDevice = 5,
  kTVMFFIRawStr = 6,
  // [Section] Static Boxed: [kTVMFFIStaticObjectBegin, kTVMFFIDynObjectBegin)
  kTVMFFIStaticObjectBegin = 64,
  kTVMFFIObject = 64,
  kTVMFFIList = 65,
  kTVMFFIDict = 66,
  kTVMFFIError = 67,
  kTVMFFIFunc = 68,
  kTVMFFIStr = 69,
  // [Section] Dynamic Boxed: [kTVMFFIDynObjectBegin, +oo)
  kTVMFFIDynObjectBegin = 128,
#ifdef __cplusplus
};
#else
} TVMFFITypeIndex;
#endif

/*!
 * \brief C-based type of all FFI object types that allocates on heap.
 * \note TVMFFIObject and TVMFFIAny share the common type_index_ header
 */
typedef struct TVMFFIObject {
  /*!
   * \brief type index of the object.
   * \note The type index of Object and Any are shared in FFI.
   */
  int32_t type_index;
  /*! \brief Reference counter of the object. */
  int32_t ref_counter;
  /*! \brief Deleter to be invoked when reference counter goes to zero. */
  void (*deleter)(struct TVMFFIObject* self);
} TVMFFIObject;

/*!
 * \brief C-based type of all on stack Any value.
 *
 * Any value can hold on stack values like int,
 * as well as reference counted pointers to object.
 */
typedef struct TVMFFIAny {
  /*!
   * \brief type index of the object.
   * \note The type index of Object and Any are shared in FFI.
   */
  int32_t type_index;
  /*! \brief length for on-stack Any object, such as small-string */
  int32_t small_len;
  union {                  // 8 bytes
    int64_t v_int64;       // integers
    double v_float64;      // floating-point numbers
    void* v_ptr;           // typeless pointers
    const char* v_c_str;   // raw C-string
    TVMFFIObject* v_obj;   // ref counted objects
    DLDataType v_dtype;    // data type
    DLDevice v_device;     // device
    char v_bytes[8];       // small string
    char32_t v_char32[2];  // small UCS4 string and Unicode
  };
} TVMFFIAny;

#ifdef __cplusplus
}  // TVM_FFI_EXTERN_C
#endif

#endif  // TVM_FFI_C_FFI_ABI_H_
