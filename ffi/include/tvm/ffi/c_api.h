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
 * \file tvm/ffi/c_api.h
 * \brief This file defines the C convention of the FFI convention
 */
#ifndef TVM_FFI_C_API_H_
#define TVM_FFI_C_API_H_

#include <dlpack/dlpack.h>
#include <stdint.h>

#if !defined(TVM_FFI_DLL) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_FFI_DLL EMSCRIPTEN_KEEPALIVE
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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
enum TVMFFITypeIndex : int32_t {
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
  kTVMFFIArray = 65,
  kTVMFFIMap = 66,
  kTVMFFIError = 67,
  kTVMFFIFunc = 68,
  kTVMFFIStr = 69,
  // [Section] Dynamic Boxed: [kTVMFFIDynObjectBegin, +oo)
  // kTVMFFIDynObject is used to indicate that the type index
  // is dynamic and needs to be looked up at runtime
  kTVMFFIDynObjectBegin = 128
#ifdef __cplusplus
};
#else
} TVMFFITypeIndex;
#endif

/*! \brief Handle to Object from C API's pov */
typedef void* TVMFFIObjectHandle;

/*!
 * \brief C-based type of all FFI object types that allocates on heap.
 * \note TVMFFIObject and TVMFFIAny share the common type_index header
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
  void (*deleter)(void* self);
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

/*! \brief Safe byte array */
typedef struct {
  int64_t num_bytes;
  const char* bytes;
} TVMFFIByteArray;

/*!
 * \brief Runtime type information for object type checking.
 */
typedef struct {
  /*!
   *\brief The runtime type index,
   * It can be allocated during runtime if the type is dynamic.
   */
  int32_t type_index;
  /*! \brief number of parent types in the type hierachy. */
  int32_t type_depth;
  /*! \brief the unique type key to identify the type. */
  const char* type_key;
  /*! \brief Cached hash value of the type key, used for consistent structural hashing. */
  uint64_t type_key_hash;
  /*!
   * \brief type_acenstors[depth] stores the type_index of the acenstors at depth level
   * \note To keep things simple, we do not allow multiple inheritance so the
   *       hieracy stays as a tree
   */
  const int32_t* type_acenstors;
} TVMFFITypeInfo;

//------------------------------------------------------------
// Section: User APIs to interact with the FFI
//------------------------------------------------------------
/*!
 * \brief Type that defines C-style safe call convention
 *
 * Safe call explicitly catches exception on function boundary.
 *
 * \param func The function handle
 * \param num_args Number if input arguments
 * \param args The input arguments to the call.
 * \param result Store output result
 *
 * \return The call return 0 if call is successful.
 *  It returns non-zero value if there is an error.
 *
 *  Possible return error of the API functions:
 *  * 0: success
 *  * -1: error happens, can be retrieved by TVMFFIGetLastError
 *  * -2: a frontend error occurred and recorded in the frontend.
 *
 * \note We decided to leverage TVMFFIGetLastError and TVMFFISetLastError
 *  for C function error propagation. This design choice, while
 *  introducing a dependency for TLS runtime, simplifies error
 *  propgation in chains of calls in compiler codegen.
 *  As we do not need to propagate error through argument but simply
 *  set them in the runtime environment.
 */
typedef int (*TVMFFISafeCallType)(void* func, int32_t num_args, const TVMFFIAny* args,
                                  TVMFFIAny* result);

/*!
 * \brief Free an object handle by decreasing reference
 * \param obj The object handle.
 * \note Internally we decrease the reference counter of the object.
 *       The object will be freed when every reference to the object are removed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIObjectFree(TVMFFIObjectHandle obj);

/*!
 * \brief Move the last error from the environment to result.
 *
 * \param result The result error.
 *
 * \note This function clears the error stored in the TLS.
 */
TVM_FFI_DLL void TVMFFIMoveFromLastError(TVMFFIAny* result);

/*!
 * \brief Set the last error in TLS, which can be fetched by TVMFFIGetLastError.
 *
 * \param error_view The error in format of any view.
 *        It can be an object, or simply a raw c_str.
 * \note
 */
TVM_FFI_DLL void TVMFFISetLastError(const TVMFFIAny* error_view);

//------------------------------------------------------------
// Section: Backend noexcept functions for internal use
//
// These functions are used internally and do not throw error
// instead the error will be logged and abort the process
// These are function are being called in startup or exit time
// so exception handling do not apply
//------------------------------------------------------------
/*!
 * \brief Get stack traceback in a string.
 * \param filaname The current file name.
 * \param func The current function
 * \param lineno The current line number
 * \return The traceback string
 *
 * \note filename func and lino are only used as a backup info, most cases they are not needed.
 *  The return value is set to const char* to be more compatible across dll boundaries.
 */
TVM_FFI_DLL const char* TVMFFITraceback(const char* filename, const char* func, int lineno);

/*!
 * \brief Initialize the type info during runtime.
 *
 *  When the function is first time called for a type,
 *  it will register the type to the type table in the runtime.
 *
 *  If the static_tindex is non-negative, the function will
 *  allocate a runtime type index.
 *  Otherwise, we will populate the type table and return the static index.
 *
 * \param type_key The type key.
 * \param static_type_index Static type index if any, can be -1, which means this is a dynamic index
 * \param num_child_slots Number of slots reserved for its children.
 * \param child_slots_can_overflow Whether to allow child to overflow the slots.
 * \param parent_type_index Parent type index, pass in -1 if it is root.
 * \param result The output type index
 *
 * \return 0 if success, -1 if error occured
 */
TVM_FFI_DLL int32_t TVMFFIGetOrAllocTypeIndex(const char* type_key, int32_t static_type_index,
                                              int32_t type_depth, int32_t num_child_slots,
                                              int32_t child_slots_can_overflow,
                                              int32_t parent_type_index);
/*!
 * \brief Get dynamic type info by type index.
 *
 * \param type_index The type index
 * \param result The output type information
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL const TVMFFITypeInfo* TVMFFIGetTypeInfo(int32_t type_index);

#ifdef __cplusplus
}  // TVM_FFI_EXTERN_C
#endif
#endif  // TVM_FFI_C_API_H_
