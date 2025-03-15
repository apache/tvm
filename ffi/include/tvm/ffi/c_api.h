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
  // [Section] On-stack POD and special types: [0, kTVMFFIStaticObjectBegin)
  // N.B. `kTVMFFIRawStr` is a string backed by a `\0`-terminated char array,
  // which is not owned by TVMFFIAny. It is required that the following
  // invariant holds:
  // - `Any::type_index` is never `kTVMFFIRawStr`
  // - `AnyView::type_index` can be `kTVMFFIRawStr`
  //
  // NOTE: kTVMFFIAny is a root type of everything
  // we include it so TypeIndex captures all possible runtime values.
  // `kTVMFFIAny` code will never appear in Any::type_index.
  // However, it may appear in field annotations during reflection.
  //
  kTVMFFIAny = -1,
  kTVMFFINone = 0,
  kTVMFFIInt = 1,
  kTVMFFIBool = 2,
  kTVMFFIFloat = 3,
  kTVMFFIOpaquePtr = 4,
  kTVMFFIDataType = 5,
  kTVMFFIDevice = 6,
  kTVMFFIDLTensorPtr = 7,
  kTVMFFIRawStr = 8,
  kTVMFFIByteArrayPtr = 9,
  // [Section] Static Boxed: [kTVMFFIStaticObjectBegin, kTVMFFIDynObjectBegin)
  // roughly order in terms of their ptential dependencies
  kTVMFFIStaticObjectBegin = 64,
  kTVMFFIObject = 64,
  kTVMFFIStr = 65,
  kTVMFFIError = 66,
  kTVMFFIFunc = 67,
  kTVMFFIArray = 68,
  kTVMFFIMap = 69,
  kTVMFFIShapeTuple = 70,
  kTVMFFINDArray = 71,
  kTVMFFIRuntimeModule = 72,
  kTVMFFIStaticObjectEnd,
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
  /*!
   * \brief length for on-stack Any object, such as small-string
   * \note This field is reserved for future compact.
   */
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
  const char* data;
  int64_t size;
} TVMFFIByteArray;

/*!
 * \brief Type that defines C-style safe call convention
 *
 * Safe call explicitly catches exception on function boundary.
 *
 * \param self The function handle
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
typedef int (*TVMFFISafeCallType)(void* self, int32_t num_args, const TVMFFIAny* args,
                                  TVMFFIAny* result);

/*!
 * \brief Getter that can take address of a field and set the result.
 * \param field The raw address of the field.
 * \param result Stores the result.
 */
typedef int (*TVMFFIFieldGetter)(void* field, TVMFFIAny* result);

/*!
 * \brief Getter that can take address of a field and set to value.
 * \param field The raw address of the field.
 * \param value The value to set.
 */
typedef int (*TVMFFIFieldSetter)(void* field, const TVMFFIAny* value);

/*!
 * \brief Information support for optional object reflection.
 */
typedef struct {
  /*! \brief The name of the field. */
  const char* name;
  /*!
   * \brief Records the static type kind of the field.
   *
   * Possible values:
   *
   *  - TVMFFITypeIndex::kTVMFFIObject for general objects
   *    - The value is nullable when kTVMFFIObject is chosen
   * - static object type kinds such as Map, Dict, String
   * - POD type index
   * - TVMFFITypeIndex::kTVMFFIAny if we don't have specialized info
   *   about the field.
   *
   * \note This information is helpful in designing serializer
   * of the field. As it helps to narrow down the type of the
   * object. It also helps to provide opportunities to enable
   * short-cut access to the field.
   */
  int32_t field_static_type_index;
  /*!
   * \brief Mark whether field is readonly.
   */
  int32_t readonly;
  /*!
   * \brief Byte offset of the field.
   */
  int64_t byte_offset;
  /*! \brief The getter to access the field. */
  TVMFFIFieldGetter getter;
  /*! \brief The setter to access the field. */
  TVMFFIFieldSetter setter;
} TVMFFIFieldInfo;

/*!
 * \brief Method information that can appear in reflection table.
 */
typedef struct {
  /*! \brief The name of the field. */
  const char* name;
  /*!
   * \brief The method wrapped as Function
   * \note The first argument to the method is always the self.
   */
  TVMFFIObjectHandle method;
} TVMFFIMethodInfo;

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
  /*! \brief number of reflection accessible fields. */
  int32_t num_fields;
  /*! \brief number of reflection acccesible methods. */
  int32_t num_methods;
  /*! \brief The reflection field information. */
  TVMFFIFieldInfo* fields;
  /*! \brief The reflection method. */
  TVMFFIMethodInfo* methods;
} TVMFFITypeInfo;

//------------------------------------------------------------
// Section: User APIs to interact with the FFI
//------------------------------------------------------------
/*!
 * \brief Free an object handle by decreasing reference
 * \param obj The object handle.
 * \note Internally we decrease the reference counter of the object.
 *       The object will be freed when every reference to the object are removed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIObjectFree(TVMFFIObjectHandle obj);

/*!
 * \brief Create a FFIFunc by passing in callbacks from C callback.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param self The resource handle of the C callback.
 * \param safe_call The C callback implementation
 * \param deleter deleter to recycle
 * \param out The output of the function.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFuncCreate(void* self, TVMFFISafeCallType safe_call,
                                 void (*deleter)(void* self), TVMFFIObjectHandle* out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFuncSetGlobal(const char* name, TVMFFIObjectHandle f, int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFuncGetGlobal(const char* name, TVMFFIObjectHandle* out);

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
 */
TVM_FFI_DLL void TVMFFISetLastError(const TVMFFIAny* error_view);

/*!
 * \brief Convert type key to type index.
 * \param type_key The key of the type.
 * \param out_tindex the corresponding type index.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFITypeKey2Index(const char* type_key, int32_t* out_tindex);

/*!
 * \brief Register type field information for rutnime reflection.
 * \param type_index The type index
 * \param info The field info to be registered.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIRegisterTypeField(int32_t type_index, const TVMFFIFieldInfo* info);

/*!
 * \brief Register type method information for rutnime reflection.
 * \param type_index The type index
 * \param info The method info to be registered.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIRegisterTypeMethod(int32_t type_index, const TVMFFIMethodInfo* info);

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
 * \return The type info
 */
TVM_FFI_DLL const TVMFFITypeInfo* TVMFFIGetTypeInfo(int32_t type_index);

#ifdef __cplusplus
}  // TVM_FFI_EXTERN_C
#endif
#endif  // TVM_FFI_C_API_H_
