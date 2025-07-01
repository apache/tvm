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

// Macros to do weak linking
#ifdef _MSC_VER
#define TVM_FFI_WEAK __declspec(selectany)
#else
#define TVM_FFI_WEAK __attribute__((weak))
#endif

// Defines two macros
// TVM_FFI_DLL: marks the function as a DLL export/import
//              depending on whether TVM_FFI_EXPORTS is defined
// TVM_FFI_DLL_EXPORT: always marks the function as a DLL export
#if !defined(TVM_FFI_DLL) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_FFI_DLL EMSCRIPTEN_KEEPALIVE
#define TVM_FFI_DLL_EXPORT EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_FFI_DLL) && defined(_MSC_VER)
#ifdef TVM_FFI_EXPORTS
#define TVM_FFI_DLL __declspec(dllexport)
#else
#define TVM_FFI_DLL __declspec(dllimport)
#endif
#define TVM_FFI_DLL_EXPORT __declspec(dllexport)
#endif
#ifndef TVM_FFI_DLL
#define TVM_FFI_DLL __attribute__((visibility("default")))
#define TVM_FFI_DLL_EXPORT __attribute__((visibility("default")))
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
  /*
   * \brief The root type of all FFI objects.
   *
   * We include it so TypeIndex captures all possible runtime values.
   * `kTVMFFIAny` code will never appear in Any::type_index.
   * However, it may appear in field annotations during reflection.
   */
  kTVMFFIAny = -1,
  /*! \brief None/nullptr value */
  kTVMFFINone = 0,
  /*! \brief POD int value */
  kTVMFFIInt = 1,
  /*! \brief POD bool value */
  kTVMFFIBool = 2,
  /*! \brief POD float value */
  kTVMFFIFloat = 3,
  /*! \brief Opaque pointer object */
  kTVMFFIOpaquePtr = 4,
  /*! \brief DLDataType */
  kTVMFFIDataType = 5,
  /*! \brief DLDevice */
  kTVMFFIDevice = 6,
  /*! \brief DLTensor* */
  kTVMFFIDLTensorPtr = 7,
  /*! \brief const char**/
  kTVMFFIRawStr = 8,
  /*! \brief TVMFFIByteArray* */
  kTVMFFIByteArrayPtr = 9,
  /*! \brief R-value reference to ObjectRef */
  kTVMFFIObjectRValueRef = 10,
  /*! \brief Start of statically defined objects. */
  kTVMFFIStaticObjectBegin = 64,
  /*!
   * \brief Object, all objects starts with TVMFFIObject as its header.
   * \note We will also add other fields
   */
  kTVMFFIObject = 64,
  /*!
   * \brief String object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
   */
  kTVMFFIStr = 65,
  /*!
   * \brief Bytes object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
   */
  kTVMFFIBytes = 66,
  /*! \brief Error object. */
  kTVMFFIError = 67,
  /*! \brief Function object. */
  kTVMFFIFunction = 68,
  /*! \brief Array object. */
  kTVMFFIArray = 69,
  /*! \brief Map object. */
  kTVMFFIMap = 70,
  /*!
   * \brief Shape object, layout = { TVMFFIObject, { const int64_t*, size_t }, ... }
   */
  kTVMFFIShape = 71,
  /*!
   * \brief NDArray object, layout = { TVMFFIObject, DLTensor, ... }
   */
  kTVMFFINDArray = 72,
  /*! \brief Runtime module object. */
  kTVMFFIModule = 73,
  kTVMFFIStaticObjectEnd,
  // [Section] Dynamic Boxed: [kTVMFFIDynObjectBegin, +oo)
  /*! \brief Start of type indices that are allocated at runtime. */
  kTVMFFIDynObjectBegin = 128
#ifdef __cplusplus
};
#else
} TVMFFITypeIndex;
#endif

/*! \brief Handle to Object from C API's pov */
typedef void* TVMFFIObjectHandle;

/*!
 * \brief C-based type of all FFI object header that allocates on heap.
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
  union {
    /*! \brief Deleter to be invoked when reference counter goes to zero. */
    void (*deleter)(struct TVMFFIObject* self);
    /*!
     * \brief auxilary field to TVMFFIObject is always 8 bytes aligned.
     * \note This helps us to ensure cross platform compatibility.
     */
    int64_t __ensure_align;
  };
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
    uint64_t v_uint64;     // uint64 repr mainly used for hashing
  };
} TVMFFIAny;

/*!
 * \brief Byte array data structure used by String and Bytes.
 *
 * String and Bytes object layout = { TVMFFIObject, TVMFFIByteArray, ... }
 *
 * \note This byte array data structure layout differs in 32/64 bit platforms.
 *       as size_t equals to the size of the pointer, use this convetion to
 *       be consistent with std::string and also avoid need to calculate padding
 *       for the size field on 32-bit platforms.
 *       The FFI binding should be careful when treating this ABI.
 */
typedef struct {
  const char* data;
  size_t size;
} TVMFFIByteArray;

/*!
 * \brief Shape cell used in shape object following header.
 */
typedef struct {
  const int64_t* data;
  size_t size;
} TVMFFIShapeCell;

/*!
 * \brief Error cell used in error object following header.
 */
typedef struct {
  /*! \brief The kind of the error. */
  TVMFFIByteArray kind;
  /*! \brief The message of the error. */
  TVMFFIByteArray message;
  /*!
   * \brief The traceback of the error.
   */
  TVMFFIByteArray traceback;
  /*!
   * \brief Function handle to update the traceback of the error.
   * \param self The self object handle.
   * \param traceback The traceback to update.
   */
  void (*update_traceback)(TVMFFIObjectHandle self, const TVMFFIByteArray* traceback);
} TVMFFIErrorCell;

/*!
 * \brief Type that defines C-style safe call convention
 *
 * Safe call explicitly catches exception on function boundary.
 *
 * \param handle The function handle
 * \param num_args Number of input arguments
 * \param args The input arguments to the call.
 * \param result Store output result.
 *
 * IMPORTANT: caller must initialize result->type_index to be kTVMFFINone,
 * or any other value smaller than kTVMFFIStaticObjectBegin.
 *
 * \return The call returns 0 if call is successful.
 *  It returns non-zero value if there is an error.
 *
 *  Possible return error of the API functions:
 *  * 0: success
 *  * -1: error happens, can be retrieved by TVMFFIErrorMoveFromRaised
 *  * -2: a frontend error occurred and recorded in the frontend.
 *
 * \note We decided to leverage TVMFFIErrorMoveFromRaised and TVMFFIErrorSetRaised
 *  for C function error propagation. This design choice, while
 *  introducing a dependency for TLS runtime, simplifies error
 *  propgation in chains of calls in compiler codegen.
 *  As we do not need to propagate error through argument but simply
 *  set them in the runtime environment.
 *
 * \sa TVMFFIErrorMoveFromRaised
 * \sa TVMFFIErrorSetRaised
 * \sa TVMFFIErrorSetRaisedFromCStr
 */
typedef int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* result);

/*!
 * \brief Object cell for function object following header.
 */
typedef struct {
  /*! \brief A C API compatible call with exception catching. */
  TVMFFISafeCallType safe_call;
} TVMFFIFunctionCell;

/*!
 * \brief Getter that can take address of a field and set the result.
 * \param field The raw address of the field.
 * \param result Stores the result.
 * \return 0 when success, nonzero when failure happens
 */
typedef int (*TVMFFIFieldGetter)(void* field, TVMFFIAny* result);

/*!
 * \brief Getter that can take address of a field and set to value.
 * \param field The raw address of the field.
 * \param value The value to set.
 * \return 0 when success, nonzero when failure happens
 */
typedef int (*TVMFFIFieldSetter)(void* field, const TVMFFIAny* value);

/*!
 * \brief Function that create a new instance of the type.
 * \param result The new object handle
 * \return 0 when success, nonzero when failure happens
 */
typedef int (*TVMFFIObjectCreator)(TVMFFIObjectHandle* result);

/*!
 * \brief bitmask of the field.
 */
#ifdef __cplusplus
enum TVMFFIFieldFlagBitMask : int32_t {
#else
typedef enum {
#endif
  /*! \brief The field is writable. */
  kTVMFFIFieldFlagBitMaskWritable = 1 << 0,
  /*! \brief The field has default value. */
  kTVMFFIFieldFlagBitMaskHasDefault = 1 << 1,
  /*! \brief The field is a static method. */
  kTVMFFIFieldFlagBitMaskIsStaticMethod = 1 << 2,
#ifdef __cplusplus
};
#else
} TVMFFIFieldFlagBitMask;
#endif

/*!
 * \brief Information support for optional object reflection.
 */
typedef struct {
  /*! \brief The name of the field. */
  TVMFFIByteArray name;
  /*! \brief The docstring about the field. */
  TVMFFIByteArray doc;
  /*! \brief The type schema of the field in JSON string. */
  TVMFFIByteArray type_schema;
  /*!
   * \brief bitmask flags of the field.
   */
  int64_t flags;
  /*! \brief The size of the field. */
  int64_t size;
  /*! \brief The alignment of the field. */
  int64_t alignment;
  /*! \brief The offset of the field. */
  int64_t offset;
  /*! \brief The getter to access the field. */
  TVMFFIFieldGetter getter;
  /*!
   * \brief The setter to access the field.
   * \note The setter is set even if the field is readonly for serialization.
   */
  TVMFFIFieldSetter setter;
  /*!
   * \brief The default value of the field, this field hold AnyView,
   *        valid when flags set kTVMFFIFieldFlagBitMaskHasDefault
   */
  TVMFFIAny default_value;
  /*!
   * \brief Records the static type kind of the field.
   *
   * Possible values:
   *
   *  - TVMFFITypeIndex::kTVMFFIObject for general objects
   *    - The value is nullable when kTVMFFIObject is chosen
   * - static object type kinds such as Map, Dict, String
   * - POD type index, note it does not give information about storage size of the field.
   * - TVMFFITypeIndex::kTVMFFIAny if we don't have specialized info
   *   about the field.
   *
   * When the value is a type index of Object type, the field is storaged as an ObjectRef.
   *
   * \note This information maybe helpful in designing serializer.
   * As it helps to narrow down the field type so we don't have to
   * print type_key for cases like POD types.
   * It also helps to provide opportunities to enable short-cut getter to ObjectRef fields.
   */
  int32_t field_static_type_index;
} TVMFFIFieldInfo;

/*!
 * \brief Method information that can appear in reflection table.
 */
typedef struct {
  /*! \brief The name of the field. */
  TVMFFIByteArray name;
  /*! \brief The docstring about the method. */
  TVMFFIByteArray doc;
  /*! \brief Optional type schema of the method in JSON string. */
  TVMFFIByteArray type_schema;
  /*! \brief bitmask flags of the method. */
  int64_t flags;
  /*!
   * \brief The method wrapped as ffi::Function, stored as AnyView.
   * \note The first argument to the method is always the self for instance methods.
   */
  TVMFFIAny method;
} TVMFFIMethodInfo;

/*!
 * \brief Extra information of object type that can be used for reflection.
 *
 * \note This information is optional and can be used to enable reflection based
 *       creation of the object.
 */
typedef struct {
  /*! \brief The docstring about the object. */
  TVMFFIByteArray doc;
  /*!
   * \brief An optional function that can create a new empty instance of the type.
   *
   * When known_fixed_size is non-zero, creator can be called
   * with nullptr passed to optional_bytes.
   *
   * \note Caller must call setter for each field to initialize the object for
   *       the final object to be in valid state.
   *
   * \note This field is optional to enable reflection based creation.
   */
  TVMFFIObjectCreator creator;
  /*!
   * \brief Total size of the object struct, if it is fixed and known.
   *
   * This field is set optional and set to 0 if not registered.
   */
  int64_t total_size;
} TVMFFITypeExtraInfo;

/*!
 * \brief Runtime type information for object type checking.
 */
typedef struct TVMFFITypeInfo {
  /*!
   *\brief The runtime type index,
   * It can be allocated during runtime if the type is dynamic.
   */
  int32_t type_index;
  /*! \brief number of parent types in the type hierachy. */
  int32_t type_depth;
  /*! \brief the unique type key to identify the type. */
  TVMFFIByteArray type_key;
  /*!
   * \brief type_acenstors[depth] stores the type_index of the acenstors at depth level
   * \note To keep things simple, we do not allow multiple inheritance so the
   *       hieracy stays as a tree
   */
  const struct TVMFFITypeInfo** type_acenstors;
  // The following fields are used for reflection
  /*! \brief Cached hash value of the type key, used for consistent structural hashing. */
  uint64_t type_key_hash;
  /*! \brief number of reflection accessible fields. */
  int32_t num_fields;
  /*! \brief number of reflection acccesible methods. */
  int32_t num_methods;
  /*! \brief The reflection field information. */
  const TVMFFIFieldInfo* fields;
  /*! \brief The reflection method. */
  const TVMFFIMethodInfo* methods;
  /*! \brief The extra information of the type. */
  const TVMFFITypeExtraInfo* extra_info;
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
 * \brief Convert type key to type index.
 * \param type_key The key of the type.
 * \param out_tindex the corresponding type index.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFITypeKeyToIndex(const TVMFFIByteArray* type_key, int32_t* out_tindex);

//-----------------------------------------------------------------------
// Section: Function calling APIs and support API for func implementation
//-----------------------------------------------------------------------
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
TVM_FFI_DLL int TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call,
                                     void (*deleter)(void* self), TVMFFIObjectHandle* out);

/*!
 * \brief Convert a AnyView to an owned Any.
 * \param any The AnyView to convert.
 * \param out The output Any, must be an empty object
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIAnyViewToOwnedAny(const TVMFFIAny* any_view, TVMFFIAny* out);

/*!
 * \brief Call a FFIFunc by passing in arguments.
 *
 * \param func The resource handle of the C callback.
 * \param args The input arguments to the call.
 * \param num_args The number of input arguments.
 * \param result The output result, caller must ensure result->type_index is set to kTVMFFINone.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* result);

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
TVM_FFI_DLL int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f,
                                        int override);

/*!
 * \brief Register the function to runtime's global table with method info.
 *
 * This is same as TVMFFIFunctionSetGlobal but with method info that can provide extra
 * metadata used in the runtime.
 *
 * \param method_info The method info to be registered.
 * \param override Whether allow override already registered function.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFunctionSetGlobalFromMethodInfo(const TVMFFIMethodInfo* method_info,
                                                      int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out);

/*!
 * \brief Move the last error from the environment to result.
 *
 * \param result The result error.
 *
 * \note This function clears the error stored in the TLS.
 */
TVM_FFI_DLL void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result);

/*!
 * \brief Set raised error in TLS, which can be fetched by TVMFFIErrorMoveFromRaised.
 *
 * \param error The error object handle
 */
TVM_FFI_DLL void TVMFFIErrorSetRaised(TVMFFIObjectHandle error);

/*!
 * \brief Set raised error in TLS, which can be fetched by TVMFFIMoveFromRaised.
 *
 * \param kind The kind of the error.
 * \param message The error message.
 * \note This is a convenient method for C API side to set error directly from string.
 */
TVM_FFI_DLL void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message);

/*!
 * \brief Create an initial error object.
 *
 * \param kind The kind of the error.
 * \param message The error message.
 * \param traceback The traceback of the error.
 * \return The created error object handle.
 * \note This function is different from other functions as it is used in error handling loop.
 *       So we do not follow normal error handling patterns via returning error code.
 */
TVM_FFI_DLL TVMFFIObjectHandle TVMFFIErrorCreate(const TVMFFIByteArray* kind,
                                                 const TVMFFIByteArray* message,
                                                 const TVMFFIByteArray* traceback);

/*!
 * \brief Check if there are any signals raised in the surrounding env.
 * \return 0 when success, nonzero when failure happens
 * \note Under python this function redirects to PyErr_CheckSignals
 */
TVM_FFI_DLL int TVMFFIEnvCheckSignals();

/*!
 * \brief Register a symbol into the from the surrounding env.
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvRegisterCAPI(const TVMFFIByteArray* name, void* symbol);

//------------------------------------------------------------
// Section: Type reflection support APIs
//------------------------------------------------------------
/*!
 * \brief Register type field information for runtime reflection.
 * \param type_index The type index
 * \param info The field info to be registered.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFITypeRegisterField(int32_t type_index, const TVMFFIFieldInfo* info);

/*!
 * \brief Register type method information for runtime reflection.
 * \param type_index The type index
 * \param info The method info to be registered.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFITypeRegisterMethod(int32_t type_index, const TVMFFIMethodInfo* info);

/*!
 * \brief Register type creator information for runtime reflection.
 * \param type_index The type index
 * \param extra_info The extra information to be registered.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFITypeRegisterExtraInfo(int32_t type_index,
                                            const TVMFFITypeExtraInfo* extra_info);

//------------------------------------------------------------
// Section: DLPack support APIs
//------------------------------------------------------------
/*!
 * \brief Produce a managed NDArray from a DLPack tensor.
 * \param from The source DLPack tensor.
 * \param require_alignment The minimum alignment requored of the data + byte_offset.
 * \param require_contiguous Boolean flag indicating if we need to check for contiguity.
 * \param out The output NDArray handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFINDArrayFromDLPack(DLManagedTensor* from, int32_t require_alignment,
                                        int32_t require_contiguous, TVMFFIObjectHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFINDArrayToDLPack(TVMFFIObjectHandle from, DLManagedTensor** out);

/*!
 * \brief Produce a managed NDArray from a DLPack tensor.
 * \param from The source DLPack tensor.
 * \param require_alignment The minimum alignment requored of the data + byte_offset.
 * \param require_contiguous Boolean flag indicating if we need to check for contiguity.
 * \param out The output NDArray handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFINDArrayFromDLPackVersioned(DLManagedTensorVersioned* from,
                                                 int32_t require_alignment,
                                                 int32_t require_contiguous,
                                                 TVMFFIObjectHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFINDArrayToDLPackVersioned(TVMFFIObjectHandle from,
                                               DLManagedTensorVersioned** out);

//---------------------------------------------------------------
// Section: dtype string support APIs.
// These APIs are used to simplify the dtype printings during FFI
//---------------------------------------------------------------

/*!
 * \brief Convert a string to a DLDataType.
 * \param str The string to convert.
 * \param out The output DLDataType.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIDataTypeFromString(const TVMFFIByteArray* str, DLDataType* out);

/*!
 * \brief Convert a DLDataType to a string.
 * \param dtype The DLDataType to convert.
 * \param out The output string.
 * \return 0 when success, nonzero when failure happens
 * \note out is a String object that needs to be freed by the caller via TVMFFIObjectFree.
         The content of string can be accessed via TVMFFIObjectGetByteArrayPtr.

 * \note The input dtype is a pointer to the DLDataType to avoid ABI compatibility issues.
 */
TVM_FFI_DLL int TVMFFIDataTypeToString(const DLDataType* dtype, TVMFFIObjectHandle* out);

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
 * \param filename The current file name.
 * \param lineno The current line number
 * \param func The current function
 * \return The traceback string
 *
 * \note filename func and lino are only used as a backup info, most cases they are not needed.
 *  The return value is set to const char* to be more compatible across dll boundaries.
 */
TVM_FFI_DLL const TVMFFIByteArray* TVMFFITraceback(const char* filename, int lineno,
                                                   const char* func);

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
TVM_FFI_DLL int32_t TVMFFITypeGetOrAllocIndex(const TVMFFIByteArray* type_key,
                                              int32_t static_type_index, int32_t type_depth,
                                              int32_t num_child_slots,
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

//---------------------------------------------------------------
// The following API defines static object field accessors
// for language bindings.
//
// They are defined in C++ inline functions for cleaner code.
// Note that they only have to do with address offset computation.
// So they can always be reimplemented in bindings when c++ is
// not available or when binding only wants to refer to the dll.
//----------------------------------------------------------------
#ifdef __cplusplus
/*!
 * \brief Get the type index of an object.
 * \param obj The object handle.
 * \return The type index.
 */
inline int32_t TVMFFIObjectGetTypeIndex(TVMFFIObjectHandle obj) {
  return static_cast<TVMFFIObject*>(obj)->type_index;
}

/*!
 * \brief Get the data pointer of a bytearray from a string or bytes object.
 * \param obj The object handle.
 * \return The data pointer.
 */
inline TVMFFIByteArray* TVMFFIBytesGetByteArrayPtr(TVMFFIObjectHandle obj) {
  return reinterpret_cast<TVMFFIByteArray*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the data pointer of a ErrorInfo from an Error object.
 * \param obj The object handle.
 * \return The data pointer.
 */
inline TVMFFIErrorCell* TVMFFIErrorGetCellPtr(TVMFFIObjectHandle obj) {
  return reinterpret_cast<TVMFFIErrorCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the data pointer of a function cell from a function object.
 * \param obj The object handle.
 * \return The data pointer.
 */
inline TVMFFIFunctionCell* TVMFFIFunctionGetCellPtr(TVMFFIObjectHandle obj) {
  return reinterpret_cast<TVMFFIFunctionCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the data pointer of a shape array from a shape object.
 * \param obj The object handle.
 * \return The data pointer.
 */
inline TVMFFIShapeCell* TVMFFIShapeGetCellPtr(TVMFFIObjectHandle obj) {
  return reinterpret_cast<TVMFFIShapeCell*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the DLTensor pointer from an NDArray object.
 * \param obj The object handle.
 * \return The DLTensor pointer.
 */
inline DLTensor* TVMFFINDArrayGetDLTensorPtr(TVMFFIObjectHandle obj) {
  return reinterpret_cast<DLTensor*>(reinterpret_cast<char*>(obj) + sizeof(TVMFFIObject));
}

/*!
 * \brief Create a DLDevice from a device type and device id.
 * \param device_type The device type.
 * \param device_id The device id.
 * \return The DLDevice.
 */
inline DLDevice TVMFFIDLDeviceFromIntPair(int32_t device_type, int32_t device_id) {
  return DLDevice{static_cast<DLDeviceType>(device_type), device_id};
}
#endif  // __cplusplus
#endif  // TVM_FFI_C_API_H_
