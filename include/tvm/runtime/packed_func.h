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
 * \file tvm/runtime/packed_func.h
 * \brief Type-erased function used across TVM API.
 */
#ifndef TVM_RUNTIME_PACKED_FUNC_H_
#define TVM_RUNTIME_PACKED_FUNC_H_

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
#include <sstream>
#endif
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/container.h>
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <utility>
#include <type_traits>


// Whether use TVM runtime in header only mode.
#ifndef TVM_RUNTIME_HEADER_ONLY
#define TVM_RUNTIME_HEADER_ONLY 0
#endif

namespace tvm {
// forward declarations
class Integer;
class PrimExpr;

namespace runtime {

/*!
 * \brief Runtime utility for getting custom type name from code
 * \param type_code Custom type code
 * \return Custom type name
 */
TVM_DLL std::string GetCustomTypeName(uint8_t type_code);

/*!
 * \brief Runtime utility for checking whether custom type is registered
 * \param type_code Custom type code
 * \return Bool representing whether type is registered
 */
TVM_DLL bool GetCustomTypeRegistered(uint8_t type_code);

/*!
 * \brief Runtime utility for parsing string of the form "custom[<typename>]"
 * \param s String to parse
 * \param scan pointer to parsing pointer, which is scanning across s
 * \return type code of custom type parsed
 */
TVM_DLL uint8_t ParseCustomDatatype(const std::string& s, const char** scan);

// forward declarations
class TVMArgs;
class TVMArgValue;
class TVMRetValue;
class TVMArgsSetter;

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of TVM.
 *  It corresponds to TVMFunctionHandle in C runtime API.
 */
class PackedFunc {
 public:
  /*!
   * \brief The internal std::function
   * \param args The arguments to the function.
   * \param rv The return value.
   *
   * \code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(TVMArgs args, TVMRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * \endcode
   */
  using FType = std::function<void (TVMArgs args, TVMRetValue* rv)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*! \brief constructor from null */
  PackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template<typename... Args>
  inline TVMRetValue operator()(Args&& ...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return body_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return body_ != nullptr;
  }

 private:
  /*! \brief internal container of packed function */
  FType body_;
};

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor "TypedPackedFunc<R(Args..)>"
 */
template<typename FType>
class TypedPackedFunc;

/*!
 * \anchor TypedPackedFuncAnchor
 * \brief A PackedFunc wrapper to provide typed function signature.
 * It is backed by a PackedFunc internally.
 *
 * TypedPackedFunc enables compile time type checking.
 * TypedPackedFunc works with the runtime system:
 * - It can be passed as an argument of PackedFunc.
 * - It can be assigned to TVMRetValue.
 * - It can be directly converted to a type-erased PackedFunc.
 *
 * Developers should prefer TypedPackedFunc over PackedFunc in C++ code
 * as it enables compile time checking.
 * We can construct a TypedPackedFunc from a lambda function
 * with the same signature.
 *
 * \code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedPackedFunc
 *  TypedPackedFunc<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to PackedFunc
 *  PackedFunc packed = ftype;
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template<typename R, typename ...Args>
class TypedPackedFunc<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedPackedFunc<R(Args...)>;
  /*! \brief default constructor */
  TypedPackedFunc() {}
  /*! \brief constructor from null */
  TypedPackedFunc(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief construct by wrap a PackedFunc
   *
   * Example usage:
   * \code
   * PackedFunc packed([](TVMArgs args, TVMRetValue *rv) {
   *   int x = args[0];
   *   *rv = x + 1;
   *  });
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(packed);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param packed The packed function
   */
  inline TypedPackedFunc(PackedFunc packed);  // NOLINT(*)
  /*!
   * \brief constructor from TVMRetValue
   * \param value The TVMRetValue
   */
  inline TypedPackedFunc(const TVMRetValue& value);  // NOLINT(*)
  /*!
   * \brief constructor from TVMArgValue
   * \param value The TVMArgValue
   */
  inline TypedPackedFunc(const TVMArgValue& value);  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TypedPackedFunc(const FLambda& typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(PackedFunc packed) {
    packed_ = packed;
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  inline R operator()(Args ...args) const;
  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator PackedFunc() const {
    return packed();
  }
  /*!
   * \return reference the internal PackedFunc
   */
  const PackedFunc& packed() const {
    return packed_;
  }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const {
    return packed_ == nullptr;
  }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return packed_ != nullptr;
  }

 private:
  friend class TVMRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template<typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

/*! \brief Arguments into TVM functions. */
class TVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  TVMArgs(const TVMValue* values,
          const int* type_codes,
          int num_args)
      : values(values),
        type_codes(type_codes),
        num_args(num_args) { }
  /*! \return size of the arguments */
  inline int size() const;
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline TVMArgValue operator[](int i) const;
};

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* TypeCode2Str(int type_code);

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
inline TVMType String2TVMType(std::string s);

/*!
 * \brief convert a TVM type to string.
 * \param t The type to be converted.
 * \return The corresponding tvm type in string.
 */
inline std::string TVMType2String(TVMType t);

// macro to check type code.
#define TVM_CHECK_TYPE_CODE(CODE, T)                           \
  CHECK_EQ(CODE, T) << " expected "                            \
  << TypeCode2Str(T) << " but get " << TypeCode2Str(CODE)      \

/*!
 * \brief Type traits for runtime type check during FFI conversion.
 * \tparam T the type to be checked.
 */
template<typename T>
struct ObjectTypeChecker {
  static bool Check(const Object* ptr) {
    using ContainerType = typename T::ContainerType;
    if (ptr == nullptr) return true;
    return ptr->IsInstance<ContainerType>();
  }
  static std::string TypeName() {
    using ContainerType = typename T::ContainerType;
    return ContainerType::_type_key;
  }
};

/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class TVMPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    TVM_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64,
             std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    if (type_code_ == kArrayHandle) return value_.v_handle;
    TVM_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator DLTensor*() const {
    if (type_code_ == kArrayHandle ||
        type_code_ == kNDArrayContainer) {
      return static_cast<DLTensor*>(value_.v_handle);
    } else {
      if (type_code_ == kNull) return nullptr;
      LOG(FATAL) << "Expect "
                 << "DLTensor* or NDArray but get "
                 << TypeCode2Str(type_code_);
      return nullptr;
    }
  }
  operator NDArray() const {
    if (type_code_ == kNull) return NDArray(ObjectPtr<Object>(nullptr));
    TVM_CHECK_TYPE_CODE(type_code_, kNDArrayContainer);
    return NDArray(NDArray::FFIDataFromHandle(
        static_cast<TVMArrayHandle>(value_.v_handle)));
  }
  operator Module() const {
    if (type_code_ == kNull) {
      return Module(ObjectPtr<Object>(nullptr));
    }
    TVM_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    return Module(
        ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  }
  operator TVMContext() const {
    TVM_CHECK_TYPE_CODE(type_code_, kTVMContext);
    return value_.v_ctx;
  }
  operator ADT() const {
    if (type_code_ == kNull) {
      return ADT(ObjectPtr<Object>(nullptr));
    }
    TVM_CHECK_TYPE_CODE(type_code_, kADTHandle);
    return ADT(
        ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  }
  int type_code() const {
    return type_code_;
  }
  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template<typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }
  // ObjectRef handling
  template<typename TObjectRef,
           typename = typename std::enable_if<
             std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  inline bool IsObjectRef() const;
  template<typename TObjectRef>
  inline TObjectRef AsObjectRef() const;
  // ObjectRef Specializations
  inline operator tvm::PrimExpr() const;
  inline operator tvm::Integer() const;

 protected:
  friend class TVMArgsSetter;
  friend class TVMRetValue;
  TVMPODValue_() : type_code_(kNull) {}
  TVMPODValue_(TVMValue value, int type_code)
      : value_(value), type_code_(type_code) {}

  /*! \brief The value */
  TVMValue value_;
  /*! \brief the type code */
  int type_code_;
};

/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and TVMValue
 *
 *  Provides utilities to do type cast into other types.
 */
class TVMArgValue : public TVMPODValue_ {
 public:
  /*! \brief default constructor */
  TVMArgValue() {}
  /*!
   * \brief constructor
   * \param value of the function
   * \param type_code The type code.
   */
  TVMArgValue(TVMValue value, int type_code)
      : TVMPODValue_(value, type_code) {
  }
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator TVMContext;
  using TVMPODValue_::operator Module;
  using TVMPODValue_::operator ADT;
  using TVMPODValue_::IsObjectRef;
  using TVMPODValue_::AsObjectRef;
  using TVMPODValue_::operator tvm::PrimExpr;
  using TVMPODValue_::operator tvm::Integer;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kTVMType) {
      return TVMType2String(operator TVMType());
    } else if (type_code_ == kBytes) {
      TVMByteArray* arr = static_cast<TVMByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else {
      TVM_CHECK_TYPE_CODE(type_code_, kStr);
      return std::string(value_.v_str);
    }
  }
  operator TVMType() const {
    if (type_code_ == kStr) {
      return String2TVMType(operator std::string());
    }
    // None type
    if (type_code_ == kNull) {
      TVMType t;
      t.code = kHandle; t.bits = 0; t.lanes = 0;
      return t;
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMType);
    return value_.v_type;
  }
  operator DataType() const {
    return DataType(operator DLDataType());
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    TVM_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  const TVMValue& value() const {
    return value_;
  }
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;
};

/*!
 * \brief Return Value container,
 *  Unlike TVMArgValue, which only holds reference and do not delete
 *  the underlying container during destruction.
 *
 *  TVMRetValue holds value and will manage the underlying containers
 *  when it stores a complicated data type.
 */
class TVMRetValue : public TVMPODValue_ {
 public:
  /*! \brief default constructor */
  TVMRetValue() {}
  /*!
   * \brief move constructor from anoter return value.
   * \param other The other return value.
   */
  TVMRetValue(TVMRetValue&& other)
      : TVMPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kNull;
  }
  /*! \brief destructor */
  ~TVMRetValue() {
    this->Clear();
  }
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator TVMContext;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator Module;
  using TVMPODValue_::operator ADT;
  using TVMPODValue_::IsObjectRef;
  using TVMPODValue_::AsObjectRef;
  using TVMPODValue_::operator tvm::PrimExpr;
  using TVMPODValue_::operator tvm::Integer;

  TVMRetValue(const TVMRetValue& other) : TVMPODValue_() {
    this->Assign(other);
  }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kTVMType) {
      return TVMType2String(operator TVMType());
    } else if (type_code_ == kBytes) {
      return *ptr<std::string>();
    }
    TVM_CHECK_TYPE_CODE(type_code_, kStr);
    return *ptr<std::string>();
  }
  operator TVMType() const {
    if (type_code_ == kStr) {
      return String2TVMType(operator std::string());
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMType);
    return value_.v_type;
  }
  operator DataType() const {
    return DataType(operator DLDataType());
  }
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    TVM_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  template<typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  // Assign operators
  TVMRetValue& operator=(TVMRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kNull;
    return *this;
  }
  TVMRetValue& operator=(double value) {
    this->SwitchToPOD(kDLFloat);
    value_.v_float64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kNull);
    value_.v_handle = value;
    return *this;
  }
  TVMRetValue& operator=(void* value) {
    this->SwitchToPOD(kHandle);
    value_.v_handle = value;
    return *this;
  }
  TVMRetValue& operator=(int64_t value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(int value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(TVMContext value) {
    this->SwitchToPOD(kTVMContext);
    value_.v_ctx = value;
    return *this;
  }
  TVMRetValue& operator=(TVMType t) {
    this->SwitchToPOD(kTVMType);
    value_.v_type = t;
    return *this;
  }
  TVMRetValue& operator=(const DataType& other) {
    return operator=(other.operator DLDataType());
  }
  TVMRetValue& operator=(bool value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::string value) {
    this->SwitchToClass(kStr, value);
    return *this;
  }
  TVMRetValue& operator=(TVMByteArray value) {
    this->SwitchToClass(kBytes, std::string(value.data, value.size));
    return *this;
  }
  TVMRetValue& operator=(NDArray other) {
    if (other.data_ != nullptr) {
      this->Clear();
      type_code_ = kNDArrayContainer;
      value_.v_handle = NDArray::FFIGetHandle(other);
      ObjectRef::FFIClearAfterMove(&other);
    } else {
      SwitchToPOD(kNull);
    }
    return *this;
  }
  TVMRetValue& operator=(Module m) {
    SwitchToObject(kModuleHandle, std::move(m.data_));
    return *this;
  }
  TVMRetValue& operator=(ADT adt) {
    SwitchToObject(kADTHandle, std::move(adt.data_));
    return *this;
  }
  TVMRetValue& operator=(PackedFunc f) {
    this->SwitchToClass(kFuncHandle, f);
    return *this;
  }
  template<typename FType>
  TVMRetValue& operator=(const TypedPackedFunc<FType>& f) {
    return operator=(f.packed());
  }
  TVMRetValue& operator=(const TVMRetValue& other) {  // NOLINT(*0
    this->Assign(other);
    return *this;
  }
  TVMRetValue& operator=(const TVMArgValue& other) {
    this->Assign(other);
    return *this;
  }
  /*!
   * \brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources is moved to front-end and
   *  the front end should take charge in managing them.
   *
   * \param ret_value The return value.
   * \param ret_type_code The return type code.
   */
  void MoveToCHost(TVMValue* ret_value,
                   int* ret_type_code) {
    // cannot move str; need specially handle.
    CHECK(type_code_ != kStr && type_code_ != kBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kNull;
  }
  /*!
   * \brief Construct a new TVMRetValue by
   *        moving from return value stored via C API.
   * \param value the value.
   * \param type_code The type code.
   * \return The created TVMRetValue.
   */
  static TVMRetValue MoveFromCHost(TVMValue value,
                                   int type_code) {
    // Can move POD and everything under the object system.
    CHECK(type_code <= kFuncHandle ||
          type_code == kNDArrayContainer);
    TVMRetValue ret;
    ret.value_ = value;
    ret.type_code_ = type_code;
    return ret;
  }
  /*! \return The value field, if the data is POD */
  const TVMValue& value() const {
    CHECK(type_code_ != kObjectHandle &&
          type_code_ != kFuncHandle &&
          type_code_ != kModuleHandle &&
          type_code_ != kADTHandle &&
          type_code_ != kStr) << "TVMRetValue.value can only be used for POD data";
    return value_;
  }
  // ObjectRef handling
  template<typename TObjectRef,
           typename = typename std::enable_if<
             std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  inline TVMRetValue& operator=(TObjectRef other);
  template<typename T,
           typename = typename std::enable_if<
             std::is_class<T>::value>::type>
  inline operator T() const;

 private:
  template<typename T>
  void Assign(const T& other) {
    switch (other.type_code()) {
      case kStr: {
        SwitchToClass<std::string>(kStr, other);
        break;
      }
      case kBytes: {
        SwitchToClass<std::string>(kBytes, other);
        break;
      }
      case kFuncHandle: {
        SwitchToClass<PackedFunc>(kFuncHandle, other);
        break;
      }
      case kModuleHandle: {
        *this = other.operator Module();
        break;
      }
      case kADTHandle: {
        *this = other.operator ADT();
        break;
      }
      case kNDArrayContainer: {
        *this = other.operator NDArray();
        break;
      }
      case kObjectHandle: {
        // Avoid operator ObjectRef as we already know it is not NDArray/Module
        SwitchToObject(
            kObjectHandle, GetObjectPtr<Object>(
                static_cast<Object*>(other.value_.v_handle)));
        break;
      }
      default: {
        SwitchToPOD(other.type_code());
        value_ = other.value_;
        break;
      }
    }
  }
  // get the internal container.
  void SwitchToPOD(int type_code) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
    }
  }
  template<typename T>
  void SwitchToClass(int type_code, T v) {
    if (type_code_ != type_code) {
      this->Clear();
      type_code_ = type_code;
      value_.v_handle = new T(v);
    } else {
      *static_cast<T*>(value_.v_handle) = v;
    }
  }
  void SwitchToObject(int type_code, ObjectPtr<Object> other) {
    if (other.data_ != nullptr) {
      this->Clear();
      type_code_ = type_code;
      // move the handle out
      value_.v_handle = other.data_;
      other.data_ = nullptr;
    } else {
      SwitchToPOD(kNull);
    }
  }
  void Clear() {
    if (type_code_ == kNull) return;
    switch (type_code_) {
      case kStr: delete ptr<std::string>(); break;
      case kFuncHandle: delete ptr<PackedFunc>(); break;
      case kNDArrayContainer: {
        NDArray::FFIDecRef(static_cast<TVMArrayHandle>(value_.v_handle));
        break;
      }
      case kModuleHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
      case kADTHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
      case kObjectHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
    }
    type_code_ = kNull;
  }
};

/*!
 * \brief Export a function with the PackedFunc signature
 *        as a PackedFunc that can be loaded by LibraryModule.
 *
 * \param ExportName The symbol name to be exported.
 * \param Function The function with PackedFunc signature.
 * \sa PackedFunc
 *
 * \code
 *
 * void AddOne_(TVMArgs args, TVMRetValue* rv) {
 *   int value = args[0];
 *   *rv = value + 1;
 * }
 * // Expose the function as "AddOne"
 * TVM_DLL_EXPORT_PACKED_FUNC(AddOne, AddOne_);
 *
 * \endcode
 */
#define TVM_DLL_EXPORT_PACKED_FUNC(ExportName, Function)                \
  extern "C" {                                                          \
  TVM_DLL int ExportName(TVMValue* args,                                \
                         int* type_code,                                \
                         int num_args,                                  \
                         TVMValue* out_value,                           \
                         int* out_type_code) {                          \
    try {                                                               \
      ::tvm::runtime::TVMRetValue rv;                                   \
      Function(::tvm::runtime::TVMArgs(                                 \
          args, type_code, num_args), &rv);                             \
      rv.MoveToCHost(out_value, out_type_code);                         \
      return 0;                                                         \
    } catch (const ::std::runtime_error& _except_) {                    \
      TVMAPISetLastError(_except_.what());                              \
      return -1;                                                        \
    }                                                                   \
  }                                                                     \
  }

/*!
 * \brief Export typed function as a PackedFunc
 *        that can be loaded by LibraryModule.
 *
 * \param ExportName The symbol name to be exported.
 * \param Function The typed function.
 * \note ExportName and Function must be different,
 *       see code examples below.
 *
 * \sa TypedPackedFunc
 *
 * \code
 *
 * int AddOne_(int x) {
 *   return x + 1;
 * }
 *
 * // Expose the function as "AddOne"
 * TVM_DLL_EXPORT_TYPED_FUNC(AddOne, AddOne_);
 *
 * // Expose the function as "SubOne"
 * TVM_DLL_EXPORT_TYPED_FUNC(SubOne, [](int x) {
 *   return x - 1;
 * });
 *
 * // The following code will cause compilation error.
 * // Because the same Function and ExortName
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, AddOne_);
 *
 * // The following code is OK, assuming the macro
 * // is in a different namespace from xyz
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, xyz::AddOne_);
 *
 * \endcode
 */
#define TVM_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                 \
  extern "C" {                                                          \
  TVM_DLL int ExportName(TVMValue* args,                                \
                         int* type_code,                                \
                         int num_args,                                  \
                         TVMValue* out_value,                           \
                         int* out_type_code) {                          \
    try {                                                               \
      auto f = Function;                                                \
      using FType = ::tvm::runtime::detail::                            \
                    function_signature<decltype(f)>::FType;             \
      ::tvm::runtime::TVMRetValue rv;                                   \
      ::tvm::runtime::detail::unpack_call_by_signature<FType>::run(     \
           f,                                                           \
           ::tvm::runtime::TVMArgs(args, type_code, num_args), &rv);    \
      rv.MoveToCHost(out_value, out_type_code);                         \
      return 0;                                                         \
    } catch (const ::std::runtime_error& _except_) {                    \
      TVMAPISetLastError(_except_.what());                              \
      return -1;                                                        \
    }                                                                   \
    }                                                                   \
  }

// implementation details
inline const char* TypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kStr: return "str";
    case kBytes: return "bytes";
    case kHandle: return "handle";
    case kNull: return "NULL";
    case kArrayHandle: return "ArrayHandle";
    case kTVMType: return "TVMType";
    case kTVMContext: return "TVMContext";
    case kFuncHandle: return "FunctionHandle";
    case kModuleHandle: return "ModuleHandle";
    case kADTHandle: return "ADTHandle";
    case kNDArrayContainer: return "NDArrayContainer";
    case kObjectHandle: return "Object";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}

#ifndef _LIBCPP_SGX_NO_IOSTREAMS
inline std::ostream& operator<<(std::ostream& os, TVMType t) {  // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool"; return os;
  }
  if (t.code < kCustomBegin) {
    os << TypeCode2Str(t.code);
  } else {
    os << "custom[" << GetCustomTypeName(t.code) << "]";
  }
  if (t.code == kHandle) return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) { // NOLINT(*)
  return os << dtype.operator DLDataType();
}

#endif

inline std::string TVMType2String(TVMType t) {
  if (t.bits == 0) return "";
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::ostringstream os;
  os << t;
  return os.str();
#else
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    return "bool";
  }
  if (t.code < kCustomBegin) {
    repr += TypeCode2Str(t.code);
  } else {
    repr += "custom[" + GetCustomTypeName(t.code) + "]";
  }
  if (t.code == kHandle) return repr;
  repr += std::to_string(static_cast<int>(t.bits));
  if (t.lanes != 1) {
    repr += "x" + std::to_string(static_cast<int>(t.lanes));
  }
  return repr;
#endif
}

inline TVMType String2TVMType(std::string s) {
  TVMType t;
  // handle None type
  if (s.length() == 0) {
    t.bits = 0; t.lanes = 0; t.code = kHandle;
    return t;
  }
  t.bits = 32; t.lanes = 1;
  const char* scan;
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;  scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt; scan = s.c_str() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat; scan = s.c_str() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.c_str() + 6;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else if (s.substr(0, 6) == "custom") {
    t.code = ParseCustomDatatype(s, &scan);
  } else {
    scan = s.c_str();
    LOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, &endpt, 10));
  }
  CHECK(endpt == s.c_str() + s.length()) << "unknown type " << s;
  return t;
}

inline TVMArgValue TVMArgs::operator[](int i) const {
  CHECK_LT(i, num_args)
      << "not enough argument passed, "
      << num_args << " passed"
      << " but request arg[" << i << "].";
  return TVMArgValue(values[i], type_codes[i]);
}

inline int TVMArgs::size() const {
  return num_args;
}

inline void PackedFunc::CallPacked(TVMArgs args, TVMRetValue* rv) const {
  body_(args, rv);
}

inline PackedFunc::FType PackedFunc::body() const {
  return body_;
}

// internal namespace
namespace detail {

template<bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template<typename T, typename ...Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I+1), F>
        ::run(f, std::forward<Args>(args)...);
  }
};

template<std::size_t I, typename F>
struct for_each_dispatcher<true, I, F>  {
  static void run(const F& f) {}  // NOLINT(*)
};

template<typename F, typename ...Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>
      ::run(f, std::forward<Args>(args)...);
}

template<typename T>
struct func_signature_helper {
  using FType = void;
};

template<typename T, typename R, typename ...Args>
struct func_signature_helper<R (T::*)(Args...)> {
  using FType = R(Args...);
};

template<typename T, typename R, typename ...Args>
struct func_signature_helper<R (T::*)(Args...) const> {
  using FType = R(Args...);
};

/*!
 * \brief template class to get function signature of a function or functor.
 * \tparam T The funtion/functor type.
 */
template<typename T>
struct function_signature {
  using FType = typename func_signature_helper<decltype(&T::operator())>::FType;
};

// handle case of function.
template<typename R, typename ...Args>
struct function_signature<R(Args...)> {
  using FType = R(Args...);
};

// handle case of function ptr.
template<typename R, typename ...Args>
struct function_signature<R (*)(Args...)> {
  using FType = R(Args...);
};
}  // namespace detail

/* \brief argument settter to PackedFunc */
class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template<typename T,
           typename = typename std::enable_if<
             std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    CHECK_LE(value,
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, const TVMArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kArrayHandle;
  }
  void operator()(size_t i, TVMContext value) const {
    values_[i].v_ctx = value;
    type_codes_[i] = kTVMContext;
  }
  void operator()(size_t i, TVMType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kTVMType;
  }
  void operator()(size_t i, DataType dtype) const {
    operator()(i, dtype.operator DLDataType());
  }
  void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kStr;
  }
  // setters for container types
  void operator()(size_t i, const std::string& value) const {
    values_[i].v_str = value.c_str();
    type_codes_[i] = kStr;
  }
  void operator()(size_t i, const TVMByteArray& value) const {
    values_[i].v_handle = const_cast<TVMByteArray*>(&value);
    type_codes_[i] = kBytes;
  }
  void operator()(size_t i, const PackedFunc& value) const {
    values_[i].v_handle = const_cast<PackedFunc*>(&value);
    type_codes_[i] = kFuncHandle;
  }
  template<typename FType>
  void operator()(size_t i, const TypedPackedFunc<FType>& value) const {
    operator()(i, value.packed());
  }
  void operator()(size_t i, const TVMRetValue& value) const {
    if (value.type_code() == kStr) {
      values_[i].v_str = value.ptr<std::string>()->c_str();
      type_codes_[i] = kStr;
    } else {
      CHECK_NE(value.type_code(), kBytes) << "not handled.";
      values_[i] = value.value_;
      type_codes_[i] = value.type_code();
    }
  }
  // ObjectRef handling
  template<typename TObjectRef,
           typename = typename std::enable_if<
             std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  inline void operator()(size_t i, const TObjectRef& value) const;

 private:
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template<typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  TVMRetValue rv;
  body_(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

namespace detail {
template<typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>
        ::run(f, args_pack, rv,
              std::forward<Args>(unpacked_args)...,
              args_pack[index]);
  }
};

template<typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template<int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename R, int nargs, typename F>
inline void unpack_call(const F& f, const TVMArgs& args, TVMRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template<typename FType>
struct unpack_call_by_signature {
};

template<typename R, typename ...Args>
struct unpack_call_by_signature<R(Args...)> {
  template<typename F>
  static void run(const F& f,
                  const TVMArgs& args,
                  TVMRetValue* rv) {
    unpack_call<R, sizeof...(Args)>(f, args, rv);
  }
};

template<typename R, typename ...Args>
inline R call_packed(const PackedFunc& pf, Args&& ...args) {
  return R(pf(std::forward<Args>(args)...));
}

template<typename R>
struct typed_packed_call_dispatcher {
  template<typename ...Args>
  static inline R run(const PackedFunc& pf, Args&& ...args) {
    return pf(std::forward<Args>(args)...);
  }
};

template<>
struct typed_packed_call_dispatcher<void> {
  template<typename ...Args>
  static inline void run(const PackedFunc& pf, Args&& ...args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed)
  : packed_(packed) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const TVMRetValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const TVMArgValue& value)
    : packed_(value.operator PackedFunc()) {}

template<typename R, typename ...Args>
template<typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const TVMArgs& args, TVMRetValue* rv) {
      detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
    });
}

template<typename R, typename ...Args>
inline R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>
      ::run(packed_, std::forward<Args>(args)...);
}

// ObjectRef related conversion handling
// Object can have four possible type codes:
//      kNDArrayContainer, kModuleHandle, kObjectHandle, kADTHandle
//
// We use type traits to eliminate un-necessary checks.
template<typename TObjectRef, typename>
inline void TVMArgsSetter::operator()(size_t i, const TObjectRef& value) const {
  if (value.defined()) {
    Object* ptr = value.data_.data_;
    if (std::is_base_of<NDArray, TObjectRef>::value ||
        (std::is_base_of<TObjectRef, NDArray>::value &&
         ptr->IsInstance<NDArray::ContainerType>())) {
      values_[i].v_handle = NDArray::FFIGetHandle(value);
      type_codes_[i] = kNDArrayContainer;
    } else if (std::is_base_of<ADT, TObjectRef>::value ||
               (std::is_base_of<TObjectRef, ADT>::value &&
                ptr->IsInstance<ADT::ContainerType>())) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kADTHandle;
    } else if (std::is_base_of<Module, TObjectRef>::value ||
               (std::is_base_of<TObjectRef, Module>::value &&
                ptr->IsInstance<Module::ContainerType>())) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kModuleHandle;
    } else {
      values_[i].v_handle = ptr;
      type_codes_[i] = kObjectHandle;
    }
  } else {
    type_codes_[i] = kNull;
  }
}

template<typename TObjectRef, typename>
inline bool TVMPODValue_::IsObjectRef() const {
  using ContainerType = typename TObjectRef::ContainerType;
  // NOTE: the following code can be optimized by constant folding.
  if (std::is_base_of<NDArray, TObjectRef>::value) {
    return type_code_ == kNDArrayContainer &&
        TVMArrayHandleToObjectHandle(
            static_cast<TVMArrayHandle>(value_.v_handle))->IsInstance<ContainerType>();
  }
  if (std::is_base_of<Module, TObjectRef>::value) {
    return type_code_ == kModuleHandle &&
        static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
  }
  if (std::is_base_of<ADT, TObjectRef>::value) {
    return type_code_ == kADTHandle &&
        static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
  }
  return
      (std::is_base_of<TObjectRef, NDArray>::value && type_code_ == kNDArrayContainer) ||
      (std::is_base_of<TObjectRef, Module>::value && type_code_ == kModuleHandle) ||
      (std::is_base_of<TObjectRef, ADT>::value && type_code_ == kADTHandle) ||
      (type_code_ == kObjectHandle &&
       ObjectTypeChecker<TObjectRef>::Check(static_cast<Object*>(value_.v_handle)));
}

template<typename TObjectRef>
inline TObjectRef TVMPODValue_::AsObjectRef() const {
  static_assert(
      std::is_base_of<ObjectRef, TObjectRef>::value,
      "Conversion only works for ObjectRef");
  using ContainerType = typename TObjectRef::ContainerType;
  if (type_code_ == kNull) return TObjectRef(ObjectPtr<Object>(nullptr));
  // NOTE: the following code can be optimized by constant folding.
  if (std::is_base_of<NDArray, TObjectRef>::value) {
    // Casting to a sub-class of NDArray
    TVM_CHECK_TYPE_CODE(type_code_, kNDArrayContainer);
    ObjectPtr<Object> data = NDArray::FFIDataFromHandle(
        static_cast<TVMArrayHandle>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expect " << ContainerType::_type_key << " but get " << data->GetTypeKey();
    return TObjectRef(data);
  }
  if (std::is_base_of<Module, TObjectRef>::value) {
    // Casting to a sub-class of Module
    TVM_CHECK_TYPE_CODE(type_code_, kModuleHandle);
    ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expect " << ContainerType::_type_key << " but get " << data->GetTypeKey();
    return TObjectRef(data);
  }
  if (std::is_base_of<ADT, TObjectRef>::value) {
    // Casting to a sub-class of ADT
    TVM_CHECK_TYPE_CODE(type_code_, kADTHandle);
    ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expect " << ContainerType::_type_key << " but get " << data->GetTypeKey();
    return TObjectRef(data);
  }

  if (type_code_ == kObjectHandle) {
    // normal object type check.
    Object* ptr = static_cast<Object*>(value_.v_handle);
    CHECK(ObjectTypeChecker<TObjectRef>::Check(ptr))
        << "Expect " << ObjectTypeChecker<TObjectRef>::TypeName()
        << " but get " << ptr->GetTypeKey();
    return TObjectRef(GetObjectPtr<Object>(ptr));
  } else if (std::is_base_of<TObjectRef, NDArray>::value &&
             type_code_ == kNDArrayContainer) {
    // Casting to a base class that NDArray can sub-class
    ObjectPtr<Object> data = NDArray::FFIDataFromHandle(
        static_cast<TVMArrayHandle>(value_.v_handle));
    return TObjectRef(data);
  } else if (std::is_base_of<TObjectRef, Module>::value &&
             type_code_ == kModuleHandle) {
    // Casting to a base class that Module can sub-class
    return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  } else if (std::is_base_of<TObjectRef, ADT>::value &&
             type_code_ == kADTHandle) {
    // Casting to a base class that ADT can sub-class
    return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  } else {
    TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);
    return TObjectRef(ObjectPtr<Object>(nullptr));
  }
}

template<typename TObjectRef, typename>
inline TVMRetValue& TVMRetValue::operator=(TObjectRef other) {
  const Object* ptr = other.get();
  if (ptr != nullptr) {
    if (std::is_base_of<NDArray, TObjectRef>::value ||
        (std::is_base_of<TObjectRef, NDArray>::value &&
         ptr->IsInstance<NDArray::ContainerType>())) {
      return operator=(NDArray(std::move(other.data_)));
    }
    if (std::is_base_of<Module, TObjectRef>::value ||
        (std::is_base_of<TObjectRef, Module>::value &&
         ptr->IsInstance<Module::ContainerType>())) {
      return operator=(Module(std::move(other.data_)));
    }
    if (std::is_base_of<ADT, TObjectRef>::value ||
        (std::is_base_of<TObjectRef, ADT>::value &&
         ptr->IsInstance<ADT::ContainerType>())) {
      return operator=(ADT(std::move(other.data_)));
    }
    SwitchToObject(kObjectHandle, std::move(other.data_));
  } else {
    SwitchToPOD(kNull);
  }
  return *this;
}

template<typename T, typename>
inline TVMArgValue::operator T() const {
  return AsObjectRef<T>();
}

template<typename T, typename>
inline TVMRetValue::operator T() const {
  return AsObjectRef<T>();
}

inline PackedFunc Module::GetFunction(const std::string& name, bool query_imports) {
  return (*this)->GetFunction(name, query_imports);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
