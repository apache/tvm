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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/variant.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Whether use TVM runtime in header only mode.
#ifndef TVM_RUNTIME_HEADER_ONLY
#define TVM_RUNTIME_HEADER_ONLY 0
#endif

namespace tvm {
namespace runtime {

// forward declarations
class TVMArgs;
class TVMArgValue;
class TVMMovableArgValueWithContext_;
class TVMRetValue;
class TVMArgsSetter;
template <typename FType>
class TypedPackedFunc;
template <typename TSignature>
struct SignaturePrinter;

/*!
 * \brief Object container class that backs PackedFunc.
 * \note Do not use this function directly, use PackedFunc.
 */
class PackedFuncObj : public Object {
 public:
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  TVM_ALWAYS_INLINE void CallPacked(TVMArgs args, TVMRetValue* rv) const;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimePackedFunc;
  static constexpr const char* _type_key = "runtime.PackedFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(PackedFuncObj, Object);

 protected:
  /*!
   * \brief Internal struct for extracting the callable method from callable type.
   */
  template <class TPackedFuncSubObj>
  struct Extractor {
    /*!
     * \brief Extracting the callable method from callable type.
     * \param obj The base packed function object class.
     * \param args The arguments
     * \param rv The return value.
     */
    static void Call(const PackedFuncObj* obj, TVMArgs args, TVMRetValue* rv);
  };

  /*! \brief The internal callable function type. */
  using FCallPacked = void(const PackedFuncObj*, TVMArgs, TVMRetValue*);

  /*!
   * \brief Constructing a packed function object from a function pointer.
   * \param f_call_pack The function pointer used to call the packed function.
   */
  explicit PackedFuncObj(FCallPacked* f_call_pack) : f_call_packed_(f_call_pack) {}

  /*! \brief Delete the default constructor explicitly. */
  PackedFuncObj() = delete;

  /*! \brief Internal callable function pointer used to call the packed function. */
  FCallPacked* f_call_packed_;
};

/*! \brief Derived object class for constructing PackedFuncObj. */
template <class TCallable>
class PackedFuncSubObj : public PackedFuncObj {
  using TStorage = typename std::remove_cv<typename std::remove_reference<TCallable>::type>::type;

 public:
  /*! \brief The type of derived object class */
  using TSelf = PackedFuncSubObj<TCallable>;
  /*!
   * \brief Derived object class for constructing PackedFuncObj.
   * \param callable The type-erased callable object.
   */
  explicit PackedFuncSubObj(TCallable callable)
      : PackedFuncObj(Extractor<TSelf>::Call), callable_(callable) {}
  /*! \brief Type-erased filed for storing callable object*/
  mutable TStorage callable_;
};

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function type of TVM.
 *  It corresponds to TVMFunctionHandle in C runtime API.
 */
class PackedFunc : public ObjectRef {
 public:
  /*! \brief Constructor from null */
  PackedFunc(std::nullptr_t null) : ObjectRef(nullptr) {}  // NOLINT(*)
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `PackedFunc`
   * \param data the internal container of packed function.
   */
  template <typename TCallable,
            typename = std::enable_if_t<
                std::is_convertible<TCallable, std::function<void(TVMArgs, TVMRetValue*)>>::value &&
                !std::is_base_of<TCallable, PackedFunc>::value>>
  explicit PackedFunc(TCallable data) {
    using ObjType = PackedFuncSubObj<TCallable>;
    data_ = make_object<ObjType>(std::forward<TCallable>(data));
  }
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
  template <typename... Args>
  inline TVMRetValue operator()(Args&&... args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  TVM_ALWAYS_INLINE void CallPacked(TVMArgs args, TVMRetValue* rv) const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return data_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return data_ != nullptr; }

  TVM_DEFINE_OBJECT_REF_METHODS(PackedFunc, ObjectRef, PackedFuncObj);
};

/*! \brief Using static function to output TypedPackedFunc signature */
using FSig = std::string();

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor "TypedPackedFunc<R(Args..)>"
 */
template <typename FType>
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
template <typename R, typename... Args>
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
   * ICHECK_EQ(ftyped(1), 2);
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
   * \brief constructor from TVMMovableArgValue_
   * \param value The TVMMovableArgValue_
   */
  inline TypedPackedFunc(TVMMovableArgValueWithContext_&& value);  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda, "add_one");
   * // call the typed version.
   * ICHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \param name the name of the lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                  FLambda, std::function<R(Args...)>>::value>::type>
  TypedPackedFunc(const FLambda& typed_lambda, std::string name) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda, name);
  }
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * This version does not take a name. It is highly recommend you use the
   * version that takes a name for the lambda.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedPackedFunc<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * ICHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                  FLambda, std::function<R(Args...)>>::value>::type>
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
   * ICHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template <typename FLambda, typename = typename std::enable_if<
                                  std::is_convertible<FLambda,
                                                      std::function<R(Args...)>>::value>::type>
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
  TVM_ALWAYS_INLINE R operator()(Args... args) const;
  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator PackedFunc() const { return packed(); }
  /*!
   * \return reference the internal PackedFunc
   */
  const PackedFunc& packed() const { return packed_; }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return packed_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return packed_ != nullptr; }

 private:
  friend class TVMRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \param name The name associated with this lambda.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template <typename FLambda>
  inline void AssignTypedLambda(FLambda flambda, std::string name);
  /*!
   * \brief Assign the packed field using a typed lambda function. This variant is for functions
   * without names.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template <typename FLambda>
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
  TVMArgs(const TVMValue* values, const int* type_codes, int num_args)
      : values(values), type_codes(type_codes), num_args(num_args) {}
  /*! \return size of the arguments */
  inline int size() const;
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline TVMArgValue operator[](int i) const;
  /*!
   * \brief Get the i-th argument and do proper type checking with detailed error messages.
   * \tparam T The expected type.
   * \param i The index
   * \return The corresponding argument value.
   */
  template <typename T>
  inline T At(int i) const;
};

/*!
 * \brief Convert argument type code to string.
 * \param type_code The input type code.
 * \return The corresponding string repr.
 */
inline const char* ArgTypeCode2Str(int type_code);

inline std::ostream& operator<<(std::ostream& os, DLDevice dev);  // NOLINT(*)

// macro to check type code.
#define TVM_CHECK_TYPE_CODE(CODE, T) \
  ICHECK_EQ(CODE, T) << "expected " << ArgTypeCode2Str(T) << " but got " << ArgTypeCode2Str(CODE)

/*!
 * \brief Type traits for runtime type check during FFI conversion.
 * \tparam T the type to be checked.
 */
template <typename T>
struct ObjectTypeChecker {
  /*!
   * \brief Check if an object matches the template type and return the
   *        mismatched type if it exists.
   * \param ptr The object to check the type of.
   * \return An Optional containing the actual type of the pointer if it does not match the
   *         template type. If the Optional does not contain a value, then the types match.
   */
  static Optional<String> CheckAndGetMismatch(const Object* ptr) {
    using ContainerType = typename T::ContainerType;
    if (ptr == nullptr) {
      if (T::_type_is_nullable) {
        return NullOpt;
      } else {
        return String("nullptr");
      }
    }
    if (ptr->IsInstance<ContainerType>()) {
      return NullOpt;
    } else {
      return String(ptr->GetTypeKey());
    }
  }
  /*!
   * \brief Check if an object matches the template type.
   * \param ptr The object to check the type of.
   * \return Whether or not the template type matches the objects type.
   */
  static bool Check(const Object* ptr) {
    using ContainerType = typename T::ContainerType;
    if (ptr == nullptr) return T::_type_is_nullable;
    return ptr->IsInstance<ContainerType>();
  }
  static std::string TypeName() {
    using ContainerType = typename T::ContainerType;
    return ContainerType::_type_key;
  }
};

// Additional overloads for PackedFunc checking.
template <typename T>
struct ObjectTypeChecker<Array<T>> {
  static Optional<String> CheckAndGetMismatch(const Object* ptr) {
    if (ptr == nullptr) {
      return NullOpt;
    }
    if (!ptr->IsInstance<ArrayNode>()) {
      return String(ptr->GetTypeKey());
    }
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (size_t i = 0; i < n->size(); i++) {
      const ObjectRef& p = (*n)[i];
      Optional<String> check_subtype = ObjectTypeChecker<T>::CheckAndGetMismatch(p.get());
      if (check_subtype.defined()) {
        return String("Array[index " + std::to_string(i) + ": " + check_subtype.value() + "]");
      }
    }
    return NullOpt;
  }
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<ArrayNode>()) return false;
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (const ObjectRef& p : *n) {
      if (!ObjectTypeChecker<T>::Check(p.get())) {
        return false;
      }
    }
    return true;
  }
  static std::string TypeName() { return "Array[" + ObjectTypeChecker<T>::TypeName() + "]"; }
};
template <typename K, typename V>
struct ObjectTypeChecker<Map<K, V>> {
  static Optional<String> CheckAndGetMismatch(const Object* ptr) {
    if (ptr == nullptr) return NullOpt;
    if (!ptr->IsInstance<MapNode>()) return String(ptr->GetTypeKey());
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : *n) {
      Optional<String> key_type = ObjectTypeChecker<K>::CheckAndGetMismatch(kv.first.get());
      Optional<String> value_type = ObjectTypeChecker<K>::CheckAndGetMismatch(kv.first.get());
      if (key_type.defined() || value_type.defined()) {
        std::string key_name =
            key_type.defined() ? std::string(key_type.value()) : ObjectTypeChecker<K>::TypeName();
        std::string value_name = value_type.defined() ? std::string(value_type.value())
                                                      : ObjectTypeChecker<V>::TypeName();
        return String("Map[" + key_name + ", " + value_name + "]");
      }
    }
    return NullOpt;
  }
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<MapNode>()) return false;
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : *n) {
      if (!ObjectTypeChecker<K>::Check(kv.first.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[" + ObjectTypeChecker<K>::TypeName() + ", " + ObjectTypeChecker<V>::TypeName() +
           ']';
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
    ICHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    ICHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kTVMNullptr) return nullptr;
    if (type_code_ == kTVMDLTensorHandle) return value_.v_handle;
    TVM_CHECK_TYPE_CODE(type_code_, kTVMOpaqueHandle);
    return value_.v_handle;
  }
  operator DLTensor*() const {
    if (type_code_ == kTVMDLTensorHandle || type_code_ == kTVMNDArrayHandle) {
      return static_cast<DLTensor*>(value_.v_handle);
    } else {
      if (type_code_ == kTVMNullptr) return nullptr;
      LOG(FATAL) << "Expected "
                 << "DLTensor* or NDArray but got " << ArgTypeCode2Str(type_code_);
      return nullptr;
    }
  }
  operator NDArray() const {
    if (type_code_ == kTVMNullptr) return NDArray(ObjectPtr<Object>(nullptr));
    TVM_CHECK_TYPE_CODE(type_code_, kTVMNDArrayHandle);
    return NDArray(NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle)));
  }
  operator Module() const {
    if (type_code_ == kTVMNullptr) {
      return Module(ObjectPtr<Object>(nullptr));
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMModuleHandle);
    return Module(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  }
  operator PackedFunc() const {
    if (type_code_ == kTVMNullptr) {
      return PackedFunc(ObjectPtr<Object>(nullptr));
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMPackedFuncHandle);
    return PackedFunc(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  }
  operator Device() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLDevice);
    return value_.v_device;
  }
  int type_code() const { return type_code_; }
  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template <typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }
  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  inline bool IsObjectRef() const;
  template <typename TObjectRef>
  inline TObjectRef AsObjectRef() const;

 protected:
  friend class TVMArgsSetter;
  friend class TVMRetValue;
  friend class TVMMovableArgValue_;
  TVMPODValue_() : type_code_(kTVMNullptr) {}
  TVMPODValue_(TVMValue value, int type_code) : value_(value), type_code_(type_code) {}

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
  TVMArgValue(TVMValue value, int type_code) : TVMPODValue_(value, type_code) {}
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator Device;
  using TVMPODValue_::operator Module;
  using TVMPODValue_::operator PackedFunc;
  using TVMPODValue_::AsObjectRef;
  using TVMPODValue_::IsObjectRef;

  // conversion operator.
  operator std::string() const {
    if (type_code_ == kTVMDataType) {
      return DLDataType2String(operator DLDataType());
    } else if (type_code_ == kTVMBytes) {
      TVMByteArray* arr = static_cast<TVMByteArray*>(value_.v_handle);
      return std::string(arr->data, arr->size);
    } else if (type_code_ == kTVMStr) {
      return std::string(value_.v_str);
    } else {
      return AsObjectRef<tvm::runtime::String>().operator std::string();
    }
  }
  template <typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  const TVMValue& value() const { return value_; }

  template <typename T, typename = typename std::enable_if<std::is_class<T>::value>::type>
  inline operator T() const;
  inline operator DLDataType() const;
  inline operator DataType() const;
};

/*!
 * \brief Internal auxiliary struct for TypedPackedFunc to indicate a movable argument.
 *
 *  We can only construct a movable argument once from a single argument position.
 *  If the argument is passed as RValue reference, the result will be moved.
 *  We should only construct a MovableArg from an argument once,
 *  as the result will can moved.
 *
 * \note For internal development purpose only.
 */
class TVMMovableArgValue_ : public TVMPODValue_ {
 public:
  TVMMovableArgValue_(TVMValue value, int type_code) : TVMPODValue_(value, type_code) {}
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator Device;
  using TVMPODValue_::operator Module;
  using TVMPODValue_::operator PackedFunc;
  // reuse conversion rule from ArgValue.
  operator std::string() const { return AsArgValue().operator std::string(); }
  template <typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  operator DLDataType() const { return AsArgValue().operator DLDataType(); }
  operator DataType() const { return AsArgValue().operator DataType(); }
  operator TVMArgValue() const { return AsArgValue(); }
  /*!
   * \brief Helper converter function.
   *  Try to move out an argument if possible,
   *  fall back to normal argument conversion rule otherwise.
   */
  template <typename T,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
  inline operator T() const;

 private:
  /*! \return The arg value repr of the value. */
  TVMArgValue AsArgValue() const { return TVMArgValue(value_, type_code_); }
};

/*!
 * \brief Internal auxiliary struct for TypedPackedFunc to indicate a movable argument with
 * additional context information (function name and argument index) for better error reporting.
 *
 * \sa MovableArgValue_
 * \note For internal development purpose only.
 */
class TVMMovableArgValueWithContext_ {
 public:
  /*!
   * \brief move constructor from another return value.
   * \param value The other return value.
   * \param type_code The code associated with the type of the value.
   * \param arg_index In a function call, this argument is at index arg_index (0-indexed).
   * \param optional_name Name of the function being called. Can be nullptr if the function is not.
   * \param f_sig Pointer to static function outputting signature of the function being called.
   * named.
   */
  TVMMovableArgValueWithContext_(TVMValue value, int type_code, int arg_index,
                                 const std::string* optional_name, FSig* f_sig)
      : value_(value, type_code),
        arg_index_(arg_index),
        optional_name_(optional_name),
        f_sig_(f_sig) {}

  template <typename T>
  operator T() const {
    try {
      return value_;  // implicit conversion happens here
    } catch (dmlc::Error& e) {
      LOG(FATAL) << "In function " << (optional_name_ == nullptr ? "<anonymous>" : *optional_name_)
                 << (f_sig_ == nullptr ? "" : (*f_sig_)()) << ": error while converting argument "
                 << arg_index_ << ": " << e.what();
      throw;  // never reached, LOG(FATAL) throws, but this silences a warning.
    }
  }

 private:
  TVMMovableArgValue_ value_;
  int arg_index_;
  const std::string* optional_name_;
  FSig* f_sig_;
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
   * \brief move constructor from another return value.
   * \param other The other return value.
   */
  TVMRetValue(TVMRetValue&& other) : TVMPODValue_(other.value_, other.type_code_) {
    other.value_.v_handle = nullptr;
    other.type_code_ = kTVMNullptr;
  }
  /*! \brief destructor */
  ~TVMRetValue() { this->Clear(); }
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  using TVMPODValue_::operator Device;
  using TVMPODValue_::operator NDArray;
  using TVMPODValue_::operator Module;
  using TVMPODValue_::operator PackedFunc;
  using TVMPODValue_::AsObjectRef;
  using TVMPODValue_::IsObjectRef;

  TVMRetValue(const TVMRetValue& other) : TVMPODValue_() { this->Assign(other); }
  // conversion operators
  operator std::string() const {
    if (type_code_ == kTVMDataType) {
      return DLDataType2String(operator DLDataType());
    } else if (type_code_ == kTVMBytes) {
      return *ptr<std::string>();
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMStr);
    return *ptr<std::string>();
  }
  operator DLDataType() const {
    if (type_code_ == kTVMStr) {
      return String2DLDataType(operator std::string());
    }
    TVM_CHECK_TYPE_CODE(type_code_, kTVMDataType);
    return value_.v_type;
  }
  operator DataType() const { return DataType(operator DLDataType()); }
  template <typename FType>
  operator TypedPackedFunc<FType>() const {
    return TypedPackedFunc<FType>(operator PackedFunc());
  }
  // Assign operators
  TVMRetValue& operator=(TVMRetValue&& other) {
    this->Clear();
    value_ = other.value_;
    type_code_ = other.type_code_;
    other.type_code_ = kTVMNullptr;
    return *this;
  }
  TVMRetValue& operator=(double value) {
    this->SwitchToPOD(kDLFloat);
    value_.v_float64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::nullptr_t value) {
    this->SwitchToPOD(kTVMNullptr);
    value_.v_handle = value;
    return *this;
  }
  TVMRetValue& operator=(void* value) {
    this->SwitchToPOD(kTVMOpaqueHandle);
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
  TVMRetValue& operator=(DLDevice value) {
    this->SwitchToPOD(kDLDevice);
    value_.v_device = value;
    return *this;
  }
  TVMRetValue& operator=(DLDataType t) {
    this->SwitchToPOD(kTVMDataType);
    value_.v_type = t;
    return *this;
  }
  TVMRetValue& operator=(const DataType& other) { return operator=(other.operator DLDataType()); }
  TVMRetValue& operator=(bool value) {
    this->SwitchToPOD(kDLInt);
    value_.v_int64 = value;
    return *this;
  }
  TVMRetValue& operator=(std::string value) {
    this->SwitchToClass(kTVMStr, value);
    return *this;
  }
  TVMRetValue& operator=(TVMByteArray value) {
    this->SwitchToClass(kTVMBytes, std::string(value.data, value.size));
    return *this;
  }
  TVMRetValue& operator=(NDArray other) {
    if (other.data_ != nullptr) {
      this->Clear();
      type_code_ = kTVMNDArrayHandle;
      value_.v_handle = NDArray::FFIGetHandle(other);
      ObjectRef::FFIClearAfterMove(&other);
    } else {
      SwitchToPOD(kTVMNullptr);
      value_.v_handle = nullptr;
    }
    return *this;
  }
  TVMRetValue& operator=(Module m) {
    SwitchToObject(kTVMModuleHandle, std::move(m.data_));
    return *this;
  }
  TVMRetValue& operator=(PackedFunc f) {
    this->SwitchToObject(kTVMPackedFuncHandle, std::move(f.data_));
    return *this;
  }
  template <typename FType>
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
  TVMRetValue& operator=(TVMMovableArgValue_&& other) {
    this->Assign(other);
    return *this;
  }
  /*!
   * \brief Move the value back to front-end via C API.
   *  This marks the current container as null.
   *  The managed resources are moved to the front-end.
   *  The front end should take charge in managing them.
   *
   * \param ret_value The return value.
   * \param ret_type_code The return type code.
   */
  void MoveToCHost(TVMValue* ret_value, int* ret_type_code) {
    // cannot move str; need specially handle.
    ICHECK(type_code_ != kTVMStr && type_code_ != kTVMBytes);
    *ret_value = value_;
    *ret_type_code = type_code_;
    type_code_ = kTVMNullptr;
  }
  /*!
   * \brief Construct a new TVMRetValue by
   *        moving from return value stored via C API.
   * \param value the value.
   * \param type_code The type code.
   * \return The created TVMRetValue.
   */
  static TVMRetValue MoveFromCHost(TVMValue value, int type_code) {
    // Can move POD and everything under the object system.
    ICHECK(type_code <= kTVMPackedFuncHandle || type_code == kTVMNDArrayHandle);
    TVMRetValue ret;
    ret.value_ = value;
    ret.type_code_ = type_code;
    return ret;
  }
  /*! \return The value field, if the data is POD */
  const TVMValue& value() const {
    ICHECK(type_code_ != kTVMObjectHandle && type_code_ != kTVMPackedFuncHandle &&
           type_code_ != kTVMModuleHandle && type_code_ != kTVMStr)
        << "TVMRetValue.value can only be used for POD data";
    return value_;
  }
  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  inline TVMRetValue& operator=(TObjectRef other);
  template <typename T, typename = typename std::enable_if<std::is_class<T>::value>::type>
  inline operator T() const;

 private:
  template <typename T>
  void Assign(const T& other) {
    switch (other.type_code()) {
      case kTVMStr: {
        SwitchToClass<std::string>(kTVMStr, other);
        break;
      }
      case kTVMBytes: {
        SwitchToClass<std::string>(kTVMBytes, other);
        break;
      }
      case kTVMPackedFuncHandle: {
        *this = other.operator PackedFunc();
        break;
      }
      case kTVMModuleHandle: {
        *this = other.operator Module();
        break;
      }
      case kTVMNDArrayHandle: {
        *this = other.operator NDArray();
        break;
      }
      case kTVMObjectHandle: {
        // Avoid operator ObjectRef as we already know it is not NDArray/Module
        SwitchToObject(kTVMObjectHandle,
                       GetObjectPtr<Object>(static_cast<Object*>(other.value_.v_handle)));
        break;
      }
      case kTVMObjectRValueRefArg: {
        operator=(other.operator ObjectRef());
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
  template <typename T>
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
      SwitchToPOD(kTVMNullptr);
      value_.v_handle = nullptr;
    }
  }
  void Clear() {
    if (type_code_ == kTVMNullptr) return;
    switch (type_code_) {
      case kTVMStr:
      case kTVMBytes:
        delete ptr<std::string>();
        break;
      case kTVMPackedFuncHandle:
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      case kTVMNDArrayHandle: {
        NDArray::FFIDecRef(static_cast<TVMArrayHandle>(value_.v_handle));
        break;
      }
      case kTVMModuleHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
      case kTVMObjectHandle: {
        static_cast<Object*>(value_.v_handle)->DecRef();
        break;
      }
    }
    type_code_ = kTVMNullptr;
  }
};

/*!
 * \brief Type trait to specify special value conversion rules from
 *        TVMArgValue and TVMRetValue.
 *
 *  The trait can be specialized to add type specific conversion logic
 *  from the TVMArgvalue and TVMRetValue.
 *
 * \tparam TObjectRef the specific ObjectRefType.
 */
template <typename TObjectRef>
struct PackedFuncValueConverter {
  /*!
   * \brief Convert a TObjectRef from an argument value.
   * \param val The argument value.
   * \return the converted result.
   */
  static TObjectRef From(const TVMArgValue& val) { return val.AsObjectRef<TObjectRef>(); }
  /*!
   * \brief Convert a TObjectRef from a return value.
   * \param val The argument value.
   * \return the converted result.
   */
  static TObjectRef From(const TVMRetValue& val) { return val.AsObjectRef<TObjectRef>(); }
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
#define TVM_DLL_EXPORT_PACKED_FUNC(ExportName, Function)                                    \
  extern "C" {                                                                              \
  TVM_DLL int ExportName(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, \
                         int* out_type_code, void* resource_handle);                        \
  int ExportName(TVMValue* args, int* type_code, int num_args, TVMValue* out_value,         \
                 int* out_type_code, void* resource_handle) {                               \
    try {                                                                                   \
      ::tvm::runtime::TVMRetValue rv;                                                       \
      Function(::tvm::runtime::TVMArgs(args, type_code, num_args), &rv);                    \
      rv.MoveToCHost(out_value, out_type_code);                                             \
      return 0;                                                                             \
    } catch (const ::std::exception& _except_) {                                            \
      TVMAPISetLastError(_except_.what());                                                  \
      return -1;                                                                            \
    }                                                                                       \
  }                                                                                         \
  }

#define TVM_MODULE_VTABLE_BEGIN(TypeKey)                                                 \
  const char* type_key() const final { return TypeKey; }                                 \
  PackedFunc GetFunction(const String& _name, const ObjectPtr<Object>& _self) override { \
    using SelfPtr = std::remove_cv_t<decltype(this)>;
#define TVM_MODULE_VTABLE_END() \
  return PackedFunc(nullptr);   \
  }
#define TVM_MODULE_VTABLE_END_WITH_DEFAULT(MemFunc) \
  {                                                 \
    auto f = (MemFunc);                             \
    return (this->*f)(_name);                       \
  }                                                 \
  }  // NOLINT(*)
#define TVM_MODULE_VTABLE_ENTRY(Name, MemFunc)                                                    \
  if (_name == Name) {                                                                            \
    return PackedFunc([_self](TVMArgs args, TVMRetValue* rv) -> void {                            \
      using Helper = ::tvm::runtime::detail::ModuleVTableEntryHelper<decltype(MemFunc)>;          \
      SelfPtr self = static_cast<SelfPtr>(_self.get());                                           \
      CHECK_EQ(args.size(), Helper::LenArgs)                                                      \
          << "Function `" << self->type_key() << "::" << Name << "` requires " << Helper::LenArgs \
          << " arguments, but got " << args.size();                                               \
      Helper::Call(rv, self, MemFunc, args, Helper::IndexSeq{});                                  \
    });                                                                                           \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                  \
  if (_name == Name) {                                                 \
    return PackedFunc([_self](TVMArgs args, TVMRetValue* rv) -> void { \
      (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);       \
    });                                                                \
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
 * // Because the same Function and ExportName
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, AddOne_);
 *
 * // The following code is OK, assuming the macro
 * // is in a different namespace from xyz
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, xyz::AddOne_);
 *
 * \endcode
 */
#define TVM_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                                     \
  extern "C" {                                                                              \
  TVM_DLL int ExportName(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, \
                         int* out_type_code, void* resource_handle) {                       \
    try {                                                                                   \
      auto f = Function;                                                                    \
      using FType = ::tvm::runtime::detail::function_signature<decltype(f)>::FType;         \
      ::tvm::runtime::TVMRetValue rv;                                                       \
      ::tvm::runtime::detail::unpack_call_by_signature<FType>::run(                         \
          f, ::tvm::runtime::TVMArgs(args, type_code, num_args), &rv);                      \
      rv.MoveToCHost(out_value, out_type_code);                                             \
      return 0;                                                                             \
    } catch (const ::std::exception& _except_) {                                            \
      TVMAPISetLastError(_except_.what());                                                  \
      return -1;                                                                            \
    }                                                                                       \
  }                                                                                         \
  }

inline TVMArgValue TVMArgs::operator[](int i) const {
  ICHECK_LT(i, num_args) << "not enough argument passed, " << num_args << " passed"
                         << " but request arg[" << i << "].";
  return TVMArgValue(values[i], type_codes[i]);
}

inline int TVMArgs::size() const { return num_args; }

template <class TPackedFuncSubObj>
void PackedFuncObj::Extractor<TPackedFuncSubObj>::Call(const PackedFuncObj* obj, TVMArgs args,
                                                       TVMRetValue* rv) {
  (static_cast<const TPackedFuncSubObj*>(obj))->callable_(args, rv);
}

TVM_ALWAYS_INLINE void PackedFuncObj::CallPacked(TVMArgs args, TVMRetValue* rv) const {
  (*f_call_packed_)(this, args, rv);
}

TVM_ALWAYS_INLINE void PackedFunc::CallPacked(TVMArgs args, TVMRetValue* rv) const {
  (static_cast<PackedFuncObj*>(data_.get()))->CallPacked(args, rv);
}

// internal namespace
inline const char* ArgTypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt:
      return "int";
    case kDLUInt:
      return "uint";
    case kDLFloat:
      return "float";
    case kTVMStr:
      return "str";
    case kTVMBytes:
      return "bytes";
    case kTVMOpaqueHandle:
      return "handle";
    case kTVMNullptr:
      return "NULL";
    case kTVMDLTensorHandle:
      return "ArrayHandle";
    case kTVMDataType:
      return "DLDataType";
    case kDLDevice:
      return "DLDevice";
    case kTVMPackedFuncHandle:
      return "FunctionHandle";
    case kTVMModuleHandle:
      return "ModuleHandle";
    case kTVMNDArrayHandle:
      return "NDArrayContainer";
    case kTVMObjectHandle:
      return "Object";
    case kTVMObjectRValueRefArg:
      return "ObjectRValueRefArg";
    default:
      LOG(FATAL) << "unknown type_code=" << static_cast<int>(type_code);
  }
  throw;
}

/*!
 * \brief The name of DLDeviceType.
 * \param type The device type.
 * \return the device name.
 */
inline const char* DLDeviceType2Str(int type) {
  switch (type) {
    case kDLCPU:
      return "cpu";
    case kDLCUDA:
      return "cuda";
    case kDLCUDAHost:
      return "cuda_host";
    case kDLCUDAManaged:
      return "cuda_managed";
    case kDLOpenCL:
      return "opencl";
    case kDLSDAccel:
      return "sdaccel";
    case kDLAOCL:
      return "aocl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kDLROCMHost:
      return "rocm_host";
    case kDLExtDev:
      return "ext_dev";
    case kDLOneAPI:
      return "oneapi";
    case kDLWebGPU:
      return "webgpu";
    case kDLHexagon:
      return "hexagon";
    case kOpenGL:
      return "opengl";
    case kDLMicroDev:
      return "microdev";
    default:
      LOG(FATAL) << "unknown type = " << type;
  }
  throw;
}

namespace detail {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I + 1), F>::run(f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void run(const F& f) {}  // NOLINT(*)
};

template <typename F, typename... Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::run(f, std::forward<Args>(args)...);
}

template <typename T>
struct ModuleVTableEntryHelper {};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...) const> {
  using MemFnType = R (T::*)(Args...) const;
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    *rv = (self->*f)(args[Is]...);
  }
};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...)> {
  using MemFnType = R (T::*)(Args...);
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    *rv = (self->*f)(args[Is]...);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...) const> {
  using MemFnType = void (T::*)(Args...) const;
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    (self->*f)(args[Is]...);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...)> {
  using MemFnType = void (T::*)(Args...);
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    (self->*f)(args[Is]...);
  }
};

namespace parameter_pack {

template <typename... EnumArgs>
struct EnumeratedParamPack {
  struct InvokeWithoutArg {
    template <template <size_t i, typename TArgument> class Functor, typename ExtraParams>
    static void F(ExtraParams&& extra_params) {
      using TExpander = int[];
      (void)TExpander{
          0,
          (Functor<EnumArgs::i, typename EnumArgs::T>::F(std::forward<ExtraParams>(extra_params)),
           0)...,
      };
    }
  };
  struct InvokeWithArg {
    template <template <size_t i, typename TArgument> class Functor, typename ExtraParams,
              typename... Params>
    static void F(ExtraParams&& extra_params, Params&&... params) {
      using TExpander = int[];
      (void)TExpander{
          0,
          (Functor<EnumArgs::i, typename EnumArgs::T>::F(std::forward<ExtraParams>(extra_params),
                                                         std::forward<Params>(params)),
           0)...,
      };
    }
  };
};

template <typename... Args>
struct EnumerateImpl {
 private:
  template <size_t _i, typename _T>
  struct Item {
    static const constexpr size_t i = _i;
    using T = _T;
  };

  template <typename...>
  struct Zipper;

  template <std::size_t... id>
  struct Zipper<std::integer_sequence<std::size_t, id...>> {
    using WithoutArg = typename EnumeratedParamPack<Item<id, Args>...>::InvokeWithoutArg;
    using WithArg = typename EnumeratedParamPack<Item<id, Args>...>::InvokeWithArg;
  };

 public:
  using WithoutArg = typename Zipper<std::index_sequence_for<Args...>>::WithoutArg;
  using WithArg = typename Zipper<std::index_sequence_for<Args...>>::WithArg;
};

template <typename... Args>
using EnumerateWithoutArg = typename EnumerateImpl<Args...>::WithoutArg;

template <typename... Args>
using EnumerateWithArg = typename EnumerateImpl<Args...>::WithArg;

template <typename... Args>
struct ParamPack {
  template <template <size_t i, typename TArgument> class Functor, typename ExtraParams>
  static void InvokeWithoutArg(ExtraParams&& extra_params) {
    EnumerateWithoutArg<Args...>::template F<Functor, ExtraParams>(
        std::forward<ExtraParams>(extra_params));
  }
};

}  // namespace parameter_pack

/*!
 * \brief Template class to get function signature of a function or functor.
 * \tparam T The function/functor type.
 */
template <typename T>
struct func_signature_helper {
  using FType = void;
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(Args...)> {
  using FType = R(Args...);
  using ParamType = parameter_pack::ParamPack<Args...>;
  using RetType = R;
  static_assert(!std::is_reference<R>::value, "TypedPackedFunc return reference");
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(Args...) const> {
  using FType = R(Args...);
  using ParamType = parameter_pack::ParamPack<Args...>;
  using RetType = R;
  static_assert(!std::is_reference<R>::value, "TypedPackedFunc return reference");
};

/*!
 * \brief Template class to get function signature of a function or functor.
 * \tparam T The function/functor type.
 */
template <typename T>
struct function_signature {
  using FType = typename func_signature_helper<decltype(&T::operator())>::FType;
  using ParamType = typename func_signature_helper<decltype(&T::operator())>::ParamType;
  using RetType = typename func_signature_helper<decltype(&T::operator())>::RetType;
};

// handle case of function.
template <typename R, typename... Args>
struct function_signature<R(Args...)> {
  using FType = R(Args...);
  using ParamType = parameter_pack::ParamPack<Args...>;
  using RetType = R;
  static_assert(!std::is_reference<R>::value, "TypedPackedFunc return reference");
};

// handle case of function ptr.
template <typename R, typename... Args>
struct function_signature<R (*)(Args...)> {
  using FType = R(Args...);
  using ParamType = detail::parameter_pack::ParamPack<Args...>;
  using RetType = R;
  static_assert(!std::is_reference<R>::value, "TypedPackedFunc return reference");
};

template <typename TSignature>
struct SignaturePrinter;

namespace type2str {

template <typename T>
struct TypeSimplifier;

template <typename T>
struct Type2Str {
  template <typename = std::enable_if_t<std::is_base_of<ObjectRef, T>::value>>
  static std::string v() {
    return T::ContainerType::_type_key;
  }
};
template <>
struct Type2Str<int> {
  static std::string v() { return "int"; }
};
template <>
struct Type2Str<double> {
  static std::string v() { return "double"; }
};
template <>
struct Type2Str<int64_t> {
  static std::string v() { return "int64_t"; }
};
template <>
struct Type2Str<uint64_t> {
  static std::string v() { return "uint64_t"; }
};
template <>
struct Type2Str<bool> {
  static std::string v() { return "bool"; }
};
template <>
struct Type2Str<void> {
  static std::string v() { return "void"; }
};
template <>
struct Type2Str<std::basic_string<char>> {
  static std::string v() { return "basic_string<char>"; }
};
template <typename K, typename V>
struct Type2Str<Map<K, V>> {
  static std::string v() {
    return "Map<" + TypeSimplifier<K>::v() + ", " + TypeSimplifier<V>::v() + ">";
  }
};
template <>
struct Type2Str<DLDevice> {
  static std::string v() { return "DLDevice"; }
};
template <>
struct Type2Str<DLTensor> {
  static std::string v() { return "DLTensor"; }
};
template <>
struct Type2Str<DataType> {
  static std::string v() { return "DataType"; }
};
template <>
struct Type2Str<DLDataType> {
  static std::string v() { return "DLDataType"; }
};
template <>
struct Type2Str<TVMRetValue> {
  static std::string v() { return "TVMRetValue"; }
};
template <>
struct Type2Str<TVMArgValue> {
  static std::string v() { return "TVMArgValue"; }
};
template <>
struct Type2Str<TVMByteArray> {
  static std::string v() { return "TVMByteArray"; }
};
template <typename FType>
struct Type2Str<TypedPackedFunc<FType>> {
  static std::string v() { return SignaturePrinter<function_signature<FType>>::F(); }
};
template <typename T>
struct Type2Str<Array<T>> {
  static std::string v() { return "Array<" + TypeSimplifier<T>::v() + ">"; }
};

/*!
 * \brief Template class to remove const, pointer and reference of original type.
 * \tparam T The original type.
 */
template <typename T>
struct TypeSimplifier {
  static std::string v() {
    using U = typename std::remove_cv<
        typename std::remove_reference<typename std::remove_pointer<T>::type>::type>::type;
    return (std::is_const<T>::value ? "const " : "") + Type2Str<U>::v() +
           (std::is_pointer<T>::value ? "*" : "") + (std::is_reference<T>::value ? "&" : "");
  }
};

}  // namespace type2str

/*!
 * \brief Template class to generate static function outputting signature of a function or functor.
 * \tparam TSignature The function/functor signature type generated by `function_signature`.
 */
template <typename TSignature>
struct SignaturePrinter {
  using ParamType = typename TSignature::ParamType;
  using RetType = typename TSignature::RetType;

  template <size_t i, typename TArgument>
  struct PrintParamType {
    static void F(std::ostream& os) {
      os << (i == 0 ? "" : ", ") << i << ": " << type2str::TypeSimplifier<TArgument>::v();
    }
  };

  static std::string F() {
    std::ostringstream oss;
    oss << "(";
    ParamType::template InvokeWithoutArg<PrintParamType>(oss);
    oss << ") -> " << type2str::TypeSimplifier<RetType>::v();
    return oss.str();
  }
};
}  // namespace detail

/* \brief argument settter to PackedFunc */
class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes) : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    ICHECK_LE(value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMNullptr;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, const TVMArgValue& value) const {
    values_[i] = value.value_;
    type_codes_[i] = value.type_code_;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMOpaqueHandle;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMDLTensorHandle;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, Device value) const {
    values_[i].v_device = value;
    type_codes_[i] = kDLDevice;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DLDataType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kTVMDataType;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DataType dtype) const {
    operator()(i, dtype.operator DLDataType());
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kTVMStr;
  }
  // setters for container types
  TVM_ALWAYS_INLINE void operator()(size_t i, const std::string& value) const {
    values_[i].v_str = value.c_str();
    type_codes_[i] = kTVMStr;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, const TVMByteArray& value) const {
    values_[i].v_handle = const_cast<TVMByteArray*>(&value);
    type_codes_[i] = kTVMBytes;
  }
  template <typename FType>
  TVM_ALWAYS_INLINE void operator()(size_t i, const TypedPackedFunc<FType>& value) const {
    operator()(i, value.packed());
  }
  void operator()(size_t i, const TVMRetValue& value) const {
    if (value.type_code() == kTVMStr) {
      values_[i].v_str = value.ptr<std::string>()->c_str();
      type_codes_[i] = kTVMStr;
    } else {
      ICHECK_NE(value.type_code(), kTVMBytes) << "not handled.";
      values_[i] = value.value_;
      type_codes_[i] = value.type_code();
    }
  }
  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, const TObjectRef& value) const {
    this->SetObject(i, value);
  }

  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef, typename std::remove_reference<TObjectRef>::type>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, TObjectRef&& value) const {
    this->SetObject(i, std::forward<TObjectRef>(value));
  }

 private:
  template <typename TObjectRef>
  inline void SetObject(size_t i, TObjectRef&& value) const;
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template <typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&&... args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes), std::forward<Args>(args)...);
  TVMRetValue rv;
  (static_cast<PackedFuncObj*>(data_.get()))
      ->CallPacked(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

template <size_t i, typename T>
struct TVMArgsSetterApply {
  static TVM_ALWAYS_INLINE void F(TVMArgsSetter* setter, T&& value) {
    (*setter)(i, std::forward<T>(value));
  }
};

template <typename... Args>
void TVM_ALWAYS_INLINE PackArgs(TVMValue* values, int* type_codes, Args&&... args) {
  TVMArgsSetter setter(values, type_codes);
  detail::parameter_pack::EnumerateWithArg<Args...>::template F<TVMArgsSetterApply>(
      &setter, std::forward<Args>(args)...);
}

namespace detail {
template <typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template <typename... Args>
  TVM_ALWAYS_INLINE static void run(const std::string* optional_name, FSig* f_sig, const F& f,
                                    const TVMArgs& args_pack, TVMRetValue* rv,
                                    Args&&... unpacked_args) {
    // construct a movable argument value
    // which allows potential move of argument to the input of F.
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>::run(
        optional_name, f_sig, f, args_pack, rv, std::forward<Args>(unpacked_args)...,
        TVMMovableArgValueWithContext_(args_pack.values[index], args_pack.type_codes[index], index,
                                       optional_name, f_sig));
  }
};

template <typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template <typename... Args>
  TVM_ALWAYS_INLINE static void run(const std::string* optional_name, FSig* f_sig, const F& f,
                                    const TVMArgs& args_pack, TVMRetValue* rv,
                                    Args&&... unpacked_args) {
    using RetType = decltype(f(std::forward<Args>(unpacked_args)...));
    if (std::is_same<RetType, R>::value) {
      *rv = f(std::forward<Args>(unpacked_args)...);
    } else {
      *rv = R(f(std::forward<Args>(unpacked_args)...));
    }
  }
};

template <int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template <typename... Args>
  TVM_ALWAYS_INLINE static void run(const std::string* optional_name, FSig* f_sig, const F& f,
                                    const TVMArgs& args_pack, TVMRetValue* rv,
                                    Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename F>
TVM_ALWAYS_INLINE void unpack_call(const std::string* optional_name, const F& f,
                                   const TVMArgs& args, TVMRetValue* rv) {
  FSig* f_sig = detail::SignaturePrinter<detail::function_signature<F>>::F;
  CHECK_EQ(nargs, args.size()) << "Function "
                               << (optional_name == nullptr ? "<anonymous>" : *optional_name)
                               << (f_sig == nullptr ? "" : (*f_sig)()) << " expects " << nargs
                               << " arguments but " << args.size() << " were provided";
  unpack_call_dispatcher<R, nargs, 0, F>::run(optional_name, f_sig, f, args, rv);
}

template <typename FType>
struct unpack_call_by_signature {};

template <typename R, typename... Args>
struct unpack_call_by_signature<R(Args...)> {
  template <typename F>
  TVM_ALWAYS_INLINE static void run(const F& f, const TVMArgs& args, TVMRetValue* rv) {
    unpack_call<R, sizeof...(Args)>(nullptr, f, args, rv);
  }
};

template <typename R, typename... Args>
TVM_ALWAYS_INLINE R call_packed(const PackedFunc& pf, Args&&... args) {
  return R(pf(std::forward<Args>(args)...));
}

template <typename R>
struct typed_packed_call_dispatcher {
  template <typename... Args>
  TVM_ALWAYS_INLINE static R run(const PackedFunc& pf, Args&&... args) {
    return pf(std::forward<Args>(args)...);
  }
};

template <>
struct typed_packed_call_dispatcher<void> {
  template <typename... Args>
  TVM_ALWAYS_INLINE static void run(const PackedFunc& pf, Args&&... args) {
    pf(std::forward<Args>(args)...);
  }
};
}  // namespace detail

template <typename R, typename... Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(PackedFunc packed) : packed_(packed) {}

template <typename R, typename... Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const TVMRetValue& value)
    : packed_(value.operator PackedFunc()) {}

template <typename R, typename... Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(const TVMArgValue& value)
    : packed_(value.operator PackedFunc()) {}

template <typename R, typename... Args>
TypedPackedFunc<R(Args...)>::TypedPackedFunc(TVMMovableArgValueWithContext_&& value)
    : packed_(value.operator PackedFunc()) {}

template <typename R, typename... Args>
template <typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda, std::string name) {
  FSig* f_sig = detail::SignaturePrinter<detail::function_signature<FType>>::F;
  packed_ = PackedFunc([flambda, name, f_sig](const TVMArgs& args, TVMRetValue* rv) {
    if (args.size() != sizeof...(Args)) {
      LOG(FATAL) << "Function " << name << (f_sig == nullptr ? "" : (*f_sig)()) << " expects "
                 << sizeof...(Args) << " arguments, but " << args.size() << " were provided.";
    }
    detail::unpack_call<R, sizeof...(Args)>(&name, flambda, args, rv);
  });
}

template <typename R, typename... Args>
template <typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  FSig* f_sig = detail::SignaturePrinter<detail::function_signature<FType>>::F;
  packed_ = PackedFunc([flambda, f_sig](const TVMArgs& args, TVMRetValue* rv) {
    if (args.size() != sizeof...(Args)) {
      LOG(FATAL) << "Function <anonymous> " << (*f_sig)() << " expects " << sizeof...(Args)
                 << " arguments, but " << args.size() << " were provided.";
    }
    detail::unpack_call<R, sizeof...(Args)>(nullptr, flambda, args, rv);
  });
}

template <typename R, typename... Args>
TVM_ALWAYS_INLINE R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>::run(packed_, std::forward<Args>(args)...);
}

template <typename T>
inline T TVMArgs::At(int i) const {
  TVMArgValue arg = operator[](i);
  try {
    return arg.operator T();
  } catch (const dmlc::Error& e) {
    LOG(FATAL) << "Argument " << i << " cannot be converted to type \""
               << tvm::runtime::detail::type2str::Type2Str<T>::v() << "\". Its type is \""
               << tvm::runtime::ArgTypeCode2Str(arg.type_code()) << "\".";
  }
  throw;
}

// ObjectRef related conversion handling
// Object can have three possible type codes:
//      kTVMNDArrayHandle, kTVMModuleHandle, kTVMObjectHandle
//
// We use type traits to eliminate un-necessary checks.
template <typename T>
inline void TVMArgsSetter::SetObject(size_t i, T&& value) const {
  using ContainerType = typename std::remove_reference<T>::type::ContainerType;
  if (value.defined()) {
    Object* ptr = value.data_.data_;
    if (std::is_base_of<NDArray::ContainerType, ContainerType>::value ||
        (std::is_base_of<ContainerType, NDArray::ContainerType>::value &&
         ptr->IsInstance<NDArray::ContainerType>())) {
      values_[i].v_handle = NDArray::FFIGetHandle(value);
      type_codes_[i] = kTVMNDArrayHandle;
    } else if (std::is_base_of<Module::ContainerType, ContainerType>::value ||
               (std::is_base_of<ContainerType, Module::ContainerType>::value &&
                ptr->IsInstance<Module::ContainerType>())) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kTVMModuleHandle;
    } else if (std::is_base_of<PackedFunc::ContainerType, ContainerType>::value ||
               (std::is_base_of<ContainerType, PackedFunc::ContainerType>::value &&
                ptr->IsInstance<PackedFunc::ContainerType>())) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kTVMPackedFuncHandle;
    } else if (std::is_rvalue_reference<decltype(value)>::value) {
      values_[i].v_handle = const_cast<Object**>(&(value.data_.data_));
      type_codes_[i] = kTVMObjectRValueRefArg;
    } else {
      values_[i].v_handle = value.data_.data_;
      type_codes_[i] = kTVMObjectHandle;
    }
  } else {
    type_codes_[i] = kTVMNullptr;
    values_[i].v_handle = nullptr;
  }
}

template <typename TObjectRef, typename>
inline bool TVMPODValue_::IsObjectRef() const {
  using ContainerType = typename TObjectRef::ContainerType;
  // NOTE: the following code can be optimized by constant folding.
  if (std::is_base_of<NDArray::ContainerType, ContainerType>::value) {
    return type_code_ == kTVMNDArrayHandle &&
           TVMArrayHandleToObjectHandle(static_cast<TVMArrayHandle>(value_.v_handle))
               ->IsInstance<ContainerType>();
  }
  if (std::is_base_of<Module::ContainerType, ContainerType>::value) {
    return type_code_ == kTVMModuleHandle &&
           static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
  }
  if (std::is_base_of<PackedFunc::ContainerType, ContainerType>::value) {
    return type_code_ == kTVMPackedFuncHandle &&
           static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
  }
  // NOTE: we don't pass NDArray and runtime::Module as RValue ref.
  if (type_code_ == kTVMObjectRValueRefArg) {
    return ObjectTypeChecker<TObjectRef>::Check(*static_cast<Object**>(value_.v_handle));
  }
  return (std::is_base_of<ContainerType, NDArray::ContainerType>::value &&
          type_code_ == kTVMNDArrayHandle) ||
         (std::is_base_of<ContainerType, Module::ContainerType>::value &&
          type_code_ == kTVMModuleHandle) ||
         (std::is_base_of<ContainerType, PackedFunc::ContainerType>::value &&
          type_code_ == kTVMPackedFuncHandle) ||
         (type_code_ == kTVMObjectHandle &&
          ObjectTypeChecker<TObjectRef>::Check(static_cast<Object*>(value_.v_handle)));
}

template <typename TObjectRef>
inline TObjectRef TVMPODValue_::AsObjectRef() const {
  static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                "Conversion only works for ObjectRef");
  using ContainerType = typename TObjectRef::ContainerType;

  if (type_code_ == kTVMNullptr) {
    CHECK(TObjectRef::_type_is_nullable)
        << "Expect a not null value of " << ContainerType::_type_key;
    return TObjectRef(ObjectPtr<Object>(nullptr));
  }
  // NOTE: the following code can be optimized by constant folding.
  if (std::is_base_of<NDArray::ContainerType, ContainerType>::value) {
    // Casting to a sub-class of NDArray
    TVM_CHECK_TYPE_CODE(type_code_, kTVMNDArrayHandle);
    ObjectPtr<Object> data =
        NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
    return TObjectRef(data);
  }
  if (std::is_base_of<Module::ContainerType, ContainerType>::value) {
    // Casting to a sub-class of Module
    TVM_CHECK_TYPE_CODE(type_code_, kTVMModuleHandle);
    ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
    return TObjectRef(data);
  }
  if (std::is_base_of<PackedFunc::ContainerType, ContainerType>::value) {
    // Casting to a sub-class of PackedFunc
    TVM_CHECK_TYPE_CODE(type_code_, kTVMPackedFuncHandle);
    ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
    CHECK(data->IsInstance<ContainerType>())
        << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
    return TObjectRef(data);
  }
  if (type_code_ == kTVMObjectHandle) {
    // normal object type check.
    Object* ptr = static_cast<Object*>(value_.v_handle);
    Optional<String> checked_type = ObjectTypeChecker<TObjectRef>::CheckAndGetMismatch(ptr);
    ICHECK(!checked_type.defined()) << "Expected " << ObjectTypeChecker<TObjectRef>::TypeName()
                                    << ", but got " << checked_type.value();
    return TObjectRef(GetObjectPtr<Object>(ptr));
  } else if (type_code_ == kTVMObjectRValueRefArg) {
    Object* ptr = *static_cast<Object**>(value_.v_handle);
    Optional<String> checked_type = ObjectTypeChecker<TObjectRef>::CheckAndGetMismatch(ptr);
    ICHECK(!checked_type.defined()) << "Expected " << ObjectTypeChecker<TObjectRef>::TypeName()
                                    << ", but got " << checked_type.value();
    return TObjectRef(GetObjectPtr<Object>(ptr));
  } else if (std::is_base_of<ContainerType, NDArray::ContainerType>::value &&
             type_code_ == kTVMNDArrayHandle) {
    // Casting to a base class that NDArray can sub-class
    ObjectPtr<Object> data =
        NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle));
    return TObjectRef(data);
  } else if (std::is_base_of<ContainerType, Module::ContainerType>::value &&
             type_code_ == kTVMModuleHandle) {
    // Casting to a base class that Module can sub-class
    return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  } else if (std::is_base_of<ContainerType, PackedFunc::ContainerType>::value &&
             type_code_ == kTVMPackedFuncHandle) {
    // Casting to a base class that PackedFunc can sub-class
    return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
  } else {
    TVM_CHECK_TYPE_CODE(type_code_, kTVMObjectHandle);
    return TObjectRef(ObjectPtr<Object>(nullptr));
  }
}

template <typename TObjectRef, typename>
inline TVMRetValue& TVMRetValue::operator=(TObjectRef other) {
  using ContainerType = typename TObjectRef::ContainerType;
  const Object* ptr = other.get();
  if (ptr != nullptr) {
    if (std::is_base_of<NDArray::ContainerType, ContainerType>::value ||
        (std::is_base_of<ContainerType, NDArray::ContainerType>::value &&
         ptr->IsInstance<NDArray::ContainerType>())) {
      return operator=(NDArray(std::move(other.data_)));
    }
    if (std::is_base_of<Module::ContainerType, ContainerType>::value ||
        (std::is_base_of<ContainerType, Module::ContainerType>::value &&
         ptr->IsInstance<Module::ContainerType>())) {
      return operator=(Module(std::move(other.data_)));
    }
    if (std::is_base_of<PackedFunc::ContainerType, ContainerType>::value ||
        (std::is_base_of<ContainerType, PackedFunc::ContainerType>::value &&
         ptr->IsInstance<PackedFunc::ContainerType>())) {
      return operator=(PackedFunc(std::move(other.data_)));
    }
    SwitchToObject(kTVMObjectHandle, std::move(other.data_));
  } else {
    SwitchToPOD(kTVMNullptr);
    value_.v_handle = nullptr;
  }
  return *this;
}

template <typename T, typename>
inline TVMArgValue::operator T() const {
  return PackedFuncValueConverter<T>::From(*this);
}

template <typename T, typename>
inline TVMMovableArgValue_::operator T() const {
  if (type_code_ == kTVMObjectRValueRefArg) {
    auto** ref = static_cast<Object**>(value_.v_handle);
    if (ObjectTypeChecker<T>::Check(*ref)) {
      return T(ObjectPtr<Object>::MoveFromRValueRefArg(ref));
    }
  }
  // fallback
  return PackedFuncValueConverter<T>::From(AsArgValue());
}

template <typename T, typename>
inline TVMRetValue::operator T() const {
  return PackedFuncValueConverter<T>::From(*this);
}

inline PackedFunc Module::GetFunction(const String& name, bool query_imports) {
  return (*this)->GetFunction(name, query_imports);
}

// specializations of PackedFuncValueConverter
template <>
struct PackedFuncValueConverter<::tvm::runtime::String> {
  static String From(const TVMArgValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }

  static String From(const TVMRetValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }
};

template <typename T>
struct PackedFuncValueConverter<Optional<T>> {
  static Optional<T> From(const TVMArgValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
  static Optional<T> From(const TVMRetValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
};

template <typename... VariantTypes>
struct PackedFuncValueConverter<Variant<VariantTypes...>> {
  using VType = Variant<VariantTypes...>;

  // Can't just take `const TVMPODValue&` as an argument, because
  // `TVMArgValue` and `TVMRetValue` have different implementations
  // for `operator std::string()`.
  template <typename PODSubclass>
  static VType From(const PODSubclass& val) {
    if (auto opt = TryAsObjectRef<VariantTypes...>(val)) {
      return opt.value();
    }

    if (auto opt = TryValueConverter<PODSubclass, VariantTypes...>(val)) {
      return opt.value();
    }

    LOG(FATAL) << "Expected one of "
               << static_cast<const std::stringstream&>(
                      (std::stringstream() << ... << VariantTypes::ContainerType::_type_key))
                      .str()
               << " but got " << ArgTypeCode2Str(val.type_code());
  }

  template <typename VarFirst, typename... VarRest>
  static Optional<VType> TryAsObjectRef(const TVMPODValue_& val) {
    if (val.IsObjectRef<VarFirst>()) {
      return VType(val.AsObjectRef<VarFirst>());
    } else if constexpr (sizeof...(VarRest)) {
      return TryAsObjectRef<VarRest...>(val);
    } else {
      return NullOpt;
    }
  }

  template <typename PODSubclass, typename VarFirst, typename... VarRest>
  static Optional<VType> TryValueConverter(const PODSubclass& val) {
    try {
      return VType(PackedFuncValueConverter<VarFirst>::From(val));
    } catch (const InternalError&) {
    }

    if constexpr (sizeof...(VarRest)) {
      return TryValueConverter<PODSubclass, VarRest...>(val);
    } else {
      return NullOpt;
    }
  }
};

inline bool String::CanConvertFrom(const TVMArgValue& val) {
  return val.type_code() == kTVMStr || val.IsObjectRef<tvm::runtime::String>();
}

inline TVMArgValue::operator DLDataType() const {
  if (String::CanConvertFrom(*this)) {
    return String2DLDataType(PackedFuncValueConverter<String>::From(*this).operator std::string());
  }
  // None type
  if (type_code_ == kTVMNullptr) {
    DLDataType t;
    t.code = kTVMOpaqueHandle;
    t.bits = 0;
    t.lanes = 0;
    return t;
  }
  TVM_CHECK_TYPE_CODE(type_code_, kTVMDataType);
  return value_.v_type;
}

inline TVMArgValue::operator DataType() const { return DataType(operator DLDataType()); }

}  // namespace runtime // NOLINT(*)
}  // namespace tvm // NOLINT(*)
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
