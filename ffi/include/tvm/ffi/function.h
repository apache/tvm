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
 * \file tvm/ffi/function.h
 * \brief A managed function in the TVM FFI.
 */
#ifndef TVM_FFI_FUNCTION_H_
#define TVM_FFI_FUNCTION_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function_details.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/**
 * Helper macro to construct a safe call
 *
 * \brief Marks the begining of the safe call that catches exception explicitly
 *
 */
#define TVM_FFI_SAFE_CALL_BEGIN() \
  try {                           \
  (void)0

/*!
 * \brief Marks the end of safe call.
 */
#define TVM_FFI_SAFE_CALL_END()                                                                \
  return 0;                                                                                    \
  }                                                                                            \
  catch (const ::tvm::ffi::Error& err) {                                                       \
    ::tvm::ffi::details::SetSafeCallRaised(err);                                               \
    return -1;                                                                                 \
  }                                                                                            \
  catch (const ::tvm::ffi::EnvErrorAlreadySet&) {                                              \
    return -2;                                                                                 \
  }                                                                                            \
  catch (const std::exception& ex) {                                                           \
    ::tvm::ffi::details::SetSafeCallRaised(::tvm::ffi::Error("InternalError", ex.what(), "")); \
    return -1;                                                                                 \
  }                                                                                            \
  TVM_FFI_UNREACHABLE()

#define TVM_FFI_CHECK_SAFE_CALL(func)                      \
  {                                                        \
    int ret_code = (func);                                 \
    if (ret_code != 0) {                                   \
      if (ret_code == -2) {                                \
        throw ::tvm::ffi::EnvErrorAlreadySet();            \
      }                                                    \
      throw ::tvm::ffi::details::MoveFromSafeCallRaised(); \
    }                                                      \
  }

/*!
 * \brief Object container class that backs ffi::Function
 * \note Do not use this function directly, use ffi::Function
 */
class FunctionObj : public Object, public TVMFFIFunctionCell {
 public:
  typedef void (*FCall)(const FunctionObj*, const AnyView*, int32_t, Any*);
  using TVMFFIFunctionCell::safe_call;
  /*! \brief A C++ style call implementation, with exception propagation in c++ style. */
  FCall call;

  TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
    this->call(this, args, num_args, result);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIFunction;
  static constexpr const char* _type_key = "object.Function";

  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(FunctionObj, Object);

 protected:
  /*! \brief Make default constructor protected. */
  FunctionObj() {}

  // Implementing safe call style
  static int SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
    TVM_FFI_SAFE_CALL_BEGIN();
    TVM_FFI_ICHECK_LT(result->type_index, TypeIndex::kTVMFFIStaticObjectBegin);
    FunctionObj* self = static_cast<FunctionObj*>(func);
    self->call(self, reinterpret_cast<const AnyView*>(args), num_args,
               reinterpret_cast<Any*>(result));
    TVM_FFI_SAFE_CALL_END();
  }

  friend class Function;
};

namespace details {
/*!
 * \brief Derived object class for constructing FunctionObj backed by a TCallable
 *
 * This is a helper class that
 */
template <typename TCallable>
class FunctionObjImpl : public FunctionObj {
 public:
  using TStorage = typename std::remove_cv<typename std::remove_reference<TCallable>::type>::type;
  /*! \brief The type of derived object class */
  using TSelf = FunctionObjImpl<TCallable>;
  /*!
   * \brief Derived object class for constructing ffi::FunctionObj.
   * \param callable The type-erased callable object.
   */
  explicit FunctionObjImpl(TCallable callable) : callable_(callable) {
    this->safe_call = SafeCall;
    this->call = Call;
  }

 private:
  // implementation of call
  static void Call(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* result) {
    (static_cast<const TSelf*>(func))->callable_(args, num_args, result);
  }

  /*! \brief Type-erased filed for storing callable object*/
  mutable TStorage callable_;
};

/*!
 * \brief Base class to provide a common implementation to redirect call to safecall
 * \tparam Derived The derived class in CRTP-idiom
 */
template <typename Derived>
struct RedirectCallToSafeCall {
  static void Call(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* rv) {
    Derived* self = static_cast<Derived*>(const_cast<FunctionObj*>(func));
    TVM_FFI_CHECK_SAFE_CALL(self->RedirectSafeCall(reinterpret_cast<const TVMFFIAny*>(args),
                                                   num_args, reinterpret_cast<TVMFFIAny*>(rv)));
  }

  static int32_t SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    Derived* self = reinterpret_cast<Derived*>(func);
    return self->RedirectSafeCall(args, num_args, rv);
  }
};

/*!
 * \brief FunctionObj specialization that leverages C-style callback definitions.
 */
class ExternCFunctionObjImpl : public FunctionObj,
                               public RedirectCallToSafeCall<ExternCFunctionObjImpl> {
 public:
  using RedirectCallToSafeCall<ExternCFunctionObjImpl>::SafeCall;

  ExternCFunctionObjImpl(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self))
      : self_(self), safe_call_(safe_call), deleter_(deleter) {
    this->call = RedirectCallToSafeCall<ExternCFunctionObjImpl>::Call;
    this->safe_call = RedirectCallToSafeCall<ExternCFunctionObjImpl>::SafeCall;
  }

  ~ExternCFunctionObjImpl() { deleter_(self_); }

  TVM_FFI_INLINE int32_t RedirectSafeCall(const TVMFFIAny* args, int32_t num_args,
                                          TVMFFIAny* rv) const {
    return safe_call_(self_, args, num_args, rv);
  }

 private:
  void* self_;
  TVMFFISafeCallType safe_call_;
  void (*deleter_)(void* self);
};

/*!
 * \brief FunctionObj specialization that wraps an external function.
 */
class ImportedFunctionObjImpl : public FunctionObj,
                                public RedirectCallToSafeCall<ImportedFunctionObjImpl> {
 public:
  using RedirectCallToSafeCall<ImportedFunctionObjImpl>::SafeCall;

  explicit ImportedFunctionObjImpl(ObjectPtr<Object> data) : data_(data) {
    this->call = RedirectCallToSafeCall<ImportedFunctionObjImpl>::Call;
    this->safe_call = RedirectCallToSafeCall<ImportedFunctionObjImpl>::SafeCall;
  }

  TVM_FFI_INLINE int32_t RedirectSafeCall(const TVMFFIAny* args, int32_t num_args,
                                          TVMFFIAny* rv) const {
    FunctionObj* func = const_cast<FunctionObj*>(static_cast<const FunctionObj*>(data_.get()));
    return func->safe_call(func, args, num_args, rv);
  }

 private:
  ObjectPtr<Object> data_;
};

// Helper class to set packed arguments
class PackedArgsSetter {
 public:
  explicit PackedArgsSetter(AnyView* args) : args_(args) {}

  // NOTE: setter needs to be very carefully designed
  // such that we do not have temp variable conversion(eg. convert from lvalue to rvalue)
  // that is why we need T&& and std::forward here
  template <typename T>
  TVM_FFI_INLINE void operator()(size_t i, T&& value) const {
    args_[i].operator=(std::forward<T>(value));
  }

 private:
  AnyView* args_;
};
}  // namespace details

/*!
 * \brief Represents arguments packed in AnyView array
 * \note This class represent packed arguments to ffi::Function
 */
class PackedArgs {
 public:
  /*!
   * \brief Constructor
   * \param data The arguments
   * \param size The number of arguments
   */
  PackedArgs(const AnyView* data, int32_t size) : data_(data), size_(size) {}

  /*! \return size of the arguments */
  int size() const { return size_; }

  /*! \return The arguments */
  const AnyView* data() const { return data_; }

  /*!
   * \brief Slice the arguments
   * \param begin The begin index
   * \param end The end index
   * \return The sliced arguments
   */
  PackedArgs Slice(int begin, int end = -1) const {
    if (end == -1) {
      end = size_;
    }
    return PackedArgs(data_ + begin, end - begin);
  }

  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  AnyView operator[](int i) const { return data_[i]; }

  /*!
   * \brief Fill the arguments into the AnyView array
   * \param data The AnyView array to store the packed arguments
   * \param args The arguments to be packed
   * \note Caller must ensure all args are alive during lifetime of data.
   *       A common pitfall is to pass in local variables that are immediately
   *       destroyed after calling Fill.
   */
  template <typename... Args>
  static void TVM_FFI_INLINE Fill(AnyView* data, Args&&... args) {
    details::for_each(details::PackedArgsSetter(data), std::forward<Args>(args)...);
  }

 private:
  /*! \brief The arguments */
  const AnyView* data_;
  /*! \brief The number of arguments */
  int32_t size_;
};

/*!
 * \brief ffi::Function  is a type-erased function.
 *  The arguments are passed by "packed format" via AnyView
 */
class Function : public ObjectRef {
 public:
  /*! \brief Constructor from null */
  Function(std::nullptr_t) : ObjectRef(nullptr) {}  // NOLINT(*)
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   * \note legacy purpose, should change to Function::FromPacked for mostfuture use.
   */
  template <typename TCallable>
  explicit Function(TCallable packed_call) {
    *this = FromPacked(packed_call);
  }
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   */
  template <typename TCallable>
  static Function FromPacked(TCallable packed_call) {
    static_assert(
        std::is_convertible_v<TCallable, std::function<void(const AnyView*, int32_t, Any*)>> ||
            std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>,
        "tvm::ffi::Function::FromPacked requires input function signature to match packed func "
        "format");
    if constexpr (std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>) {
      auto wrapped_call = [packed_call](const AnyView* args, int32_t num_args,
                                        Any* rv) mutable -> void {
        PackedArgs args_pack(args, num_args);
        packed_call(args_pack, rv);
      };
      return FromPackedInternal(wrapped_call);
    } else {
      return FromPackedInternal(packed_call);
    }
  }
  /*!
   * \brief Import a possibly externally defined function to this dll
   * \param other Function defined in another dynamic library.
   *
   * \note This function will redirect the call to safe_call in other.
   *  It will try to detect if the function is already from the same DLL
   *  and directly return the original function if so.
   *
   * \return The imported function.
   */
  static Function ImportFromExternDLL(Function other) {
    const FunctionObj* other_func = static_cast<const FunctionObj*>(other.get());
    // the other function comes from the same dll, no action needed
    if (other_func->safe_call == &(FunctionObj::SafeCall) ||
        other_func->safe_call == &(details::ImportedFunctionObjImpl::SafeCall) ||
        other_func->safe_call == &(details::ExternCFunctionObjImpl::SafeCall)) {
      return other;
    }
    // the other function coems from a different library
    Function func;
    func.data_ = make_object<details::ImportedFunctionObjImpl>(std::move(other.data_));
    return func;
  }
  /*!
   * \brief Create ffi::Function from a C style callbacks.
   * \param self Resource handle to the function
   * \param safe_call The safe_call definition in C.
   * \param deleter The deleter to release the resource of self.
   * \return The created function.
   */
  static Function FromExternC(void* self, TVMFFISafeCallType safe_call,
                              void (*deleter)(void* self)) {
    // the other function coems from a different library
    Function func;
    func.data_ = make_object<details::ExternCFunctionObjImpl>(self, safe_call, deleter);
    return func;
  }
  /*!
   * \brief Get global function by name
   * \param name The function name
   * \return The global function.
   * \note This function will return std::nullopt if the function is not found.
   */
  static std::optional<Function> GetGlobal(std::string_view name) {
    TVMFFIObjectHandle handle;
    TVMFFIByteArray name_arr{name.data(), name.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionGetGlobal(&name_arr, &handle));
    if (handle != nullptr) {
      return Function(
          details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<Object*>(handle)));
    } else {
      return std::nullopt;
    }
  }

  static std::optional<Function> GetGlobal(const std::string& name) {
    return GetGlobal(std::string_view(name.data(), name.length()));
  }

  static std::optional<Function> GetGlobal(const String& name) {
    return GetGlobal(std::string_view(name.data(), name.length()));
  }

  static std::optional<Function> GetGlobal(const char* name) {
    return GetGlobal(std::string_view(name));
  }
  /*!
   * \brief Get global function by name and throw an error if it is not found.
   * \param name The name of the function
   * \return The global function
   * \note This function will throw an error if the function is not found.
   */
  static Function GetGlobalRequired(std::string_view name) {
    std::optional<Function> res = GetGlobal(name);
    if (!res.has_value()) {
      TVM_FFI_THROW(ValueError) << "Function " << name << " not found";
    }
    return res.value();
  }

  static Function GetGlobalRequired(const std::string& name) {
    return GetGlobalRequired(std::string_view(name.data(), name.length()));
  }

  static Function GetGlobalRequired(const String& name) {
    return GetGlobalRequired(std::string_view(name.data(), name.length()));
  }

  static Function GetGlobalRequired(const char* name) {
    return GetGlobalRequired(std::string_view(name));
  }
  /*!
   * \brief Set global function by name
   * \param name The name of the function
   * \param func The function
   * \param override Whether to override when there is duplication.
   */
  static void SetGlobal(std::string_view name, Function func, bool override = false) {
    TVMFFIByteArray name_arr{name.data(), name.size()};
    TVM_FFI_CHECK_SAFE_CALL(
        TVMFFIFunctionSetGlobal(&name_arr, details::ObjectUnsafe::GetHeader(func.get()), override));
  }
  /*!
   * \brief List all global names
   * \return A vector of all global names
   * \note This function do not depend on Array so core do not have container dep.
   */
  static std::vector<String> ListGlobalNames() {
    Function fname_functor =
        GetGlobalRequired("ffi.FunctionListGlobalNamesFunctor")().cast<Function>();
    std::vector<String> names;
    int len = fname_functor(-1).cast<int>();
    for (int i = 0; i < len; ++i) {
      names.push_back(fname_functor(i).cast<String>());
    }
    return names;
  }
  /**
   * \brief Remove a global function by name
   * \param name The name of the function
   */
  static void RemoveGlobal(const String& name) {
    static Function fremove = GetGlobalRequired("ffi.FunctionRemoveGlobal");
    fremove(name);
  }
  /*!
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   */
  template <typename TCallable>
  static Function FromTyped(TCallable callable) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable](const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
      details::unpack_call<typename FuncInfo::RetType>(
          std::make_index_sequence<FuncInfo::num_args>{}, nullptr, callable, args, num_args, rv);
    };
    return FromPackedInternal(call_packed);
  }
  /*!
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   * \param name optional name attacked to the function.
   */
  template <typename TCallable>
  static Function FromTyped(TCallable callable, std::string name) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable, name](const AnyView* args, int32_t num_args,
                                        Any* rv) mutable -> void {
      details::unpack_call<typename FuncInfo::RetType>(
          std::make_index_sequence<FuncInfo::num_args>{}, &name, callable, args, num_args, rv);
    };
    return FromPackedInternal(call_packed);
  }
  /*!
   * \brief Call function by directly passing in unpacked arguments.
   *
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallFFIFunction(tvm::ffi::Function f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template <typename... Args>
  TVM_FFI_INLINE Any operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    AnyView args_pack[kArraySize];
    PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
    Any result;
    static_cast<FunctionObj*>(data_.get())->CallPacked(args_pack, kNumArgs, &result);
    return result;
  }
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
    static_cast<FunctionObj*>(data_.get())->CallPacked(args, num_args, result);
  }
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param result The return value.
   */
  TVM_FFI_INLINE void CallPacked(PackedArgs args, Any* result) const {
    static_cast<FunctionObj*>(data_.get())->CallPacked(args.data(), args.size(), result);
  }

  /*! \return Whether the packed function is nullptr */
  TVM_FFI_INLINE bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS(Function, ObjectRef, FunctionObj);

  class Registry;

 private:
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   */
  template <typename TCallable>
  static Function FromPackedInternal(TCallable packed_call) {
    using ObjType = typename details::FunctionObjImpl<TCallable>;
    Function func;
    func.data_ = make_object<ObjType>(std::forward<TCallable>(packed_call));
    return func;
  }
};

/*!
 * \brief Please refer to \ref TypedFunctionAnchor "TypedFunction<R(Args..)>"
 */
template <typename FType>
class TypedFunction;

/*!
 * \anchor TypedFunctionAnchor
 * \brief A ffi::Function wrapper to provide typed function signature.
 * It is backed by a ffi::Function internally.
 *
 * TypedFunction enables compile time type checking.
 * TypedFunction works with the runtime system:
 * - It can be passed as an argument of ffi::Function.
 * - It can be assigned to ffi::Any.
 * - It can be directly converted to a type-erased ffi::Function.
 *
 * Developers should prefer TypedFunction over ffi::Function in C++ code
 * as it enables compile time checking.
 * We can construct a TypedFunction from a lambda function
 * with the same signature.
 *
 * \code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedFunction
 *  TypedFunction<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to ffi::Function
 *  ffi::Function packed = ftype;
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template <typename R, typename... Args>
class TypedFunction<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedFunction<R(Args...)>;
  /*! \brief default constructor */
  TypedFunction() {}
  /*! \brief constructor from null */
  TypedFunction(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief constructor from a function
   * \param packed The function
   */
  TypedFunction(Function packed) : packed_(packed) {}  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedFunction<int(int)> ftyped(typed_lambda, "add_one");
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \param name the name of the lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                  FLambda, std::function<R(Args...)>>::value>::type>
  TypedFunction(FLambda typed_lambda, std::string name) {  // NOLINT(*)
    packed_ = Function::FromTyped(typed_lambda, name);
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
   * TypedFunction<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda, typename = typename std::enable_if<std::is_convertible<
                                  FLambda, std::function<R(Args...)>>::value>::type>
  TypedFunction(const FLambda& typed_lambda) {  // NOLINT(*)
    packed_ = Function::FromTyped(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedFunction<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
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
    packed_ = Function::FromTyped(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from ffi::Function.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(Function packed) {
    packed_ = std::move(packed);
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  TVM_FFI_INLINE R operator()(Args... args) const {
    if constexpr (std::is_same_v<R, void>) {
      packed_(std::forward<Args>(args)...);
    } else {
      Any res = packed_(std::forward<Args>(args)...);
      if constexpr (std::is_same_v<R, Any>) {
        return res;
      } else {
        return std::move(res).cast<R>();
      }
    }
  }
  /*!
   * \brief convert to ffi::Function
   * \return the internal ffi::Function
   */
  operator Function() const { return packed(); }
  /*!
   * \return reference the internal ffi::Function
   */
  const Function& packed() const& { return packed_; }
  /*!
   * \return r-value reference the internal ffi::Function
   */
  constexpr Function&& packed() && { return std::move(packed_); }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return packed_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return packed_ != nullptr; }

 private:
  /*! \brief The internal packed function */
  Function packed_;
};

template <typename FType>
inline constexpr bool use_default_type_traits_v<TypedFunction<FType>> = false;

template <typename FType>
struct TypeTraits<TypedFunction<FType>> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFunction;

  static TVM_FFI_INLINE void CopyToAnyView(const TypedFunction<FType>& src, TVMFFIAny* result) {
    TypeTraits<Function>::CopyToAnyView(src.packed(), result);
  }

  static TVM_FFI_INLINE void MoveToAny(TypedFunction<FType> src, TVMFFIAny* result) {
    TypeTraits<Function>::MoveToAny(std::move(src.packed()), result);
  }

  static TVM_FFI_INLINE bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIFunction;
  }

  static TVM_FFI_INLINE TypedFunction<FType> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return TypedFunction<FType>(TypeTraits<Function>::CopyFromAnyViewAfterCheck(src));
  }

  static TVM_FFI_INLINE std::optional<TypedFunction<FType>> TryCastFromAnyView(
      const TVMFFIAny* src) {
    std::optional<Function> opt = TypeTraits<Function>::TryCastFromAnyView(src);
    if (opt.has_value()) {
      return TypedFunction<FType>(*std::move(opt));
    } else {
      return std::nullopt;
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() { return details::FunctionInfo<FType>::Sig(); }
};

/*! \brief Registry for global function */
class Function::Registry {
 public:
  /*! \brief constructor */
  explicit Registry(const char* name) : name_(name) {}

  /*!
   * \brief Set body to be to use the packed convention.
   *
   * \tparam FLambda The signature of the function.
   * \param f The body of the function.
   */
  template <typename FLambda>
  Registry& set_body_packed(FLambda f) {
    return Register(ffi::Function::FromPacked(f));
  }
  /*!
   * \brief set the body of the function to the given function.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * int multiply(int x, int y) {
   *   return x * y;
   * }
   *
   * TVM_FFI_REGISTER_GLOBAL("multiply")
   * .set_body_typed(multiply); // will have type int(int, int)
   *
   * // will have type int(int, int)
   * TVM_FFI_REGISTER_GLOBAL("sub")
   * .set_body_typed([](int a, int b) -> int { return a - b; });
   *
   * \endcode
   *
   * \param f The function to forward to.
   * \tparam FLambda The signature of the function.
   */
  template <typename FLambda>
  Registry& set_body_typed(FLambda f) {
    return Register(Function::FromTyped(f, name_));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // objectRef subclass:
   * struct Example : ObjectRef {
   *    int DoThing(int x);
   * }
   * TVM_FFI_REGISTER_GLOBAL("Example_DoThing")
   * .set_body_method(&Example::DoThing); // will have type int(self, int)
   *
   * // Object subclass:
   * struct Example : Object {
   *    int DoThing(int x);
   * }
   *
   * TVM_FFI_REGISTER_GLOBAL("Example_DoThing")
   * .set_body_method(&Example::DoThing); // will have type int(self, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...)) {
    static_assert(std::is_base_of_v<ObjectRef, T> || std::is_base_of_v<Object, T>,
                  "T must be derived from ObjectRef or Object");
    if constexpr (std::is_base_of_v<ObjectRef, T>) {
      auto fwrap = [f](T target, Args... params) -> R {
        // call method pointer
        return (target.*f)(params...);
      };
      return Register(ffi::Function::FromTyped(fwrap, name_));
    }
    if constexpr (std::is_base_of_v<Object, T>) {
      auto fwrap = [f](const T* target, Args... params) -> R {
        // call method pointer
        return (const_cast<T*>(target)->*f)(params...);
      };
      return Register(ffi::Function::FromTyped(fwrap, name_));
    }
    return *this;
  }

  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...) const) {
    static_assert(std::is_base_of_v<ObjectRef, T> || std::is_base_of_v<Object, T>,
                  "T must be derived from ObjectRef or Object");
    if constexpr (std::is_base_of_v<ObjectRef, T>) {
      auto fwrap = [f](const T target, Args... params) -> R {
        // call method pointer
        return (target.*f)(params...);
      };
      return Register(ffi::Function::FromTyped(fwrap, name_));
    }
    if constexpr (std::is_base_of_v<Object, T>) {
      auto fwrap = [f](const T* target, Args... params) -> R {
        // call method pointer
        return (target->*f)(params...);
      };
      return Register(ffi::Function::FromTyped(fwrap, name_));
    }
    return *this;
  }

 protected:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  Registry& Register(Function f) {
    Function::SetGlobal(name_, f);
    return *this;
  }

  /*! \brief name of the function */
  const char* name_;
};

/*!
 * \brief helper function to get type index from key
 */
inline int32_t TypeKeyToIndex(std::string_view type_key) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  return type_index;
}

#define TVM_FFI_FUNC_REG_VAR_DEF \
  static inline TVM_FFI_ATTRIBUTE_UNUSED ::tvm::ffi::Function::Registry& __##TVMFFIFuncReg

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_FFI_REGISTER_GLOBAL("MyAdd")
 *   .set_body_typed([](int a, int b) {
 *      return a + b;
 *   });
 * \endcode
 */
#define TVM_FFI_REGISTER_GLOBAL(OpName) \
  TVM_FFI_STR_CONCAT(TVM_FFI_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::ffi::Function::Registry(OpName)

/*!
 * \brief Export typed function as a SafeCallType symbol.
 *
 * \param ExportName The symbol name to be exported.
 * \param Function The typed function.
 * \note ExportName and Function must be different,
 *       see code examples below.
 *
 * \sa ffi::TypedFunction
 *
 * \code
 *
 * int AddOne_(int x) {
 *   return x + 1;
 * }
 *
 * // Expose the function as "AddOne"
 * TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne, AddOne_);
 *
 * // Expose the function as "SubOne"
 * TVM_FFI_DLL_EXPORT_TYPED_FUNC(SubOne, [](int x) {
 *   return x - 1;
 * });
 *
 * // The following code will cause compilation error.
 * // Because the same Function and ExportName
 * // TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne_, AddOne_);
 *
 * // The following code is OK, assuming the macro
 * // is in a different namespace from xyz
 * // TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne_, xyz::AddOne_);
 *
 * \endcode
 */
#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                        \
  extern "C" {                                                                     \
  TVM_FFI_DLL_EXPORT int ExportName(void* self, TVMFFIAny* args, int32_t num_args, \
                                    TVMFFIAny* result) {                           \
    TVM_FFI_SAFE_CALL_BEGIN();                                                     \
    using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;        \
    static std::string name = #ExportName;                                         \
    ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(                  \
        std::make_index_sequence<FuncInfo::num_args>{}, &name, Function,           \
        reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,              \
        reinterpret_cast<::tvm::ffi::Any*>(result));                               \
    TVM_FFI_SAFE_CALL_END();                                                       \
  }                                                                                \
  }
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_FUNCTION_H_
