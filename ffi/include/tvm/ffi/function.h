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

#include <string>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Object container class that backs ffi::Function
 * \note Do not use this function directly, use ffi::Function
 */
class FunctionObj : public Object {
 public:
  typedef void (*FCall)(const FunctionObj*, int32_t, const AnyView*, Any*);
  /*! \brief A C++ style call implementation */
  FCall call;
  /*! \brief A C API compatible call with exception catching. */
  TVMFFISafeCallType safe_call;

  TVM_FFI_INLINE void CallPacked(int32_t num_args, const AnyView* args, Any* result) const {
    this->call(this, num_args, args, result);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIFunc;
  static constexpr const char* _type_key = "object.Function";

  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(FunctionObj, Object);

 protected:
  /*! \brief Make default constructor protected. */
  FunctionObj() {}

  // Implementing safe call style
  static int32_t SafeCall(void* func, int32_t num_args, const TVMFFIAny* args, TVMFFIAny* result) {
    FunctionObj* self = static_cast<FunctionObj*>(func);
    try {
      self->call(self, num_args, reinterpret_cast<const AnyView*>(args),
                 reinterpret_cast<Any*>(result));
      return 0;
    } catch (const tvm::ffi::Error& err) {
      Any(std::move(err)).MoveToTVMFFIAny(result);
      return 1;
    } catch (const std::runtime_error& err) {
      Any(tvm::ffi::Error("RuntimeError", err.what(), "")).MoveToTVMFFIAny(result);
      return 1;
    }
    TVM_FFI_UNREACHABLE();
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
   * \brief Derived object class for constructing PackedFuncObj.
   * \param callable The type-erased callable object.
   */
  explicit FunctionObjImpl(TCallable callable) : callable_(callable) {
    this->call = Call;
    this->safe_call = SafeCall;
  }

 private:
  // implementation of call
  static void Call(const FunctionObj* func, int32_t num_args, const AnyView* args, Any* result) {
    (static_cast<const TSelf*>(func))->callable_(num_args, args, result);
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
  static void Call(const FunctionObj* func, int32_t num_args, const AnyView* args, Any* rv) {
    Derived* self = static_cast<Derived*>(const_cast<FunctionObj*>(func));
    int ret_code = self->RedirectSafeCall(num_args, reinterpret_cast<const TVMFFIAny*>(args),
                                          reinterpret_cast<TVMFFIAny*>(rv));
    if (ret_code != 0) {
      if (std::optional<tvm::ffi::Error> err = rv->TryAs<tvm::ffi::Error>()) {
        throw std::move(*err);
      } else {
        TVM_FFI_THROW(RuntimeError) << "Error encountered when calling a tvm::ffi::Function";
      }
    }
  }

  static int32_t SafeCall(void* func, int32_t num_args, const TVMFFIAny* args, TVMFFIAny* rv) {
    Derived* self = reinterpret_cast<Derived*>(func);
    return self->RedirectSafeCall(num_args, args, rv);
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

  TVM_FFI_INLINE int32_t RedirectSafeCall(int32_t num_args, const TVMFFIAny* args,
                                          TVMFFIAny* rv) const {
    return safe_call_(self_, num_args, args, rv);
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

  TVM_FFI_INLINE int32_t RedirectSafeCall(int32_t num_args, const TVMFFIAny* args,
                                          TVMFFIAny* rv) const {
    FunctionObj* func = const_cast<FunctionObj*>(static_cast<const FunctionObj*>(data_.get()));
    return func->safe_call(func, num_args, args, rv);
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
 * \brief ffi::Function  is a type-erased function.
 *  The arguments are passed by packed format.
 */
class Function : public ObjectRef {
 public:
  /*! \brief Constructor from null */
  Function(std::nullptr_t) : ObjectRef(nullptr) {}  // NOLINT(*)
                                                    /*!
                                                     * \brief Constructing a packed function from a callable type
                                                     *        whose signature is consistent with `PackedFunc`
                                                     * \param packed_call The packed function signature
                                                     */
  template <typename TCallable>
  static Function FromPacked(TCallable packed_call) {
    static_assert(
        std::is_convertible_v<TCallable, std::function<void(int32_t, const AnyView*, Any*)>>,
        "tvm::ffi::Function::FromPacked requires input function signature to match packed func "
        "format");
    using ObjType = typename details::FunctionObjImpl<TCallable>;
    Function func;
    func.data_ = make_object<ObjType>(std::forward<TCallable>(packed_call));
    return func;
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
    if (other_func->safe_call == FunctionObj::SafeCall ||
        other_func->safe_call == details::ImportedFunctionObjImpl::SafeCall ||
        other_func->safe_call == details::ExternCFunctionObjImpl::SafeCall) {
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
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   */
  template <typename TCallable>
  static Function FromUnpacked(TCallable callable) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable](int32_t num_args, const AnyView* args, Any* rv) -> void {
      details::unpack_call<typename FuncInfo::RetType, FuncInfo::num_args>(nullptr, callable,
                                                                           num_args, args, rv);
    };
    return FromPacked(call_packed);
  }
  /*!
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   * \param name optional name attacked to the function.
   */
  template <typename TCallable>
  static Function FromUnpacked(TCallable callable, std::string name) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable, name](int32_t num_args, const AnyView* args, Any* rv) -> void {
      details::unpack_call<typename FuncInfo::RetType, FuncInfo::num_args>(&name, callable,
                                                                           num_args, args, rv);
    };
    return FromPacked(call_packed);
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
    details::for_each(details::PackedArgsSetter(args_pack), std::forward<Args>(args)...);
    Any result;
    static_cast<FunctionObj*>(data_.get())->CallPacked(kNumArgs, args_pack, &result);
    return result;
  }
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  TVM_FFI_INLINE void CallPacked(int32_t num_args, const AnyView* args, Any* result) const {
    static_cast<FunctionObj*>(data_.get())->CallPacked(num_args, args, result);
  }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS(Function, ObjectRef, FunctionObj);
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_FUNCTION_H_
