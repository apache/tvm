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
 * \file tvm/ffi/function_details.h
 * \brief Implements the funciton signature reflection
 */
#ifndef TVM_FFI_FUNCTION_DETAILS_H_
#define TVM_FFI_FUNCTION_DETAILS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <string>
#include <tuple>
#include <utility>

namespace tvm {
namespace ffi {
namespace details {

template <typename ArgType>
struct Arg2Str {
  template <size_t i>
  static TVM_FFI_INLINE void Apply(std::ostream& os) {
    using Arg = std::tuple_element_t<i, ArgType>;
    if constexpr (i != 0) {
      os << ", ";
    }
    os << i << ": " << Type2Str<Arg>::v();
  }
  template <size_t... I>
  static TVM_FFI_INLINE void Run(std::ostream& os, std::index_sequence<I...>) {
    using TExpander = int[];
    (void)TExpander{0, (Apply<I>(os), 0)...};
  }
};

template <typename T>
static constexpr bool ArgSupported =
    (std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, Any> ||
     std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, AnyView> ||
     TypeTraitsNoCR<T>::enabled);

// NOTE: return type can only support non-reference managed returns
template <typename T>
static constexpr bool RetSupported =
    (std::is_same_v<T, Any> || std::is_void_v<T> || TypeTraits<T>::enabled);

template <typename R, typename... Args>
struct FuncFunctorImpl {
  using FType = R(Args...);
  using ArgType = std::tuple<Args...>;
  using RetType = R;
  /*! \brief total number of arguments*/
  static constexpr size_t num_args = sizeof...(Args);
  /*! \brief Whether this function can be converted to ffi::Function via FromUnpacked */
  static constexpr bool unpacked_supported = (ArgSupported<Args> && ...) && (RetSupported<R>);

  static TVM_FFI_INLINE std::string Sig() {
    using IdxSeq = std::make_index_sequence<sizeof...(Args)>;
    std::ostringstream ss;
    ss << "(";
    Arg2Str<std::tuple<Args...>>::Run(ss, IdxSeq{});
    ss << ") -> " << Type2Str<R>::v();
    return ss.str();
  }
};

template <typename T>
struct FunctionInfoHelper;

template <typename T, typename R, typename... Args>
struct FunctionInfoHelper<R (T::*)(Args...)> : FuncFunctorImpl<R, Args...> {};
template <typename T, typename R, typename... Args>
struct FunctionInfoHelper<R (T::*)(Args...) const> : FuncFunctorImpl<R, Args...> {};

/*!
 * \brief Template class to get function signature of a function or functor.
 * \tparam T The function/functor type.
 * \note We need a decltype redirection because this helps lambda types.
 */
template <typename T>
struct FunctionInfo : FunctionInfoHelper<decltype(&T::operator())> {};

template <typename R, typename... Args>
struct FunctionInfo<R(Args...)> : FuncFunctorImpl<R, Args...> {};
template <typename R, typename... Args>
struct FunctionInfo<R (*)(Args...)> : FuncFunctorImpl<R, Args...> {};

/*! \brief Using static function to output TypedPackedFunc signature */
typedef std::string (*FGetFuncSignature)();

template <typename T>
TVM_FFI_INLINE Optional<T> as(AnyView arg) {
  return arg.as<T>();
}
template <>
TVM_FFI_INLINE Optional<Any> as<Any>(AnyView arg) {
  return Any(arg);
}
template <>
TVM_FFI_INLINE Optional<AnyView> as<AnyView>(AnyView arg) {
  return arg;
}

template <typename T>
TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* source) {
  return TypeTraits<T>::GetMismatchTypeInfo(source);
}

template <>
TVM_FFI_INLINE std::string GetMismatchTypeInfo<Any>(const TVMFFIAny* source) {
  return TypeIndexToTypeKey(source->type_index);
}

template <>
TVM_FFI_INLINE std::string GetMismatchTypeInfo<AnyView>(const TVMFFIAny* source) {
  return TypeIndexToTypeKey(source->type_index);
}

/*!
 * \brief Auxilary argument value with context for error reporting
 */
class ArgValueWithContext {
 public:
  /*!
   * \brief move constructor from another return value.
   * \param args The argument list
   * \param arg_index In a function call, this argument is at index arg_index (0-indexed).
   * \param optional_name Name of the function being called. Can be nullptr if the function is not.
   * \param f_sig Pointer to static function outputting signature of the function being called.
   * named.
   */
  TVM_FFI_INLINE ArgValueWithContext(const AnyView* args, int32_t arg_index,
                                     const std::string* optional_name, FGetFuncSignature f_sig)
      : args_(args), arg_index_(arg_index), optional_name_(optional_name), f_sig_(f_sig) {}

  template <typename Type>
  TVM_FFI_INLINE operator Type() {
    using TypeWithoutCR = std::remove_const_t<std::remove_reference_t<Type>>;
    Optional<TypeWithoutCR> opt = as<TypeWithoutCR>(args_[arg_index_]);
    if (!opt.has_value()) {
      TVMFFIAny any_data = args_[arg_index_].CopyToTVMFFIAny();
      TVM_FFI_THROW(TypeError) << "Mismatched type on argument #" << arg_index_
                               << " when calling: `"
                               << (optional_name_ == nullptr ? "" : *optional_name_)
                               << (f_sig_ == nullptr ? "" : (*f_sig_)()) << "`. Expected `"
                               << Type2Str<TypeWithoutCR>::v() << "` but got `"
                               << GetMismatchTypeInfo<TypeWithoutCR>(&any_data) << '`';
    }
    return *std::move(opt);
  }

 private:
  const AnyView* args_;
  int32_t arg_index_;
  const std::string* optional_name_;
  FGetFuncSignature f_sig_;
};

template <typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template <typename... Args>
  TVM_FFI_INLINE static void run(const std::string* optional_name, FGetFuncSignature f_sig,
                                 const F& f, const AnyView* args, int32_t num_args, Any* rv,
                                 Args&&... unpacked_args) {
    // construct a movable argument value
    // which allows potential move of argument to the input of F.
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>::run(
        optional_name, f_sig, f, args, num_args, rv, std::forward<Args>(unpacked_args)...,
        ArgValueWithContext(args, index, optional_name, f_sig));
  }
};

template <typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template <typename... Args>
  TVM_FFI_INLINE static void run(const std::string*, FGetFuncSignature, const F& f, const AnyView*,
                                 int32_t, Any* rv, Args&&... unpacked_args) {
    using RetType = decltype(f(std::forward<Args>(unpacked_args)...));
    if constexpr (std::is_same_v<RetType, R>) {
      *rv = f(std::forward<Args>(unpacked_args)...);
    } else {
      *rv = R(f(std::forward<Args>(unpacked_args)...));
    }
  }
};

template <int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template <typename... Args>
  TVM_FFI_INLINE static void run(const std::string*, FGetFuncSignature, const F& f, const AnyView*,
                                 int32_t, Any*, Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename F>
TVM_FFI_INLINE void unpack_call(const std::string* optional_name, const F& f, const AnyView* args,
                                int32_t num_args, Any* rv) {
  using FuncInfo = FunctionInfo<F>;
  FGetFuncSignature f_sig = FuncInfo::Sig;
  static_assert(FuncInfo::unpacked_supported,
                "The function signature cannot support unpacked call");
  if (nargs != num_args) {
    TVM_FFI_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                             << (optional_name == nullptr ? "" : *optional_name)
                             << (f_sig == nullptr ? "" : (*f_sig)()) << "`. Expected " << nargs
                             << " but got " << num_args << " arguments";
  }
  unpack_call_dispatcher<R, nargs, 0, F>::run(optional_name, f_sig, f, args, num_args, rv);
}

template <typename FType>
struct unpack_call_by_signature {};

template <typename R, typename... Args>
struct unpack_call_by_signature<R(Args...)> {
  template <typename F>
  TVM_FFI_INLINE static void run(const F& f, const AnyView* args, int32_t num_args, Any* rv) {
    unpack_call<R, sizeof...(Args)>(nullptr, f, args, num_args, rv);
  }
};

template <typename R>
struct typed_packed_call_dispatcher {
  template <typename F, typename... Args>
  TVM_FFI_INLINE static R run(const F& f, Args&&... args) {
    return f(std::forward<Args>(args)...);
  }
};

template <>
struct typed_packed_call_dispatcher<void> {
  template <typename F, typename... Args>
  TVM_FFI_INLINE static void run(const F& f, Args&&... args) {
    f(std::forward<Args>(args)...);
  }
};
}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_FUNCTION_DETAILS_H_
