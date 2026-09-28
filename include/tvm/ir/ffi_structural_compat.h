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

/*! \file ffi_structural_compat.h
 * \brief Keep TVM's raw structural hooks compatible with typed tvm-ffi traversal.
 */
#ifndef TVM_IR_FFI_STRUCTURAL_COMPAT_H_
#define TVM_IR_FFI_STRUCTURAL_COMPAT_H_

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace structural_compat {

template <typename T>
class ExpectedReturn {
 public:
  explicit ExpectedReturn(ffi::Expected<T>&& value) : value_(std::move(value)) {}

  operator TVMFFIAny() && noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(value_));
  }

  operator ffi::Expected<T>() && noexcept { return std::move(value_); }

 private:
  ffi::Expected<T> value_;
};

class ErrorReturn {
 public:
  explicit ErrorReturn(ffi::Unexpected<ffi::Error>&& value) : value_(std::move(value)) {}

  operator TVMFFIAny() && noexcept {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        ffi::Expected<ffi::Any>(std::move(value_)));
  }

  template <typename T>
  operator ffi::Expected<T>() && noexcept {
    return std::move(value_);
  }

 private:
  ffi::Unexpected<ffi::Error> value_;
};

template <typename T>
auto VisitReturn(T&& result) {
  if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                               ffi::Optional<ffi::VisitInterrupt>>) {
    return std::forward<T>(result);
  } else {
    return ExpectedReturn(std::forward<T>(result));
  }
}

}  // namespace structural_compat
}  // namespace tvm

// tvm-ffi's macros now return Expected directly. TVM still has raw ABI hooks,
// so select the representation from the enclosing hook's declared return type.
#undef TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN
#define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Result)                                \
  do {                                                                            \
    auto&& tvm_ffi_res_ = (Result);                                               \
    if (TVM_FFI_PREDICT_FALSE(                                                    \
            ::tvm::ffi::details::StructuralVisitNeedEarlyReturn(tvm_ffi_res_))) { \
      return ::tvm::structural_compat::VisitReturn(::std::move(tvm_ffi_res_));    \
    }                                                                             \
  } while (0)

#undef TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN
#define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                   \
  do {                                                                \
    auto&& tvm_ffi_res_ = (Result);                                   \
    if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {               \
      return ::tvm::structural_compat::ErrorReturn(                   \
          ::tvm::ffi::Unexpected(::std::move(tvm_ffi_res_).error())); \
    }                                                                 \
  } while (0)

#undef TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)               \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                        \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                \
  if constexpr (!::tvm::ffi::type_subsumes_v<::tvm::ffi::Expected<Type>, decltype(Result)>) { \
    if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>(          \
            ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))) {                         \
      return ::tvm::structural_compat::ErrorReturn(                                           \
          ::tvm::ffi::details::SMutateDeclaredTypeError());                                   \
    }                                                                                         \
  }                                                                                           \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                        \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                            \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))

#endif  // TVM_IR_FFI_STRUCTURAL_COMPAT_H_
