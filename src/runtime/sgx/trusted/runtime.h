/*!
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <tvm/runtime/packed_func.h>
#include <string>
#include "../common.h"

namespace tvm {
namespace runtime {
namespace sgx {

template<typename... Args>
inline TVMRetValue OCallPackedFunc(std::string name, Args&& ...args) {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  TVMValue ret_val;
  int ret_type_code;
  TVM_SGX_CHECKED_CALL(tvm_ocall_packed_func(name.c_str(),
                                             values,
                                             type_codes,
                                             kNumArgs,
                                             &ret_val,
                                             &ret_type_code));
  TVMRetValue* rv = new TVMRetValue();
  *rv = TVMArgValue(ret_val, ret_type_code);
  return *rv;
}

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
