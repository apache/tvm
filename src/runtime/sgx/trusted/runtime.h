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
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <tvm/runtime/packed_func.h>
#include <string>
#include <utility>
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
