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
 * \brief Topi utility function
 * \file topi/utils.h
 */
#ifndef TVM_TOPI_UTILS_H_
#define TVM_TOPI_UTILS_H_

#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace topi {

using namespace tvm::runtime;

/*! \brief Canonicalize an argument that may be Array<Expr> or int to Array<Expr> */
inline Optional<Array<Integer>> ArrayOrInt(AnyView arg) {
  if (arg == nullptr) {
    return std::nullopt;
  }
  if (auto opt_int = arg.try_cast<int>()) {
    Array<Integer> result;
    result.push_back(opt_int.value());
    return result;
  } else {
    return arg.cast<Array<Integer>>();
  }
}
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_UTILS_H_
