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
 * \file topi/util.h
 */
#ifndef TOPI_UTIL_H_
#define TOPI_UTIL_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/packed_func.h>

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

/*! \brief Canonicalize an argument that may be Array<Expr> or int to Array<Expr> */
inline Array<Integer> ArrayOrInt(TVMArgValue arg) {
  if (arg.type_code() == kDLInt || arg.type_code() == kDLUInt) {
    Array<Integer> result;
    result.push_back(arg.operator int());
    return result;
  } else {
    return arg;
  }
}
}  // namespace topi
#endif  // TOPI_UTIL_H_
