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
 * \file ir_utils.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_TRANSFORM_IR_UTILS_H_
#define TVM_TIR_TRANSFORM_IR_UTILS_H_

#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/with_context.h>
#include <tvm/runtime/device_api.h>
#include <tvm/sym/int_set.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Get the type that is passed around TVM ffi::Function API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline PrimType APIType(const PrimType& t) {
  TVM_FFI_ICHECK(!t.IsVoid()) << "Cannot pass void type through packed API.";
  TVM_FFI_ICHECK_EQ(t.lanes(), 1) << "Cannot pass vector type through packed API.";
  if (t.MatchesCode(DLDataTypeCode::kDLBool, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)) {
    return PrimType::Int(64);
  }
  TVM_FFI_ICHECK(t.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat));
  return PrimType::Float(64);
}

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(Stmt stmt);

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_IR_UTILS_H_
