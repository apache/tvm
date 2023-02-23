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
 * \file tvm/relax/op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef TVM_RELAX_OP_ATTR_TYPES_H_
#define TVM_RELAX_OP_ATTR_TYPES_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief Infer output struct info given the call
 *
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 */
using FInferStructInfo =
    runtime::TypedPackedFunc<StructInfo(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Packed function implementation for operators. The relax operator will be lowered to
 * this packed function call during codegen.
 */
using FCallPacked = String;

/*!
 * \brief The function type of a legalization function, which takes a
 * BlockBuilder and the Call to be legalized, and outputs the legalization
 * result Expr.
 * \param bb The BlockBuilder context.
 * \param call The call to be legalized.
 */
using FLegalize = runtime::TypedPackedFunc<Expr(const BlockBuilder& bb, const Call& call)>;

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_
