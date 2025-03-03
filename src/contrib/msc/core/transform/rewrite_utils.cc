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
 * \file src/contrib/msc/core/transform/rewrite_utils.cc
 */
#include "rewrite_utils.h"

#include <set>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

Var RewriteUtils::ReEmit(BlockBuilder builder, const String& name, const Expr& expr) {
  expr->span = SpanUtils::SetAttr(expr->span, msc_attr::kName, name);
  return builder->Emit(expr, name);
}

Var RewriteUtils::MakeCall(BlockBuilder builder, const String& name, Expr op, Array<Expr> args,
                           Attrs attrs) {
  const auto& call = Call(op, args, attrs);
  return ReEmit(builder, name, call);
}

Expr RewriteUtils::MakeConstant(BlockBuilder builder, const String& name, double value,
                                const DataType& dtype, size_t ndim) {
  const auto& data = support::FloatImmToNDArray(FloatImm(dtype, value));
  Span span = SpanUtils::CreateWithAttr(msc_attr::kName, name);
  const auto& constant = Constant(data, NullOpt, span);
  if (ndim == 0) {
    return constant;
  }
  static const Op& reshape_op = Op::Get("relax.reshape");
  Array<PrimExpr> exp_shape(ndim, Integer(1));
  return MakeCall(builder, name + "_exp", reshape_op, {constant, ShapeExpr(exp_shape)});
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
