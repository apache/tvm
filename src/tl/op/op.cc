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
 * \file tl/op/op.cc
 *
 * Define operators usd in tile library.
 */

#include "op.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;

TIR_REGISTER_TL_OP(RegionOp, region)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

std::unique_ptr<Operator> ParseOperator(Call call, BufferMap vmap) {
  auto op_map = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  Op op = call->op.as<Op>().value();
  if (op_map.count(op)) {
    Operator* ptr = static_cast<Operator*>(op_map[op](call->args, vmap));
    ICHECK(ptr != nullptr);
    return std::unique_ptr<Operator>(ptr);
  }
  return nullptr;
}

std::unique_ptr<Operator> ParseOperator(Stmt stmt, BufferMap vmap) {
  if (stmt.as<Evaluate>() && stmt.as<EvaluateNode>()->value.as<CallNode>()) {
    auto call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    return ParseOperator(GetRef<Call>(call), vmap);
  }
  return nullptr;
}

Var GetVarFromAccessPtr(const PrimExpr& expr) {
  auto call = expr.as<CallNode>();
  ICHECK(call);
  ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
  auto var = call->args[1].as<VarNode>();
  ICHECK(var);
  return GetRef<Var>(var);
}

RegionOp::RegionOp(Array<PrimExpr> args, BufferMap vmap) {
  size_t n = args.size();
  size_t ndim = n - 2;
  auto load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim);
  buffer_ = load->buffer;
  access_mask_ = static_cast<int>(*as_const_int(args[1]));
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr min = load->indices[i];
    PrimExpr extent = args[2 + i];
    ranges_.push_back(Range::FromMinExtent(min, extent));
  }
}

bool RegionOp::IsFullRegion() const {
  for (size_t i = 0; i < ranges_.size(); i++) {
    if (!is_zero(ranges_[i]->min)) return false;
    if (!StructuralEqual()(ranges_[i]->extent, buffer_->shape[i])) return false;
  }
  return true;
}

Stmt Operator::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  ICHECK(0) << "Not Implemented Lower method.";
  return Evaluate(0);
}

Stmt Operator::Canonialize(const CanonializeArgs& T, arith::Analyzer* analyzer) const { return {}; }

LayoutMap Operator::InferLayout(const LayoutInferArgs& T, InferLevel level) { return {}; }

}  // namespace tl
}  // namespace tvm
