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
 * \file side_effect.cc
 * \brief side effect analysis
 */
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tirx {

class ExprSideEffect : public StmtExprVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView e) final {
    if (kind_ == CallEffectKind::kUpdateState) return std::nullopt;
    if (e.as<OpaqueExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(e);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) final {
    // Preserve expression-only traversal: the source need not be a TIRx BufferVar.
    this->UpdateEffect(CallEffectKind::kReadState);
    for (const auto& index : op->indices) {
      if (auto result = Visit(index)) return result;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    static auto op_call_effect = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");

    if (te::IsTensorLoad(ffi::GetRef<Call>(op))) {
      this->UpdateEffect(CallEffectKind::kReadState);
    } else if (auto opt = op->op.as<Op>()) {
      this->UpdateEffect(static_cast<CallEffectKind>(op_call_effect[opt.value()]));
    } else {
      this->UpdateEffect(CallEffectKind::kOpaque);
    }
    return StmtExprVisitor::Visit_(op);
  }

  void UpdateEffect(CallEffectKind effect_kind) {
    if (effect_kind > CallEffectKind::kUpdateState) {
      effect_kind = CallEffectKind::kUpdateState;
    }
    if (effect_kind > kind_) {
      kind_ = effect_kind;
    }
  }

  CallEffectKind kind_{CallEffectKind::kPure};
};

CallEffectKind SideEffect(const PrimExpr& e) {
  auto visitor = ffi::make_object<ExprSideEffect>();
  visitor->Visit(e);
  return visitor->kind_;
}

}  // namespace tirx
}  // namespace tvm
