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

/*! \file side_effect.cc
 *  \brief Runtime effect properties of shared expressions.
 */
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/op.h>

namespace tvm {
namespace {

class ExprSideEffect : public ExprVisitor {
 public:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
    // Effects describe evaluated expressions, not expressions embedded in types.
    if (value.as<const TypeNode*>()) return std::nullopt;
    return ExprVisitor::Visit(value);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* node) final {
    UpdateEffect(CallEffectKind::kReadState);
    return ExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* node) final {
    static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    if (auto op = node->op.as<Op>()) {
      UpdateEffect(static_cast<CallEffectKind>(effects[*op]));
    } else {
      UpdateEffect(CallEffectKind::kOpaque);
    }
    if (kind == CallEffectKind::kUpdateState) return VisitInterrupt();
    return ExprVisitor::Visit_(node);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* node) final {
    // Lane counts are type metadata rather than evaluated expression children.
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->base));
    return this->Visit(node->stride);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* node) final {
    return this->Visit(node->value);
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* node) final {
    // Preserve indices-first effect/error order, including nonconstant indices.
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->indices));
    return this->Visit(node->vectors);
  }

  CallEffectKind kind{CallEffectKind::kPure};

 private:
  void UpdateEffect(CallEffectKind effect) {
    if (effect > CallEffectKind::kUpdateState) effect = CallEffectKind::kUpdateState;
    if (effect > kind) kind = effect;
  }
};

}  // namespace

CallEffectKind SideEffect(const Expr& expr) {
  ExprSideEffect visitor;
  visitor.Visit(expr);
  return visitor.kind;
}

}  // namespace tvm
