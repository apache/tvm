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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>

#include <algorithm>

namespace tvm {

CallEffectKind SideEffect(const Expr& expr) {
  static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  CallEffectKind kind = CallEffectKind::kPure;
  ffi::StructuralVisit(
      expr,
      [&](const CallNode* node,
          ffi::StructuralVisitorObj* visitor) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        auto effect = static_cast<CallEffectKind>(
            effects.get(node->op, static_cast<TCallEffectKind>(CallEffectKind::kOpaque)));
        kind = std::max(kind, std::min(effect, CallEffectKind::kUpdateState));
        if (kind == CallEffectKind::kUpdateState) return ffi::VisitInterrupt();
        return visitor->DefaultVisitExpected(node);
      },
      [&](const TensorLoadNode* node,
          ffi::StructuralVisitorObj* visitor) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        kind = std::max(kind, CallEffectKind::kReadState);
        return visitor->DefaultVisitExpected(node);
      },
      [](const TypeNode*,
         ffi::StructuralVisitorObj*) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        // Effects describe evaluated expressions, not expressions embedded in types.
        return std::nullopt;
      });
  return kind;
}

}  // namespace tvm
