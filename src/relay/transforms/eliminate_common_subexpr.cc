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
 *
 * \file eliminate_common_subexpr.cc
 * \brief Combine common subexpressions.
 *
 * This is an optimization pass that eliminates common subexpressions. During the pass, it tries
 * to replace an expression with a previously appeared expression with the same input and
 * attributes. The fskip callback argument allows us to skip specific expressions.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <unordered_map>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(runtime::TypedPackedFunc<bool(Expr)> fskip): fskip_(fskip) {}

  Expr VisitExpr_(const CallNode* call) final {
    static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");
    Expr new_expr = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_expr.as<CallNode>();
    CHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    StructuralEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr || op_stateful.get(GetRef<Op>(op), false)) {
      return new_expr;
    }
    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_call->op);
    if (it != expr_map_.end()) {
      for (const CallNode* candidate : it->second) {
        bool is_equivalent = true;
        if (!attrs_equal(new_call->attrs, candidate->attrs)) {
          continue;
        }
        for (size_t i = 0; i < new_call->args.size(); i++) {
          if (!new_call->args[i].same_as(candidate->args[i]) &&
              !IsEqualScalar(new_call->args[i], candidate->args[i])) {
            is_equivalent = false;
            break;
          }
        }
        if (!is_equivalent) continue;
        return GetRef<Call>(candidate);
      }
    }
    expr_map_[new_call->op].push_back(new_call);
    return new_expr;
  }

  std::unordered_map<Expr, std::vector<const CallNode*>, ObjectHash, ObjectEqual> expr_map_;
  runtime::TypedPackedFunc<bool(Expr)> fskip_;
};

Expr EliminateCommonSubexpr(const Expr& expr, PackedFunc callback) {
  return CommonSubexprEliminator(callback)(expr);
}

namespace transform {

Pass EliminateCommonSubexpr(PackedFunc fskip) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(EliminateCommonSubexpr(f, fskip));
  };
  return CreateFunctionPass(pass_func, 3, "EliminateCommonSubexpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.EliminateCommonSubexpr")
.set_body_typed(EliminateCommonSubexpr);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
