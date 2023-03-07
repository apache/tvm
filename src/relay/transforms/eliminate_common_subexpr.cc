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

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class CommonSubexprEliminator : public MixedModeMutator {
 public:
  explicit CommonSubexprEliminator(runtime::TypedPackedFunc<bool(Expr)> fskip) : fskip_(fskip) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    ICHECK(new_call);
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
      for (const Expr& candidate_expr : it->second) {
        if (const CallNode* candidate = candidate_expr.as<CallNode>()) {
          bool is_equivalent = true;
          if (!attrs_equal(new_call->attrs, candidate->attrs)) {
            continue;
          }
          for (size_t i = 0; i < new_call->args.size(); i++) {
            if (!IsEquivalent(new_call->args[i], candidate->args[i])) {
              is_equivalent = false;
              break;
            }
          }
          if (!is_equivalent) continue;
          return GetRef<Call>(candidate);
        }
      }
    }
    expr_map_[new_call->op].push_back(new_expr);
    return new_expr;
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) final {
    Expr new_expr = post;
    const TupleGetItemNode* new_tuple_item = new_expr.as<TupleGetItemNode>();
    ICHECK(new_tuple_item);

    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_tuple_item->tuple);
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const TupleGetItemNode* candidate = candidate_expr.as<TupleGetItemNode>()) {
          if (new_tuple_item->index == candidate->index) {
            return GetRef<Expr>(candidate);
          }
        }
      }
    }
    expr_map_[new_tuple_item->tuple].push_back(new_expr);
    return new_expr;
  }

  std::unordered_map<Expr, std::vector<Expr>, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  runtime::TypedPackedFunc<bool(Expr)> fskip_;

 private:
  bool IsEquivalent(const Expr& arg, const Expr& candidate_arg) {
    if (arg->IsInstance<TupleNode>() && candidate_arg->IsInstance<TupleNode>()) {
      const TupleNode* arg_node = arg.as<TupleNode>();
      const TupleNode* candidate_arg_node = candidate_arg.as<TupleNode>();

      if (arg_node->fields.size() != candidate_arg_node->fields.size()) {
        return false;
      }

      for (size_t i = 0; i < arg_node->fields.size(); i++) {
        if (!arg_node->fields[i].same_as(candidate_arg_node->fields[i]) &&
            !IsEqualScalar(arg_node->fields[i], candidate_arg_node->fields[i])) {
          return false;
        }
      }
    } else {
      if (!arg.same_as(candidate_arg) && !IsEqualScalar(arg, candidate_arg)) {
        return false;
      }
    }

    return true;
  }
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
