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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/eliminate_common_subexpr.cc
 * \brief Eliminrate common subexpression pass.
 *
 * Currently it removes common subexpressions within a DataflowBlock.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

/*!
 * \brief Check if two expressions are equal scalars.
 * \param a The expression to be checked.
 * \param b The expression to be checked
 * \return Whether two expressions are equal scalars.
 */
static bool IsEqualScalar(const Expr& a, const Expr& b) {
  const auto* constant_a = a.as<ConstantNode>();
  const auto* constant_b = b.as<ConstantNode>();
  if (!constant_a || !constant_b || !constant_a->is_scalar() || !constant_b->is_scalar()) {
    return false;
  }
  return tvm::StructuralEqual()(a, b);
}

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(runtime::TypedPackedFunc<bool(Expr)> fskip) : fskip_(fskip) {}

 private:
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    auto post = VisitExpr(GetRef<Call>(call_node));
    auto new_val = Rewrite_(call_node, post);
    return ExprMutator::VisitBinding_(binding, new_val.as<CallNode>());
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    auto post = VisitExpr(GetRef<Tuple>(val));
    return ExprMutator::VisitBinding_(binding, val);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    auto post = VisitExpr(GetRef<TupleGetItem>(val));
    auto new_val = Rewrite_(val, post);
    return ExprMutator::VisitBinding_(binding, new_val.as<TupleGetItemNode>());
  }

 private:
  Expr Rewrite_(const CallNode* call, const Expr& post) {
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    ICHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    StructuralEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr) {
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

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) {
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
  std::unordered_map<Expr, std::vector<Expr>, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  runtime::TypedPackedFunc<bool(Expr)> fskip_;
};

DataflowBlock EliminateCommonSubexpr(const DataflowBlock& df_block, PackedFunc fskip) {
  CommonSubexprEliminator mutator(fskip);
  return Downcast<DataflowBlock>(mutator.VisitBindingBlock(df_block));
}

namespace transform {

Pass EliminateCommonSubexpr(runtime::TypedPackedFunc<bool(Expr)> fskip) {
  runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock df_block, IRModule m, PassContext pc) {
        return Downcast<DataflowBlock>(EliminateCommonSubexpr(df_block, fskip));
      };
  return CreateDataflowBlockPass(pass_func, 1, "EliminateCommonSubexpr", {});
}

TVM_REGISTER_GLOBAL("relax.transform.EliminateCommonSubexpr")
    .set_body_typed(EliminateCommonSubexpr);

}  // namespace transform

}  // namespace relax
}  // namespace tvm