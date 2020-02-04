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
 * \file src/relay/pass/merge_composite.cc
 * \brief Merges expressions matching patterns into functions marked
 * as 'composite'. This is primarily intended to be used alongside the
 * external codegen infrastructure to support the case where multiple
 * Relay operators map to a single external operator.
 */

#include <tvm/te/operation.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace merge_composite {


class MergeCompositeWrapper : public ExprMutator {
 public:
  explicit MergeCompositeWrapper(const std::string& pattern_name, const Expr& pattern)
    : pattern_name_(pattern_name), pattern_(pattern) {}
  /*!
   * \brief Determine if a pattern and a subgraph have the same call structure.
   * \param pattern the root call node of the pattern.
   * \param root the root call node of the subgraph.
   * \return Whether they match.
   */
  bool MatchPattern(const Call& pattern, const Call& root) {
    if (!pattern->op->IsInstance<OpNode>() || !root->op->IsInstance<OpNode>())
      return false;
    if (pattern->op.as<OpNode>()->name != root->op.as<OpNode>()->name)
      return false;
    if (pattern->args.size() != root->args.size())
      return false;

    unsigned int i = 0;
    for (const auto& arg : pattern->args) {
      if (arg->IsInstance<CallNode>()) {
        if (!root->args[i]->IsInstance<CallNode>())
          return false;
        if (!MatchPattern(Downcast<Call>(arg), Downcast<Call>(root->args[i])))
          return false;
      }
      i++;
    }
    return true;
  }

  Expr ExtractPattern(const Var& pattern, const Expr& root,
          Map<std::string, Array<Expr>>* var_map) {
    if (var_map->find(pattern->name_hint()) == var_map->end()) {
      auto free_var = VarNode::make(pattern->name_hint(), Type());
      var_map->Set(pattern->name_hint(), Array<Expr>({free_var, root}));
      return std::move(free_var);
    } else {
      return (*var_map)[pattern->name_hint()][0];
    }
  }

  Expr ExtractPattern(const Constant& pattern, const Expr& root,
          Map<std::string, Array<Expr>>* var_map) {
    return root;
  }

  Expr ExtractPattern(const Call& pattern, const Call& root,
          Map<std::string, Array<Expr>>* var_map) {
    Expr empty_expr;
    if (!pattern->op->IsInstance<OpNode>() || !root->op->IsInstance<OpNode>())
      return empty_expr;
    if (pattern->op.as<OpNode>()->name != root->op.as<OpNode>()->name)
      return empty_expr;
    if (pattern->args.size() != root->args.size())
      return empty_expr;

    unsigned int i = 0;
    Array<Expr> new_args;
    for (const auto& arg : pattern->args) {
      if (arg->IsInstance<CallNode>()) {
        new_args.push_back(ExtractPattern(Downcast<Call>(arg),
                                          Downcast<Call>(root->args[i]),
                                          var_map));
      }
      if (arg->IsInstance<VarNode>()) {
        new_args.push_back(ExtractPattern(Downcast<Var>(arg),
                                          root->args[i],
                                          var_map));
      }
      if (arg->IsInstance<ConstantNode>()) {
        new_args.push_back(ExtractPattern(Downcast<Constant>(arg),
                                          root->args[i],
                                          var_map));
      }
      i++;
    }

    auto new_call = CallNode::make(root->op, new_args, root->attrs);
    return std::move(new_call);
  }

  Expr VisitExpr_(const CallNode* cn) {
    Call call = GetRef<Call>(cn);
    if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      CHECK(func.defined());
      const auto name_node = FunctionGetAttr(func, attr::kComposite).as<tir::StringImmNode>();
      // don't step into existing composite functions
      if (name_node->value != "") {
        tvm::Array<tvm::relay::Expr> new_args;
        for (const auto& arg : call->args) {
          auto new_e = this->Mutate(arg);
          new_args.push_back(new_e);
        }
        return CallNode::make(call->op, new_args, call->attrs);
      }
    }

    Expr expr = ExprMutator::VisitExpr_(cn);
    call = Downcast<Call>(expr);
    if (!call->op->IsInstance<OpNode>())
      return std::move(call);

    Op op = Downcast<Op>(call->op);
    CHECK(op.defined());
    Call pattern = Downcast<Call>(pattern_);
    if (Downcast<Op>(pattern->op)->name != op->name)
      return std::move(call);

    if (MatchPattern(pattern, call)) {
      Map<std::string, Array<Expr>> args_map;
      auto extract = ExtractPattern(pattern, call, &args_map);
      auto free_vars = FreeVars(extract);
      Function new_func = FunctionNode::make(free_vars, extract,
              call->checked_type_, {}, Attrs());
      new_func = FunctionSetAttr(new_func, attr::kComposite,
                                 tir::StringImmNode::make(pattern_name_));
      new_func = FunctionSetAttr(new_func, attr::kPrimitive,
          tvm::Integer(1));
      Array<Expr> args;
      for (const auto& free_var : free_vars) {
        args.push_back(args_map[free_var->name_hint()][1]);
      }
      auto new_call = CallNode::make(new_func, args);
      return std::move(new_call);
    }

    return std::move(call);
  }

 private:
  std::string pattern_name_;
  Expr pattern_;
};

Expr MergeComposite(const Expr& expr,
    const Array<tir::StringImm>& pattern_names, const Array<Expr>& patterns) {
  CHECK(pattern_names.size() == patterns.size());
  Expr merged_expr = expr;
  // merge the patterns one-by-one in order
  for (size_t i = 0; i < patterns.size(); i++) {
    std::string pattern_name = pattern_names[i]->value;
    Expr pattern = patterns[i];
    merged_expr = MergeCompositeWrapper(pattern_name, pattern).Mutate(merged_expr);
  }
  return merged_expr;
}

}  // namespace merge_composite

namespace transform {

Pass MergeComposite(const tvm::Array<tir::StringImm>& pattern_names,
    const tvm::Array<Expr>& patterns) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::merge_composite::MergeComposite(f, pattern_names, patterns));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "MergeComposite", {});
  return func_pass;
}

TVM_REGISTER_GLOBAL("relay._transform.MergeComposite")
.set_body_typed(MergeComposite);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
