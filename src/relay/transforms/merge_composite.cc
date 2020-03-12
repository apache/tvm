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
 * \file src/relay/transforms/merge_composite.cc
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

  Expr ExtractPattern(const Var& pattern, const Expr& root,
          Map<std::string, Array<Expr>>* var_map) {
    if (var_map->find(pattern->name_hint()) == var_map->end()) {
      // if we haven't encountered this var yet, make a new free var and associate
      // it with the value at 'root'
      auto free_var = VarNode::make(pattern->name_hint(), Type());
      var_map->Set(pattern->name_hint(), Array<Expr>({free_var, root}));
      return std::move(free_var);
    } else {
      // if we have encountered this var already, return the free var that was created
      auto vars = (*var_map)[pattern->name_hint()];
      auto free_var = vars[0];
      auto graph_expr = vars[1];
      // make sure to first check they both map to the same node in the graph
      if (graph_expr != root) {
        return Expr();
      }
      return (*var_map)[pattern->name_hint()][0];
    }
  }

  Expr ExtractPattern(const Constant& pattern, const Expr& root,
          Map<std::string, Array<Expr>>* var_map) {
    return root;
  }

  /*!
   * \brief Try and extract a given pattern from a graph as a subgraph.
   * \param pattern The pattern to extract.
   * \param root The graph to extract from.
   * \param var_map A map between free vars in the subgraph and nodes in the graph.
   * \return The extracted subgraph.
   *
   * \note How does this work?
   *
   * A pattern consists of Relay expression containing only operator call nodes, constants
   * and free variables. The free variables indicate where the pattern can 'attach' in your
   * graph. This function takes the final call node of the pattern and the call node currently
   * being traversed in the Relay graph. It traverses through the pattern in lockstep with call node
   * from the graph (referred to as the 'root' node here) to check they're identical. If at any point
   * they differ, an empty expression is returned to signify the extract failed. If a free var is
   * reached in the pattern, the corresponding value in the root is associated with the name of the
   * free var (via the var_map) so that when we construct the composite function, the inputs match
   * up correctly with the rest of the graph. The return value of this function when successful is
   * a new Relay expression ready to be wrapped into a composite function.
   */
  Expr ExtractPattern(const Call& pattern, const Call& root,
          Map<std::string, Array<Expr>>* var_map, Map<Expr, Expr>* call_map) {
    // check to make sure both calls are to operators (not functions)
    if (!pattern->op->IsInstance<OpNode>() || !root->op->IsInstance<OpNode>())
      return Expr();
    if (pattern->op.as<OpNode>()->name != root->op.as<OpNode>()->name)
      return Expr();

    unsigned int i = 0;
    Array<Expr> new_args;
    for (const auto& arg : pattern->args) {
      Expr new_arg;
      if (arg->IsInstance<CallNode>()) {
        // if we've already processed this call node, return the previous result
        if (call_map->find(arg) != call_map->end()) {
          new_arg = (*call_map)[arg];
        } else {
          // fail if the root argument is not also a call node
          if (!root->args[i]->IsInstance<CallNode>()) {
            return Expr();
          }
          // if it's a call node, recursively call this function
          new_arg = ExtractPattern(Downcast<Call>(arg),
                                  Downcast<Call>(root->args[i]),
                                  var_map, call_map);
          call_map->Set(arg, new_arg);
        }
      } else if (arg->IsInstance<VarNode>()) {
        // if there's a var in the pattern, it must be a free var
        // so call the function to update the var_map
        new_arg = ExtractPattern(Downcast<Var>(arg),
                                 root->args[i],
                                 var_map);
      } else if (arg->IsInstance<ConstantNode>()) {
        // if there's a constant, simply get the corresponding
        // value of the constant from the root
        new_arg = ExtractPattern(Downcast<Constant>(arg),
                                 root->args[i],
                                 var_map);
      }
      if (!new_arg.defined()) {
        return Expr();
      }
      new_args.push_back(new_arg);
      i++;
    }
    return CallNode::make(root->op, new_args, root->attrs);
  }

  Expr VisitExpr_(const CallNode* cn) {
    Call call = GetRef<Call>(cn);
    if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      CHECK(func.defined());
      const auto name_node =
          func->GetAttr<tir::StringImm>(attr::kComposite);
      // don't step into existing composite functions
      if (name_node.defined() && name_node->value != "") {
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

    // only call patterns are supported
    Call pattern = Downcast<Call>(pattern_);
    CHECK(pattern.defined());
    Map<std::string, Array<Expr>> args_map;
    Map<Expr, Expr> call_map;
    auto extract = ExtractPattern(pattern, call, &args_map, &call_map);
    if (extract.defined()) {
      auto free_vars = FreeVars(extract);
      // make the composite function
      auto f = Function(free_vars, extract, call->checked_type_, {}, DictAttrs());
      f = WithAttr(std::move(f), attr::kComposite, tir::StringImmNode::make(pattern_name_));
      // find the expressions associated with the free vars using the args_map
      // this tells us which expressions should be given as inputs to the composite function
      Array<Expr> args;
      for (const auto& free_var : free_vars) {
        args.push_back(args_map[free_var->name_hint()][1]);
      }
      auto new_call = CallNode::make(f, args);
      return std::move(new_call);
    }
    return std::move(call);
  }

 private:
  /*! \brief The name of the pattern to match */
  std::string pattern_name_;
  /*! \brief The pattern to match */
  Expr pattern_;
};

Expr MergeComposite(const Expr& expr,
    const Array<tir::StringImm>& pattern_names, const Array<Expr>& patterns) {
  CHECK_EQ(pattern_names.size(), patterns.size());
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
