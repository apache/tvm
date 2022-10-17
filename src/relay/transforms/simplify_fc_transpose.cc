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
 * \file simplify_fc_transpose.cc
 *
 * \brief Mutate ```y = nn.dense(x, tranpose(w, [1, 0]))``` to
 *        ```y = nn.dense(x, wt)```
 */
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

// Find name of weight in ```y = nn.dense(x, tranpose(w, [1, 0]))```
class FCTransposeVisitor : private ExprVisitor {
 public:
  FCTransposeVisitor() : dense_op_(Op::Get("nn.dense")), transpose_op_(Op::Get("transpose")) {}

  Array<String> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  void VisitExpr_(const CallNode* n) final {
    if (n->op == dense_op_) {
      const auto weight = n->args[1].as<CallNode>();
      if (weight) {
        if (weight->op == transpose_op_) {
          if (weight->args[0].as<VarNode>()) {
            const auto arg = weight->args[0].as<VarNode>();
            memo_.push_back(arg->name_hint());
          }
        }
      }
    }
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }

  const Op& dense_op_;
  const Op& transpose_op_;
  Array<String> memo_;
};  // SearchDenseOpWeight

Array<String> SearchFCTranspose(const Expr& e) { return FCTransposeVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_fc_transpose").set_body_typed(SearchFCTranspose);

// Mutate ```y = nn.dense(x, tranpose(w, [1, 0]))``` to ```y = nn.dense(x, wt)```
class FCTransposeMutator : public ExprRewriter {
 public:
  explicit FCTransposeMutator(const Array<ObjectRef>& target_weights)
      : dense_op_(Op::Get("nn.dense")), transpose_op_(Op::Get("transpose")) {
    for (size_t i = 0; i < target_weights.size(); ++i) {
      ICHECK(target_weights[i]->IsInstance<runtime::StringObj>());
      std::string k = target_weights[i].as<runtime::StringObj>()->data;
      target_weights_.emplace(k);
    }
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == dense_op_) {
      const auto data = post.as<CallNode>()->args[0];
      const auto weight = pre->args[1].as<CallNode>();
      if (weight) {
        if (weight->op == transpose_op_) {
          const auto arg = weight->args[0];
          if (arg.as<VarNode>()) {
            const auto& arg_node = arg.as<VarNode>();
            ICHECK_GT(target_weights_.count(arg_node->name_hint()), 0);
            const auto& tt = arg_node->type_annotation.as<TensorTypeNode>();
            auto wt_type = TensorType({tt->shape[1], tt->shape[0]}, tt->dtype);
            Var wt(arg_node->name_hint() + ".T", wt_type);
            return Call(dense_op_, {data, wt}, pre->attrs, pre->type_args);
          }
        }
      }
    }
    return post;
  }

 private:
  // Cached op
  const Op& dense_op_;
  const Op& transpose_op_;
  std::unordered_set<std::string> target_weights_;
};  // class DenseToSparseDenseAlter

Expr SimplifyFCTranspose(const Expr& e, const Array<ObjectRef>& target_weights) {
  auto rewriter = FCTransposeMutator(target_weights);
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass SimplifyFCTranspose(const Array<ObjectRef>& target_weights) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warning
        auto f0 = Downcast<Function>(SimplifyFCTranspose(f, target_weights));
        Array<Var> wt_params = FreeVars(f0);
        auto f1 = WithFields(f0, wt_params);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : wt_params) {
          params.push_back(var);
        }
        return WithFields(f1, params);
      };
  return CreateFunctionPass(pass_func, 4, "SimplifyFCTranspose", {"DeadCodeElimination"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyFCTranspose").set_body_typed(SimplifyFCTranspose);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
