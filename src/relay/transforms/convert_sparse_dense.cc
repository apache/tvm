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
 * \file convert_sparse_dense.cc
 *
 * \brief Mutate dense operator to sparse dense operator
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

// Search dense op weight name from Expr
class DenseOpWeightVisitor : private ExprVisitor {
 public:
  DenseOpWeightVisitor() : dense_op_(Op::Get("nn.dense")) {}

  Array<String> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  void VisitExpr_(const CallNode* n) final {
    if (n->op == dense_op_) {
      const auto weight = n->args[1].as<VarNode>();
      if (weight) {
        memo_.push_back(weight->name_hint());
      }
    }
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }
  // Cache op
  const Op& dense_op_;

  Array<String> memo_;
};  // SearchDenseOpWeight

Array<String> SearchDenseOpWeight(const Expr& e) { return DenseOpWeightVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_dense_op_weight").set_body_typed(SearchDenseOpWeight);

// Mutate ```nn.dense``` to ```nn.sparse_dense```
class DenseToSparseDenseMutator : public ExprRewriter {
 public:
  DenseToSparseDenseMutator(const Array<ObjectRef>& weight_name,
                            const Array<Array<PrimExpr> >& weight_shape)
      : dense_op_(Op::Get("nn.dense")), sparse_dense_op_(Op::Get("nn.sparse_dense")) {
    CHECK_EQ(weight_name.size(), weight_shape.size());
    for (size_t i = 0; i < weight_name.size(); ++i) {
      CHECK(weight_name[i]->IsInstance<runtime::StringObj>());
      std::string k = weight_name[i].as<runtime::StringObj>()->data;
      const auto& ws = weight_shape[i];
      std::vector<int> v(ws.size());
      for (size_t j = 0; j < ws.size(); ++j) {
        v[j] = ws[j].as<IntImmNode>()->value;
      }
      target_weights_.emplace(k, v);
    }
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == dense_op_) {
      const auto weight = pre->args[1].as<VarNode>();
      if (weight) {
        if (target_weights_.count(weight->name_hint())) {
          const auto& prefix = weight->name_hint();
          const auto& ws = target_weights_.at(prefix);
          const auto data = post.as<CallNode>()->args[0];
          auto ws_data_type =
              relay::TensorType({ws.at(0), ws.at(1), ws.at(2)}, DataType::Float(32));
          auto ws_indices_type = relay::TensorType({ws.at(3)}, DataType::Int(32));
          auto ws_indptr_type = relay::TensorType({ws.at(4)}, DataType::Int(32));
          Var weight_data(prefix + ".data", ws_data_type);
          Var weight_indices(prefix + ".indices", ws_indices_type);
          Var weight_indptr(prefix + ".indptr", ws_indptr_type);

          return Call(sparse_dense_op_, {data, weight_data, weight_indices, weight_indptr});
        }
      }
    }
    return post;
  }

 private:
  // Cached op
  const Op& dense_op_;
  const Op& sparse_dense_op_;
  std::unordered_map<std::string, std::vector<int> > target_weights_;
};  // class DenseToSparseDenseAlter

Expr DenseToSparse(const Expr& e, const Array<ObjectRef>& weight_name,
                   const Array<Array<PrimExpr> >& weight_shape) {
  auto rewriter = DenseToSparseDenseMutator(weight_name, weight_shape);
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass DenseToSparse(const Array<ObjectRef>& weight_name,
                   const Array<Array<PrimExpr> >& weight_shape) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 = Downcast<Function>(DenseToSparse(f, weight_name, weight_shape));
        Array<Var> sparse_params = FreeVars(f0);
        auto f1 = Function(sparse_params,
                        f0->body,
                        f0->ret_type,
                        f0->type_params,
                        f0->attrs);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : sparse_params) {
          params.push_back(var);
        }
        return Function(params,
                        f1->body,
                        f1->ret_type,
                        f1->type_params,
                        f1->attrs);
      };
  return CreateFunctionPass(pass_func, 4, "DenseToSparse", {"DeadCodeElimination"});
}

TVM_REGISTER_GLOBAL("relay._transform.DenseToSparse").set_body_typed(DenseToSparse);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
