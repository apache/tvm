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
 * \file convert_sparse_conv2d.cc
 *
 * \brief Mutate conv2d operator to sparse conv2d operator
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

// Search conv2d op weight name from Expr
class Conv2dOpWeightVisitor : private ExprVisitor {
 public:
  Conv2dOpWeightVisitor() : conv2d_op_(Op::Get("nn.conv2d")) {}

  Array<String> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  void VisitExpr_(const CallNode* n) final {
    if (n->op == conv2d_op_) {
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
  const Op& conv2d_op_;

  Array<String> memo_;
};  // SearchConv2dOpWeight

Array<String> SearchConv2dOpWeight(const Expr& e) { return Conv2dOpWeightVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_conv2d_op_weight").set_body_typed(SearchConv2dOpWeight);

// Mutate ```nn.conv2d``` to ```nn.sparse_conv2d```
class Conv2dToSparseConv2dMutator : public ExprRewriter {
 public:
  Conv2dToSparseConv2dMutator(const Array<ObjectRef>& weight_name,
                              const Array<Array<PrimExpr>>& weight_shape, const String& layout)
      : conv2d_op_(Op::Get("nn.conv2d")), sparse_conv2d_op_(Op::Get("nn.sparse_conv2d")) {
    ICHECK_EQ(weight_name.size(), weight_shape.size());
    layout_ = layout;
    for (size_t i = 0; i < weight_name.size(); ++i) {
      ICHECK(weight_name[i]->IsInstance<runtime::StringObj>());
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
    if (pre->op == conv2d_op_) {
      const auto weight = pre->args[1].as<VarNode>();
      if (weight) {
        if (target_weights_.count(weight->name_hint())) {
          const auto& prefix = weight->name_hint();
          const auto& ws = target_weights_.at(prefix);
          const auto data = post.as<CallNode>()->args[0];
          relay::TensorType ws_data_type, ws_indices_type, ws_indptr_type;
          if (ws.size() == 5) {
            ws_data_type = relay::TensorType({ws.at(0), ws.at(1), ws.at(2)}, DataType::Float(32));
            ws_indices_type = relay::TensorType({ws.at(3)}, DataType::Int(32));
            ws_indptr_type = relay::TensorType({ws.at(4)}, DataType::Int(32));
          } else if (ws.size() == 4) {
            ws_data_type = relay::TensorType({ws.at(0), ws.at(1)}, DataType::Float(32));
            ws_indices_type = relay::TensorType({ws.at(2)}, DataType::Int(32));
            ws_indptr_type = relay::TensorType({ws.at(3)}, DataType::Int(32));
          }
          Var weight_data(prefix + ".data", ws_data_type);
          Var weight_indices(prefix + ".indices", ws_indices_type);
          Var weight_indptr(prefix + ".indptr", ws_indptr_type);
          auto attrs = make_object<SparseConv2DAttrs>();
          attrs->layout = std::move(layout_);
          return Call(sparse_conv2d_op_, {data, weight_data, weight_indices, weight_indptr},
                      Attrs(attrs));
        }
      }
    }
    return post;
  }

 private:
  // Cached op
  const Op& conv2d_op_;
  const Op& sparse_conv2d_op_;
  std::unordered_map<std::string, std::vector<int>> target_weights_;
  String layout_;
};  // class Conv2dToSparseConv2dAlter

Expr Conv2dToSparse(const Expr& e, const Array<ObjectRef>& weight_name,
                    const Array<Array<PrimExpr>>& weight_shape, const String& layout) {
  auto rewriter = Conv2dToSparseConv2dMutator(weight_name, weight_shape, layout);
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass Conv2dToSparse(const Array<ObjectRef>& weight_name, const Array<Array<PrimExpr>>& weight_shape,
                    const String& layout) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 = Downcast<Function>(Conv2dToSparse(f, weight_name, weight_shape, layout));
        Array<Var> sparse_params = FreeVars(f0);
        auto f1 = Function(sparse_params, f0->body, f0->ret_type, f0->type_params, f0->attrs);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : sparse_params) {
          params.push_back(var);
        }
        return Function(params, f1->body, f1->ret_type, f1->type_params, f1->attrs);
      };
  return CreateFunctionPass(pass_func, 4, "Conv2dToSparse", {"DeadCodeElimination"});
}

TVM_REGISTER_GLOBAL("relay._transform.Conv2dToSparse").set_body_typed(Conv2dToSparse);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
