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
 * \brief Mutate dense  conv2d operator to sparse  conv2d operator
 */
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include "../op/nn/convolution_make.h"

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

// Search dense op weight name from Expr
class Conv2dOpWeightVisitor : private ExprVisitor {
 public:
  Conv2dOpWeightVisitor() : conv2d_op_(Op::Get("nn.conv2d")) {}

  Array<String> Search(const Expr& expr) {
    printf("running search");
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
    else {
      std::cout << "is not a nn.conv2d op: " << n->op << std::endl;
    }
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }
  // Cache op
  const Op& conv2d_op_;

  Array<String> memo_;
};  // SearchDenseOpWeight

Array<String> SearchConv2dOpWeight(const Expr& e) { return Conv2dOpWeightVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_conv2d_op_weight").set_body_typed(SearchConv2dOpWeight);

// Mutate ```nn.conv2d``` to ```nn.conv2d_sparse```
class Conv2dToSparseConv2dMutator : public ExprRewriter {
 public:
  Conv2dToSparseConv2dMutator(const Array<ObjectRef>& weight_name,
                                      const Array<Array<PrimExpr> >& weight_shape)
    : conv2d_op_(Op::Get("nn.conv2d")), sparse_dense_op_(Op::Get("nn.conv2d_sparse")) {
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
    if (pre->op == conv2d_op_) {
      const auto weight = pre->args[1].as<VarNode>();
      if (weight) {
        if (target_weights_.count(weight->name_hint())) {
          const auto& prefix = weight->name_hint();
          const auto& ws = target_weights_.at(prefix);
          const auto data = post.as<CallNode>()->args[0];
          auto ws_data_type =
              relay::TensorType({ws.at(0)}, DataType::Float(32));

          auto ws_indices_type = relay::TensorType({ws.at(1)}, DataType::Int(32));
          auto ws_indptr_type = relay::TensorType({ws.at(2)}, DataType::Int(32));
          Var weight_data(prefix + ".data", ws_data_type);
          Var weight_indices(prefix + ".indices", ws_indices_type);
          Var weight_indptr(prefix + ".indptr", ws_indptr_type);


          auto my_attr = std::move(pre->attrs.as<Conv2DAttrs>());
          auto attrs = make_object<Conv2DAttrs>();
          attrs->strides = std::move(my_attr->strides);
          attrs->padding = std::move(my_attr->padding);
          attrs->dilation = std::move(my_attr->dilation);
          attrs->groups = 1;
          attrs->channels = std::move(my_attr->channels);
          attrs->kernel_size = std::move(my_attr->kernel_size);
          attrs->data_layout = std::move(my_attr->data_layout);
          attrs->kernel_layout = std::move(my_attr->kernel_layout);
          attrs->out_layout = std::move(my_attr->out_layout);
          attrs->out_dtype = std::move(my_attr->out_dtype);


          return Call(sparse_dense_op_, {data, weight_data, weight_indices, weight_indptr}, Attrs(attrs), {});

        }
      }
    }
    return post;
  }

 private:
  // Cached op
  const Op& conv2d_op_;
  const Op& sparse_dense_op_;
  std::unordered_map<std::string, std::vector<int> > target_weights_;
};

Expr Conv2dToSparse(const Expr& e, const Array<ObjectRef>& weight_name,
                    const Array<Array<PrimExpr> >& weight_shape) {
  auto rewriter = Conv2dToSparseConv2dMutator(weight_name, weight_shape);
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass Conv2dToSparse(const Array<ObjectRef>& weight_name,
                        const Array<Array<PrimExpr> >& weight_shape) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 = Downcast<Function>(Conv2dToSparse(f, weight_name, weight_shape));

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
