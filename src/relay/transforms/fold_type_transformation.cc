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
 * \file src/relay/transforms/fold_type_transformation.cc
 * \brief A pass for transforming relay graph function
 * signatures such that when a function-level inputs is
 * transformed by a subsequent cast or quantize operation,
 * that operation is folded into the signature itself.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/qnn/attrs.h>

namespace tvm {
namespace relay {

/*! \brief This class transforms a relay module's function signature
 * such that when a function-level input is transformed by a subsequent
 * "cast" or "qnn.quantize" operation, that operation is folded into
 * the signature itself. For example,
 * 
 * def @main(%data: Tensor[(1, 3, 224, 224), float32]) {
 *   %0 = qnn.quantize(%data, 2f, 0, out_dtype="uint8");
 *   add(%0, %0)
 * }
 * 
 * would be transformed to
 * 
 * def @main(%data: Tensor[(1, 3, 224, 224), uint8]) {
 *   add(%0, %0)
 * }
 * 
 * Note that now it is the user's responsibility to modify their
 * input pre-processing pipeline to satisfy the new signature's
 * constraints. 
 * 
 * For this pass to fold a type transformation, the following conditions
 * must be met:
 *   - The relay module must contain only a single function.
 *   - The type of each function-level input is transformed only once
 *     per program.
 *   - The type transformation operation must be either a "cast"
 *     or "qnn.quantize".
 */
class FoldTypeTransformationRewriter : public MixedModeMutator {
 protected:
  Expr Rewrite_(const CallNode* pre_call_node, const Expr& post) final {
    const CallNode* post_call_node = post.as<CallNode>();
    CHECK(post_call_node) << "Expected a CallNode, but got " << post;

    Expr cur_op = pre_call_node->op;
    for (auto arg : pre_call_node->args) {
      auto maybe_var_node = arg.as<VarNode>();
      if (maybe_var_node) {
        auto var = Downcast<Var>(arg);
        auto it = input_transform_map_.find(var);
        if (it != input_transform_map_.end()) {
          // Checks that the function-level input var hasn't been an arg
          // to a CallNode yet.
          CHECK(!it->second) << "Function input with name '" << var->name_hint()
                             << "' is fed into more than one call; "
                             << "aborting transformation";

          it->second = pre_call_node;

          // Get the type to transform the function signature to
          DataType out_dtype;
          if (cur_op == cast_op_) {
            auto attrs = pre_call_node->attrs.as<CastAttrs>();
            out_dtype = attrs->dtype;
          } else if (cur_op == quantize_op_) {
            auto attrs = pre_call_node->attrs.as<qnn::QuantizeAttrs>();
            out_dtype = attrs->out_dtype;
          } else {
            CHECK(false) << "FoldTypeTransformation will only fold cast and "
                         << "quantize type transformations";
          }

          // Mutate the var node type
          VarNode* var_node = reinterpret_cast<VarNode*>(maybe_var_node);
          const TensorTypeNode* anno = var_node->type_annotation.as<TensorTypeNode>();
          auto mut_anno = reinterpret_cast<TensorTypeNode*>(anno);
          auto shape = anno->shape;
          mut_anno->dtype = out_dtype;

          return GetRef<Expr>(var_node);
        } else {
          LOG(WARNING) << "Variable '" << var->name_hint() << "' encountered"
                       << " but wasn't registered as a function-level input";
        }
      }
    }

    return Call(cur_op, post_call_node->args, pre_call_node->attrs, pre_call_node->type_args,
                pre_call_node->span);
  }

  Expr VisitExpr_(const FunctionNode* node) {
    function_count_++;
    if (function_count_ > 1) {
      CHECK(false) << "FoldTypeTransformation is supported for only single-function graphs";
    }

    for (auto param : node->params) {
      input_transform_map_.insert(std::pair<Var, const CallNode*>(param, NULL));
    }
    auto body = this->Mutate(node->body);

    return Function(node->params, body, node->ret_type, node->type_params, node->attrs, node->span);
  }

  const Op cast_op_ = Op::Get("cast");
  const Op quantize_op_ = Op::Get("qnn.quantize");

 private:
  // Maps function-level input to the first-encountered call node within
  // the function that takes in that input.
  std::map<Var, const CallNode*> input_transform_map_;

  // Tracks number of functions in this program.
  int function_count_;
};

Expr FoldTypeTransformation(const Expr& expr, const IRModule& mod) {
  return FoldTypeTransformationRewriter().Mutate(expr);
}

namespace transform {

Pass FoldTypeTransformation() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldTypeTransformation(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "FoldTypeTransformation", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldTypeTransformation")
    .set_body_typed(FoldTypeTransformation);

}  // namespace transform

}  // namespace relay
}  // namespace tvm

