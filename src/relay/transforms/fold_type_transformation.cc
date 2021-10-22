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
 * \brief A pass for taking transforming relay graph function
 * signatures.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/qnn/attrs.h>

#include <tvm/ir/type.h>


namespace tvm {
namespace relay {

/* Description of FoldTypeTransformation
TODO
*/

// class HeaderMutator : public ExprMutator {

// }
using namespace tvm::tir;

class FoldTypeTransformationRewriter : public MixedModeMutator {
  int count = 0;
 protected:
  Expr Rewrite_(const CallNode* pre_call_node, const Expr& post) final {
    const CallNode* post_call_node = post.as<CallNode>();
    CHECK(post_call_node) << "Expected a CallNode, but got " << post;

    // std::cout << "pre call node " << pre_call_node->op << std::endl;
    // std::cout << "pre call node " << pre_call_node->args << std::endl;
    // std::cout << "post expr " << post << std::endl;
    // CHECK(false) << "temp";

    Expr cur_op = post_call_node->op;

    for (auto arg : pre_call_node->args) {
      auto maybe_var_node = arg.as<VarNode>();
      if (maybe_var_node) {
        std::string var_name = maybe_var_node->name_hint();

        std::cout << "num map elements START " << input_transform_map_.size() << std::endl;
        auto var = Downcast<Var>(arg);
        input_transform_map_.insert(std::pair<Var, const CallNode*>(var, pre_call_node));

        auto it = input_transform_map_.find(var);
        if (it != input_transform_map_.end()) {
          // Checks that the function-level input var hasn't been an arg
          // to a CallNode yet.
          CHECK(!it->second) << "input with name '" << var->name_hint() << "' is fed into more than one call, aborting transformation";

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
            CHECK(false) << "FoldTypeTransformation will only fold cast and quantize type transformations for function inputs.";
          }

          // Mutate the var node type
          VarNode* var_node = (VarNode*)maybe_var_node;
          const TensorTypeNode* anno = var_node->type_annotation.as<TensorTypeNode>();
          auto mut_anno = (TensorTypeNode*) anno;
          auto shape = anno->shape;
          mut_anno->dtype = out_dtype;

          // TODO: Instead of mutating the var node in-place, create a new var node.
          // This also requires updating the function signature. Need to store the var node
          // in the input_transform_map_ probably, then update the function once all
          // Rewrite_ calls are complete.

          return GetRef<Expr>(var_node);
        } else {
          std::cout << "Did not find var with name " << var->name_hint() << " in the map" << std::endl;
        }
      }
    }

    return Call(cur_op, post_call_node->args, pre_call_node->attrs, pre_call_node->type_args, pre_call_node->span);
  }


  // Expr VisitExpr_(const CallNode* node) {
  //   // this iterates from the bottom of the program up
  //   Op op = Downcast<Op>(node->op);
  //   std::cout << "op name " << op->name << std::endl;

  //   for (auto arg : pre_call_node->args) {
  //     auto maybe_var_node = arg.as<VarNode>();
  //     if (maybe_var_node) {
  //       std::string var_name = maybe_var_node->name_hint();
  //       auto it = unvisited_input_names_.find(var_name);
  //       if (it != unvisited_input_names_.end()) {
  //         CHECK(cur_op == cast_op_) << "Expected a cast op, but got " << cur_op;

  //         std::cout << "call attrs " << pre_call_node->attrs << std::endl;
  //         auto attrs = pre_call_node->attrs.as<CastAttrs>();
  //         auto dtype = attrs->dtype;

  //         auto this_is_a_thing = DataType::Int(32);

  //         unvisited_input_names_.erase(it);
  //         std::cout << "Removing " << var_name << " from unvisited input names" << std::endl;
  //       }
  //     }
  //   }

  //   Expr expr;
  //   if (op == quantize_op_) {// || op == cast_op_) {
  //     expr = GetRef<Expr>(node);
  //     std::cout << "at a quantize op" << std::endl;
  //     // Get the type input names of the op
  //     auto inputs = node->args;
  //     std::cout << "INPUTS SI<<<<<<<<<<<<<<<<<<<<<z " << inputs.size() << std::endl;
  //     auto expr = inputs[0];

  //     auto tensor_node = expr.as<TensorTypeNode>();
  //     // auto node = expr.as<CallNode>();

  //     std::cout << "node ptr " << tensor_node << std::endl;

  //     expr = ExprMutator::VisitExpr_(node);
  //   } else {
  //     expr = ExprMutator::VisitExpr_(node);
  //   }

  //   // static const Op& op = Op::Get("nn.batch_flatten");
  //   // return Call(oexpr
  // }

  Expr VisitExpr_(const FunctionNode* node) {
    function_count_++;
    if (function_count_ > 1) {
      CHECK(false) << "FoldTypeTransformation is supported for only single-function graphs";
    }

    tvm::Array<TypeVar> ty_params;
    bool all_ty_params_unchanged = true;

    for (auto ty_param : node->type_params) {
      TypeVar new_ty_param = Downcast<TypeVar>(VisitType(ty_param));
      ty_params.push_back(new_ty_param);
      all_ty_params_unchanged &= new_ty_param.same_as(ty_param);

      std::cout << "type param" << ty_param << std::endl;
      std::cout << "all params unchanged " << all_ty_params_unchanged << std::endl;
    }

    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (auto param : node->params) {
      Var new_param = Downcast<Var>(this->Mutate(param));
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
      // std::cout << "param " << param << std::endl;
      std::string name = param->name_hint();
      unvisited_input_names_.insert(name);

      input_transform_map_.insert(std::pair<Var, const CallNode*>(param, NULL));

      std::cout << "all params unchanked " << all_params_unchanged << std::endl;
    }

    auto ret_type = this->VisitType(node->ret_type);
    auto body = this->Mutate(node->body);

    // std::cout << "ret type" << node->ret_type << std::endl;
    // std::cout << "num type params" << params.size() << std::endl;
    // std::cout << "num type params" << node->params.size() << std::endl;

    std::cout << "params unchanged ? " << all_params_unchanged << "  " << all_ty_params_unchanged << std::endl;
      std::cout << "body same? " << body.same_as(node->body) << std::endl;
    if (all_ty_params_unchanged && all_params_unchanged && ret_type.same_as(node->ret_type) &&
        body.same_as(node->body)) {
      return GetRef<Expr>(node);
    } else {
      auto f = Function(params, body, ret_type, ty_params, node->attrs, node->span);
      std::cout << "are we in here" << std::endl;
      return f;
    }
  }

  const Op cast_op_ = Op::Get("cast");
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");

 private:
  // An input name is removed from this set when we visit a call node that
  // references the corresponding input. For this pass, we expect that
  // program-level inputs are only referenced once. 
  std::unordered_set<std::string> unvisited_input_names_;
 
  // Maps function-level input to the first-encountered call node within
  // the function that takes in that input.
  std::map<Var, const CallNode*> input_transform_map_;
  // std::map<Var, std::tuple<const CallNode*, const VarNode*>> input_transform_map_;

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