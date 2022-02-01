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
 * \file scalar_to_tensor_constant.cc
 * \brief Converts scalar constant into tensor constant for binary ops of CMSIS-NN
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../op/make_op.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This Mutator finds all partitioned functions meant for CMSIS-NN binary ops.
 * Then, it substitutes the scalar constants with tensor constants. It makes the shape of this
 * new constant same as that of the neighbouring constant of the other binary operand. The
 * expectation is that the ExtractConstant pass would later extract this tensor constant out of the
 * global partitioned function, thus making the entire global partitioned and its composite function
 * constant free. This makes the TIR generation for binary ops via CMSIS-NN independent of
 * constants.
 */
class ScalarToTensorConstantMutator : public MixedModeMutator {
 public:
  explicit ScalarToTensorConstantMutator(const IRModule& mod) : mod_(mod) {}

 private:
  using MixedModeMutator::VisitExpr_;

  // Here is an example with the annotated scalar constant:
  // def @tvmgen_default_cmsis_nn_main_1(%cmsis_nn_input: Tensor[], Inline=1, Compiler="cmsis-nn",
  //                                     global_symbol="tvmgen_default_cmsis_nn_main",
  //                                     Primitive=1) -> Tensor[] {
  //   %56 = fn (%input0: _scalar_constant_, %input1: Tensor[],
  //             PartitionedFromPattern="qnn.mul_", Composite="cmsis-nn.qnn_mul") -> Tensor[] {
  //     qnn.mul(%input0, %input1, scale0, zero_point0,
  //              scale1, zero_point_1, output_scale, output_zero_point)
  //   };
  //   %56(meta[relay.Constant] /* _scalar constant_ */, %cmsis-nn_input)
  // }
  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr final_call = post;
    call = post.as<CallNode>();

    // Create a new variable argument that is of the same shape as the neighbouring argument
    // in the binary op. This needs to be done only when one of the arguments is a scalar.
    if (call->op.as<OpNode>()) {
      final_call = ReplaceScalarWithTensorVariable(GetRef<Call>(call));
    }

    if (auto* glob_var_node = call->op.as<GlobalVarNode>()) {
      GlobalVar global_var = GetRef<GlobalVar>(glob_var_node);
      Function func = Downcast<Function>(mod_->Lookup(global_var));
      auto compiler_name = func->GetAttr<String>(::tvm::relay::attr::kCompiler);
      if (!compiler_name.defined() || compiler_name != "cmsis-nn") {
        return final_call;
      }
      auto new_body = VisitExpr(func->body);
      if (new_body.same_as(func->body)) {
        return final_call;
      }
      Function new_func = WithFields(func, FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
      mod_->Update(global_var, new_func);
      final_call = Call(global_var, call->args);
    }

    // Substitute scalar constant with a tensor constant in the call to composite function
    // comprising partitioned binary ops. Shape of the new constant should be same as its
    // neighbouring tensor's shape.
    if (auto* func_node = call->op.as<FunctionNode>()) {
      Function func = GetRef<Function>(func_node);
      auto func_name = func->GetAttr<String>(attr::kComposite);
      if (func_name.defined() &&
          (func_name == "cmsis-nn.qnn_add" || func_name == "cmsis-nn.qnn_mul")) {
        final_call = ReplaceScalarWithTensorConstant(GetRef<Call>(call), func);
      }
    }

    return final_call;
  }

  // Replaces scalar variable with a tensor variable with same shape as that of the neibouring
  // operand tensor in a binary op
  Call ReplaceScalarWithTensorVariable(Call call) {
    const OpNode* opnode = call->op.as<OpNode>();
    if (opnode == nullptr) {
      return call;
    }
    String op_name = opnode->name;
    Array<Expr> new_args;
    for (uint32_t i = 0; i < call->args.size(); ++i) {
      Expr arg = call->args[i];
      new_args.push_back(arg);
      if (!arg->checked_type_.defined()) {
        continue;
      }
      auto* arg_type = arg->type_as<TensorTypeNode>();
      if (arg_type->shape.size() != 0 || arg.as<ConstantNode>()) {
        continue;
      }
      String arg_name = arg.as<VarNode>()->name_hint();
      int tensor_arg_id = (i + 1) % 2;
      Expr tensor_arg = call->args[tensor_arg_id];
      if (!tensor_arg->checked_type_.defined()) {
        continue;
      }
      TensorType tensor_type = GetRef<TensorType>(tensor_arg->type_as<TensorTypeNode>());
      new_args.Set(i, Var(arg_name, tensor_type));
    }
    return Call(call->op, new_args, call->attrs, {});
  }

  // Makes tensor constant of same shape as tensor_arg with values from scalar_arg
  Call ReplaceScalarWithTensorConstant(Call call, Function func) {
    Array<Expr> new_args;
    for (uint32_t i = 0; i < call->args.size(); ++i) {
      new_args.push_back(call->args[i]);
      Expr scalar_arg = call->args[i];
      if (!scalar_arg->checked_type_.defined()) {
        continue;
      }
      Array<PrimExpr> scalar_shape = scalar_arg->type_as<TensorTypeNode>()->shape;
      if (scalar_shape.size() != 0 || scalar_arg.as<ConstantNode>() == nullptr) {
        continue;
      }
      int tensor_arg_id = (i + 1) % 2;
      Expr tensor_arg = call->args[tensor_arg_id];
      if (!tensor_arg->checked_type_.defined()) {
        continue;
      }
      TensorType tensor_type = GetRef<TensorType>(tensor_arg->type_as<TensorTypeNode>());
      std::vector<int64_t> tensor_shape;
      for (auto& dim : tensor_type->shape) {
        tensor_shape.push_back(qnn::get_const_int(dim));
      }
      int8_t scalar_value = GetScalarFromConstant<int8_t>(scalar_arg);
      int tensor_num_elements = qnn::get_const_int(tensor_type->Size());
      std::vector<int8_t> tensor_values(tensor_num_elements, scalar_value);
      Constant tensor_constant =
          MakeConstantTensor<int8_t>(DataType::Int(8), tensor_shape, tensor_values);
      new_args.Set(i, tensor_constant);
    }
    auto new_body = VisitExpr(func->body);
    Function new_func = WithFields(func, FreeVars(new_body), new_body, func->ret_type,
                                   FreeTypeVars(new_body, mod_), func->attrs);
    return Call(new_func, new_args);
  }

 private:
  IRModule mod_;
};

IRModule ScalarToTensorConstant(const IRModule& mod) {
  auto mutator = ScalarToTensorConstantMutator(mod);
  Function main_func = Downcast<Function>(mod->Lookup("main"));
  auto new_main_body = mutator.VisitExpr(main_func->body);
  if (!new_main_body.same_as(main_func->body)) {
    auto main_var = mod->GetGlobalVar("main");
    auto new_main_func = Function(main_func->params, new_main_body, main_func->ret_type,
                                  main_func->type_params, main_func->attrs);
    mod->Update(main_var, new_main_func);
  }
  return mod;
}

transform::Pass ScalarToTensorConstantPass() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return ScalarToTensorConstant(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "ScalarToTensorConstant", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.ScalarToTensorConstants")
    .set_body_typed(ScalarToTensorConstantPass);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
