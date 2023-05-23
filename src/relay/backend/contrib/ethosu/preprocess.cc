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
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../op/make_op.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*!
 * \brief This expression rewriter will traverse the graph to find calls
 * to all external functions. If they have multiple inputs and/or
 * multiple outputs, the following has to be done :
 * 1) If multiple inputs are present, they needed to be concat before the call.
 * 2) Inside the external function they need to be split again to their original inputs.
 * 3) If there are multiple outputs, they need to be concat at the end of external function.
 * 4) Then, the concat output again need to be split and made the original tuple output in the
 * main.
 */
class ExternalFuncIOHandler : public ExprRewriter {
 public:
  explicit ExternalFuncIOHandler(const IRModule& module) : module_(module) {}
  int count = 0;

  Function InferType(const Function& expr, const IRModule& m) {
    IRModule mod(m);
    mod->Update(mod->GetGlobalVar("main"), expr);
    mod = transform::InferType()(mod);
    return Downcast<Function>(mod->Lookup("main"));
  }

  /*!
   * \brief This function will take shape and compute
   * the scalar size value for it to be use to create
   * flat single dimensional tensors.
   */
  int64_t CalcSize(const Array<Integer>& shape) {
    int size = 1;
    for (auto dim_size : shape) {
      size = size * Downcast<Integer>(dim_size)->value;
    }
    return size;
  }

  /*!
   * \brief This will take a tensor and create a flattened
   * tensor to be used by the concat.
   */
  Expr CreateFlattenTensor(const Expr& input) {
    auto ishape = Downcast<Array<Integer>>(Downcast<TensorType>(input->checked_type())->shape);
    int flatten_size = CalcSize(ishape);
    Array<Integer> output_shape = {Integer(flatten_size)};
    return MakeReshape(input, output_shape);
  }

  /*!
   * \brief This will take flattened tensors and create
   * a single concat'd tensor.
   */
  Expr CreateConcatTensor(const Array<Expr>& inputs) {
    auto tuple = Tuple(inputs);
    return MakeConcatenate(tuple, 0);
  }

  /*!
   * \brief This will take a flattened concat'd tensor and use the original inputs shapes
   * to recreate a Tuple of the original set of tensors.
   */
  Expr CreateSplitReshapedTensors(const Expr& input, const Array<Expr>& original_args) {
    Array<Array<Integer>> shapes;
    Array<Integer> flatten_tensor_sizes;
    Array<IndexExpr> split_indices;
    Array<Expr> rets;

    int total_size = 0;
    for (auto orig_arg : original_args) {
      auto shape = Downcast<Array<Integer>>(Downcast<TensorType>(orig_arg->checked_type())->shape);
      shapes.push_back(shape);
      flatten_tensor_sizes.push_back(CalcSize(shape));
      if (total_size != 0) {
        split_indices.push_back(total_size);
      }
      total_size += CalcSize(shape);
    }
    auto split_outs = MakeSplit(input, split_indices, 0);
    for (unsigned int i = 0; i < shapes.size(); i++) {
      auto split_out = TupleGetItem(split_outs, i);
      split_out->checked_type_ = original_args[i]->checked_type_;
      rets.push_back(MakeReshape(split_out, shapes[i]));
    }
    return Tuple(rets);
  }

  /*!
   * \brief Modify the external function to split the input as the original compute
   * as required originally. Moreover, the outputs will be flattened and concat'd
   * to make a single output. Finaly, the external function should only have a single input
   * and a single output.
   */
  Function ModifyExternalFunction(const Function& func, const GlobalVar& gv,
                                  const DataType& dtype) {
    Array<Expr> inputs;
    Var ifms;
    if (func->params.size() > 1) {
      Array<Array<Integer>> shapes;
      Array<Integer> flatten_tensor_sizes;
      Array<IndexExpr> split_indices;

      auto func_name = gv->name_hint;
      int total_size = 0;
      for (auto input : func->params) {
        auto shape = Downcast<Array<Integer>>(Downcast<TensorType>(input->checked_type())->shape);
        shapes.push_back(shape);
        auto flat_size = CalcSize(shape);
        flatten_tensor_sizes.push_back(flat_size);
        if (total_size != 0) {
          split_indices.push_back(total_size);
        }
        total_size += flat_size;
      }
      Array<PrimExpr> ifms_shape = {total_size};
      ifms = Var(func_name + "_ifms", TensorType(ifms_shape, dtype));
      auto split_outs = MakeSplit(ifms, split_indices, 0);
      for (unsigned int i = 0; i < shapes.size(); i++) {
        auto split_out = TupleGetItem(split_outs, i);
        split_out->checked_type_ = func->params[i]->checked_type();
        inputs.push_back(MakeReshape(split_out, shapes[i]));
      }
    } else {
      CHECK_EQ(func->params.size(), 1);
      inputs.push_back(func->params[0]);
      ifms = func->params[0];
    }
    Map<Var, Expr> bind_map;
    CHECK_EQ(func->params.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      bind_map.Set(func->params[i], inputs[i]);
    }
    auto core_compute_expr = Bind(func->body, bind_map);

    // Creation of wrapper inside the external function
    Array<Var> params = {ifms};
    if (func->body->IsInstance<TupleNode>()) {
      auto tuple_out = func->body.as<TupleNode>();
      Array<Expr> reshaped_outputs;
      for (unsigned int i = 0; i < tuple_out->fields.size(); i++) {
        auto out = Downcast<Tuple>(core_compute_expr)->fields[i];
        out->checked_type_ = tuple_out->fields[i]->checked_type_;
        reshaped_outputs.push_back(CreateFlattenTensor(out));
      }
      auto concat_out = CreateConcatTensor(reshaped_outputs);
      auto f = Function(params, concat_out, concat_out->checked_type_, {}, func->attrs);
      return InferType(f, this->module_);
    } else {
      auto f =
          Function(params, core_compute_expr, core_compute_expr->checked_type_, {}, func->attrs);
      return InferType(f, this->module_);
    }
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    auto post_call = Downcast<Call>(post);

    if (auto optional_glb_var = post_call->op.as<GlobalVar>()) {
      auto glb_var = optional_glb_var.value();
      auto func = Downcast<Function>(module_->functions[glb_var]);

      // If the number of inputs and output are 1 --> no need to do anything
      if (post_call->args.size() == 1 && !func->body->IsInstance<TupleNode>()) {
        return post;
      }
      if (auto compiler = func->GetAttr<String>(attr::kCompiler)) {
        if (compiler == "ethos-u") {
          auto ext_input = std::move(post_call->args[0]);
          auto arg_dtype = Downcast<TensorType>(post_call->args[0]->checked_type())->dtype;
          if (post_call->args.size() > 1) {
            Array<Expr> reshaped_inputs;
            for (const auto& arg : post_call->args) {
              // All arguments should be of same data type
              CHECK_EQ(arg_dtype, Downcast<TensorType>(arg->checked_type())->dtype)
                  << "Currently NPU external functions require all inputs to be of same data "
                     "type";
              reshaped_inputs.push_back(CreateFlattenTensor(arg));
            }
            ext_input = CreateConcatTensor(reshaped_inputs);
          }
          auto ext_func = ModifyExternalFunction(func, glb_var, arg_dtype);
          Array<Expr> new_args = {ext_input};
          module_->Add(glb_var, ext_func);
          Expr new_call = Call(glb_var, new_args);
          if (func->body->IsInstance<TupleNode>()) {
            auto orginal_tuple_out = Downcast<Tuple>(func->body);
            new_call = CreateSplitReshapedTensors(new_call, orginal_tuple_out->fields);
          }
          return std::move(new_call);
        }
      }
    }
    return post;
  }

 private:
  IRModule module_;
};

IRModule PreprocessExternalFuncIO_(const IRModule& module) {
  ExternalFuncIOHandler ex_func_io_handle(module);
  auto func = Downcast<Function>(module->Lookup("main"));
  auto preprocessed = PostOrderRewrite(func, &ex_func_io_handle);
  module->Update(module->GetGlobalVar("main"), Downcast<Function>(preprocessed));
  return module;
}

}  // namespace ethosu
}  // namespace contrib

/*!
 * \brief This is a pre-processing pass for all NPU external functions.
 * Currently, the NPU runtime module expects a single input and a single output.
 * Therefore, this pass will concat the inputs pre-call, split again inside ext. func,
 * concat the output inside ext. func and re-split again after the call.
 */

namespace transform {
Pass PreprocessExternalFuncIO() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pre_processed_ext_func =
      [=](IRModule m, PassContext pc) {
        auto _m = contrib::ethosu::PreprocessExternalFuncIO_(m);
        return _m;
      };
  auto preprocess_pass =
      CreateModulePass(pre_processed_ext_func, 0, "PreprocessExternalFuncIO", {});
  return Sequential({preprocess_pass, InferType()});
}

TVM_REGISTER_GLOBAL("relay.ext.ethos-u.PreprocessExternalFuncIO")
    .set_body_typed(transform::PreprocessExternalFuncIO);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
