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
 * Changed by Kappes Johannes @2023
 */

/*!
 *
 * \file conv2d_checksum_extension.cc
 *
 * \brief Extend each conv2d with a checksum generation (Hari et. al.)
 */
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

// Find conv2d operations in dataflowgraph and bring them into an array
class Conv2dVisitor : private ExprVisitor {
 public:
  Conv2dVisitor() : conv2d_op(Op::Get("nn.conv2d")) {}

  Array<ObjectRef> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  bool supportedInputTensorType(const TensorTypeNode* arg){
    return ((arg->dtype == DataType::Int(8))  ||
            (arg->dtype == DataType::Int(16)) ||
            (arg->dtype == DataType::UInt(8)) ||
            (arg->dtype == DataType::UInt(16)));
  }

    bool supportedOutputTensorType(const Conv2DAttrs* arg){
    return ((arg->out_dtype == DataType::Int(32))  ||
            (arg->out_dtype == DataType::Int(64))  ||
            (arg->out_dtype == DataType::UInt(32)) ||
            (arg->out_dtype == DataType::UInt(64)));
  }


  void VisitExpr_(const CallNode* n) final {
    if (n->op == conv2d_op) {
      // detect (u)int8/16 * (u)int8/16 * -> (u)int32/64 conv2d
      auto attr = n->attrs.as<Conv2DAttrs>();
      auto input  = n->args[0]->type_as<TensorTypeNode>();
      auto weight = n->args[1]->type_as<TensorTypeNode>();
      ICHECK(attr);
      ICHECK(input);
      ICHECK(weight);
        if(supportedInputTensorType(input)  && 
           supportedInputTensorType(weight) &&
          (supportedOutputTensorType(attr))){
          memo_.push_back(GetRef<Call>(n));
        }
    }
    // iterate deeper levels
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }

  const Op& conv2d_op;
  Array<ObjectRef> memo_;  // Array for all already existing conv2d operation
};

Array<ObjectRef> SearchConv2d(const Expr& e) { return Conv2dVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_conv2d").set_body_typed(SearchConv2d);

// We dont want to exchange single nodes in the graph => No Mutation

namespace transform {

ObjectPtr<Conv2DAttrs> create_depthwise_conv_attr(const Conv2DAttrs* orig_conv_attr,  PrimExpr P, PrimExpr Q, IntImm C){
  auto depthwise_conv_attr = make_object<Conv2DAttrs>();
  depthwise_conv_attr->strides = orig_conv_attr->strides;
  depthwise_conv_attr->padding = orig_conv_attr->padding;
  depthwise_conv_attr->dilation = orig_conv_attr->dilation;
  depthwise_conv_attr->groups = C->value;
  depthwise_conv_attr->kernel_size = {P, Q};
  depthwise_conv_attr->data_layout = orig_conv_attr->data_layout;
  depthwise_conv_attr->kernel_layout = orig_conv_attr->kernel_layout;
  depthwise_conv_attr->out_layout = orig_conv_attr->out_layout;
  depthwise_conv_attr->out_dtype = orig_conv_attr->out_dtype;
  depthwise_conv_attr->channels = C;
  return depthwise_conv_attr;
}


ObjectPtr<Conv2DAttrs> create_vec_vec_prod_attr(Shape weight_shape){
  auto vec_vec_prod_attr = make_object<Conv2DAttrs>();
  vec_vec_prod_attr->strides = {1,1};
  vec_vec_prod_attr->padding = {0, 0};
  vec_vec_prod_attr->dilation = {1,1};
  vec_vec_prod_attr->groups = 1;
  vec_vec_prod_attr->kernel_size = {weight_shape[2], weight_shape[3]}; //SR
  vec_vec_prod_attr->data_layout   = "NCHW";
  vec_vec_prod_attr->kernel_layout = "OIHW";
  vec_vec_prod_attr->out_layout = "NCHW";
  vec_vec_prod_attr->out_dtype = DataType::Int(64);
  vec_vec_prod_attr->channels = {1};
  return vec_vec_prod_attr;
}





IRModule Extend2DConv(const IRModule& mod) {
  // required for Add function for module
  tvm::Map<GlobalVar, Function> updates;

  auto funcs = mod->functions;  // unorderd_map with global var(function name) and function
  for (const auto& ele : funcs) {
    ICHECK_EQ(FreeVars(ele.second).size(), 0);
    if (const auto* n = ele.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);
      Array<ObjectRef> conv_array = SearchConv2d(func->body);
      auto first_exp = func->body;
      Array<tvm::relay::Expr> output_expr;
      
      // get existing expression tree
      if (func->body.as<Tuple>()) {
        first_exp = Downcast<Tuple>(func->body); 
      } else if (func->body.as<Call>()) {
        first_exp = Downcast<Call>(func->body);
      } else {
        ICHECK_EQ(1, 0) << "func->body should be either Call or Tuple node";
      }
      output_expr.push_back(first_exp);

      //Save ops reference to reduce access time
      const Op& conv2d_op = Op::Get("nn.conv2d");
      const Op& cast_op = Op::Get("cast");
      const Op& sum_op = Op::Get("sum");
      const Op& neq_op = Op::Get("not_equal");

      // Add Checksum calc for each conv2d op in conv_array
      for (const auto& it : conv_array) {
        Call origin_conv2d = Downcast<Call>(it);
        const auto* input_tensor  = origin_conv2d->args[0]->type_as<TensorTypeNode>();
        const auto* weight_tensor = origin_conv2d->args[1]->type_as<TensorTypeNode>();

        ICHECK(input_tensor != nullptr);
        ICHECK(weight_tensor != nullptr);

        const auto input = Downcast<Var>(origin_conv2d->args[0]);
        const auto weight = Downcast<Var>(origin_conv2d->args[1]);

        //Filter Checksum and input fmap checksum(Ic) requires 32-bit precision (Hari et. al.) => Cast each input tensor to 32-bit dtype
        auto cast_attr_32bit = make_object<CastAttrs>();
        cast_attr_32bit->dtype = DataType::Int(32);
        Call input_32bit(cast_op, {input}, Attrs(cast_attr_32bit));
        Call weight_32bit(cast_op, {weight}, Attrs(cast_attr_32bit));
        // Reduced output fmaps require 64 bit precision

        auto cast_attr_64bit = make_object<CastAttrs>();
        cast_attr_64bit->dtype = DataType::Int(64);
        Call origin_conv2d_64bit(cast_op, {origin_conv2d}, Attrs(cast_attr_64bit));


        
        // sum operator has reduction attributes
        auto reduce_axis_attrs = make_object<ReduceAttrs>();
        reduce_axis_attrs->axis = {0};       // look in sry/relay/op/tensor/reduce.cc l.451 for example
        reduce_axis_attrs->keepdims = true;  // 4D -> 4D We want a Conv2D (vec*vec) vec = (C,H,W)
        reduce_axis_attrs->exclude  = false;

        /// Implement depth-wise conv with conv2d Operation (Group arg splits input into C seperate batches)
        const auto* orig_conv_attr = origin_conv2d->attrs.as<Conv2DAttrs>();
        ICHECK(orig_conv_attr != nullptr);
        //bool is_depthwise_conv = IsDepthwiseConv(origin_conv2d,orig_conv_attr,orig_conv_attr->kernel_layout);
        auto weight_shape = weight_tensor->shape; // KCSR
        auto input_shape = input_tensor->shape;   // NCHW

        // calculate output shape for the according Matrix(P,Q and C inferred from input shape)
        //  (Input Size – ((Filter Size – 1)Dilation Factor + 1) + 2Padding)/Stride + 1
        PrimExpr P = input_shape[2] - weight_shape[2] + 1;
        PrimExpr Q = input_shape[3] - weight_shape[3] + 1;
        IntImm C = Downcast<IntImm>(input_shape[1]); //need to do this as no conv frim PrimExpr to int exists ffs
        PrimExpr One{1};



        auto depthwise_conv_attr = create_depthwise_conv_attr(orig_conv_attr, P, Q, C);
        auto depthwise_kernel = Ones({C, One, P, Q}, DataType::Int(8));

        Call depthwise_conv(conv2d_op, {input, depthwise_kernel}, Attrs{depthwise_conv_attr});
        Call batchwise_sum_input(sum_op, {depthwise_conv}, Attrs(reduce_axis_attrs));  // use batch as axis for depthwise sum
        
        Call filterwise_sum_input(sum_op, {weight_32bit},  Attrs(reduce_axis_attrs));  // add each element of indiv filter
  

        // Simple Vector-Vector dot product of 3D Tensors (Checksum dot product)
        auto vec_vec_prod_attr = create_vec_vec_prod_attr(weight_shape);
        Call vec_vec_prod(conv2d_op, {batchwise_sum_input, filterwise_sum_input}, Attrs(vec_vec_prod_attr));
        
        
        //elemwise sum operation
        auto reduce_elemwise_attrs = make_object<ReduceAttrs>();
        reduce_elemwise_attrs->axis = {0,1,2,3};  // look in sry/relay/op/tensor/reduce.cc l.451 for example
        reduce_elemwise_attrs->keepdims = false;  // 4D -> 1D
        reduce_elemwise_attrs->exclude  = false;

        //Reduce 4D Tensor into 64-bit scalar 
        Call output_checksum(sum_op, {origin_conv2d_64bit}, Attrs(reduce_elemwise_attrs));
        Call vec_vec_prod_right_dim(sum_op, {vec_vec_prod}, Attrs(reduce_elemwise_attrs));

        //Final comparision, which is then one additional output in the output tuple
        Call comp(neq_op, {vec_vec_prod_right_dim, output_checksum});

        output_expr.push_back(comp);
      }
      Tuple new_func_body(output_expr);
      Array<Type> return_array = {func->ret_type};
      //first elem==original element
      TensorType comp_output({}, DataType::Bool(1)); //boolean type has dim=0
      for(uint i=1; i < output_expr.size(); i++){
        return_array.push_back(comp_output);
      }
      TupleType final_ret_type(return_array);
      Function extended_func(func->params, new_func_body, final_ret_type, func->type_params);
      
      updates.Set(ele.first, Downcast<Function>(extended_func));

      VLOG(1) << "Print out all conv2d which need a treatment:" << std::endl
              << PrettyPrint(conv_array) << std::endl
              << "Print out return type of new function:" << std::endl
              << PrettyPrint(func->ret_type) << std::endl
              << "and the function:" << std::endl
              << PrettyPrint(extended_func) << std::endl;
    }
    // Use implemented function to update each global var/ func pair
    for (auto pair : updates) {
      mod->Add(pair.first, pair.second, true);
    }
  }
  return mod;
}



Pass Extend2DConv() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [&](IRModule m, PassContext pc) { return Extend2DConv(m); };

  return CreateModulePass(pass_func, 0, "Extend2DConv", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.Extend2DConv").set_body_typed([](){return Extend2DConv();});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
