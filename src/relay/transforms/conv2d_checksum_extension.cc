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
#include <tvm/tir/data_layout.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <tuple>

#include "pattern_utils.h"


/* Description of Conv2d_checksum_extension
 *
 * The purpose of this pass is to find quantized conv2D operators, which use only integers as in/output. These Operators are then extended 
 * with multiple Checksum Calculations after Hari. et. Al.(FIC/conv2D) and Uzair. et. Al.(FICwd/depthwise_conv2D) to find soft fault during 
 * convolution and return each Checksum result for each Conv2D operator
 *
 *  x       w
 *   \     /
 *    \   /
 *    conv2D
 *   depth_conv2d
 *      |
 *      |
 *      y
 *
 * and convert them into subgraphs with actual integer operations on x and w
 *
 * The pass does this via a 2 level approach:
 *
 * The recursive search function fills an array with quantized "normal" Conv2D and depthwise_conv2d relay nodes.
 *
 * After that, a new Call Node is attached to the the top level of the function for each found operator, forming a Top level output tuple.
 * For Readability reasons x,w is copied. x,w,ones is u(int)8/16 
 * Filter Checksum and input fmap checksum(Ic) requires 32-bit precision (Hari et. al.)
 * Reduced output fmaps and vector-vector dot product require 64 bit precision.
 * The used Algorithm is showed examplaraly for the data_layout = NCHW, kernel_layout = OIHW, dilitation/striding/padding = default
 * Normal Conv2D:
 * 
 *      x(N,C,H,W)   w(K,C,S,R)   ones(C,1,P,Q)    x(N,C,H,W)     w(K,C,S,R)
 *           |           |              \            /                 |
 *           |           |               \          /                  |
 *           |           |            depthwise_conv2d           cast("32bit")
 *            \         /                    |                         |
 *             \       /                     |                         |
 *              \     /                   sum(Axis={0},           sum(Axis={0},
 *               \   /                    keepdims=true)          keepdims=true)                         
 *               conv2D                     (1,C,S,R)            /
 *               /  \                            \              /
 *              /    \                            \            /
 *             /      \                            \          /
 *            /     cast("64bit")              conv2D((1,C,S,R)*(1,C,S,R))
 *           /          \                             /
 *          /            \                           /
 *         |              \                         /
 *         |            sum(Axis=            sum(Axis=
 *         |           {0,1,2,3})           {0,1,2,3}) //Solely 4D->1D
 *         |                 \               /
 *         |                  \             /
 *      y(N,K,P,Q)               not_equal
 *
 *
 */

/*
*
 * Depthwise Conv2D
 *
 *      x(N,C,H,W)   w(C,1,S,R)   ones(C,1,P,Q)    x(N,C,H,W)   w(C,1,S,R)
 *           |           |              \            /                 |
 *           |           |               \          /                  |
 *           |           |            depthwise_conv2d          reshape(w(C,1,S,R)->w(1,C,S,R))
 *            \         /                 (N,C,S,R)                   |
 *             \       /                     |                        |
 *              \     /                      |                       /
 *               \   /                       \                      /
 *          depth_conv2D                 sum(Axis={0},          cast("32bit")
 *               /  \                     keepdims=true)          /
 *              /    \                           \               /
 *             /      \                           \             /
 *            /     cast("64bit")              conv2D(1,C,S,R)*(1,C,S,R)
 *           /          \                             /
 *          /            \                           /
 *         |              \                         /
 *         |            sum(Axis=            sum(Axis=
 *         |           {0,1,2,3})           {0,1,2,3}) //Solely 4D->1D
 *         |                 \               /
 *         |                  \             /
 *      y(N,C,P,Q)               not_equal
 *
 *
 */


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
  //only supports 8bit dtypes as 16 bit is too big to be supported for accurate checksums
  bool supportedInputTensorType(const TensorTypeNode* arg){
    return ((arg->dtype == DataType::Int(8))  ||
            (arg->dtype == DataType::UInt(8)));
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
      if(supportedInputTensorType(input)  && supportedInputTensorType(weight) && supportedOutputTensorType(attr)){
        memo_.push_back(GetRef<Call>(n));
      }
   }
    // iterate deeper levels
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }
  const Op& conv2d_op;
  Array<ObjectRef> memo_;            // Array for all already existing conv2d operation/including depthwise
};

Array<ObjectRef> SearchConv2d(const Expr& e) { return Conv2dVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_conv2d").set_body_typed(SearchConv2d);

// We dont want to exchange single nodes in the graph => No Mutation

namespace transform {

///searches in data layout at which position each tensor dimension is: returns in ({pos_N, pos_C, pos_H, pos_W};)  form
struct tensor_dim_pos
{
  int pos_N, pos_C, pos_H, pos_W;
};



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

/// @brief create attributes for a tensor/tensor product with a generic 4D layout (data and kernel layout has to be corresponding)
/// @param weight_shape
/// @return
ObjectPtr<Conv2DAttrs> create_ten_ten_prod_attr(Shape weight_shape, const Conv2DAttrs* orig_conv_attr,tensor_dim_pos dim_pos ){
  auto vec_vec_prod_attr = make_object<Conv2DAttrs>();
  vec_vec_prod_attr->strides = {1,1};
  vec_vec_prod_attr->padding = {0, 0, 0, 0};
  vec_vec_prod_attr->dilation = {1,1};
  vec_vec_prod_attr->groups = 1;
  vec_vec_prod_attr->kernel_size = {weight_shape[dim_pos.pos_H], weight_shape[dim_pos.pos_W]}; //SR
  vec_vec_prod_attr->data_layout   = orig_conv_attr->data_layout;
  vec_vec_prod_attr->kernel_layout = orig_conv_attr->kernel_layout;
  vec_vec_prod_attr->out_layout = orig_conv_attr->out_layout;
  vec_vec_prod_attr->out_dtype = DataType::Int(64);
  vec_vec_prod_attr->channels = {1};
  return vec_vec_prod_attr;
}


Shape infer_output_shape_conv2d(const Call origin_conv2d, tensor_dim_pos dim_pos){
        const auto* input_tensor  = origin_conv2d->args[0]->type_as<TensorTypeNode>();
        const auto* weight_tensor = origin_conv2d->args[1]->type_as<TensorTypeNode>();

        ICHECK(input_tensor != nullptr);
        ICHECK(weight_tensor != nullptr);

        auto weight_shape = weight_tensor->shape;
        auto input_shape = input_tensor->shape;

        // need std::array to use [] access as position might vary on dim_pos of input_shape
        std::array<PrimExpr,4> one_array;
        // calculate output shape for the according Matrix(P,Q and C inferred from input shape)
        //  (Input Size – ((Filter Size – 1)Dilation Factor + 1) + 2Padding)/Stride + 1
        one_array[dim_pos.pos_H] = input_shape[dim_pos.pos_H] - weight_shape[dim_pos.pos_H] + 1;
        one_array[dim_pos.pos_W] = input_shape[dim_pos.pos_W] - weight_shape[dim_pos.pos_W] + 1;
        one_array[dim_pos.pos_N] = input_shape[dim_pos.pos_C]; //switch between C and N for depthwise Conv
        one_array[dim_pos.pos_C] = PrimExpr(1);   //channel needs to be one

        return Shape(one_array.begin(), one_array.end());

}


tvm::relay::TShapeDataDependent infer_weight_for_tensor_tensor_dot(const Call origin_conv2d, tensor_dim_pos dim_pos){
        const auto* weight_tensor = origin_conv2d->args[1]->type_as<TensorTypeNode>();
        ICHECK(weight_tensor != nullptr);

        auto weight_shape = weight_tensor->shape;
        std::array<IntImm,4> new_shape;

        new_shape[dim_pos.pos_H] =  Downcast<IntImm>(weight_shape[dim_pos.pos_H]);
        new_shape[dim_pos.pos_W] =  Downcast<IntImm>(weight_shape[dim_pos.pos_W]);
        new_shape[dim_pos.pos_N] =  Downcast<IntImm>(weight_shape[dim_pos.pos_C]); //switch between C and N for same dimensions
        new_shape[dim_pos.pos_C] =  Downcast<IntImm>(weight_shape[dim_pos.pos_N]);

        return {static_cast<int32_t>(new_shape[0]->value), static_cast<int32_t>(new_shape[1]->value),
                static_cast<int32_t>(new_shape[2]->value), static_cast<int32_t>(new_shape[3]->value)};
}


//searches data_layout for position of each Dimension position
tensor_dim_pos infer_tensor_dim_pos(const Conv2DAttrs* orig_conv_attr){
  ICHECK_EQ(orig_conv_attr->data_layout.length(), 4) <<  "data layout needs to be 4-dimensional";
   //String is a ref to std::string like string_view?
  int pos_N = static_cast<std::string>(orig_conv_attr->data_layout).find("N");
  int pos_C = static_cast<std::string>(orig_conv_attr->data_layout).find("C");
  int pos_H = static_cast<std::string>(orig_conv_attr->data_layout).find("H");
  int pos_W = static_cast<std::string>(orig_conv_attr->data_layout).find("W");
  return tensor_dim_pos{pos_N, pos_C, pos_H, pos_W};
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
      Array<ObjectRef> conv2D_array = SearchConv2d(func->body);
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
      const Op& reshape = Op::Get("reshape");


        //////////////STATIC ATTR:
        //Cast Attr
        auto cast_attr_32bit = make_object<CastAttrs>();
        
        auto cast_attr_64bit = make_object<CastAttrs>();
        cast_attr_64bit->dtype = DataType::Int(64);
        ///SUM Attr
        // sum up elements Kernel/batchwise
        auto reduce_axis_attrs = make_object<ReduceAttrs>();
        reduce_axis_attrs->keepdims = true;  // 4D -> 4D We want a Conv2D (tensor*tensor) tensor = (1,C,H,W)
        reduce_axis_attrs->exclude  = false;
        //elemwise sum operation
        auto reduce_elemwise_attrs = make_object<ReduceAttrs>();
        reduce_elemwise_attrs->axis = {0,1,2,3};  // look in sry/relay/op/tensor/reduce.cc l.451 for example
        reduce_elemwise_attrs->keepdims = false;  // 4D -> 1D
        reduce_elemwise_attrs->exclude  = false;


      // Add Checksum calc for each conv2d op in conv2D_array
      for (const auto& it : conv2D_array) {
        Call origin_conv2d = Downcast<Call>(it);
        const auto* input_tensor  = origin_conv2d->args[0]->type_as<TensorTypeNode>();
        const auto* weight_tensor = origin_conv2d->args[1]->type_as<TensorTypeNode>();

        ICHECK(input_tensor != nullptr);
        ICHECK(weight_tensor != nullptr);

        const auto* orig_conv_attr = origin_conv2d->attrs.as<Conv2DAttrs>();
        ICHECK(orig_conv_attr != nullptr);
        //search layout string for position of N,C,H,W in data layout
        auto dim_pos = infer_tensor_dim_pos(orig_conv_attr);
        reduce_axis_attrs->axis = {dim_pos.pos_N}; //Get the N-th dimension in any layout

        const auto input  = origin_conv2d->args[0];
        const auto weight = origin_conv2d->args[1];

/*
*    ones(C,1,P,Q)    x(N,C,H,W)
 *        \                /
 *         \              /
 *          \            /
 *           \          /
 *         depthwise_conv2d
 *               |
 *               |
 *            sum(Axis={0},
 *            keepdims=true)
 *                  \
 *                   \
*/
       
        /// Implement depth-wise conv with conv2d Operation (Group arg splits input into C seperate batches)
        Shape one_tensor = infer_output_shape_conv2d(origin_conv2d, dim_pos); //C1PQ for NCHW, but supports also other layouts


        auto depthwise_conv_attr = create_depthwise_conv_attr(orig_conv_attr, one_tensor[dim_pos.pos_H], one_tensor[dim_pos.pos_W], Downcast<IntImm>(one_tensor[dim_pos.pos_N]));
        auto depthwise_kernel = Ones(one_tensor, input_tensor->dtype);

        Call depthwise_conv(conv2d_op, {input, depthwise_kernel}, Attrs{depthwise_conv_attr});


/*
*                     standard conv2D:                                   depthwise conv2D:
 *                             w(K,C,S,R)                                      w(C,1,S,R)
 *                                 |                                               |
 *                                 |                                               |
 *                                 |                                               |
 *                                 |                                               |
 *                                 |                                               |
 *                            cast("32bit")                           reshape(w(C,1,S,R)->w(1,C,S,R))
 *                                 |                                              |
 *                                 |                                              |
 *       |                         |                            |                /
 *    sum(Axis={0},           sum(Axis={0},                     \               /
 *    keepdims=true)          keepdims=true)      vs.       sum(Axis={0},     cast(32bit)
 *          \                /                              keepdims=true)    /
 *           \              /                                    \           /
 *            \            /                                      \         /
 *             \          /                                   conv2D(1,C,S,R)*(1,C,S,R)
 *                conv2D                                            /
 *                /                                                /
 *               /                                                /
*/      
       
        Call batchwise_sum_input(sum_op, {depthwise_conv}, Attrs(reduce_axis_attrs));  // use batch as axis for depthwise sum
        Call filterwise_sum_input;
        bool depth = IsDepthwiseConv(origin_conv2d, orig_conv_attr, orig_conv_attr->kernel_layout);

        if(weight_tensor->dtype.is_int()){
          cast_attr_32bit->dtype = DataType::Int(32);
        }else{
          cast_attr_32bit->dtype = DataType::UInt(32);
        }

        if(!depth){
          // normal Conv
          Call weight_32bit(cast_op, {weight}, Attrs(cast_attr_32bit));
          filterwise_sum_input = Call(sum_op, {weight_32bit},  Attrs(reduce_axis_attrs));  // add each element of indiv filter
        }else{
          //depthwise Conv
          auto reshape_attrs = make_object<ReshapeAttrs>();  // reshape(w(C,1,S,R)->w(1,C,S,R)) for NCHW
          reshape_attrs->newshape = infer_weight_for_tensor_tensor_dot(origin_conv2d, dim_pos);
          auto filterwise_sum_8bit = Call(reshape, {weight},  Attrs(reshape_attrs));
          filterwise_sum_input = Call(cast_op, {filterwise_sum_8bit}, Attrs(cast_attr_32bit));
        }
        // Simple Vector-Vector dot product of 3D Tensors (Checksum dot product)
        auto weight_shape = origin_conv2d->args[1]->type_as<TensorTypeNode>()->shape;
        auto vec_vec_prod_attr = create_ten_ten_prod_attr(weight_shape, orig_conv_attr, dim_pos);
        Call vec_vec_prod(conv2d_op, {batchwise_sum_input, filterwise_sum_input}, Attrs(vec_vec_prod_attr));
        


/*
*
 *  cast("64bit")
 *       \                       /
 *        \                     /
 *         \                   /
 *      sum(Axis=         sum(Axis=
 *     {0,1,2,3})        {0,1,2,3}) //Solely 4D->1D
 *           \               /
 *            \             /
 *               not_equal
*/
        //Reduce 4D Tensor into 64-bit scalar 
        Call origin_conv2d_64bit(cast_op, {origin_conv2d}, Attrs(cast_attr_64bit));
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
              << PrettyPrint(conv2D_array) << std::endl
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
