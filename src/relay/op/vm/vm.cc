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
 * \file src/relay/op/vm/vm.cc
 * \brief Dialect operators for Relay VM.
 */

#include "vm.h"

#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/vm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/data_type.h>
#include <tvm/topi/elemwise.h>

#include <utility>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

// vm.shape_func
TVM_REGISTER_NODE_TYPE(ShapeFuncAttrs);

RELAY_REGISTER_OP("vm.shape_of")
    .describe(R"code(Get the shape of an input tensor.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The input tensor")
    .add_type_rel("ShapeOf", ShapeOfRel)
    .set_attrs_type_key("relay.attrs.ShapeOfAttrs")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

Expr ShapeOf(Expr expr) {
  auto attrs = make_object<ShapeOfAttrs>();
  attrs->dtype = DataType::Int(64);
  static const Op& op = Op::Get("vm.shape_of");
  return Call(op, {expr}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vm.shape_of").set_body_typed(ShapeOf);

Expr ShapeFunc(Expr func, Expr inputs, Expr outputs, Array<tvm::Integer> is_input) {
  static const Op& op = Op::Get("vm.shape_func");
  auto attrs = make_object<ShapeFuncAttrs>();
  attrs->is_input = is_input;
  return Call(op, {func, inputs, outputs}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vm.shape_func").set_body_typed(ShapeFunc);

bool ShapeFuncRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4u);
  auto shape_func_attrs = attrs.as<ShapeFuncAttrs>();
  ICHECK(shape_func_attrs != nullptr) << "Internal compiler error";

  auto func_type = types[0].as<FuncTypeNode>();
  ICHECK(func_type != nullptr);

  auto tuple = TupleType(func_type->arg_types);
  auto in_types = FlattenTupleType(tuple);
  auto out_types = FlattenTupleType(func_type->ret_type);
  Array<Integer> is_input;
  for (size_t i = 0; i < func_type->arg_types.size(); ++i) {
    auto const& aty = func_type->arg_types[i];
    size_t num_types = 1;
    if (aty.as<TupleTypeNode>()) {
      num_types = FlattenTupleType(aty).size();
    }
    for (size_t j = 0; j < num_types; ++j) {
      is_input.push_back(shape_func_attrs->is_input[i]);
    }
  }

  Array<Type> shape_func_ins, shape_func_outs;
  for (size_t i = 0; i < in_types.size(); i++) {
    auto in_type = in_types[i];

    if (is_input[i]) {
      shape_func_ins.push_back(in_type);
    } else {
      auto shape = RankShape(in_type->shape);
      shape_func_ins.push_back(TensorType(shape, DataType::Int(64)));
    }
  }

  for (auto out_type : out_types) {
    auto rank_shape = RankShape(out_type->shape);
    shape_func_outs.push_back(TensorType(rank_shape, DataType::Int(64)));
  }

  auto input_type = TupleType(shape_func_ins);
  auto output_type = TupleType(shape_func_outs);

  reporter->Assign(types[1], input_type);
  reporter->Assign(types[2], output_type);
  reporter->Assign(types[3], TupleType::Empty());

  return true;
}

RELAY_REGISTER_OP("vm.shape_func")
    .describe(R"code(Get the shape of a tensor.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("func", "Function", "The operation to call")
    .add_argument("ins", "Tuple", "The input tensors.")
    .add_argument("outs", "Tuple", "The output tensors.")
    .set_attrs_type_key("relay.attrs.ShapeFuncAttrs")
    .add_type_rel("ShapeFuncRel", ShapeFuncRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// vm.invoke_tvm_op
bool InvokeTVMOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4u);
  auto func_type = types[0].as<FuncTypeNode>();
  ICHECK(func_type != nullptr) << "input must be operator with known type";
  auto input_type = types[1].as<TupleTypeNode>();
  auto output_type = types[2].as<TupleTypeNode>();
  ICHECK(input_type != nullptr)
      << "internal invariant violated: invoke_tvm_op inputs must be a tuple";
  ICHECK(output_type != nullptr)
      << "internal invariant violated: invoke_tvm_op outputs must be a tuple";
  Type ex_output;
  if (func_type->ret_type.as<TensorTypeNode>()) {
    ex_output = TupleType({func_type->ret_type});
  } else {
    ICHECK(func_type->ret_type.as<TupleTypeNode>()) << "should be tuple type";
    ex_output = func_type->ret_type;
  }
  auto ex_input = TupleType(func_type->arg_types);
  reporter->Assign(ex_input, GetRef<Type>(input_type));
  reporter->Assign(ex_output, GetRef<Type>(output_type));
  reporter->Assign(types[3], TupleType::Empty());
  return true;
}

Expr InvokeTVMOp(Expr func, Expr inputs, Expr outputs) {
  return Call(Op::Get("vm.invoke_tvm_op"), {func, inputs, outputs}, Attrs());
}

TVM_REGISTER_GLOBAL("relay.op.vm.invoke_tvm_op").set_body_typed(InvokeTVMOp);

RELAY_REGISTER_OP("vm.invoke_tvm_op")
    .describe(R"code(Invoke an operation compiled by TVM.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("op", "Function", "The operation to call")
    .add_argument("ins", "Tuple", "The input tensors.")
    .add_argument("outs", "Tuple", "The output tensors.")
    .add_type_rel("InvokeTVMOp", InvokeTVMOpRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// vm.reshape
TVM_REGISTER_NODE_TYPE(ReshapeTensorAttrs);

bool ReshapeTensorRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3u);
  auto reshape_attrs = attrs.as<ReshapeTensorAttrs>();
  ICHECK(reshape_attrs);
  auto tt = types[0].as<TensorTypeNode>();
  ICHECK(tt) << "input must be tensor type";
  reporter->Assign(types[2], TensorType(reshape_attrs->newshape, tt->dtype));
  return true;
}

RELAY_REGISTER_OP("vm.reshape_tensor")
    .describe(R"code(Use VM reshape_tensor instruction to reshape the tensor.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor")
    .add_argument("shape", "Tensor", "The output shape tensor")
    .add_type_rel("ReshapeTensor", ReshapeTensorRel)
    .set_attrs_type_key("relay.attrs.ReshapeTensorAttrs")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

Expr ReshapeTensor(Expr data, Expr shape, Array<PrimExpr> newshape) {
  static const Op& op = Op::Get("vm.reshape_tensor");
  auto attrs = make_object<ReshapeTensorAttrs>();
  attrs->newshape = std::move(newshape);
  return Call(op, {data, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vm.reshape_tensor").set_body_typed(ReshapeTensor);

}  // namespace relay
}  // namespace tvm
