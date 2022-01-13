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

// shape_of
// register ShapeOfAttrs here to make sure it has been registered when vm.shape_of uses it
TVM_REGISTER_NODE_TYPE(ShapeOfAttrs);

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
  return Call(op, {std::move(expr)}, Attrs(std::move(attrs)), {});
}

TVM_REGISTER_GLOBAL("relay.op.vm.shape_of").set_body_typed(ShapeOf);

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
    ICHECK(func_type->ret_type.as<TupleTypeNode>())
        << "expecting function result to be tuple type. Types:" << std::endl
        << PrettyPrint(types);
    ex_output = func_type->ret_type;
  }
  auto ex_input = TupleType(func_type->arg_types);
  reporter->Assign(ex_input, GetRef<Type>(input_type));
  reporter->Assign(ex_output, GetRef<Type>(output_type));
  reporter->Assign(types[3], TupleType::Empty());
  return true;
}

Expr InvokeTVMOp(Expr func, Expr inputs, Expr outputs, DictAttrs attrs) {
  static const Op& op = Op::Get("vm.invoke_tvm_op");
  return Call(op, {std::move(func), std::move(inputs), std::move(outputs)}, std::move(attrs));
}

TVM_REGISTER_GLOBAL("relay.op.vm.invoke_tvm_op")
    .set_body_typed([](Expr func, Expr inputs, Expr outputs, DictAttrs attrs) {
      return InvokeTVMOp(std::move(func), std::move(inputs), std::move(outputs), std::move(attrs));
    });

RELAY_REGISTER_OP("vm.invoke_tvm_op")
    .describe(R"code(Invoke an operation compiled by TVM.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("op", "Function", "The operation to call")
    .add_argument("ins", "Tuple", "The input tensors.")
    .add_argument("outs", "Tuple", "The output tensors.")
    .add_type_rel("InvokeTVMOp", InvokeTVMOpRel)
    .set_attrs_type_key("DictAttrs")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
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
  return Call(op, {std::move(data), std::move(shape)}, Attrs(std::move(attrs)), {});
}

TVM_REGISTER_GLOBAL("relay.op.vm.reshape_tensor").set_body_typed(ReshapeTensor);

}  // namespace relay
}  // namespace tvm
