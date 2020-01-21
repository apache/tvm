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
 * \file src/relay/op/memory/memory.cc
 * \brief Operators for manifest shape-aware memory allocation in Relay.
 */

#include <topi/elemwise.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/memory.h>

#include "../op_common.h"
#include "../../pass/infer_layout_util.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(AllocTensorAttrs);
TVM_REGISTER_NODE_TYPE(ShapeFuncAttrs);

// The passing value in attrs and args doesn't seem super great.
// We should consider a better solution, i.e the type relation
// being able to see the arguments as well?
TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_storage")
    .set_body_typed([](Expr size, Expr alignment, DataType dtype) {
      auto attrs = make_object<AllocTensorAttrs>();
      attrs->dtype = dtype;
      static const Op& op = Op::Get("memory.alloc_storage");
      return CallNode::make(op, {size, alignment}, Attrs(attrs), {});
    });

bool AllocStorageRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3u);
  auto size_type = types[0];
  auto tensor_type = size_type.as<TensorTypeNode>();
  CHECK(tensor_type != nullptr);
  CHECK_EQ(tensor_type->dtype, DataType::Int(64));
  CHECK_EQ(tensor_type->shape.size(), 0);
  auto align_type = types[1];
  auto align_ttype = align_type.as<TensorTypeNode>();
  CHECK(align_ttype != nullptr);
  CHECK_EQ(align_ttype->dtype, DataType::Int(64));
  CHECK_EQ(align_ttype->shape.size(), 0);
  auto mod = reporter->GetModule();
  CHECK(mod.defined());
  auto storage_name = mod->GetGlobalTypeVar("Storage");
  auto storage = TypeCall(storage_name, {});
  reporter->Assign(types[2], storage);
  return true;
}

RELAY_REGISTER_OP("memory.alloc_storage")
    .describe(R"code(Explicitly allocate storage to be used by tensors.)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("size", "Tensor", "The size of the storage to allocate.")
    .add_argument("alignment", "Tensor", "The alignment of the storage.")
    .add_type_rel("AllocStorage", AllocStorageRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype, const Target& target) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_tensor")
    .set_body_typed(
        [](Expr storage, tvm::relay::Expr shape, DataType dtype, Array<IndexExpr> assert_shape) {
          auto attrs = make_object<AllocTensorAttrs>();
          attrs->dtype = dtype;
          if (assert_shape.defined()) {
            attrs->assert_shape = assert_shape;
          } else {
            attrs->const_shape = Downcast<Constant>(shape);
          }
          static const Op& op = Op::Get("memory.alloc_tensor");
          return CallNode::make(op, {storage, shape}, Attrs(attrs), {});
        });

std::vector<int64_t> FromConstShape(Constant konst) {
  runtime::NDArray shape = konst->data;
  std::vector<int64_t> raw_shape;
  DLTensor tensor = shape.ToDLPack()->dl_tensor;
  CHECK_EQ(tensor.ndim, 1u);
  CHECK_EQ(tensor.dtype.code, 0U)
    << "found " << tensor.dtype.code;

  CHECK(tensor.dtype.bits == 64 || tensor.dtype.bits == 32)
    << "found " << static_cast<int>(tensor.dtype.bits);

  if (tensor.dtype.bits == 32) {
    const int32_t* int_ptr = reinterpret_cast<int32_t*>(tensor.data);
    for (auto i = 0; i < tensor.shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  } else if (tensor.dtype.bits == 64) {
    const int64_t* int_ptr = reinterpret_cast<int64_t*>(tensor.data);
    for (auto i = 0; i < tensor.shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  }

  return raw_shape;
}

bool AllocTensorRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3u);
  auto alloc_attrs = attrs.as<AllocTensorAttrs>();
  CHECK(alloc_attrs != nullptr) << "must be alloc_tensor attributes";
  // First argument should be storage.
  auto mod = reporter->GetModule();
  CHECK(mod.defined());
  auto storage_name = mod->GetGlobalTypeVar("Storage");
  auto storage = relay::TypeCall(storage_name, {});
  reporter->Assign(types[0], storage);
  // Second argument should be shape tensor.
  auto tt = types[1].as<TensorTypeNode>();
  CHECK(tt != nullptr) << "must be tensor type";
  auto rank = tt->shape[0].as<tvm::IntImmNode>();
  CHECK(rank != nullptr);
  auto dims = rank->value;

  // Constant node case.
  Type alloc_type;
  if (alloc_attrs->const_shape.defined()) {
    auto con = alloc_attrs->const_shape;
    auto sh = FromConstShape(con);
    Array<IndexExpr> out_shape;
    for (auto i = 0u; i < dims; i++) {
      out_shape.push_back(tvm::Integer(sh[i]));
    }
    alloc_type = TensorType(out_shape, alloc_attrs->dtype);
  } else {
    CHECK(alloc_attrs->assert_shape.defined())
        << "the assert_shape must be set when const_shape is not";
    alloc_type = TensorType(alloc_attrs->assert_shape, alloc_attrs->dtype);
    return true;
  }

  reporter->Assign(types[2], alloc_type);
  return true;
}

RELAY_REGISTER_OP("memory.alloc_tensor")
    .describe(R"code(Explicitly allocate storage to be used by tensors.)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("storage", "Storage", "The storage to allocate from.")
    .add_argument("shape", "Tensor", "The shape of the tensor to allocate.")
    .add_type_rel("AllocTensor", AllocTensorRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype, const Target& target) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

bool InvokeTVMOPRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4u);
  auto func_type = types[0].as<FuncTypeNode>();
  CHECK(func_type != nullptr) << "input must be operator with known type";
  auto input_type = types[1].as<TupleTypeNode>();
  auto output_type = types[2].as<TupleTypeNode>();
  CHECK(input_type != nullptr)
      << "internal invariant violated: invoke_tvm_op inputs must be a tuple";
  CHECK(output_type != nullptr)
      << "internal invariant violated: invoke_tvm_op outputs must be a tuple";
  Type ex_output;
  if (func_type->ret_type.as<TensorTypeNode>()) {
    ex_output = TupleType({func_type->ret_type});
  } else {
    CHECK(func_type->ret_type.as<TupleTypeNode>()) << "should be tuple type";
    ex_output = func_type->ret_type;
  }
  auto ex_input = TupleType(func_type->arg_types);
  reporter->Assign(ex_input, GetRef<Type>(input_type));
  reporter->Assign(ex_output, GetRef<Type>(output_type));
  reporter->Assign(types[3], TupleType::Empty());
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.memory._make.invoke_tvm_op")
    .set_body_typed(
        [](Expr func, Expr inputs, Expr outputs) {
          return CallNode::make(Op::Get("memory.invoke_tvm_op"), {func, inputs, outputs}, Attrs());
        });

RELAY_REGISTER_OP("memory.invoke_tvm_op")
    .describe(R"code(Invoke an operation compiled by TVM.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("op", "Function", "The operation to call")
    .add_argument("ins", "Tuple", "The input tensors.")
    .add_argument("outs", "Tuple", "The output tensors.")
    .add_type_rel("InvokeTVMOP", InvokeTVMOPRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype, const Target& target) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

bool KillRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2u);
  // TODO(@jroesch): should only support tensors.
  reporter->Assign(types[1], TupleType::Empty());
  return true;
}

RELAY_REGISTER_OP("memory.kill")
    .describe(R"code(Mark a tensor for release to the allocator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("to_free", "Tensor", "The tensor to free.")
    .add_type_rel("Kill", KillRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype, const Target& target) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.memory._make.shape_func")
    .set_body_typed(
      [](Expr func, Expr inputs, Expr outputs, Array<tvm::Integer> is_input) {
      static const Op& op = Op::Get("memory.shape_func");
      auto attrs = make_object<ShapeFuncAttrs>();
      attrs->is_input = is_input;
      return CallNode::make(op, {func, inputs, outputs}, Attrs(attrs), {});
    });

static void FlattenTypeAux(const Type& type, std::vector<TensorType>* out) {
  if (auto tt = type.as<TensorTypeNode>()) {
    out->push_back(GetRef<TensorType>(tt));
  } else if (auto tuple_ty = type.as<TupleTypeNode>()) {
    for (auto field : tuple_ty->fields) {
      FlattenTypeAux(field, out);
    }
  } else {
    LOG(FATAL) << "unsupported " << type;
  }
}

std::vector<TensorType> FlattenType(const Type& type) {
  std::vector<TensorType> out;
  FlattenTypeAux(type, &out);
  return out;
}

Expr PackByType(const Type& t, const Array<Expr>& exprs) {
  LOG(FATAL) << "NYI";
  return Expr();
}

bool ShapeFuncRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4u);
  auto shape_func_attrs = attrs.as<ShapeFuncAttrs>();
  CHECK(shape_func_attrs != nullptr) << "Internal compiler error";

  auto func_type = types[0].as<FuncTypeNode>();
  CHECK(func_type != nullptr);

  auto tuple = TupleType(func_type->arg_types);
  auto in_types = FlattenType(tuple);
  auto out_types = FlattenType(func_type->ret_type);

  Array<Type> shape_func_ins, shape_func_outs;
  for (size_t i = 0; i < in_types.size(); i++) {
    auto in_type = in_types[i];

    if (shape_func_attrs->is_input[i]) {
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

RELAY_REGISTER_OP("memory.shape_func")
    .describe(R"code(Get the shape of a tensor.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("tensor", "Tensor", "The tensor to retrieve the shape for.")
    .add_type_rel("ShapeFuncRel", ShapeFuncRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype, const Target& target) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

}  // namespace relay
}  // namespace tvm
