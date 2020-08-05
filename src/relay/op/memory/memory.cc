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

#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/data_type.h>
#include <tvm/topi/elemwise.h>

#include "../../transforms/infer_layout_util.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(AllocStorageAttrs);
TVM_REGISTER_NODE_TYPE(AllocTensorAttrs);

// The passing value in attrs and args doesn't seem super great.
// We should consider a better solution, i.e the type relation
// being able to see the arguments as well?
TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_storage")
    .set_body_typed([](Expr size, Expr alignment, TVMContext ctx, DataType dtype_hint) {
      auto attrs = make_object<AllocStorageAttrs>();
      attrs->dtype = dtype_hint;
      attrs->device_id = ctx.device_id;
      attrs->device_type = ctx.device_type;
      static const Op& op = Op::Get("memory.alloc_storage");
      return Call(op, {size, alignment}, Attrs(attrs), {});
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
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_tensor")
    .set_body_typed([](Expr storage, Expr offset, tvm::relay::Expr shape, DataType dtype,
                       Array<IndexExpr> assert_shape) {
      auto attrs = make_object<AllocTensorAttrs>();
      attrs->dtype = dtype;
      if (assert_shape.defined()) {
        attrs->assert_shape = assert_shape;
      } else {
        attrs->const_shape = Downcast<Constant>(shape);
      }
      static const Op& op = Op::Get("memory.alloc_tensor");
      return Call(op, {storage, offset, shape}, Attrs(attrs), {});
    });

std::vector<int64_t> FromConstShape(Constant konst) {
  runtime::NDArray shape = konst->data;
  std::vector<int64_t> raw_shape;
  CHECK_EQ(shape->ndim, 1u);
  CHECK_EQ(shape->dtype.code, 0U) << "The dtype of constant shape must be int32 or int64, but got "
                                  << runtime::DLDataType2String(shape->dtype);
  CHECK(shape->dtype.bits == 64 || shape->dtype.bits == 32)
      << "The dtype of constant shape must be int32 or int64, but got"
      << runtime::DLDataType2String(shape->dtype);

  if (shape->dtype.bits == 32) {
    const int32_t* int_ptr = reinterpret_cast<int32_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  } else if (shape->dtype.bits == 64) {
    const int64_t* int_ptr = reinterpret_cast<int64_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  }

  return raw_shape;
}

bool AllocTensorRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4u);
  auto alloc_attrs = attrs.as<AllocTensorAttrs>();
  CHECK(alloc_attrs != nullptr) << "must be alloc_tensor attributes";
  // First argument should be storage.
  auto mod = reporter->GetModule();
  CHECK(mod.defined());
  auto storage_name = mod->GetGlobalTypeVar("Storage");
  auto storage = relay::TypeCall(storage_name, {});
  reporter->Assign(types[0], storage);
  // Second argument should be the offset.
  auto offset_type = types[1].as<TensorTypeNode>();
  CHECK(offset_type != nullptr) << "must be a scalar type";

  // Third argument should be shape tensor.
  auto tt = types[2].as<TensorTypeNode>();
  CHECK(tt != nullptr) << "must be tensor type";

  // Be careful about having to allocate scalars.
  int64_t dims = 0;
  if (tt->shape.size() != 0) {
    auto rank = tt->shape[0].as<tvm::IntImmNode>();
    CHECK(rank != nullptr);
    dims = rank->value;
  }

  // Constant node case.
  Type alloc_type;
  if (alloc_attrs->const_shape.defined()) {
    auto con = alloc_attrs->const_shape;
    auto sh = FromConstShape(con);
    CHECK_EQ(sh.size(), dims);
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

  reporter->Assign(types[3], alloc_type);
  return true;
}

RELAY_REGISTER_OP("memory.alloc_tensor")
    .describe(R"code(Explicitly allocate storage to be used by tensors.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("storage", "Storage", "The storage to allocate from.")
    .add_argument("offset", "Tensor", "The offset into the backing storage.")
    .add_argument("shape", "Tensor", "The shape of the tensor to allocate.")
    .add_type_rel("AllocTensor", AllocTensorRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
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
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

static void FlattenTupleTypeAux(const Type& type, std::vector<TensorType>* out) {
  if (auto tt = type.as<TensorTypeNode>()) {
    out->push_back(GetRef<TensorType>(tt));
  } else if (auto tuple_ty = type.as<TupleTypeNode>()) {
    for (auto field : tuple_ty->fields) {
      FlattenTupleTypeAux(field, out);
    }
  } else {
    LOG(FATAL) << "unsupported " << type;
  }
}

std::vector<TensorType> FlattenTupleType(const Type& type) {
  std::vector<TensorType> out;
  FlattenTupleTypeAux(type, &out);
  return out;
}

static void FromTupleTypeAux(const Type& type, const Expr& expr, std::vector<Expr>* out) {
  if (type.as<TensorTypeNode>()) {
    out->push_back(expr);
  } else if (auto tuple_ty = type.as<TupleTypeNode>()) {
    for (size_t i = 0; i < tuple_ty->fields.size(); i++) {
      FromTupleTypeAux(tuple_ty->fields[i], TupleGetItem(expr, i), out);
    }
  } else {
    LOG(FATAL) << "unsupported " << type;
  }
}

std::vector<Expr> FromTupleType(const Type& type, const Expr& expr) {
  std::vector<Expr> out;
  FromTupleTypeAux(type, expr, &out);
  return out;
}

static void ToTupleTypeAux(const Type& type, const std::vector<Expr>& exprs, int* index,
                           std::vector<Expr>* out) {
  if (type.as<TensorTypeNode>()) {
    out->push_back(exprs[*index]);
    *index += 1;
  } else if (auto tuple_ty = type.as<TupleTypeNode>()) {
    std::vector<Expr> tuple_out;
    for (size_t i = 0; i < tuple_ty->fields.size(); i++) {
      ToTupleTypeAux(tuple_ty->fields[i], exprs, index, &tuple_out);
    }
    out->push_back(Tuple(tuple_out));
  } else {
    LOG(FATAL) << "unsupported " << type;
  }
}

// Pack the sequence of expressions according to the provided TupleType.
Expr ToTupleType(const Type& t, const std::vector<Expr>& exprs) {
  if (t.as<TensorTypeNode>() && exprs.size() == 1) {
    return exprs[0];
  } else {
    std::vector<Expr> out;
    int index = 0;
    ToTupleTypeAux(t, exprs, &index, &out);
    return out[0];
  }
}

TVM_REGISTER_GLOBAL("relay.op.memory._make.FlattenTupleType").set_body_typed([](Type type) {
  auto types = FlattenTupleType(type);
  return Array<Type>(types.begin(), types.end());
});

TVM_REGISTER_GLOBAL("relay.op.memory._make.FromTupleType").set_body_typed([](Type type, Expr expr) {
  auto exprs = FromTupleType(type, expr);
  return Array<Expr>(exprs.begin(), exprs.end());
});

TVM_REGISTER_GLOBAL("relay.op.memory._make.ToTupleType")
    .set_body_typed([](Type t, Array<Expr> array) {
      return ToTupleType(t, std::vector<Expr>(array.begin(), array.end()));
    });

}  // namespace relay
}  // namespace tvm
