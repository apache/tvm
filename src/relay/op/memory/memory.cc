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

#include "memory.h"

#include <tvm/node/node.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/data_type.h>
#include <tvm/topi/elemwise.h>

#include <utility>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../annotation/annotation.h"
#include "../op_common.h"
#include "../type_relations.h"
#include "on_device.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(AllocStorageAttrs);
TVM_REGISTER_NODE_TYPE(AllocTensorAttrs);

// The passing value in attrs and args doesn't seem super great.
// We should consider a better solution, i.e the type relation
// being able to see the arguments as well?
Expr AllocStorage(Expr size, Expr shape, Expr alignment, VirtualDevice virtual_device,
                  DataType dtype_hint) {
  auto attrs = make_object<AllocStorageAttrs>();
  attrs->dtype = dtype_hint;
  attrs->virtual_device = std::move(virtual_device);
  static const Op& op = Op::Get("memory.alloc_storage");
  return Call(op, {std::move(size), std::move(shape), std::move(alignment)},
              Attrs(std::move(attrs)), {});
}

TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_storage").set_body_typed(AllocStorage);

bool AllocStorageRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4u);
  auto size_type = types[0];
  auto tensor_type = size_type.as<TensorTypeNode>();
  ICHECK(tensor_type != nullptr);
  ICHECK_EQ(tensor_type->dtype, DataType::Int(64));
  ICHECK_EQ(tensor_type->shape.size(), 0);

  // Tensor shape
  auto tt = types[1].as<TensorTypeNode>();
  ICHECK(tt != nullptr) << "must be tensor type";

  auto align_type = types[2];
  auto align_ttype = align_type.as<TensorTypeNode>();
  ICHECK(align_ttype != nullptr);
  ICHECK_EQ(align_ttype->dtype, DataType::Int(64));
  ICHECK_EQ(align_ttype->shape.size(), 0);
  auto mod = reporter->GetModule();
  ICHECK(mod.defined());
  auto storage_name = mod->GetGlobalTypeVar("Storage");
  auto storage = TypeCall(storage_name, {});
  reporter->Assign(types[3], storage);
  return true;
}

RELAY_REGISTER_OP("memory.alloc_storage")
    .describe(R"code(Explicitly allocate storage to be used by tensors.)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("size", "Tensor", "The size of the storage to allocate.")
    .add_argument("shape", "Tensor", "The shape of the storage to allocate.")
    .add_argument("alignment", "Tensor", "The alignment of the storage.")
    .add_type_rel("AllocStorage", AllocStorageRel)
    .set_attrs_type_key("relay.attrs.AllocStorageAttrs")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

const Op& MemoryAllocTensorOp() {
  static const Op& op = Op::Get("memory.alloc_tensor");
  return op;
}

Expr AllocTensor(Expr storage, Expr offset, Expr shape, DataType dtype,
                 Array<IndexExpr> assert_shape) {
  auto attrs = make_object<AllocTensorAttrs>();
  attrs->dtype = dtype;
  if (assert_shape.defined()) {
    attrs->assert_shape = assert_shape;
  } else {
    // Look through any on_device for the shape argument expression.
    const auto* constant_node = AsIgnoringOnDevice<ConstantNode>(shape);
    ICHECK(constant_node);
    attrs->const_shape = GetRef<Constant>(constant_node);
  }
  return Call(MemoryAllocTensorOp(), {storage, offset, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.memory._make.alloc_tensor").set_body_typed(AllocTensor);

std::vector<int64_t> FromConstShape(Constant konst) {
  runtime::NDArray shape = konst->data;
  std::vector<int64_t> raw_shape;
  ICHECK_EQ(shape->ndim, 1u);
  ICHECK_EQ(shape->dtype.code, 0U) << "The dtype of constant shape must be int32 or int64, but got "
                                   << runtime::DLDataType2String(shape->dtype);
  ICHECK(shape->dtype.bits == 64 || shape->dtype.bits == 32)
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
  ICHECK_EQ(types.size(), 4u);
  auto alloc_attrs = attrs.as<AllocTensorAttrs>();
  ICHECK(alloc_attrs != nullptr) << "must be alloc_tensor attributes";
  // First argument should be storage.
  auto mod = reporter->GetModule();
  ICHECK(mod.defined());
  auto storage_name = mod->GetGlobalTypeVar("Storage");
  auto storage = relay::TypeCall(storage_name, {});
  reporter->Assign(types[0], storage);
  // Second argument should be the offset.
  auto offset_type = types[1].as<TensorTypeNode>();
  ICHECK(offset_type != nullptr) << "must be a scalar type";

  // Third argument should be shape tensor.
  auto tt = types[2].as<TensorTypeNode>();
  ICHECK(tt != nullptr) << "must be tensor type";

  // Be careful about having to allocate scalars.
  int64_t dims = 0;
  if (tt->shape.size() != 0) {
    auto rank = tt->shape[0].as<tvm::IntImmNode>();
    ICHECK(rank != nullptr);
    dims = rank->value;
  }

  // Constant node case.
  Type alloc_type;
  if (alloc_attrs->const_shape.defined()) {
    auto con = alloc_attrs->const_shape;
    auto sh = FromConstShape(con);
    ICHECK_EQ(sh.size(), dims);
    Array<IndexExpr> out_shape;
    for (auto i = 0u; i < dims; i++) {
      out_shape.push_back(tvm::Integer(sh[i]));
    }
    alloc_type = TensorType(out_shape, alloc_attrs->dtype);
  } else {
    ICHECK(alloc_attrs->assert_shape.defined())
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
    .set_attrs_type_key("relay.attrs.AllocTensorAttrs")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

bool KillRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2u);
  // TODO(@jroesch): should only support tensors.
  reporter->Assign(types[1], TupleType::Empty());
  return true;
}

RELAY_REGISTER_OP("memory.kill")
    .describe(R"code(Mark a variable for release to the allocator.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("to_free", "Variable", "The variable to free.")
    .add_type_rel("Kill", KillRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

static void FlattenTupleTypeAux(const Type& type, std::vector<TensorType>* out) {
  if (auto tt = type.as<TensorType>()) {
    out->push_back(tt.value());
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
