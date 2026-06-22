/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inspect.cc
 * \brief Operators to access runtime DLTensor parameters
 */

#include "inspect.h"

#include <tvm/ffi/cast.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>

#include <tuple>

namespace tvm {
namespace relax {
namespace inspect {

TensorType GetTensorArgInfo(const Call& call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1, TypeError)
      << "Operator " << call->op << " expects one argument, "
      << "but received " << call->args.size() << " arguments: " << call->args;

  const auto& arg = call->args[0];
  auto ty = GetType(arg);

  auto tensor_ty = ty.as<TensorType>();
  TVM_FFI_CHECK(tensor_ty, TypeError) << "Operator " << call->op << " expects a tensor argument, "
                                      << "but argument " << arg << " has type " << ty;

  return tensor_ty.value();
}

std::tuple<TensorType, ffi::Optional<int64_t>> GetTensorArgInfoWithIndex(const Call& call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 2, TypeError)
      << "Operator " << call->op << " expects two arguments, "
      << "but received " << call->args.size() << " arguments: " << call->args;
  const auto& arg = call->args[0];
  const auto& axis = call->args[1];

  auto tensor_ty = arg->ty.as<TensorTypeNode>();
  TVM_FFI_CHECK(tensor_ty, TypeError)
      << "Operator " << call->op << " expects arguments (tensor, axis), "
      << "but the first argument " << arg << " in expression " << call << " has type " << arg->ty;

  auto axis_ty = axis->ty.as<PrimTypeNode>();
  TVM_FFI_CHECK(axis_ty, TypeError)
      << "Operator " << call->op << " expects arguments (tensor, axis), "
      << "but the second argument " << arg << " in expression " << call << " has type " << axis->ty;

  ffi::Optional<int64_t> int_imm_axis = std::nullopt;
  if (const auto* prim_value = axis.as<PrimValueNode>()) {
    if (const auto* int_imm = prim_value->value.as<IntImmNode>()) {
      int_imm_axis = int_imm->value;
    }
  }

  if (int_imm_axis) {
    TVM_FFI_ICHECK_GE(int_imm_axis.value(), 0);
  }
  if (int_imm_axis && !tensor_ty->IsUnknownNdim()) {
    TVM_FFI_CHECK_LT(int_imm_axis.value(), tensor_ty->ndim, ValueError)
        << "Expression " << call << " attempts to access " << arg << ".shape["
        << int_imm_axis.value() << "]"
        << ", but " << arg << ".shape only has " << tensor_ty->ndim << " elements";
  }

  return {ffi::GetRef<TensorType>(tensor_ty), int_imm_axis};
}

DataType GetTensorDataType(const Call& call) { return GetTensorArgInfo(call)->dtype; }

tirx::PrimFunc GetDLTensorField(tirx::builtin::TVMStructFieldKind field, DataType field_dtype) {
  tirx::Var dlpack_handle("dlpack_handle", DataType::Handle());

  tirx::Var value("value", field_dtype);

  tirx::Stmt body = tirx::SeqStmt(
      {tirx::Bind(value, tirx::Call(field_dtype, tirx::builtin::tvm_struct_get(),
                                    {dlpack_handle, IntImm::Int32(0), IntImm::Int32(field)})),
       tirx::Evaluate(tvm::ret(value))});

  DictAttrs attrs({{"tirx.is_scheduled", true}, {"tirx.is_host_func", true}});

  tirx::PrimFunc func(ffi::Array<tirx::Var>{dlpack_handle}, body, tvm::PrimType(field_dtype), {},
                      attrs);

  FuncType ty({TensorType(DataType::Void(), kUnknownNDim)}, PrimType(field_dtype));
  func->ty = ty;

  return func;
}

Expr NormalizeToKnownPrimValue(const BlockBuilder&, Call call) { return call; }

//// relax.tensor_dtype_code

Expr tensor_dtype_code(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_code");
  return Call(op, {expr});
}

Type InferTypeTensorDtypeCode(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(8);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

Expr LegalizeTensorDtypeCode(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimType>(call->ty)->dtype;

  Expr arg = call->args[0];
  tirx::PrimFunc getter =
      GetDLTensorField(tirx::builtin::TVMStructFieldKind::kDLTensorTypeCode, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_code");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_code")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorDtypeCode)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeCode)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_dtype_bits

Expr tensor_dtype_bits(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_bits");
  return Call(op, {expr});
}

Type InferTypeTensorDtypeBits(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(8);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

Expr LegalizeTensorDtypeBits(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimType>(call->ty)->dtype;

  Expr arg = call->args[0];
  tirx::PrimFunc getter =
      GetDLTensorField(tirx::builtin::TVMStructFieldKind::kDLTensorTypeBits, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_bits");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_bits")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorDtypeBits)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeBits)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_dtype_lanes

Expr tensor_dtype_lanes(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_lanes");
  return Call(op, {expr});
}

Type InferTypeTensorDtypeLanes(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(16);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

Expr LegalizeTensorDtypeLanes(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimType>(call->ty)->dtype;

  Expr arg = call->args[0];
  tirx::PrimFunc getter =
      GetDLTensorField(tirx::builtin::TVMStructFieldKind::kDLTensorTypeLanes, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_lanes");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_lanes")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorDtypeLanes)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeLanes)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_ndim

Expr tensor_ndim(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_ndim");
  return Call(op, {expr});
}

Type InferTypeTensorNDim(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(32);

  auto ty = GetTensorArgInfo(call);
  if (ty->IsUnknownNdim()) {
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

Expr LegalizeTensorNDim(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimType>(call->ty)->dtype;

  Expr arg = call->args[0];
  tirx::PrimFunc getter =
      GetDLTensorField(tirx::builtin::TVMStructFieldKind::kDLTensorNDim, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_ndim");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_ndim")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorNDim)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorNDim)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_shape_i

Expr tensor_shape_i(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_shape_i");
  return Call(op, {expr});
}

Type InferTypeTensorShape(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(64);

  auto [tensor_ty, int_imm_axis] = GetTensorArgInfoWithIndex(call);

  auto tensor_shape = tensor_ty->GetShape();

  if (int_imm_axis && tensor_shape.defined()) {
    return PrimType(tensor_shape.value()[int_imm_axis.value()].dtype());
  } else {
    return PrimType(dlpack_type);
  }
}

Expr LegalizeTensorShape(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimType>(call->ty)->dtype;

  tirx::PrimFunc getter = [&]() -> tirx::PrimFunc {
    tirx::Var dlpack_handle("dlpack_handle", DataType::Handle());
    tirx::Var axis("axis", DataType::Int(64));

    tirx::Var ndim("ndim", DataType::Int(32));

    tirx::Buffer shape_buffer = tirx::decl_buffer({ndim}, field_dtype, "shape");

    tirx::Var extent("extent", field_dtype);

    tirx::Stmt body = tirx::SeqStmt(
        {tirx::AssertStmt(0 <= axis, tirx::StringImm("RuntimeError"),
                          {tirx::StringImm("Specified axis may not be negative")}),
         tirx::Bind(ndim,
                    tirx::Call(ndim->dtype, tirx::builtin::tvm_struct_get(),
                               {dlpack_handle, IntImm::Int32(0),
                                IntImm::Int32(tirx::builtin::TVMStructFieldKind::kDLTensorNDim)})),
         tirx::AssertStmt(
             axis < tvm::cast(axis->dtype, ndim), tirx::StringImm("RuntimeError"),
             {tirx::StringImm(
                 "Specified axis may not be larger than the tensor's dimensionality")}),
         tirx::Bind(shape_buffer->data,
                    tirx::Call(DataType::Handle(), tirx::builtin::tvm_struct_get(),
                               {dlpack_handle, IntImm::Int32(0),
                                IntImm::Int32(tirx::builtin::TVMStructFieldKind::kDLTensorShape)})),
         tirx::DeclBuffer(shape_buffer), tirx::Bind(extent, tirx::BufferLoad(shape_buffer, {axis})),
         tirx::Evaluate(tvm::ret(extent))});

    DictAttrs attrs({{"tirx.is_scheduled", true}, {"tirx.is_host_func", true}});

    tirx::PrimFunc func({dlpack_handle, axis}, body, tvm::PrimType(field_dtype), {}, attrs);

    FuncType ty({TensorType(DataType::Void(), kUnknownNDim), PrimType(axis->dtype)},
                PrimType(field_dtype));
    func->ty = ty;
    return func;
  }();

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_shape_i");
  return Call(gvar_getter, call->args);
}

TVM_REGISTER_OP("relax.inspect.tensor_shape_i")
    .set_num_inputs(2)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .add_argument("axis", "Prim(int64)", "The axis whose extent should be returned")
    .set_attr<FInferType>("FInferType", InferTypeTensorShape)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorShape)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_stride_i

Expr tensor_stride_i(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_stride_i");
  return Call(op, {expr});
}

Type InferTypeTensorStride(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(64);

  auto [tensor_ty, int_imm_axis] = GetTensorArgInfoWithIndex(call);

  auto opt_tensor_shape = tensor_ty->GetShape();

  if (int_imm_axis && opt_tensor_shape.defined()) {
    // As of 2024-03-14, Relax does not have an explicit
    // representation for striding in `TensorType`.  The
    // `FLegalize` function for most operators is implemented in terms
    // of `topi`, and is then converted from TE to `tirx::PrimFunc`
    // using `tvm::tirx::CreatePrimFunc`.  The `te::Tensor` is
    // converted to a `tirx::Buffer` in `RewriteStageToBlock`, and uses
    // the default empty list for the strides.  The empty strides
    // represent a compact data array.
    //
    // Therefore, while Relax does not explicitly represent the
    // striding of a tensor, it implicitly requires compact striding
    // for any legalizable Tensor.
    auto tensor_shape = opt_tensor_shape.value();
    PrimExpr stride = IntImm::Int64(1);
    for (size_t axis = int_imm_axis.value() + 1; axis < tensor_shape.size(); axis++) {
      stride = stride * tensor_shape[axis];
    }
    return PrimType(stride.dtype());
  } else {
    return PrimType(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_stride_i")
    .set_num_inputs(2)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .add_argument("axis", "Prim(int64)", "The axis whose extent should be returned")
    .set_attr<FInferType>("FInferType", InferTypeTensorStride)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_byte_offset

Expr tensor_byte_offset(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_byte_offset");
  return Call(op, {expr});
}

Type InferTypeTensorByteOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(64);

  auto tensor_ty = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_ty->GetShape();
  if (opt_tensor_shape.defined()) {
    // Relax implicitly requires that the byte offset is zero for any
    // legalizable tensor.  See InferTypeTensorStride for full
    // explanation.
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_byte_offset")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorByteOffset)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

//// relax.tensor_elem_offset

Expr tensor_elem_offset(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_elem_offset");
  return Call(op, {expr});
}

Type InferTypeTensorElemOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(64);

  auto tensor_ty = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_ty->GetShape();
  if (opt_tensor_shape.defined()) {
    // Relax implicitly requires that the element offset is zero for
    // any legalizable tensor.  See InferTypeTensorStride for
    // full explanation.
    return PrimType(dlpack_type);
  } else {
    return PrimType(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_elem_offset")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferType>("FInferType", InferTypeTensorElemOffset)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<bool>("FPurity", true);

}  // namespace inspect
}  // namespace relax
}  // namespace tvm
