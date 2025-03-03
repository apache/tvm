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

#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <tuple>

namespace tvm {
namespace relax {
namespace inspect {

TensorStructInfo GetTensorArgInfo(const Call& call) {
  CHECK_EQ(call->args.size(), 1) << "TypeError: "
                                 << "Operator " << call->op << " expects one argument, "
                                 << "but received " << call->args.size()
                                 << " arguments: " << call->args;

  const auto& arg = call->args[0];
  auto sinfo = GetStructInfo(arg);

  auto tensor_sinfo = sinfo.as<TensorStructInfo>();
  CHECK(tensor_sinfo) << "TypeError: "
                      << "Operator " << call->op << " expects a tensor argument, "
                      << "but argument " << arg << " has struct info " << sinfo;

  return tensor_sinfo.value();
}

std::tuple<TensorStructInfo, PrimStructInfo> GetTensorArgInfoWithIndex(const Call& call) {
  CHECK_EQ(call->args.size(), 2) << "TypeError: "
                                 << "Operator " << call->op << " expects two arguments, "
                                 << "but received " << call->args.size()
                                 << " arguments: " << call->args;
  const auto& arg = call->args[0];
  const auto& axis = call->args[1];

  auto tensor_sinfo = arg->struct_info_.as<TensorStructInfoNode>();
  CHECK(tensor_sinfo) << "TypeError: "
                      << "Operator " << call->op << " expects arguments (tensor, axis), "
                      << "but the first argument " << arg << " in expression " << call
                      << " has struct info " << arg->struct_info_;

  auto axis_sinfo = axis->struct_info_.as<PrimStructInfoNode>();
  CHECK(axis_sinfo) << "TypeError: "
                    << "Operator " << call->op << " expects arguments (tensor, axis), "
                    << "but the second argument " << arg << " in expression " << call
                    << " has struct info " << axis->struct_info_;

  auto int_imm_axis = axis_sinfo->value.as<IntImmNode>();

  if (int_imm_axis) {
    CHECK_GE(int_imm_axis->value, 0);
  }
  if (int_imm_axis && !tensor_sinfo->IsUnknownNdim()) {
    CHECK_LT(int_imm_axis->value, tensor_sinfo->ndim)
        << "ValueError: "
        << "Expression " << call << " attempts to access " << arg << ".shape["
        << int_imm_axis->value << "]"
        << ", but " << arg << ".shape only has " << tensor_sinfo->ndim << " elements";
  }

  return {GetRef<TensorStructInfo>(tensor_sinfo), GetRef<PrimStructInfo>(axis_sinfo)};
}

DataType GetTensorDataType(const Call& call) { return GetTensorArgInfo(call)->dtype; }

tir::PrimFunc GetDLTensorField(tir::builtin::TVMStructFieldKind field, DataType field_dtype) {
  tir::Var dlpack_handle("dlpack_handle", DataType::Handle());

  tir::Var value("value", field_dtype);

  tir::LetStmt body(
      value,
      tir::Call(field_dtype, tir::builtin::tvm_struct_get(),
                {dlpack_handle, IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), field)}),
      tir::Evaluate(tvm::ret(value)));

  DictAttrs attrs({{"tir.is_scheduled", Bool(true)}, {"tir.is_host", Bool(true)}});

  tir::PrimFunc func(Array<tir::Var>{dlpack_handle}, body, PrimType(field_dtype), {}, attrs);

  FuncStructInfo sinfo({TensorStructInfo(DataType::Void(), kUnknownNDim)},
                       PrimStructInfo(field_dtype));
  func->struct_info_ = sinfo;

  return func;
}

Expr NormalizeToKnownPrimValue(const BlockBuilder&, Call call) {
  if (auto prim_sinfo = call->struct_info_.as<PrimStructInfoNode>()) {
    if (prim_sinfo->value.defined()) {
      return PrimValue(prim_sinfo->value.value());
    }
  }
  return call;
}

//// relax.tensor_dtype_code

Expr tensor_dtype_code(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_code");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorDtypeCode(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(8);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimStructInfo(dlpack_type);
  } else {
    return PrimStructInfo(IntImm(dlpack_type, dtype.code()));
  }
}

Expr LegalizeTensorDtypeCode(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimStructInfo>(call->struct_info_)->dtype;

  Expr arg = call->args[0];
  tir::PrimFunc getter =
      GetDLTensorField(tir::builtin::TVMStructFieldKind::kArrTypeCode, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_code");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_code")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorDtypeCode)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeCode)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_dtype_bits

Expr tensor_dtype_bits(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_bits");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorDtypeBits(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(8);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimStructInfo(dlpack_type);
  } else {
    return PrimStructInfo(IntImm(dlpack_type, dtype.bits()));
  }
}

Expr LegalizeTensorDtypeBits(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimStructInfo>(call->struct_info_)->dtype;

  Expr arg = call->args[0];
  tir::PrimFunc getter =
      GetDLTensorField(tir::builtin::TVMStructFieldKind::kArrTypeBits, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_bits");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_bits")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorDtypeBits)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeBits)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_dtype_lanes

Expr tensor_dtype_lanes(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_dtype_lanes");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorDtypeLanes(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(16);

  DataType dtype = GetTensorDataType(call);
  if (dtype.is_void()) {
    return PrimStructInfo(dlpack_type);
  } else {
    return PrimStructInfo(IntImm(dlpack_type, dtype.lanes()));
  }
}

Expr LegalizeTensorDtypeLanes(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimStructInfo>(call->struct_info_)->dtype;

  Expr arg = call->args[0];
  tir::PrimFunc getter =
      GetDLTensorField(tir::builtin::TVMStructFieldKind::kArrTypeLanes, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_lanes");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_dtype_lanes")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorDtypeLanes)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorDtypeLanes)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_ndim

Expr tensor_ndim(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_ndim");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorNDim(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(32);

  auto sinfo = GetTensorArgInfo(call);
  if (sinfo->IsUnknownNdim()) {
    return PrimStructInfo(dlpack_type);
  } else {
    return PrimStructInfo(IntImm(dlpack_type, sinfo->ndim));
  }
}

Expr LegalizeTensorNDim(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimStructInfo>(call->struct_info_)->dtype;

  Expr arg = call->args[0];
  tir::PrimFunc getter = GetDLTensorField(tir::builtin::TVMStructFieldKind::kArrNDim, field_dtype);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_ndim");
  return Call(gvar_getter, {arg});
}

TVM_REGISTER_OP("relax.inspect.tensor_ndim")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorNDim)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorNDim)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_shape_i

Expr tensor_shape_i(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_shape_i");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorShape(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(64);

  auto [tensor_sinfo, axis_sinfo] = GetTensorArgInfoWithIndex(call);

  auto tensor_shape = tensor_sinfo->GetShape();
  auto int_imm_axis = axis_sinfo->value.as<IntImmNode>();

  if (int_imm_axis && tensor_shape.defined()) {
    return PrimStructInfo(tensor_shape.value()[int_imm_axis->value]);
  } else {
    return PrimStructInfo(dlpack_type);
  }
}

Expr LegalizeTensorShape(const BlockBuilder& bb, const Call& call) {
  auto field_dtype = Downcast<PrimStructInfo>(call->struct_info_)->dtype;

  tir::PrimFunc getter = [&]() -> tir::PrimFunc {
    tir::Var dlpack_handle("dlpack_handle", DataType::Handle());
    tir::Var axis("axis", DataType::Int(64));

    tir::Var ndim("ndim", DataType::Int(32));

    tir::Buffer shape_buffer = tir::decl_buffer({ndim}, field_dtype, "shape");

    tir::Var extent("extent", field_dtype);

    tir::Stmt body = tir::Evaluate(tvm::ret(extent));

    body = tir::LetStmt(extent, tir::BufferLoad(shape_buffer, {axis}), body);
    body = tir::DeclBuffer(shape_buffer, body);
    body = tir::LetStmt(
        shape_buffer->data,
        tir::Call(DataType::Handle(), tir::builtin::tvm_struct_get(),
                  {dlpack_handle, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), tir::builtin::TVMStructFieldKind::kArrShape)}),
        body);

    body = tir::AssertStmt(
        axis < tvm::cast(axis->dtype, ndim),
        tir::StringImm("Specified axis may not be larger than the tensor's dimensionality"), body);

    body = tir::LetStmt(
        ndim,
        tir::Call(ndim->dtype, tir::builtin::tvm_struct_get(),
                  {dlpack_handle, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), tir::builtin::TVMStructFieldKind::kArrNDim)}),
        body);

    body = tir::AssertStmt(0 <= axis, tir::StringImm("Specified axis may not be negative"), body);

    DictAttrs attrs({{"tir.is_scheduled", Bool(true)}, {"tir.is_host", Bool(true)}});

    tir::PrimFunc func({dlpack_handle, axis}, body, PrimType(field_dtype), {}, attrs);

    FuncStructInfo sinfo(
        {TensorStructInfo(DataType::Void(), kUnknownNDim), PrimStructInfo(axis->dtype)},
        PrimStructInfo(field_dtype));
    func->struct_info_ = sinfo;
    return func;
  }();

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_shape_i");
  return Call(gvar_getter, call->args);
}

TVM_REGISTER_OP("relax.inspect.tensor_shape_i")
    .set_num_inputs(2)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .add_argument("axis", "Prim(int64)", "The axis whose extent should be returned")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorShape)
    .set_attr<FLegalize>("FLegalize", LegalizeTensorShape)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_stride_i

Expr tensor_stride_i(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_stride_i");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorStride(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::Int(64);

  auto [tensor_sinfo, axis_sinfo] = GetTensorArgInfoWithIndex(call);

  auto opt_tensor_shape = tensor_sinfo->GetShape();
  auto int_imm_axis = axis_sinfo->value.as<IntImmNode>();

  if (int_imm_axis && opt_tensor_shape.defined()) {
    // As of 2024-03-14, Relax does not have an explicit
    // representation for striding in `TensorStructInfo`.  The
    // `FLegalize` function for most operators is implemented in terms
    // of `topi`, and is then converted from TE to `tir::PrimFunc`
    // using `tvm::tir::CreatePrimFunc`.  The `te::Tensor` is
    // converted to a `tir::Buffer` in `RewriteStageToBlock`, and uses
    // the default empty list for the strides.  The empty strides
    // represent a compact data array.
    //
    // Therefore, while Relax does not explicitly represent the
    // striding of a tensor, it implicitly requires compact striding
    // for any legalizable Tensor.
    auto tensor_shape = opt_tensor_shape.value();
    PrimExpr stride = IntImm(DataType::Int(64), 1);
    for (size_t axis = int_imm_axis->value + 1; axis < tensor_shape.size(); axis++) {
      stride = stride * tensor_shape[axis];
    }
    return PrimStructInfo(stride);
  } else {
    return PrimStructInfo(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_stride_i")
    .set_num_inputs(2)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .add_argument("axis", "Prim(int64)", "The axis whose extent should be returned")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorStride)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_byte_offset

Expr tensor_byte_offset(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_byte_offset");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorByteOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(64);

  auto tensor_sinfo = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_sinfo->GetShape();
  if (opt_tensor_shape.defined()) {
    // Relax implicitly requires that the byte offset is zero for any
    // legalizable tensor.  See InferStructInfoTensorStride for full
    // explanation.
    return PrimStructInfo(IntImm(dlpack_type, 0));
  } else {
    return PrimStructInfo(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_byte_offset")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorByteOffset)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

//// relax.tensor_elem_offset

Expr tensor_elem_offset(Expr expr) {
  static const Op& op = Op::Get("relax.inspect.tensor_elem_offset");
  return Call(op, {expr});
}

StructInfo InferStructInfoTensorElemOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = DataType::UInt(64);

  auto tensor_sinfo = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_sinfo->GetShape();
  if (opt_tensor_shape.defined()) {
    // Relax implicitly requires that the element offset is zero for
    // any legalizable tensor.  See InferStructInfoTensorStride for
    // full explanation.
    return PrimStructInfo(IntImm(dlpack_type, 0));
  } else {
    return PrimStructInfo(dlpack_type);
  }
}

TVM_REGISTER_OP("relax.inspect.tensor_elem_offset")
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The tensor to be inspected")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTensorElemOffset)
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FNormalize>("FNormalize", NormalizeToKnownPrimValue)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace inspect
}  // namespace relax
}  // namespace tvm
