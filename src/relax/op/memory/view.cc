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
 * \file view.cc
 * \brief Operator to view an existing tensor.
 */

#include "view.h"

namespace tvm {
namespace relax {

/* relax.op.memory.view */
Expr view(Expr x, Optional<Expr> shape, Optional<Expr> dtype, Optional<Expr> relative_byte_offset) {
  Tuple void_expr(Array<Expr>{});

  static const Op& op = Op::Get("relax.memory.view");
  return Call(op, {
                      x,
                      shape.value_or(void_expr),
                      dtype.value_or(void_expr),
                      relative_byte_offset.value_or(void_expr),
                  });
}

TVM_REGISTER_GLOBAL("relax.op.memory.view").set_body_typed(view);

StructInfo InferStructInfoView(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 4) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Operator " << call->op << " should receive 4 arguments, "
                     << "but received " << call->args);
  }
  Expr arg_data = call->args[0];
  Expr arg_shape = call->args[1];
  Expr arg_dtype = call->args[2];
  Expr arg_relative_byte_offset = call->args[3];

  TensorStructInfo data_sinfo = [&]() -> TensorStructInfo {
    StructInfo sinfo = GetStructInfo(arg_data);
    if (auto opt = sinfo.as<TensorStructInfo>()) {
      return opt.value();
    } else {
      LOG(FATAL) << "TypeError: "
                 << "Operator " << call->op << " expects first argument to be a tensor, "
                 << "but received " << arg_data << " with type " << sinfo;
    }
  }();
  auto view_shape_sinfo = [&]() -> const ShapeStructInfoNode* {
    StructInfo sinfo = GetStructInfo(arg_shape);
    if (HasVoidStructInfo(arg_shape)) {
      // No shape change is applied.  The input tensor's shape is
      // kept as-is.
      return nullptr;
    } else if (auto ptr = sinfo.as<ShapeStructInfoNode>()) {
      // The `R.view` operation returns a different shape.
      return ptr;
    } else {
      LOG(FATAL) << "TypeError: "
                 << "Operator " << call->op << " expects second argument to be a ShapeExpr, "
                 << "or a void-type (empty relax tuple), "
                 << "but received " << arg_shape << " with type " << sinfo;
    }
  }();

  auto view_dtype = [&]() -> std::optional<DataType> {
    StructInfo sinfo = GetStructInfo(arg_dtype);

    if (HasVoidStructInfo(arg_dtype)) {
      // No datatype change is applied.  The input tensor's dtype is
      // kept as-is.
      return std::nullopt;
    }

    Expr arg_value = arg_dtype;
    while (auto arg_var = arg_value.as<Var>()) {
      if (auto bound_value = ctx->LookupBinding(arg_var.value())) {
        arg_value = bound_value.value();
      } else {
        break;
      }
    }

    // In general, StructInfo inference should only depend on the
    // StructInfo of the arguments, and not on the arguments
    // themselves.  However, `relax::DataTypeImm` uses
    // `ObjectStructInfo`, so we need to inspect the argument itself
    // in this case.
    if (auto dtype_imm = arg_value.as<DataTypeImmNode>()) {
      // We know the datatype for the view.
      return dtype_imm->value;
    } else if (sinfo.as<ObjectStructInfoNode>()) {
      // The view changes the datatype, but we don't know what it is
      // being changed into.
      return DataType::Void();
    } else {
      LOG(FATAL) << "TypeError: "
                 << "Operator " << call->op
                 << " expects the dtype argument to be a relax::DataTypeImm, "
                 << "but received " << arg_dtype << " with type " << sinfo;
    }
  }();

  auto view_relative_byte_offset = [&]() -> Optional<PrimExpr> {
    StructInfo sinfo = GetStructInfo(arg_relative_byte_offset);

    if (HasVoidStructInfo(arg_relative_byte_offset)) {
      // No byte offset is specified, so no change is applied.
      return IntImm(DataType::Int(64), 0);
    } else if (auto prim_sinfo = sinfo.as<PrimStructInfoNode>()) {
      CHECK_EQ(prim_sinfo->dtype, DataType::Int(64))
          << "TypeError: "
          << "Operator " << call->op
          << " expects the relative_byte_offset to be a 64-bit integer, but received "
          << arg_relative_byte_offset << ", which has type " << sinfo;
      if (prim_sinfo->value.defined()) {
        // An offset of known value is applied.  The known value may
        // be dynamic.
        return prim_sinfo->value.value();
      } else {
        // An offset of unknown value is applied.
        return NullOpt;
      }
    } else {
      LOG(FATAL) << "TypeError: "
                 << "Operator " << call->op << " expects the relative_byte_offset argument "
                 << "to be a Relax PrimValue.  "
                 << "However, expression " << call << " provides relative_byte_offset of "
                 << arg_relative_byte_offset << ", which has type " << sinfo;
    }
  }();

  Optional<Array<PrimExpr>> input_shape = data_sinfo->GetShape();

  Optional<Array<PrimExpr>> output_shape = NullOpt;
  int output_ndim = kUnknownNDim;
  if (view_shape_sinfo && view_shape_sinfo->values.defined()) {
    output_shape = view_shape_sinfo->values.value();
  } else if (view_shape_sinfo) {
    output_ndim = view_shape_sinfo->ndim;
  } else if (input_shape) {
    output_shape = input_shape;
  } else {
    output_ndim = data_sinfo->ndim;
  }

  DataType output_dtype = view_dtype.value_or(data_sinfo->dtype);

  // Helper function, returns the number of bytes per vectorized
  // element.  Cannot use `DataType::bytes`, as it returns the
  // number of bytes per scalar element.
  auto get_size_bytes = [](const DataType& dtype) -> Optional<IntImm> {
    if (dtype.is_void()) {
      return NullOpt;
    } else {
      auto size_bits = dtype.bits() * dtype.lanes();
      return IntImm(DataType::Int(64), (size_bits + 7) / 8);
    }
  };

  // Helper function, returns the number of elements in an array,
  // given the shape of that array.
  auto get_num_elements = [&ctx](const Optional<Array<PrimExpr>>& shape) -> Optional<PrimExpr> {
    if (!shape.defined()) {
      return NullOpt;
    }

    PrimExpr num_elements = Integer(1);
    for (const auto& dim : shape.value()) {
      num_elements *= dim;
    }
    return ctx->GetAnalyzer()->Simplify(num_elements);
  };

  Optional<PrimExpr> input_nelements = get_num_elements(input_shape);
  Optional<PrimExpr> output_nelements = get_num_elements(output_shape);

  Optional<IntImm> input_element_size = get_size_bytes(data_sinfo->dtype);
  Optional<IntImm> output_element_size = get_size_bytes(output_dtype);

  if (input_nelements && output_nelements && input_element_size && output_element_size &&
      view_relative_byte_offset) {
    // The shapes and dtype of input and output are known.  We know
    // the byte_offset that is applied, and can verify that the view
    // does not overrun the bounds of the original array.

    PrimExpr input_nbytes = input_nelements.value() * input_element_size.value();
    PrimExpr output_nbytes = output_nelements.value() * output_element_size.value();
    PrimExpr view_end = output_nbytes + view_relative_byte_offset.value();

    if (ctx->GetAnalyzer()->CanProve(output_nbytes + view_relative_byte_offset.value() >
                                     input_nbytes)) {
      LOG(FATAL) << "ValueError: "
                 << "Views into an array must not exceed the bounds of the array being viewed.  "
                 << "However, expression " << call << " attempted to create view of type "
                 << TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype)
                 << " with relative byte offset " << view_relative_byte_offset
                 << ", viewing into the array " << arg_data << " of type " << data_sinfo << ".  "
                 << "The end of the view would occur at byte " << view_end
                 << ", relative to the start of array " << arg_data << ", but " << arg_data
                 << " is only " << input_nbytes << " long.";
    }

  } else if (input_nelements && output_nelements && input_element_size && output_element_size) {
    // The shapes and dtype of input and output are known.  However,
    // we don't know if the `byte_offset` is being adjusted.  We can
    // still check validate using the size of the view.  If the view
    // is larger than the original array, then it would overrun its
    // bounds regardless of the `relative_byte_offset` being applied.

    PrimExpr input_nbytes = input_nelements.value() * input_element_size.value();
    PrimExpr output_nbytes = output_nelements.value() * output_element_size.value();

    if (ctx->GetAnalyzer()->CanProve(output_nbytes > input_nbytes)) {
      LOG(FATAL) << "ValueError: "
                 << "Views into an array must not exceed the bounds of the array being viewed.  "
                 << "However, expression " << call << " attempted to create view of type "
                 << TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype)
                 << " from input array of type " << data_sinfo << ".  "
                 << "This view would increase the size from " << output_nbytes << " bytes to "
                 << output_nbytes << " bytes.";
    }

  } else if (input_element_size && output_element_size && !view_shape_sinfo) {
    // The output view has a known dtype, which is different from the
    // known dtype of the input array.  Because the view's shape is
    // the same as the original array, when counted in number of
    // elements, an increase to the per-element size would cause the
    // view to be larger than the original array.

    CHECK_GE(input_element_size.value()->value, output_element_size.value()->value)
        << "ValueError: "
        << "Operator " << call->op
        << " may not produce a view that exceeds the bounds of the original array.  "
        << "In expression " << call << " the data type is changed from " << data_sinfo->dtype
        << " to " << view_dtype.value() << ", increasing the size per element from "
        << input_element_size << " bytes to " << output_element_size << " bytes.  "
        << "Consider providing a new shape for the R.view.";
  } else if (input_nelements && output_nelements && !view_dtype) {
    // The shape is being updated, while keeping the datatype the
    // same.  Even though we don't know the size of each element, we
    // know it must be the same for the input and output arrays.  An
    // increase to the number of elements would cause the view to be
    // larger than the original array, regardless of the size of each
    // individual element.

    if (ctx->GetAnalyzer()->CanProve(output_nelements.value() > input_nelements.value())) {
      LOG(FATAL) << "ValueError: "
                 << "Views into an array must not exceed the bounds of the array being viewed.  "
                 << "However, expression " << call << " attempted to view array " << arg_data
                 << " (shape = " << input_shape << ", " << input_nelements << " elements) as shape "
                 << output_shape << " with " << output_nelements << " elements.";
    }
  } else if (view_relative_byte_offset && !view_shape_sinfo && !view_dtype) {
    // The byte_offset is being updated, but neither the shape nor the
    // dtype is changing.  Any non-zero offset will cause the view to
    // overrun the bounds of the original array.
    if (ctx->GetAnalyzer()->CanProve(view_relative_byte_offset.value() > 0)) {
      LOG(FATAL) << "ValueError: "
                 << "Views into an array must not exceed the bounds of the array being viewed.  "
                 << "However, expression " << call << " attempted to offset the view by "
                 << view_relative_byte_offset << " bytes, "
                 << "without reducing either the number of elements in the view "
                 << "or the size of each element.";
    }
  }

  if (output_shape.defined()) {
    return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype, data_sinfo->vdevice);
  } else {
    return TensorStructInfo(output_dtype, output_ndim, data_sinfo->vdevice);
  }
}

TVM_REGISTER_GLOBAL("tvm.relax.struct_info.infer_view_sinfo").set_body_typed(InferStructInfoView);

Expr LowerBuiltinView(const BlockBuilder& bb, const Call& call) {
  Expr data = call->args[0];
  Expr shape = call->args[1];
  Expr dtype = call->args[2];
  Expr relative_byte_offset = call->args[3];

  if (HasVoidStructInfo(shape) && HasVoidStructInfo(dtype) &&
      HasVoidStructInfo(relative_byte_offset)) {
    // Special-case, no change is required by the view.
    return data;
  }

  // Prior to legalization, it is useful to use void-type argument to
  // specify "no change".  This allows for better shape inference when
  // a pass updates the input `data` tensor.  However, when we
  // legalize the `R.view`, we must provide an explicit parameters.

  if (HasVoidStructInfo(shape)) {
    auto data_shape = data->struct_info_.as<TensorStructInfo>().value()->GetShape();
    CHECK(data_shape.defined())
        << "Legalization of " << call->op
        << " requires that either the output shape be explicitly specified, "
        << "or the input shape is known.  "
        << "However, in expression " << call << ", no output shape is specified, "
        << "and the input " << data << " of type " << data->struct_info_ << " has unknown shape.";
    shape = ShapeExpr(data_shape.value());
  }

  if (HasVoidStructInfo(dtype)) {
    auto data_dtype = data->struct_info_.as<TensorStructInfo>().value()->dtype;
    CHECK(!data_dtype.is_void())
        << "Legalization of " << call->op
        << " requires that either the output dtype be explicitly specified, "
        << "or the input dtype is known.  "
        << "However, in expression " << call << ", no output dtype is specified, "
        << "and the input " << data << " of type " << data->struct_info_ << " has unknown dtype.";
    dtype = relax::DataTypeImm(data_dtype);
  }

  if (HasVoidStructInfo(relative_byte_offset)) {
    relative_byte_offset = relax::PrimValue::Int64(0);
  }

  StructInfoDeriveFunc infer_sinfo_env_func;
  infer_sinfo_env_func = EnvFunc::Get("tvm.relax.struct_info.infer_view_sinfo");
  auto runtime_view_sinfo = FuncStructInfo::OpaqueFunc(infer_sinfo_env_func, true);

  ExternFunc runtime_view_func("runtime.TVMArrayCreateView", runtime_view_sinfo);

  return Call(runtime_view_func, {data, shape, dtype, relative_byte_offset});
}

TVM_REGISTER_OP("relax.memory.view")
    .set_num_inputs(4)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The view's shape.")
    .add_argument("dtype", "DataType", "The view's data type.")
    .add_argument("relative_byte_offset", "Prim(\"int64\")",
                  "The view's byte offset, relative to the input tensor's byte offset.")
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoView)
    .set_attr<Bool>("FPurity", Bool(true))
    .set_attr<FLowerBuiltin>("FLowerBuiltin", LowerBuiltinView);

Expr ensure_zero_offset(const Expr& x) {
  static const Op& op = Op::Get("relax.memory.ensure_zero_offset");
  return Call(op, {x});
}

TVM_REGISTER_GLOBAL("relax.op.memory.ensure_zero_offset").set_body_typed(ensure_zero_offset);

StructInfo InferStructInfoEnsureZeroOffset(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Operator " << call->op << " should receive 1 argument, "
                     << "but received " << call->args);
  }
  return GetStructInfo(call->args[0]);
}

Expr LowerBuiltinEnsureZeroOffset(const BlockBuilder& bb, const Call& call) {
  const ExternFunc builtin_ensure_zero_offset_{"vm.builtin.ensure_zero_offset"};
  return Call(builtin_ensure_zero_offset_, call->args, Attrs(), {GetStructInfo(call)});
}

TVM_REGISTER_OP("relax.memory.ensure_zero_offset")
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<Bool>("RequiresArgumentShapes", Bool(false))
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEnsureZeroOffset)
    .set_attr<Bool>("FPurity", Bool(true))
    .set_attr<FLowerBuiltin>("FLowerBuiltin", LowerBuiltinEnsureZeroOffset);

}  // namespace relax
}  // namespace tvm
