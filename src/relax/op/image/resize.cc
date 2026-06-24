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
 * \file resize.cc
 * \brief Image resize operators.
 */

#include "resize.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() { Resize2DAttrs::RegisterReflection(); }
TVM_FFI_STATIC_INIT_BLOCK() { Resize3DAttrs::RegisterReflection(); }

/* relax.resize2d */

Expr resize2d(Expr data, Expr size, ffi::Array<FloatImm> roi, ffi::String layout,
              ffi::String method, ffi::String coordinate_transformation_mode,
              ffi::String rounding_method, double cubic_alpha, int cubic_exclude,
              double extrapolation_value, ffi::Optional<DLDataType> out_dtype) {
  ffi::ObjectPtr<Resize2DAttrs> attrs = ffi::make_object<Resize2DAttrs>();
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype.value_or((DLDataType{kDLOpaqueHandle, 0, 0}));

  static const Op& op = Op::Get("relax.image.resize2d");
  return Call(op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.resize2d", resize2d);
}

Type InferTypeResize2D(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Resize2D expects 2 arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeNode>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprNode>();
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize2D expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize2D expects the given output image size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  if (size_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Resize2D expects the given output image size to "
                                             "be a 2-dim shape, while the given one has ndim "
                                          << size_ty->ndim;
  }

  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                    /*tgt_layout=*/"NCHW",     //
                                                    /*tensor_name=*/"data");

  PrimType out_dtype = attrs->out_dtype == DLDataType{kDLOpaqueHandle, 0, 0}
                           ? data_ty->dtype
                           : PrimType(attrs->out_dtype);

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, ffi::GetRef<TensorType>(data_ty), data_layout);
  if (!data_shape.defined() || size_value == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  out_NCHW_shape.Set(2, size_value->values[0]);
  out_NCHW_shape.Set(3, size_value->values[1]);

  ffi::Array<PrimExpr> out_shape = data2NCHW.BackwardShape(out_NCHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutResize2d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.image.resize2d");
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout;
  ffi::ObjectPtr<Resize2DAttrs> new_attrs = ffi::make_object<Resize2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for resize2d.
    SLayout desired_data_layout = (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(4), attrs->layout, desired_data_layout);
    new_attrs->layout = (*it).second[0];
  } else {
    // We dont have a desired layout for resize2d, propagate from the input instead.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    // Not handling sub indexing now.
    if (data_layout->layout.ndim() != data_layout->layout.ndim_primal()) {
      data_layout = LayoutDecision(InitialLayout(4));
    }
    new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), data_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, InitialNLayout(call->args[1])}, {data_layout},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.image.resize2d")
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("size", "Shape", "The output image shape.")
    .set_attr<FInferType>("FInferType", InferTypeResize2D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutResize2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.resize3d */

Expr resize3d(Expr data, Expr size, ffi::Array<FloatImm> roi, ffi::String layout,
              ffi::String method, ffi::String coordinate_transformation_mode,
              ffi::String rounding_method, double cubic_alpha, int cubic_exclude,
              double extrapolation_value, ffi::Optional<DLDataType> out_dtype) {
  ffi::ObjectPtr<Resize3DAttrs> attrs = ffi::make_object<Resize3DAttrs>();
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype.value_or((DLDataType{kDLOpaqueHandle, 0, 0}));

  static const Op& op = Op::Get("relax.image.resize3d");
  return Call(op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.resize3d", resize3d);
}

Type InferTypeResize3D(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Resize3D expects 2 arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeNode>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprNode>();
  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize3D expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Resize3D expects the given output image size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  if (size_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Resize3D expects the given output image size to "
                                             "be a 3-dim shape, while the given one has ndim "
                                          << size_ty->ndim;
  }

  const auto* attrs = call->attrs.as<Resize3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                     /*tgt_layout=*/"NCDHW",    //
                                                     /*tensor_name=*/"data");

  PrimType out_dtype = attrs->out_dtype == DLDataType{kDLOpaqueHandle, 0, 0}
                           ? data_ty->dtype
                           : PrimType(attrs->out_dtype);

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, ffi::GetRef<TensorType>(data_ty), data_layout);
  if (!data_shape.defined() || size_value == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_NCDHW_shape(data_NCDHW_shape);
  out_NCDHW_shape.Set(2, size_value->values[0]);
  out_NCDHW_shape.Set(3, size_value->values[1]);
  out_NCDHW_shape.Set(4, size_value->values[2]);

  ffi::Array<PrimExpr> out_shape = data2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

InferLayoutOutput InferLayoutResize3d(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.image.resize3d");
  const auto* attrs = call->attrs.as<Resize3DAttrs>();
  TVM_FFI_ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout;
  ffi::ObjectPtr<Resize3DAttrs> new_attrs = ffi::make_object<Resize3DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    SLayout desired_data_layout = (*it).second[0];
    TVM_FFI_ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(5), attrs->layout, desired_data_layout);
    new_attrs->layout = (*it).second[0];
  } else {
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    if (data_layout->layout.ndim() != data_layout->layout.ndim_primal()) {
      data_layout = LayoutDecision(InitialLayout(5));
    }
    new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(5), data_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, InitialNLayout(call->args[1])}, {data_layout},
                           Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.image.resize3d")
    .set_attrs_type<Resize3DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("size", "Shape", "The output image shape.")
    .set_attr<FInferType>("FInferType", InferTypeResize3D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutResize3d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.grid_sample */

TVM_FFI_STATIC_INIT_BLOCK() { GridSampleAttrs::RegisterReflection(); }

Expr grid_sample(Expr data, Expr grid, ffi::String method, ffi::String layout,
                 ffi::String padding_mode, bool align_corners) {
  ffi::ObjectPtr<GridSampleAttrs> attrs = ffi::make_object<GridSampleAttrs>();
  attrs->method = std::move(method);
  attrs->layout = std::move(layout);
  attrs->padding_mode = std::move(padding_mode);
  attrs->align_corners = align_corners;

  static const Op& op = Op::Get("relax.image.grid_sample");
  return Call(op, {std::move(data), std::move(grid)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.grid_sample", grid_sample);
}

Type InferTypeGridSample(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "GridSample expects two arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  const auto* grid_ty = GetTypeAs<TensorTypeNode>(call->args[1]);

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GridSample expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (grid_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "GridSample expects the grid to be a Tensor, while the given grid is "
        << call->args[1]->GetTypeKey();
  }

  const auto* attrs = call->attrs.as<GridSampleAttrs>();

  // grid_sample supports both 2D (NCHW) and 3D (NCDHW) sampling. The frontend
  // sets attrs->layout to "NCDHW" for the volumetric case; everything else is
  // treated as the 2D NCHW path so existing behavior is preserved.
  const bool is_ncdhw = (attrs->layout == "NCDHW");

  auto [data_layout, data2tgt] = CheckTensorLayout(call, ctx, attrs->layout,
                                                   /*tgt_layout=*/is_ncdhw ? "NCDHW" : "NCHW",
                                                   /*tensor_name=*/"data");

  PrimType out_dtype = data_ty->dtype;

  ffi::Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, ffi::GetRef<TensorType>(data_ty), data_layout);
  const auto* grid_shape = grid_ty->shape.as<ShapeExprNode>();

  if (!data_shape.defined() || grid_shape == nullptr) {
    return TensorType(out_dtype, data_layout.ndim(), data_ty->vdevice);
  }

  ffi::Array<PrimExpr> data_tgt_shape = data2tgt.ForwardShape(data_shape.value()->values);
  ffi::Array<PrimExpr> out_tgt_shape(data_tgt_shape);
  if (is_ncdhw) {
    // grid (TVM layout) is [N, 3, D_out, H_out, W_out], output is
    // [N, C, D_out, H_out, W_out]; the spatial extents are grid->values[2:].
    out_tgt_shape.Set(2, grid_shape->values[2]);  // D_out
    out_tgt_shape.Set(3, grid_shape->values[3]);  // H_out
    out_tgt_shape.Set(4, grid_shape->values[4]);  // W_out
  } else {
    // grid (TVM layout) is [N, 2, H_out, W_out], output is [N, C, H_out, W_out]
    out_tgt_shape.Set(2, grid_shape->values[2]);  // H_out
    out_tgt_shape.Set(3, grid_shape->values[3]);  // W_out
  }

  ffi::Array<PrimExpr> out_shape = data2tgt.BackwardShape(out_tgt_shape);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

TVM_REGISTER_OP("relax.image.grid_sample")
    .set_attrs_type<GridSampleAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("grid", "Tensor", "The grid tensor for sampling.")
    .set_attr<FInferType>("FInferType", InferTypeGridSample)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.image.affine_grid */

Expr affine_grid(Expr data, Expr size, bool align_corners) {
  ffi::ObjectPtr<AffineGridAttrs> attrs = ffi::make_object<AffineGridAttrs>();
  attrs->align_corners = align_corners;
  static const Op& op = Op::Get("relax.image.affine_grid");
  return Call(op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() { AffineGridAttrs::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.image.affine_grid", affine_grid);
}

Type InferTypeAffineGrid(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "AffineGrid expects two arguments, while the given number of arguments is "
        << call->args.size();
  }

  const auto* data_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  const auto* size_ty = GetTypeAs<ShapeTypeNode>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprNode>();

  if (data_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "AffineGrid expects the input data to be a Tensor, while the given data is "
        << call->args[0]->GetTypeKey();
  }
  if (size_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "AffineGrid expects the target size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey();
  }
  if (size_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "AffineGrid expects the target size to be a 2-dim shape, while the given "
           "one has ndim "
        << size_ty->ndim;
  }

  // data should be 3-D: [batch, 2, 3]
  if (data_ty->ndim != -1 && data_ty->ndim != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "AffineGrid expects the input data to be 3-D (batch, 2, 3), but got ndim "
        << data_ty->ndim;
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  if (data_shape != nullptr) {
    // Check that the affine matrix has shape [batch, 2, 3]
    if (data_shape->values.size() >= 2) {
      auto* dim1 = data_shape->values[1].as<IntImmNode>();
      if (dim1 != nullptr && dim1->value != 2) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "AffineGrid expects the second dimension of input to be 2, but got " << dim1->value;
      }
    }
    if (data_shape->values.size() >= 3) {
      auto* dim2 = data_shape->values[2].as<IntImmNode>();
      if (dim2 != nullptr && dim2->value != 3) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "AffineGrid expects the third dimension of input to be 3, but got " << dim2->value;
      }
    }
  }

  PrimType out_dtype = data_ty->dtype;

  if (data_shape == nullptr || size_value == nullptr) {
    return TensorType(out_dtype, /*ndim=*/4, data_ty->vdevice);
  }

  // Output shape: [batch, 2, target_height, target_width]
  ffi::Array<PrimExpr> out_shape;
  out_shape.push_back(data_shape->values[0]);  // batch
  out_shape.push_back(IntImm::Int64(2));       // 2 (spatial dimensions)
  out_shape.push_back(size_value->values[0]);  // target_height
  out_shape.push_back(size_value->values[1]);  // target_width

  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

TVM_REGISTER_OP("relax.image.affine_grid")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input affine matrix tensor.")
    .add_argument("size", "Shape", "The target output shape (H, W).")
    .set_attrs_type<AffineGridAttrs>()
    .set_attr<FInferType>("FInferType", InferTypeAffineGrid)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
