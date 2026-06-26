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
 * \file ternary.cc
 * \brief ternary operators.
 */

#include "ternary.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace relax {

Type InferTypeEwiseFMA(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType t1 = input_ty[0];
  TensorType t2 = input_ty[1];
  TensorType t3 = input_ty[2];

  int ndim = kUnknownNDim;
  if (!t1->IsUnknownNdim()) {
    ndim = t1->ndim;
  }
  if (!t2->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t2->ndim;
    } else if (t2->ndim != ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The 3 arguments of EwiseFMA must have the same number of dimensions";
    }
  }
  if (!t3->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t3->ndim;
    } else if (t3->ndim != ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The 3 arguments of EwiseFMA must have the same number of dimensions";
    }
  }

  ffi::Optional<PrimType> output_dtype = std::nullopt;
  if (t1->IsUnknownDtype() || t2->IsUnknownDtype() || t3->IsUnknownDtype()) {
    output_dtype = std::nullopt;
  } else if (t1->dtype != t2->dtype || t2->dtype != t3->dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call) << "Data types " << t1->dtype << ", " << t2->dtype
                                         << ", and " << t3->dtype << " must be equal for EwiseFMA";
  } else {
    output_dtype = t1->dtype;
  }

  VDevice vdev = VDevice();
  for (int i = 0; i < 3; ++i) {
    if (input_ty[i]->vdevice.defined()) {
      if (!vdev.defined()) {
        vdev = input_ty[i]->vdevice.value();
      } else if (input_ty[i]->vdevice.value()->target.defined()) {
        // mismatch
        if (input_ty[i]->vdevice.value() != vdev) {
          vdev = VDevice();
          break;
        }
      }
    }
  }

  auto* s1 = t1->shape.as<ShapeExprNode>();
  auto* s2 = t2->shape.as<ShapeExprNode>();
  auto* s3 = t3->shape.as<ShapeExprNode>();
  arith::Analyzer analyzer = ctx->GetAnalyzer();
  if (s1 && s2 && s3) {
    ffi::Array<PrimExpr> output_shape;
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim1 = s1->values[i];
      PrimExpr dim2 = s2->values[i];
      PrimExpr dim3 = s3->values[i];
      if (analyzer->CanProveEqual(dim1, dim2) && analyzer->CanProveEqual(dim2, dim3)) {
        output_shape.push_back(dim1);
      } else {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "The 3 arguments of EwiseFMA must have the same shape";
      }
    }
    if (vdev.defined()) {
      return TensorType(ShapeExpr(output_shape), output_dtype, vdev);
    }
    return TensorType(ShapeExpr(output_shape), output_dtype);
  } else if (t1->shape.defined() && t1->shape.same_as(t2->shape) && t1->shape.same_as(t3->shape)) {
    if (vdev.defined()) {
      return TensorType(t1->shape.value(), output_dtype, vdev);
    }
    return TensorType(t1->shape.value(), output_dtype);
  }
  if (vdev.defined()) {
    return TensorType(output_dtype, ndim, vdev);
  }
  return TensorType(output_dtype, ndim);
}

InferLayoutOutput InferLayoutEwiseFMA(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));

  LayoutDecision layout0 = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision layout1 = GetLayoutDecision(var_layout_map, call->args[1]);
  LayoutDecision layout2 = GetLayoutDecision(var_layout_map, call->args[2]);
  LayoutDecision layout = layout0;
  if (NLayoutEqual()(layout1, layout2)) {
    layout = layout1;
  }
  return InferLayoutOutput({layout, layout, layout}, {layout}, Attrs(call->attrs));
}

TVM_REGISTER_OP("relax.ewise_fma")
    .set_num_inputs(3)
    .add_argument("x1", "Tensor", "The left hand operand of the multiplication")
    .add_argument("x2", "Tensor", "The right hand operand of the multiplication")
    .add_argument("x3", "Tensor", "The operand of the addition")
    .set_attr<FInferType>("FInferType", InferTypeEwiseFMA)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutEwiseFMA)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

Expr ewise_fma(Expr x1, Expr x2, Expr x3) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {x1, x2, x3}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ewise_fma", ewise_fma);
}

}  // namespace relax
}  // namespace tvm
