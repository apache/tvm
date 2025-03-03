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

namespace tvm {
namespace relax {

StructInfo InferStructInfoEwiseFMA(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo t1 = input_sinfo[0];
  TensorStructInfo t2 = input_sinfo[1];
  TensorStructInfo t3 = input_sinfo[2];

  int ndim = kUnknownNDim;
  if (!t1->IsUnknownNdim()) {
    ndim = t1->ndim;
  }
  if (!t2->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t2->ndim;
    } else if (t2->ndim != ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
  }
  if (!t3->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t3->ndim;
    } else if (t3->ndim != ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
  }

  DataType output_dtype;
  if (t1->IsUnknownDtype() || t2->IsUnknownDtype() || t3->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t1->dtype != t2->dtype || t2->dtype != t3->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << t1->dtype << ", " << t2->dtype << ", and " << t3->dtype
                     << " must be equal for EwiseFMA");
  } else {
    output_dtype = t1->dtype;
  }

  VDevice vdev = VDevice();
  for (int i = 0; i < 3; ++i) {
    if (input_sinfo[i]->vdevice.defined()) {
      if (!vdev.defined()) {
        vdev = input_sinfo[i]->vdevice.value();
      } else if (input_sinfo[i]->vdevice.value()->target.defined()) {
        // mismatch
        if (input_sinfo[i]->vdevice.value() != vdev) {
          vdev = VDevice();
          break;
        }
      }
    }
  }

  auto* s1 = t1->shape.as<ShapeExprNode>();
  auto* s2 = t2->shape.as<ShapeExprNode>();
  auto* s3 = t3->shape.as<ShapeExprNode>();
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  if (s1 && s2 && s3) {
    Array<PrimExpr> output_shape;
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim1 = s1->values[i];
      PrimExpr dim2 = s2->values[i];
      PrimExpr dim3 = s3->values[i];
      if (analyzer->CanProveEqual(dim1, dim2) && analyzer->CanProveEqual(dim2, dim3)) {
        output_shape.push_back(dim1);
      } else {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "The 3 arguments of EwiseFMA must have the same shape");
      }
    }
    if (vdev.defined()) {
      return TensorStructInfo(ShapeExpr(output_shape), output_dtype, vdev);
    }
    return TensorStructInfo(ShapeExpr(output_shape), output_dtype);
  } else if (t1->shape.defined() && t1->shape.same_as(t2->shape) && t1->shape.same_as(t3->shape)) {
    if (vdev.defined()) {
      return TensorStructInfo(t1->shape.value(), output_dtype, vdev);
    }
    return TensorStructInfo(t1->shape.value(), output_dtype);
  }
  if (vdev.defined()) {
    return TensorStructInfo(output_dtype, ndim, vdev);
  }
  return TensorStructInfo(output_dtype, ndim);
}

InferLayoutOutput InferLayoutEwiseFMA(const Call& call,
                                      const Map<String, Array<String>>& desired_layouts,
                                      const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

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
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEwiseFMA)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutEwiseFMA)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

Expr ewise_fma(Expr x1, Expr x2, Expr x3) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {x1, x2, x3}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.ewise_fma").set_body_typed(ewise_fma);

}  // namespace relax
}  // namespace tvm
