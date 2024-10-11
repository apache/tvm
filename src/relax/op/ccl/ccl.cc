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

#include "ccl.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.ccl.allreduce */
TVM_REGISTER_NODE_TYPE(AllReduceAttrs);

Expr allreduce(Expr x, String op_type, bool in_group) {
  ObjectPtr<AllReduceAttrs> attrs = make_object<AllReduceAttrs>();
  attrs->op_type = std::move(op_type);
  attrs->in_group = std::move(in_group);

  static const Op& op = Op::Get("relax.ccl.allreduce");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ccl.allreduce").set_body_typed(allreduce);

StructInfo InferStructInfoAllReduce(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  return input_sinfo;
}

TVM_REGISTER_OP("relax.ccl.allreduce")
    .set_attrs_type<AllReduceAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "Input to which allreduce will be applied.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAllReduce)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.ccl.allgather */
TVM_REGISTER_NODE_TYPE(AllGatherAttrs);

Expr allgather(Expr x, int num_workers, bool in_group) {
  ObjectPtr<AllGatherAttrs> attrs = make_object<AllGatherAttrs>();
  attrs->num_workers = std::move(num_workers);
  attrs->in_group = std::move(in_group);

  static const Op& op = Op::Get("relax.ccl.allgather");
  return Call(op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ccl.allgather").set_body_typed(allgather);

StructInfo InferStructInfoAllGather(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<AllGatherAttrs>();
  int num_workers = attrs->num_workers;

  DataType output_dtype = input_sinfo->dtype;
  auto input_shape = input_sinfo->GetShape();
  if (!input_shape.defined()) {
    return input_sinfo;
  }
  Array<PrimExpr> output_shape = input_shape.value();
  output_shape.Set(0, floor(output_shape[0] * num_workers));
  return TensorStructInfo(ShapeExpr(output_shape), output_dtype, input_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.ccl.allgather")
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "Input to which allgather will be applied.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAllGather)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.ccl.broadcast_from_worker0 */
Expr broadcast_from_worker0(Expr x) {
  static const Op& op = Op::Get("relax.ccl.broadcast_from_worker0");
  return Call(op, {std::move(x)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ccl.broadcast_from_worker0").set_body_typed(broadcast_from_worker0);

StructInfo InferStructInfoBroadcastFromZero(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  return input_sinfo;
}

TVM_REGISTER_OP("relax.ccl.broadcast_from_worker0")
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "Input to be broadcast.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBroadcastFromZero)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.ccl.scatter_from_worker0 */
TVM_REGISTER_NODE_TYPE(ScatterCollectiveAttrs);

Expr scatter_from_worker0(Expr data, int num_workers, int axis) {
  ObjectPtr<ScatterCollectiveAttrs> attrs = make_object<ScatterCollectiveAttrs>();
  attrs->num_workers = std::move(num_workers);
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("relax.ccl.scatter_from_worker0");

  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ccl.scatter_from_worker0").set_body_typed(scatter_from_worker0);

StructInfo InferStructInfoScatter(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  DataType output_dtype = input_sinfo->dtype;

  const auto* attrs = call->attrs.as<ScatterCollectiveAttrs>();
  int num_workers = attrs->num_workers;

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  auto input_shape = input_sinfo->GetShape();
  CHECK(input_shape.defined()) << "input tensor of scatter_from_worker0 should have defined shape.";

  if (analyzer->CanProve(floormod(input_shape.value()[0], PrimExpr(num_workers))) != 0) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "scatter_from_worker0 expects the size of axis 0 of input tensor to be "
                        "divisible by the "
                        "num_workers. However, the axis 0 of input tensor is "
                     << input_shape.value() << " while num_workers is " << num_workers);
  }

  Array<PrimExpr> output_shape = input_shape.value();
  output_shape.Set(attrs->axis, div(output_shape[attrs->axis], num_workers));
  return TensorStructInfo(ShapeExpr(output_shape), output_dtype, input_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.ccl.scatter_from_worker0")
    .set_num_inputs(1)
    .add_argument("x", "Tensor",
                  "The buffer to be divided into equal parts and sent to each worker accordingly.")
    .set_attrs_type<ScatterCollectiveAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoScatter)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
