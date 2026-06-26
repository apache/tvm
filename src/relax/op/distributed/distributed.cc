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
 * \file distributed.cc
 * \brief Redistribute operator.
 */

#include "distributed.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/attrs/ccl.h>
#include <tvm/topi/einsum.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../ccl/ccl.h"

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() { DistributionAttrs::RegisterReflection(); }

/* relax.dist.annotate_sharding */

Expr annotate_sharding(Expr input, distributed::DeviceMesh device_mesh,
                       distributed::Placement placement) {
  ffi::ObjectPtr<DistributionAttrs> attrs = ffi::make_object<DistributionAttrs>();
  attrs->device_mesh = device_mesh;
  attrs->placement = placement;

  static const Op& op = Op::Get("relax.dist.annotate_sharding");
  return Call(op, {std::move(input)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dist.annotate_sharding", annotate_sharding);
}

Type InferTypeAnnotateSharding(const Call& call, const BlockBuilder& ctx) {
  return GetType(call->args[0]);
}

TVM_REGISTER_OP("relax.dist.annotate_sharding")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeAnnotateSharding)
    .set_attr<FInferType>("dist.FInferType", InferTypeAnnotateSharding)
    .set_attr<bool>("FPurity", true);

/* relax.dist.redistribute */

Expr redistribute(Expr input, distributed::DeviceMesh device_mesh,
                  distributed::Placement placement) {
  ffi::ObjectPtr<DistributionAttrs> attrs = ffi::make_object<DistributionAttrs>();
  attrs->device_mesh = device_mesh;
  attrs->placement = placement;

  static const Op& op = Op::Get("relax.dist.redistribute");
  return Call(op, {std::move(input)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dist.redistribute", redistribute);
}

Type InferDistTypeRedistribute(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<DistributionAttrs>();
  const auto* ty = GetTypeAs<distributed::DTensorTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(ty);
  return distributed::DTensorType(ty->tensor_ty, attrs->device_mesh, attrs->placement);
}

TVM_REGISTER_OP("relax.dist.redistribute")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The input tensor.")
    .set_attr<FInferType>("dist.FInferType", InferDistTypeRedistribute)
    .set_attr<bool>("FPurity", true);

Type InferTypeCallTIRLocalView(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exactly 1 output type.";
  }
  TVM_FFI_ICHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "call_tir_local_view expects the first argument to be a GlobalVar referring to a TIR "
         "PrimFunc. "
      << "However, gets " << call->args[0];
  return call->ty_args[0];
}

TVM_REGISTER_OP("relax.dist.call_tir_local_view")
    .set_num_inputs(3)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferType>("FInferType", InferTypeCallTIRLocalView)
    .set_attr<bool>("FPurity", true);

Expr MakeCallTIRLocalView(Expr func, Tuple args, ffi::Array<distributed::DTensorType> out_ty_list,
                          ffi::Optional<Expr> packed_ints) {
  for (const distributed::DTensorType& ty : out_ty_list) {
    const auto* shape = ty->tensor_ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir_local_view should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op& op = Op::Get("relax.dist.call_tir_local_view");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_ty});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_ty});
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dist.call_tir_local_view", MakeCallTIRLocalView);
}

Type InferTypeRtoS(const Call& call, const BlockBuilder& ctx) {
  TensorType input_ty = GetUnaryInputTensorType(call, ctx);
  ffi::Optional<PrimType> output_dtype = input_ty->dtype;

  const auto* attrs = call->attrs.as<ScatterCollectiveAttrs>();
  int num_workers = attrs->num_workers;

  arith::Analyzer analyzer = ctx->GetAnalyzer();
  auto input_shape = input_ty->GetShape();
  TVM_FFI_ICHECK(input_shape.defined())
      << "input tensor of redistribute_replica_to_shard should have defined shape.";

  if (analyzer->CanProve(floormod(input_shape.value()[attrs->axis], PrimExpr(num_workers))) != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "redistribute_replica_to_shard expects the size of axis " << attrs->axis
        << " of input tensor to be "
           "divisible by the "
           "num_workers. However, the axis "
        << attrs->axis << " of input tensor is " << input_shape.value()[attrs->axis]
        << " while num_workers is " << num_workers;
  }

  ffi::Array<PrimExpr> output_shape = input_shape.value();
  output_shape.Set(attrs->axis, div(output_shape[attrs->axis], num_workers));
  return TensorType(ShapeExpr(output_shape), output_dtype, input_ty->vdevice);
}

Type InferDistTypeRtoS(const Call& call, const BlockBuilder& ctx) {
  using namespace distributed;
  ffi::Array<DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 1);
  DTensorType input_dtensor_ty = input_dtensor_tys[0];
  TensorType tensor_ty = input_dtensor_ty->tensor_ty;
  const auto* attrs = call->attrs.as<ScatterCollectiveAttrs>();
  int num_workers = attrs->num_workers;
  arith::Analyzer analyzer = ctx->GetAnalyzer();
  auto input_shape = tensor_ty->GetShape();
  TVM_FFI_ICHECK(input_shape.defined())
      << "input tensor of redistribute_replica_to_shard should have defined shape.";

  if (analyzer->CanProve(floormod(input_shape.value()[attrs->axis], PrimExpr(num_workers))) != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "redistribute_replica_to_shard expects the size of axis " << attrs->axis
        << " of input tensor to be "
           "divisible by the "
           "num_workers. However, the axis "
        << attrs->axis << " of input tensor is " << input_shape.value()[attrs->axis]
        << " while num_workers is " << num_workers;
  }

  DeviceMesh device_mesh = input_dtensor_ty->device_mesh;
  // FIXME: this is a hack where there's only 1d mesh
  TVM_FFI_ICHECK(device_mesh->shape.size() == 1);
  TVM_FFI_ICHECK(input_dtensor_ty->placement->dim_specs[0]->kind == PlacementSpecKind::kReplica);
  return DTensorType(tensor_ty, device_mesh,
                     Placement::FromText("S[" + std::to_string(attrs->axis) + "]"));
}

Expr redistribute_replica_to_shard(Expr input, int num_workers, int axis) {
  ffi::ObjectPtr<ScatterCollectiveAttrs> attrs = ffi::make_object<ScatterCollectiveAttrs>();
  attrs->num_workers = std::move(num_workers);
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("relax.dist.redistribute_replica_to_shard");

  return Call(op, {std::move(input)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.dist.redistribute_replica_to_shard",
                        redistribute_replica_to_shard);
}

TVM_REGISTER_OP("relax.dist.redistribute_replica_to_shard")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The buffer to be sliced.")
    .set_attrs_type<ScatterCollectiveAttrs>()
    .set_attr<FInferType>("FInferType", InferTypeRtoS)
    .set_attr<FInferType>("dist.FInferType", InferDistTypeRtoS)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
