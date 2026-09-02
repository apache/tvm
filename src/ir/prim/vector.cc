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

/*! \file vector.cc */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op.h>

#include <optional>

namespace tvm {
namespace prim {
namespace {
// File-local helper: returns the vscale multiplier if `lanes` is of the form
// `multiplier * vscale()` or `vscale() * multiplier`, nullopt otherwise.
std::optional<int> ExtractVscaleFactor(const PrimExpr& lanes) {
  auto is_vscale = [](const PrimExpr& e) -> bool {
    if (const auto* call = e.as<CallNode>()) {
      return call->op.same_as(prim::builtin::vscale());
    }
    return false;
  };
  if (const auto* mul = lanes.as<MulNode>()) {
    if (const auto* imm = mul->a.as<IntImmNode>(); imm && is_vscale(mul->b)) {
      return static_cast<int>(imm->value);
    }
    if (const auto* imm = mul->b.as<IntImmNode>(); imm && is_vscale(mul->a)) {
      return static_cast<int>(imm->value);
    }
  }
  return std::nullopt;
}
}  // namespace
// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(base.defined());
  TVM_FFI_ICHECK(stride.defined());
  PrimType base_ty = base.ty();
  PrimType stride_ty = stride.ty();
  TVM_FFI_ICHECK(base_ty.IsScalar());
  TVM_FFI_ICHECK(stride_ty.IsScalar());
  if (stride_ty != base_ty) {
    stride = cast(base_ty, stride);
  }

  ffi::ObjectPtr<RampNode> node = ffi::make_object<RampNode>();
  auto* lanes_as_int = lanes.as<IntImmNode>();
  if (lanes_as_int) {
    int lanes = static_cast<int>(lanes_as_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = base_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(base_ty.code(), base_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->base = base;
  node->stride = stride;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.prim.Ramp", [](PrimExpr base, PrimExpr stride, PrimExpr lanes,
                                           Span span) { return Ramp(base, stride, lanes, span); });
}

// Broadcast
Broadcast::Broadcast(PrimExpr value, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_ty = value.ty();
  TVM_FFI_ICHECK(value_ty.IsScalar());

  ffi::ObjectPtr<BroadcastNode> node = ffi::make_object<BroadcastNode>();
  auto* lanes_int = lanes.as<IntImmNode>();
  if (lanes_int) {
    int lanes = static_cast<int>(lanes_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = value_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(value_ty.code(), value_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = node;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.prim.Broadcast", [](PrimExpr value, PrimExpr lanes, Span span) {
    return Broadcast(value, lanes, span);
  });
}

// Shuffle
Shuffle::Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0U);
  TVM_FFI_ICHECK_NE(indices.size(), 0U);

  PrimType base_type = vectors[0].ty().WithLanes(1);
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    PrimType val_ty = val.ty();
    TVM_FFI_ICHECK(val_ty.WithLanes(1)->dtype == base_type->dtype);
    total_lanes += val_ty.lanes();
  }
  TVM_FFI_ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ffi::ObjectPtr<ShuffleNode> node = ffi::make_object<ShuffleNode>();
  node->ExprNode::ty = base_type.WithLanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

PrimExpr Shuffle::Concat(ffi::Array<PrimExpr> vectors, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  ffi::Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.ty().lanes(); ++i) {
      indices.push_back(IntImm::Int32(index++));
    }
  }
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {IntImm::Int32(index)}, span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.prim.Shuffle",
                        [](ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
                          return Shuffle(vectors, indices, span);
                        });
}

}  // namespace prim
}  // namespace tvm
