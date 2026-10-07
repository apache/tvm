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

#include <tvm/ir/prim/op.h>
#include <tvm/te/operation.h>

namespace tvm::prim {
using s_tir::IterVar;
PrimExpr sum(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = prim::Add(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), 0, span);
  te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr all(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  TVM_FFI_ICHECK(source.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument for tvm::all, but received " << source << " of type "
      << source.ty();
  PrimVar x("x", source.ty(), span), y("y", source.ty());
  PrimExpr result = prim::And(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), true, span);
  te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr any(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  TVM_FFI_ICHECK(source.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument for tvm::any, but received " << source << " of type "
      << source.ty();
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = prim::Or(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), false, span);
  te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

}  // namespace tvm::prim

namespace tvm {
PrimExpr max(PrimExpr source, ffi::Array<s_tir::IterVar> rdom, ffi::Array<PrimExpr> init,
             Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = prim::Max(x, y, span);
  PrimExpr identity_element = prim::min_value(source.ty(), span);
  te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr min(PrimExpr source, ffi::Array<s_tir::IterVar> rdom, ffi::Array<PrimExpr> init,
             Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = prim::Min(x, y, span);
  PrimExpr identity_element = prim::max_value(source.ty(), span);
  te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

}  // namespace tvm

namespace tvm::prim {
PrimExpr prod(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  if (source.ty().MatchesCode(DLDataTypeCode::kDLBool)) {
    // Bool product (prod) has the same truth table as logical AND.  Reuse all() to
    // avoid lowering bool prod through Mul, which LLVM codegen does not support.
    return all(source, rdom, init, span);
  } else {
    // For non-bool types, we lower prod through Mul.
    PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
    PrimExpr result = prim::Mul(x, y, span);
    PrimExpr identity_element = MakeConst(source.ty(), 1, span);
    te::CommReducer combiner = te::CommReducer({x}, {y}, {result}, {identity_element}, span);
    return te::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
  }
}

}  // namespace tvm::prim
