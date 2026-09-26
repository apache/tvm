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

#include "unary.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeUnaryCheck(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeUnary<false>(call, ctx,
                                   [](const TensorType& input_ty) { return PrimType::Bool(); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.abs")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.acos")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.acosh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.asin")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.asinh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.atan")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.atanh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.bitwise_not")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.ceil")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.cos")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.cosh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.exp")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.floor")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.log")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.logical_not")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.negative")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.round")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.rsqrt")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.sigmoid")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.sign")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.sin")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.sinh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.square")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<false>);

  OpDef("relax.sqrt")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.tan")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.tanh")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.erf")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryArith<true>);

  OpDef("relax.isfinite")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryCheck);

  OpDef("relax.isinf")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryCheck);

  OpDef("relax.isnan")
      .set_attr<FInferType>("dist.FInferType", InferDistTypeUnaryCheck);
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
