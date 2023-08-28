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

Expr allreduce(Expr x, String op_type) {
  ObjectPtr<AllReduceAttrs> attrs = make_object<AllReduceAttrs>();
  attrs->op_type = std::move(op_type);

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

}  // namespace relax
}  // namespace tvm
