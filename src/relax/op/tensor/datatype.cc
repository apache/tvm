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
 * \file datatype.cc
 * \brief Datatype operators.
 */

#include "datatype.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.astype */
TVM_REGISTER_NODE_TYPE(AstypeAttrs);

Expr astype(Expr x, DataType dtype) {
  ObjectPtr<AstypeAttrs> attrs = make_object<AstypeAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.astype");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.astype").set_body_typed(astype);

StructInfo InferStructInfoAstype(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<AstypeAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.astype")
    .set_attrs_type<AstypeAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAstype);

}  // namespace relax
}  // namespace tvm
