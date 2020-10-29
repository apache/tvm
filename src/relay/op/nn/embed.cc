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
 * \file embed.cc
 * \brief Property def of nn.embed operator.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(EmbedAttrs);

bool EmbedRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3) << "Embed shape relation takes three arguments: the embedding table, "
                                "the indices, and the output";
  const auto* table = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  ICHECK_EQ(table->shape.size(), 2) << "Embed table must be a dimension 2 tensor.";
  ICHECK_EQ(indices->shape.size(), 1) << "Embed indices must be a one dimensional vector.";
  ICHECK(indices->dtype.is_int() || indices->dtype.is_uint()) << "Embed indices must be integers.";

  reporter->Assign(types[2], TensorType({indices->shape[0], table->shape[1]}, table->dtype));
  return true;
}

Expr MakeEmbed(Expr table, Expr indices) {
  auto attrs = make_object<EmbedAttrs>();
  static const Op& op = Op::Get("nn.embed");
  return Call(op, {table, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.embed").set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 2>(MakeEmbed, args, rv);
});

RELAY_REGISTER_OP("nn.embed")
    .describe(R"code(Lookup of indices in an embedding table.

- **table**: M x N tensor
- **indices**: K long tensor of indices into `table`
- **out**: K x N tensor

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EmbedAttrs>()
    .set_num_inputs(2)
    .add_argument("table", "2D Tensor", "Embedding table.")
    .add_argument("indices", "1D Tensor", "Indices to lookup.")
    .set_support_level(1)
    .add_type_rel("Embed", EmbedRel);

TVM_REGISTER_NODE_TYPE(EmbedGradAttrs);

bool EmbedGradRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4) << "EmbedGrad shape relation takes four arguments: the embedding "
                                "table, the indices, the gradient, and the output";
  const auto* table = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  const auto* grad = types[2].as<TensorTypeNode>();
  ICHECK_EQ(table->shape.size(), 2) << "EmbedGrad table must be a dimension 2 tensor.";
  ICHECK_EQ(indices->shape.size(), 1) << "EmbedGrad indices must be a one dimensional vector.";
  ICHECK_EQ(grad->shape.size(), 2) << "EmbedGrad grad must be a dimension 2 tensor.";
  ICHECK(indices->dtype.is_int() || indices->dtype.is_uint())
      << "EmbedGrad indices must be integers.";
  reporter->AssertEQ(table->shape[0], grad->shape[0]);
  reporter->AssertEQ(table->shape[1], grad->shape[1]);

  reporter->Assign(types[3], TensorType(table->shape, table->dtype));
  return true;
}

Expr MakeEmbedGrad(Expr table, Expr indices, Expr grad) {
  auto attrs = make_object<EmbedGradAttrs>();
  static const Op& op = Op::Get("nn.embed_grad");
  return Call(op, {table, indices, grad}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.embed_grad")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 3>(MakeEmbedGrad, args, rv);
    });

RELAY_REGISTER_OP("nn.embed_grad")
    .describe(R"code(Gradient of Embed

- **table**: M x N tensor
- **indices**: K long tensor of indices into `table`
- **grad**: K x N tensor of the gradient
- **out**: M x N tensor

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EmbedGradAttrs>()
    .set_num_inputs(3)
    .add_argument("table", "2D Tensor", "EmbedGradding table.")
    .add_argument("indices", "1D Tensor", "Indices to lookup.")
    .add_argument("grad", "2D Tensor", "Gradient.")
    .set_support_level(1)
    .add_type_rel("EmbedGrad", EmbedGradRel);

}  // namespace relay
}  // namespace tvm
