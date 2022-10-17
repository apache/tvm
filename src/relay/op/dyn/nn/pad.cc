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
 * \file pad.cc
 * \brief Implementation of dynamic pad
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/op.h>
#include <tvm/topi/nn.h>

#include <vector>

#include "../../make_op.h"
#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace dyn {

// relay.dyn.nn.pad

bool PadRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
            const TypeReporter& reporter) {
  // types = [data_type, pad_width_type, pad_value_type, ret_type]
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const auto* pad_width = types[1].as<TensorTypeNode>();
  if (pad_width == nullptr) return false;

  const auto* pad_value = types[2].as<TensorTypeNode>();
  if (pad_value == nullptr) return false;

  int data_rank = data->shape.size();
  ICHECK(data_rank) << "Data shape must have static rank";

  int pad_width_rank = pad_width->shape.size();
  ICHECK_EQ(pad_width_rank, 2) << "Pad width must be 2D";

  const PadAttrs* param = attrs.as<PadAttrs>();
  ICHECK(param != nullptr);

  std::vector<IndexExpr> oshape;
  for (int i = 0; i < data_rank; i++) {
    oshape.push_back(Any());
  }

  reporter->Assign(types[3], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> PadCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  const auto* param = attrs.as<PadAttrs>();
  ICHECK(param);

  auto data = inputs[0];
  auto pad_width = inputs[1];

  te::Tensor cast_pad_value = topi::cast(inputs[2], inputs[0]->dtype);
  const PrimExpr& pad_value = cast_pad_value(Array<PrimExpr>());

  Array<IndexExpr> pad_before;
  Array<IndexExpr> pad_after;

  for (int i = 0; i < pad_width->shape[0].as<IntImmNode>()->value; ++i) {
    pad_before.push_back(pad_width[i][0]);
    pad_after.push_back(pad_width[i][1]);
  }

  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype != nullptr);

  return Array<te::Tensor>{topi::pad(inputs[0], pad_before, pad_after, pad_value, "T_pad",
                                     topi::kElementWise, param->pad_mode,
                                     &out_type.as<TensorTypeNode>()->shape)};
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakePad(Expr data, Expr pad_width, Expr pad_value, String pad_mode) {
  auto attrs = make_object<PadAttrs>();
  attrs->pad_mode = std::move(pad_mode);
  static const Op& op = Op::Get("dyn.nn.pad");
  return Call(op, {data, pad_width, pad_value}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.nn._make.pad").set_body_typed(MakePad);

RELAY_REGISTER_OP("dyn.nn.pad")
    .describe(R"code(Pad for n-D tensor.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<PadAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Tensor that will be padded")
    .add_argument("pad_width", "Tensor", "Tensor of how much to pad by")
    .add_argument("pad_val", "double", "The value to fill the padded area with")
    .set_support_level(2)
    .add_type_rel("DynamicPad", PadRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FTVMCompute>("FTVMCompute", PadCompute);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
