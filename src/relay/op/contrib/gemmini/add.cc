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
 * \file src/relay/op/contrib/gemmini/add.cc
 * \brief Add operator definition for Gemmini.
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/relay/op.h>

#include "../../../qnn/op/op_common.h"
#include "../../../qnn/utils.h"
#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace gemmini {

/*! \brief Attributes used by the Gemmini Add operators */
struct GemminiAddAttrs : public tvm::AttrsNode<GemminiAddAttrs> {
  Expr ifm1_scale;
  Expr ifm1_offset;
  Expr ifm2_scale;
  Expr ifm2_offset;
  Expr ofm_scale;
  Expr ofm_offset;
  Array<PrimExpr> shape;

  TVM_DECLARE_ATTRS(GemminiAddAttrs, "relay.attrs.GemminiAddAttrs") {
    TVM_ATTR_FIELD(ifm1_scale).describe("Input feature map 1 quantization scale");
    TVM_ATTR_FIELD(ifm1_offset).describe("Input feature map 1 quantization offset");
    TVM_ATTR_FIELD(ifm2_scale).describe("Input feature map 2 quantization scale");
    TVM_ATTR_FIELD(ifm2_offset).describe("Input feature map 2 quantization offset");
    TVM_ATTR_FIELD(ofm_scale).describe("Output feature map quantization scale");
    TVM_ATTR_FIELD(ofm_offset).describe("Output feature map quantization offset");
    TVM_ATTR_FIELD(shape).describe("Output shape");
  }
};

TVM_REGISTER_NODE_TYPE(GemminiAddAttrs);

bool GemminiAddRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  const int ifm1_index = 0;
  const int ifm2_index = 1;
  const int result_index = 3;
  ICHECK_EQ(types.size(), result_index + 1);

  const auto* ifm1 = types[ifm1_index].as<TensorTypeNode>();
  const auto* ifm2 = types[ifm2_index].as<TensorTypeNode>();
  ICHECK(ifm1 != nullptr) << "ifm1 cannot be nullptr.";
  ICHECK(ifm2 != nullptr) << "ifm2 cannot be nullptr.";

  const auto* param = attrs.as<GemminiAddAttrs>();
  ICHECK(param != nullptr) << "GemminiAddAttrs cannot be nullptr.";

  DataType ofm_dtype = DataType::Int(8);

  // Assign ofm type
  Array<IndexExpr> ofm_shape({ifm1->shape[0], ifm2->shape[1], ifm2->shape[2], ifm2->shape[3]});
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeGemminiAdd(Expr ifm1, Expr ifm2, Expr ifm1_scale, Expr ifm1_offset, Expr ifm2_scale,
                    Expr ifm2_offset, Expr ofm_scale, Expr ofm_offset, Array<PrimExpr> shape) {
  auto attrs = make_object<GemminiAddAttrs>();
  attrs->ifm1_scale = std::move(ifm1_scale);
  attrs->ifm1_offset = std::move(ifm1_offset);
  attrs->ifm2_scale = std::move(ifm2_scale);
  attrs->ifm2_offset = std::move(ifm2_offset);
  attrs->ofm_scale = std::move(ofm_scale);
  attrs->ofm_offset = std::move(ofm_offset);
  attrs->shape = std::move(shape);

  static const Op& op = Op::Get("contrib.gemmini.add");

  auto requantized_ifm1 = ifm1;

  auto requantized_ifm2 = ifm2;

  auto ofm_offset_tensor = Full(attrs->ofm_offset, attrs->shape, DataType::Float(32));
  auto ifm1_offset_tensor = Multiply(Divide(attrs->ifm1_scale, attrs->ofm_scale),
                                     Cast(attrs->ifm1_offset, DataType::Float(32)));
  auto ifm2_offset_tensor = Multiply(Divide(attrs->ifm2_scale, attrs->ofm_scale),
                                     Cast(attrs->ifm2_offset, DataType::Float(32)));
  ofm_offset_tensor = Subtract(Subtract(ofm_offset_tensor, ifm1_offset_tensor), ifm2_offset_tensor);

  auto final_offset_tensor = tvm::relay::qnn::RequantizeOrUpcast(
      ofm_offset_tensor, MakeConstantScalar(DataType::Float(32), 1),
      MakeConstantScalar(DataType::Float(32), 0), MakeConstantScalar(DataType::Float(32), 1),
      MakeConstantScalar(DataType::Float(32), 0), attrs->shape, -1);

  auto add_output =
      Call(op, {requantized_ifm1, requantized_ifm2, final_offset_tensor}, Attrs(attrs), {});
  return add_output;
}

TVM_REGISTER_GLOBAL("relay.op._make.gemmini_add").set_body_typed(MakeGemminiAdd);

RELAY_REGISTER_OP("contrib.gemmini.add")
    .describe("Gemmini Add operator.")
    .set_attrs_type<GemminiAddAttrs>()
    .set_num_inputs(3)
    .add_argument("ifm1", "Tensor", "The Input 1 Feature Map tensor.")
    .add_argument("ifm2", "Tensor", "The Input 2 Feature Map tensor.")
    .add_argument("ofm_offset_tensor", "Tensor", "The output offset tensor.")
    .set_support_level(11)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("GemminiAdd", GemminiAddRel);

}  // namespace gemmini
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
