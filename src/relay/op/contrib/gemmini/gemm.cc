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
 * \file src/relay/op/contrib/gemmini/gemm.cc
 * \brief GEMM operator definition for Gemmini.
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/relay/op.h>

#include "../../../qnn/utils.h"
#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace gemmini {

/*! \brief Attributes used by the Gemmini GEMM operator */
struct GemminiGEMMAttrs : public tvm::AttrsNode<GemminiGEMMAttrs> {
  Expr ifm_scale;
  Expr ifm_offset;
  Expr bias_scale;
  Expr bias_offset;
  Expr ofm_scale;
  Expr ofm_offset;

  TVM_DECLARE_ATTRS(GemminiGEMMAttrs, "relay.attrs.GemminiGEMMAttrs") {
    TVM_ATTR_FIELD(ifm_scale).describe("Data quantization scale");
    TVM_ATTR_FIELD(ifm_offset).describe("Data quantization offset");
    TVM_ATTR_FIELD(bias_scale).describe("Bias quantization scale");
    TVM_ATTR_FIELD(bias_offset).describe("Bias quantization offset");
    TVM_ATTR_FIELD(ofm_scale).describe("Output quantization scale");
    TVM_ATTR_FIELD(ofm_offset).describe("Output quantization offset");
  }
};

TVM_REGISTER_NODE_TYPE(GemminiGEMMAttrs);

bool GemminiGEMMRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  const int ifm1_index = 0;
  const int ifm2_index = 1;
  const int bias_index = 2;
  const int result_index = 3;

  const auto* ifm1 = types[ifm1_index].as<TensorTypeNode>();
  const auto* ifm2 = types[ifm2_index].as<TensorTypeNode>();
  const auto* bias = types[bias_index].as<TensorTypeNode>();
  if (ifm1 == nullptr) return false;
  if (ifm2 == nullptr) return false;
  if (bias == nullptr) return false;

  const auto* param = attrs.as<GemminiGEMMAttrs>();
  ICHECK(param != nullptr) << "GemminiGEMMAttrs cannot be nullptr.";

  DataType ofm_dtype = DataType::Int(8);

  // Assign ofm type
  Array<IndexExpr> ofm_shape({ifm1->shape[0], ifm2->shape[1]});
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeGemminiGEMM(Expr data, Expr weights, Expr bias, Expr ifm_scale, Expr ifm_offset,
                     Expr bias_scale, Expr bias_offset, Expr ofm_scale, Expr ofm_offset) {
  auto attrs = make_object<GemminiGEMMAttrs>();
  attrs->ifm_scale = std::move(ifm_scale);
  attrs->ifm_offset = std::move(ifm_offset);
  attrs->bias_scale = std::move(bias_scale);
  attrs->bias_offset = std::move(bias_offset);
  attrs->ofm_scale = std::move(ofm_scale);
  attrs->ofm_offset = std::move(ofm_offset);

  static const Op& op = Op::Get("contrib.gemmini.gemm");

  auto weights_transposed = MakeTranspose(weights, {1, 0});
  auto reduced_t3 = Sum(Cast(weights_transposed, DataType::Int(32)), {0}, false, false);
  auto term3 = Multiply(attrs->ifm_offset, reduced_t3);

  auto scale = Divide(attrs->bias_scale, attrs->ofm_scale);
  auto bias_fix = Divide(Cast(attrs->ofm_offset, DataType::Float(32)), scale);

  auto new_bias = Add(Subtract(bias, term3), Cast(bias_fix, DataType::Int(32)));

  auto gemm_output = Call(op, {data, weights_transposed, new_bias}, Attrs(attrs), {});
  return gemm_output;
}

TVM_REGISTER_GLOBAL("relay.op._make.gemmini_gemm").set_body_typed(MakeGemminiGEMM);

RELAY_REGISTER_OP("contrib.gemmini.gemm")
    .describe("Gemmini GEMM operator")
    .set_attrs_type<GemminiGEMMAttrs>()
    .set_num_inputs(3)
    .add_argument("ifm1", "Tensor", "The Input Feature Map tensor.")
    .add_argument("ifm2", "Tensor", "The Weights tensor.")
    .add_argument("bias", "Tensor", "The bias tensor")
    .set_support_level(11)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("GemminiGEMM", GemminiGEMMRel);

}  // namespace gemmini
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
