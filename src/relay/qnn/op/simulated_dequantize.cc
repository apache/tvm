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
 * \file src/relay/qnn/op/simulated_dequantize.cc
 * \brief QNN simulated dequantize operator. Mimics the behavior
 * of QNN dequantize in floating point with added flexibility.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

bool SimulatedDequantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                            const TypeReporter& reporter) {
  // types = [data_type, datatype_type, scale_type, zp_type, ret_type]
  ICHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* dtype = types[1].as<TensorTypeNode>();

  if ((data == nullptr) || (dtype == nullptr)) {
    return false;
  }

  // assign output type
  reporter->Assign(types[4], TensorType(data->shape, data->dtype));
  return true;
}

Expr MakeSimulatedDequantize(Expr data, Expr in_dtype, Expr input_scale, Expr input_zero_point,
                             int axis) {
  auto attrs = make_object<DequantizeAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("qnn.simulated_dequantize");
  return Call(op, {data, in_dtype, input_scale, input_zero_point}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.simulated_dequantize")
    .describe(R"code(Simulates the functionality of qnn.dequantize but allows more flexible
    dynamic input type conversion and always operates on float values.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<DequantizeAttrs>()
    .set_num_inputs(4)
    .add_argument("data", "Tensor", "The tensor to dequantize.")
    .add_argument("in_dtype", "Tensor",
                  "A code corresponding to the type of quantization to convert from.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_support_level(11)
    .add_type_rel("QNNSimulatedDequantize", SimulatedDequantizeRel);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.simulated_dequantize")
    .set_body_typed(MakeSimulatedDequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
