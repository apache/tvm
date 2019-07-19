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
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of qauntized nn operators.
 */

#include <tvm/relay/qnn/attrs.h>
#include "../../../op/nn/nn.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.qnn.dense
TVM_REGISTER_NODE_TYPE(QDenseAttrs);

// Positional relay function to create quantized dense operator used by frontend FFI.
Expr MakeQuantizedDense(Expr data,
                        Expr weight,
                        IndexExpr units,
                        int32_t input_zero_point,
                        int32_t kernel_zero_point,
                        DataType out_dtype) {
  auto attrs = make_node<QDenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  attrs->input_zero_point = input_zero_point;
  attrs->kernel_zero_point = kernel_zero_point;
  static const Op& op = Op::Get("qnn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.qnn.op._make.dense")
.set_body_typed(MakeQuantizedDense);

RELAY_REGISTER_OP("qnn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.QDenseAttrs")
.set_num_inputs(2)
.add_argument("data", "quantized nD Tensor", "Input data.")
.add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
.set_support_level(10)
.add_type_rel("QDense", DenseRel<QDenseAttrs, DenseType::kQuantizedDense>);

} // namespace qnn
} // namespace relay
} // namespace tvm
