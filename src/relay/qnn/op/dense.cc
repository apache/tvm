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
 *  Copyright (c) 2019 by Contributors
 * \file src/relay/qnn/op/dense.cc
 * \brief Property def of qnn dense operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../op/nn/nn.h"
#include "../../pass/pattern_util.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.dense
TVM_REGISTER_NODE_TYPE(QnnDenseAttrs);

bool QnnDenseRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<QnnDenseAttrs>();
  CHECK(param != nullptr) << "QnnConv2DAttrs cannot be nullptr.";
  CHECK(data->dtype == Int(8) || data->dtype == UInt(8))
    << "Expected quantized dense type(int8, uint8) for input but was " <<  data->dtype;
  CHECK(weight->dtype == Int(8) || weight->dtype == UInt(8))
    << "Expected quantized dense type(int8, uint8) for weight but was " <<  weight->dtype;
  CHECK(param->out_dtype == Int(32))
    << "Expected quantized dense type(int32) for output but was " <<  param->out_dtype;
  CHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";
  return DenseRel<QnnDenseAttrs>(types, num_inputs, attrs, reporter);
}

// Positional relay function to create quantized dense operator used by frontend FFI.
Expr MakeQuantizedDense(Expr data,
                        Expr weight,
                        IndexExpr units,
                        int32_t input_zero_point,
                        int32_t kernel_zero_point,
                        DataType out_dtype) {
  auto attrs = make_node<QnnDenseAttrs>();
  attrs->units = std::move(units);
  attrs->out_dtype = out_dtype;
  attrs->input_zero_point = input_zero_point;
  attrs->kernel_zero_point = kernel_zero_point;
  static const Op& op = Op::Get("qnn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

/**
 * \brief Lowers Qnn convolution in terms of core operators in relay.
 * Mathematically it is equals to -
 * Dense((quantized_input - input_zero_point;int32), (quantized_kernel - kernel_zero_point; int32))
 *
 * \param attrs QnnDenseAttrs for Qnn Dense layer.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The data types of input and output.
 * \reutrn The sequence of Relay ops for qnn cov2d op.
 */
Expr QnnDenseCanonicalize(const Attrs& attrs,
                          const Array<Expr>& new_args,
                          const Array<tvm::relay::Type>& arg_types) {
  CHECK_EQ(new_args.size(), 2);
  Expr quantized_data = new_args[0];
  Expr quantized_kernel = new_args[1];
  const auto* qnn_dense_attrs = attrs.as<QnnDenseAttrs>();
  Expr quantized_data_int32 = Cast(quantized_data, Int(32));
  if (qnn_dense_attrs->input_zero_point != 0) {
    quantized_data_int32 = Subtract(quantized_data_int32,
                                    MakeConstantScalar(Int(32),
                                    qnn_dense_attrs->input_zero_point));
  }
  Expr quantized_kernel_int32 = Cast(quantized_kernel, Int(32));
  if (qnn_dense_attrs->kernel_zero_point != 0) {
    quantized_kernel_int32 = Subtract(quantized_kernel_int32,
                                      MakeConstantScalar(Int(32),
                                      qnn_dense_attrs->kernel_zero_point));
  }
  Expr int32_dense = Dense(quantized_data_int32,
                           quantized_kernel_int32,
                           qnn_dense_attrs->units,
                           qnn_dense_attrs->out_dtype);
  return int32_dense;
}

RELAY_REGISTER_OP("qnn.dense")
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.
- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.qnn.QnnDenseAttrs")
.set_num_inputs(2)
.add_argument("data", "quantized nD Tensor", "Input data.")
.add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
.set_support_level(11)
.add_type_rel("QDense", DenseRel<QnnDenseAttrs>)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnDenseCanonicalize);

TVM_REGISTER_API("relay.qnn.op._make.dense")
.set_body_typed(MakeQuantizedDense);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
