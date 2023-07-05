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
 * \file correlation.cc
 * \brief Correlation operators
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/op.h>
#include <tvm/topi/nn.h>

#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.nn.correlation
TVM_REGISTER_NODE_TYPE(CorrelationAttrs);

InferCorrectLayoutOutput CorrelationInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* params = attrs.as<CorrelationAttrs>();
  Layout layout{params->layout};
  return InferCorrectLayoutOutput({layout, layout}, {layout}, attrs);
}

// Positional relay function to create correlation operator
// used by frontend FFI.
Expr MakeCorrelation(Expr data1, Expr data2, int kernel_size, int max_displacement, int stride1,
                     int stride2, Array<IndexExpr> padding, bool is_multiply, String layout) {
  auto attrs = make_object<CorrelationAttrs>();
  attrs->kernel_size = kernel_size;
  attrs->max_displacement = max_displacement;
  attrs->stride1 = stride1;
  attrs->stride2 = stride2;
  attrs->padding = std::move(padding);
  attrs->is_multiply = is_multiply;
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.correlation");
  return Call(op, {data1, data2}, Attrs(attrs), {});
}

bool CorrelationRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data1 = types[0].as<TensorTypeNode>();
  const auto* data2 = types[1].as<TensorTypeNode>();
  if (data1 == nullptr || data2 == nullptr) return false;

  const CorrelationAttrs* param = attrs.as<CorrelationAttrs>();
  ICHECK(param != nullptr);
  ICHECK_EQ(param->layout, "NCHW") << "layout not supported.";
  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  IndexExpr padded_height = data1->shape[2] + pad_h;
  IndexExpr padded_width = data2->shape[3] + pad_w;
  int kernel_radius = (param->kernel_size - 1) / 2;
  int border_size = param->max_displacement + kernel_radius;
  int displacement_radius = param->max_displacement / param->stride2;
  int displacement_size = 2 * displacement_radius + 1;
  int out_channel = displacement_size * displacement_size;
  IndexExpr out_height =
      indexdiv((padded_height - 2 * border_size + param->stride1 - 1), param->stride1);
  IndexExpr out_width =
      indexdiv((padded_width - 2 * border_size + param->stride1 - 1), param->stride1);
  Array<tvm::PrimExpr> oshape{data1->shape[0], out_channel, out_height, out_width};
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, data1->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.correlation").set_body_typed(MakeCorrelation);

RELAY_REGISTER_OP("nn.correlation")
    .describe(R"code(Applies correlation to inputs.

The correlation layer performs multiplicative patch comparisons between two feature maps.
Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,
the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.

For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and
:math:`x_{2}` in the second map is then defined as:

.. math::
   c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>

for a square patch of size :math:`K:=2k+1`.

Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other
data. For this reason, it has no training weights.

Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.

Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
centered around :math:`x_{1}`.

The final output is defined by the following expression:

.. math::
  out[n, q, i, j] = c(x_{i, j}, x_{q})

where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<CorrelationAttrs>()
    .set_num_inputs(2)
    .add_argument("data1", "Tensor", "Input data1 to the correlation.")
    .add_argument("data2", "Tensor", "Input data2 to the correlation.")
    .set_support_level(2)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", CorrelationInferCorrectLayout)
    .add_type_rel("Correlation", CorrelationRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

}  // namespace relay
}  // namespace tvm
