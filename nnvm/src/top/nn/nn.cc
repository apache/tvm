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
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <tvm/operation.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/layout.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/dense.h"
#include "topi/nn.h"
#include "topi/nn/softmax.h"

namespace nnvm {
namespace top {

using tvm::Var;
using tvm::Expr;
using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;

// dense
DMLC_REGISTER_PARAMETER(DenseParam);

inline bool DenseInferShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  // reverse infer
  if ((*out_shape)[0].ndim() != 0) {
    TShape dshape = (*out_shape)[0];
    dshape[dshape.ndim() - 1] = 0;
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kData, dshape);
  }
  dim_t num_inputs = 0;
  if ((*in_shape)[DenseParam::kData].ndim() != 0) {
    TShape oshape = (*in_shape)[DenseParam::kData];
    num_inputs = oshape[oshape.ndim() - 1];
    oshape[oshape.ndim() - 1] = param.units;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kWeight,
                          TShape({param.units, num_inputs}));
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kBias, TShape({param.units}));
  }
  return true;
}

NNVM_REGISTER_OP(dense)
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, units)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(DenseParam::__FIELDS__())
.set_attr_parser(ParamParser<DenseParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<DenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<DenseParam>)
.set_attr<FInferShape>("FInferShape", DenseInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
// leave weight & bias layout undefined
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    const DenseParam& param = nnvm::get<DenseParam>(n->attrs.parsed);

    NodeEntry data_grad = MakeNode("matmul",
                                   n->attrs.name + "_data_grad",
                                   {ograds[0], n->inputs[DenseParam::kWeight]});
    NodeEntry w_grad_sub = MakeNode("matmul",
                                     n->attrs.name + "_weight_grad_sub0",
                                     {ograds[0], n->inputs[DenseParam::kData]},
                                     {{"transpose_a", "true"}});
    TShape w_reduce_axis = {0, -1};
    std::ostringstream w_oss; w_oss << w_reduce_axis;
    NodeEntry w_grad = MakeNode("sum", n->attrs.name + "_weight_grad",
                                {w_grad_sub},
                                {{"axis", w_oss.str()}, {"exclude", "true"}});
    std::vector<NodeEntry> grads = {data_grad, w_grad};

    if (param.use_bias) {
      TShape axis = {-1};
      std::ostringstream b_oss; b_oss << axis;
      grads.push_back(MakeNode("sum", n->attrs.name + "_bias_grad",
                      {ograds[0]},
                      {{"axis", b_oss.str()}, {"exclude", "true"}}));
    }
    return grads;
})
.set_support_level(1);

// relu
NNVM_REGISTER_ELEMWISE_UNARY_OP(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(input, 0)

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::relu(inputs[0], 0.0f) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = relu(x)
    // grad = indicator(x > 0) * ograd
    NodeEntry sub0 = MakeNode("zeros_like", n->attrs.name + "_sub0",
                              {n->inputs[0]});
    NodeEntry sub1 = MakeNode("greater", n->attrs.name + "_sub1",
                              {n->inputs[0], sub0}, {{"exclude", "true"}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad",
               {ograds[0], sub1})
    };
})
.set_support_level(1);

// dropout
DMLC_REGISTER_PARAMETER(DropoutParam);

NNVM_REGISTER_OP(dropout)
.describe(R"(Applies dropout operation to input array.

- During training, each element of the input is set to zero with probability p.
  The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  sum of the input unchanged.

)" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input to which dropout will be applied")
.add_arguments(DropoutParam::__FIELDS__())
.set_attr_parser(ParamParser<DropoutParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DropoutParam>)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 2>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_attr<FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
    return 1;
  })
.set_attr<FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "mask"};
  })
.set_support_level(1);

// batchnorm
DMLC_REGISTER_PARAMETER(BatchNormParam);

inline bool BatchNormInferShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape>* in_shape,
                                std::vector<TShape>* out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 5U)
      << "Input:[data, gamma, beta, moving_mean, moving_var]";
  CHECK_EQ(out_shape->size(), 3U);
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  CHECK((size_t)param.axis < dshape.Size());

  TShape bshape({dshape[param.axis]});
  if (in_shape->at(1).ndim() == 0) in_shape->at(1) = bshape;
  if (in_shape->at(2).ndim() == 0) in_shape->at(2) = bshape;
  if (in_shape->at(3).ndim() == 0) in_shape->at(3) = bshape;
  if (in_shape->at(4).ndim() == 0) in_shape->at(4) = bshape;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  out_shape->at(1) = in_shape->at(3);
  out_shape->at(2) = in_shape->at(4);
  return true;
}

inline bool BatchNormCorrectLayout(const NodeAttrs& attrs,
                                   std::vector<Layout> *in_layouts,
                                   const std::vector<Layout> *last_in_layouts,
                                   std::vector<Layout> *out_layouts) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  CHECK_EQ(in_layouts->size(), 5U);
  CHECK_EQ(last_in_layouts->size(), 5U);
  CHECK_EQ(out_layouts->size(), 3U);

  Layout data_layout = in_layouts->at(0);
  const Layout& origin_data_layout = last_in_layouts->at(0);
  Layout param_layout("C");
  if (data_layout.defined()) {
    if (data_layout.indexof('C') != param.axis) {
      CHECK(origin_data_layout.defined())
        << "Channel in data layout " << data_layout
        << " is not at index " << param.axis;
      // convert it to the original one.
      data_layout = origin_data_layout;
      NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
    } else if (data_layout.indexof('c') >= 0 &&
               static_cast<uint32_t>(data_layout.indexof('c')) != (data_layout.ndim()-1)) {
      CHECK(origin_data_layout.defined())
        << "sub-channel c in data layout " << data_layout
        << " does not at the final dimension";
      // convert it to the original one.
      data_layout = origin_data_layout;
      NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
    } else {
      for (Layout::LayoutDim axis : data_layout) {
        if (Layout::is_subdim(axis) && axis != 'c') {
          CHECK(origin_data_layout.defined())
            << "sub-axis other than c appears in data layout " << data_layout;
          // convert it to the original one.
          data_layout = origin_data_layout;
          NNVM_ASSIGN_LAYOUT(*in_layouts, 0, origin_data_layout);
          break;
        }
      }
    }

    // decide the param layout
    if (data_layout.defined()) {
      auto channel_block = data_layout.subsizeof('C');
      if (channel_block > 0) {
        param_layout = param_layout.split('C', 1, channel_block);
      }
    }
  }

  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 1, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 2, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 3, param_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 4, param_layout);

  NNVM_ASSIGN_LAYOUT(*out_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*out_layouts, 1, param_layout);
  NNVM_ASSIGN_LAYOUT(*out_layouts, 2, param_layout);
  return true;
}

NNVM_REGISTER_OP(batch_norm)
.describe(R"(Batch normalization layer (Ioffe and Szegedy, 2014).
Normalizes the input at each batch, i.e. applies a transformation
that maintains the mean activation close to 0 and the activation
standard deviation close to 1.

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta`` have shape *(k,)*.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

.. note::
    This operator can be optimized away for inference.
)" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input to which dropout will be applied")
.add_argument("gamma", "Tensor", "The gamma scale factor")
.add_argument("beta", "Tensor", "The beta offset factor")
.add_argument("moving_mean", "Tensor", "running mean of input")
.add_argument("moving_var", "Tensor", "running variance of input")
.add_arguments(BatchNormParam::__FIELDS__())
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BatchNormParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", BatchNormCorrectLayout)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<FInferShape>("FInferShape", BatchNormInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<5, 3>)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
  })
.set_attr<FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "mean", "var"};
  })
.set_attr<FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
    return 1;
  })
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{3, 4};
  })
.set_support_level(1);

// softmax
DMLC_REGISTER_PARAMETER(SoftmaxParam);

NNVM_REGISTER_OP(softmax)
.describe(R"code(Computes softmax.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__())
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SoftmaxParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
    return Array<Tensor>{ topi::nn::softmax(inputs[0], param.axis) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // grad_x = grad_y dot jacobian of softmax
    //
    // jacobian of softmax
    // [-y1y1 + y1, -y1y2,        ...    ]
    // [ ...      , -y2y2 + y2,   ...    ]
    // [ ...                      ...    ]
    // [ ...                  ,-ynyn + yn]
    //
    // grad_x =
    // [-y1*(ograd1*y1 - ograd1 + ograd2*y2 + ...),
    //  -y2*(ograd1*y1 - ograd2 + ograd2*y2 + ...),
    //  ...
    //  -yn*(ograd1*y1 - ogradn + ograd2*y2 + ...)]

    // grad_x = ograd elemwise_mul output
    // grad_x = sum(grad_x, keepdim, axis)
    // grad_x = grad_x broadcast_mul output
    // grad_x = neg grad_x
    // grad_x = grad_x + ograd elemwise_mul output
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(n->attrs.parsed);
    NodeEntry output =  NodeEntry{n, 0, 0};
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub0", {ograds[0], output});
    NodeEntry sub1 = MakeNode("sum", n->attrs.name + "_grad_sub1", {sub0},
                              {{"axis", std::to_string(param.axis)}, {"keepdims", "true"}});
    NodeEntry sub2 = MakeNode("broadcast_mul", n->attrs.name + "_grad_sub2", {sub1, output});
    return std::vector<NodeEntry> {
      MakeNode("elemwise_sub", n->attrs.name + "_grad", {sub0, sub2})
    };
});

// log_softmax
NNVM_REGISTER_OP(log_softmax)
.describe(R"code(Computes log softmax.

.. math:: \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__())
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SoftmaxParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
    CHECK(param.axis == -1 || param.axis == static_cast<int32_t>(inputs[0].ndim()) - 1)
        << "log_softmax currently only works on last dimension";
    return Array<Tensor>{ topi::nn::log_softmax(inputs[0]) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // grad_x = grad_y dot jacobian of logsoftmax
    //
    // jacobian of logsoftmax
    // [-y1 + 1, -y2,        ...    ]
    // [ ...   , -y2 + 1,    ...    ]
    // [ ...                 ...    ]
    // [ ...                ,-yn + 1]
    //
    // grad_x =
    // [ograd1 - exp(y1)*(ograd1 + ... + ogradn),
    //  ograd2 - exp(y2)*(ograd1 + ... + ogradn),
    //  ...
    //  ogradn - exp(yn)*(ograd1 + ... + ogradn)]

    // grad_x = sum(ograd, keepdim, axis)
    // sigma = exp(output)
    // grad_x = grad_x elemwise_mul sigma
    // grad_x = neg grad_x
    // grad_x = grad_x + ograd
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(n->attrs.parsed);
    NodeEntry output =  NodeEntry{n, 0, 0};
    NodeEntry sub0 = MakeNode("sum", n->attrs.name + "_grad_sub0", {ograds[0]},
                              {{"axis", std::to_string(param.axis)}, {"keepdims", "true"}});
    NodeEntry sub1 = MakeNode("exp", n->attrs.name + "_grad_sub1", {output});
    NodeEntry sub2 = MakeNode("broadcast_mul", n->attrs.name + "_grad_sub2", {sub0, sub1});
    return std::vector<NodeEntry> {
      MakeNode("elemwise_sub", n->attrs.name + "_grad", {ograds[0], sub2})
    };
})
.set_support_level(1);

// leaky_relu
DMLC_REGISTER_PARAMETER(LeakyReLUParam);

NNVM_REGISTER_OP(leaky_relu)
.describe(R"code(Leaky version of a Rectified Linear Unit.

`y = x > 0 ? x : alpha * x`

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(LeakyReLUParam::__FIELDS__())
.set_attr_parser(ParamParser<LeakyReLUParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<LeakyReLUParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
    return Array<Tensor>{ topi::leaky_relu(inputs[0], param.alpha) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = leak_relu(x)
    // grad = indicator(x > 0) + alpha * indicator(x < 0)
    const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(n->attrs.parsed);
    NodeEntry zero = MakeNode("zeros_like", n->attrs.name + "_grad_zero",
                              {n->inputs[0]});
    NodeEntry sub0 = MakeNode("greater", n->attrs.name + "_pos_grad",
                              {n->inputs[0], zero});
    NodeEntry sub1 = MakeNode("less", n->attrs.name + "_neg_grad",
                              {n->inputs[0], zero});
    NodeEntry sub2 = MakeNode("__mul_scalar__", n->attrs.name + "_neg_mul_2",
                              {sub1},
                              {{"scalar", std::to_string(param.alpha)}});
    NodeEntry sub3 = MakeNode("elemwise_add", n->attrs.name + "_sub3", {sub0, sub2});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {ograds[0], sub3})
    };
})
.set_support_level(1);

// prelu
DMLC_REGISTER_PARAMETER(PReLUParam);

inline bool PReluInferShape(const nnvm::NodeAttrs &attrs,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape) {
  const PReLUParam &param = nnvm::get<PReLUParam>(attrs.parsed);
  TShape dshape = in_shape->at(0);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 0, dshape);

  // The case of parametric relu
  CHECK(size_t(param.axis) < dshape.Size())
      << "Wrong axis ("  << param.axis << ")value.";

  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 1, TShape({dshape[param.axis]}));

  TShape oshape(dshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

inline bool PReluCorrectLayout(const NodeAttrs& attrs,
                               std::vector<Layout> *in_layouts,
                               const std::vector<Layout> *last_in_layouts,
                               std::vector<Layout> *out_layouts) {
  const PReLUParam& param = nnvm::get<PReLUParam>(attrs.parsed);
  CHECK_EQ(in_layouts->size(), 2U);
  CHECK_EQ(last_in_layouts->size(), 2U);
  CHECK_EQ(out_layouts->size(), 1U);

  const Layout& data_layout = last_in_layouts->at(0).defined() ?
                              last_in_layouts->at(0) : in_layouts->at(0);
  if (data_layout.defined()) {
    CHECK(data_layout.indexof('C') == param.axis && !data_layout.contains('c'))
      << "Channel in data layout " << data_layout
      << " is not at index " << param.axis;
  }

  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, data_layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 1, Layout("C"));
  NNVM_ASSIGN_LAYOUT(*out_layouts, 0, data_layout);

  return true;
}

NNVM_REGISTER_OP(prelu)
.describe(R"code(Parametric version of a Rectified Linear Unit.
It accepts two arguments: an input ``x`` and a channelwise slope ``alpha``
and computes the output as :math:`PReLU(x) y = x > 0 ? x : alpha * x`,
where :math:`*` is an channelwise multiplication for each sample in the

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_argument("alpha", "Tensor", "Input channelwise alpha.")
.add_arguments(PReLUParam::__FIELDS__())
.set_attr_parser(ParamParser<PReLUParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", PReluInferShape)
.set_attr<FCorrectLayout>("FCorrectLayout", PReluCorrectLayout)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "alpha"};
  })
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const PReLUParam& param = nnvm::get<PReLUParam>(attrs.parsed);
    return Array<Tensor>{ topi::prelu(inputs[0], inputs[1], param.axis)};
  })
.set_support_level(4);

DMLC_REGISTER_PARAMETER(PadParam);

inline bool PadInferShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_shape,
                          std::vector<TShape>* out_shape) {
  const PadParam& param = nnvm::get<PadParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() == 0) return false;
  CHECK_EQ(param.pad_width.ndim(), dshape.ndim());
  TShape oshape = dshape;
  for (uint32_t i = 0; i < dshape.ndim(); i++) {
    CHECK_EQ(param.pad_width[i].ndim(), 2U);
    int pad_before = param.pad_width[i][0];
    int pad_after = param.pad_width[i][1];
    oshape[i] = dshape[i] + pad_before + pad_after;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(pad)
.describe(R"code(Pad for n-D tensor.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "n-D Tensor", "Input data.")
.add_arguments(PadParam::__FIELDS__())
.set_attr_parser(ParamParser<PadParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<PadParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", PadInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const PadParam& param = nnvm::get<PadParam>(attrs.parsed);
    auto pad_width = param.pad_width;
    CHECK(pad_width.ndim() == inputs[0]->shape.size() &&
      pad_width[0].ndim() == 2)
      << "Illegal pad_width";
    Array<tvm::Expr> pad_before;
    for (size_t i = 0; i < pad_width.ndim(); ++i) {
      pad_before.push_back(tvm::make_const(tvm::Int(32), pad_width[i][0]));
    }
    Array<tvm::Expr> pad_after;
    for (size_t i = 0; i < pad_width.ndim(); ++i) {
      pad_after.push_back(tvm::make_const(tvm::Int(32), pad_width[i][1]));
    }
    return Array<Tensor>{ topi::pad(inputs[0], pad_before, pad_after,
                          tvm::make_const(inputs[0]->dtype, param.pad_value)) };
})
.set_support_level(1);

// layout transformer
DMLC_REGISTER_PARAMETER(LayoutTransformParam);

inline bool LayoutTransformInferShape(const NodeAttrs& attrs,
                                      std::vector<TShape>* in_attrs,
                                      std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const LayoutTransformParam& param = nnvm::get<LayoutTransformParam>(attrs.parsed);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  const TShape &oshape = ConvertLayout(dshape,
                                       Layout(param.src_layout),
                                       Layout(param.dst_layout));
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(__layout_transform__)
.describe(R"code(Transform the input data layout.

For transforming from NCHW to N16cHWC, the `__layout_transform__` operator reshapes
the input array by output[n, c, h, w, C] = data[n, C*16+c, h, w]

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(LayoutTransformParam::__FIELDS__())
.set_attr_parser(ParamParser<LayoutTransformParam>)
.set_attr<FInferShape>("FInferShape", LayoutTransformInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>(
  "FCorrectLayout", [](const NodeAttrs& attrs,
                     std::vector<Layout> *ilayouts,
                     const std::vector<Layout> *last_ilayouts,
                     std::vector<Layout> *olayouts) {
    const LayoutTransformParam& param = nnvm::get<LayoutTransformParam>(attrs.parsed);
    CHECK_EQ(ilayouts->size(), 1U);
    CHECK_EQ(olayouts->size(), 1U);
    NNVM_ASSIGN_LAYOUT(*ilayouts, 0, Layout(param.src_layout));
    NNVM_ASSIGN_LAYOUT(*olayouts, 0, Layout(param.dst_layout));
    return true;
})
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& outputs) {
    const LayoutTransformParam& param = nnvm::get<LayoutTransformParam>(attrs.parsed);
    return Array<Tensor>{
      topi::layout_transform(inputs[0], param.src_layout, param.dst_layout)
    };
})
.set_support_level(1);

DMLC_REGISTER_PARAMETER(LRNParam);

inline bool LRNInferShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_shape,
                          std::vector<TShape>* out_shape) {
  TShape dshape = (*in_shape)[0];
  TShape oshape = dshape;

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(lrn)
.describe(R"code(LRN layer)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<LRNParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", LRNInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(1);

DMLC_REGISTER_PARAMETER(L2NormalizeParam);

inline bool L2NormalizeInferShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape>* in_shape,
                                  std::vector<TShape>* out_shape) {
  TShape dshape = (*in_shape)[0];
  TShape oshape = dshape;

  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(l2_normalize)
.describe(R"code(L2NORMALIZE layer)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.set_attr_parser(ParamParser<L2NormalizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<L2NormalizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", L2NormalizeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
