/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/dense.h"
#include "topi/nn.h"
#include "topi/nn/softmax.h"

namespace nnvm {
namespace top {

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
    // grad = indicator(x > 0)
    NodeEntry zero = MakeNode("zeros_like", n->attrs.name + "_grad_zero",
                              {n->inputs[0]});
    return std::vector<NodeEntry>{
      MakeNode("greater", n->attrs.name + "_grad",
               {n->inputs[0], zero}, {{"exclude", "true"}})
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
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 1, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 2, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 3, bshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 4, bshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 1, bshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 2, bshape);
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
.set_attr<FMutateInputs>("FListMutateInputs", [](const NodeAttrs& attrs) {
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
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
    CHECK_EQ(param.axis, -1) << "Currently only axis=-1 is supported";
    return Array<Tensor>{ topi::nn::softmax(inputs[0]) };
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
    // [-y1*(ograd1*y1 - 1 + ograd2*y2 + ..., -y2*(ograd1*y1 - 1 + ograd2*y2, ..., ...]]

    // grad_x = ograd elemwise_mul output
    // grad_x = sum(grad_x, keepdim, axis)
    // grad_x = grad_x broadcast_mul output
    // grad_x = neg grad_x
    // grad_x = grad_x + output
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(n->attrs.parsed);
    NodeEntry output =  NodeEntry{n, 0, 0};
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub0", {ograds[0], output});
    NodeEntry sub1 = MakeNode("sum", n->attrs.name + "_grad_sub1", {sub0},
                              {{"axis", std::to_string(param.axis)}, {"keepdims", "true"}});
    NodeEntry sub2 = MakeNode("broadcast_mul", n->attrs.name + "_grad_sub2", {sub1, output});
    NodeEntry sub3 = MakeNode("negative", n->attrs.name + "_grad_sub3", {sub2});
    return std::vector<NodeEntry> {
      MakeNode("elemwise_add", n->attrs.name + "_grad", {sub3, output})
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
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
    CHECK_EQ(param.axis, -1) << "Currently only axis=-1 is supported";
    return Array<Tensor>{ topi::nn::log_softmax(inputs[0]) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // grad_x = grad_y dot jacobian of softmax
    //
    // jacobian of softmax
    // [-y1 + 1, -y2,        ...    ]
    // [ ...   , -y2 + 1,    ...    ]
    // [ ...                 ...    ]
    // [ ...                ,-yn + 1]
    //
    // grad_x =
    // [-(ograd1*y1 - 1 + ograd2*y2 + ..., -(ograd1*y1 - 1 + ograd2*y2, ..., ...]]

    // grad_x = ograd elemwise_mul output
    // grad_x = sum(grad_x, keepdim, axis)
    // grad_x = neg grad_x
    // grad_x = grad_x + ones_like(grad_x)
    // grad_x = expand_dims(grad_x, axis)
    const SoftmaxParam& param = nnvm::get<SoftmaxParam>(n->attrs.parsed);
    NodeEntry output =  NodeEntry{n, 0, 0};
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub0", {ograds[0], output});
    NodeEntry sub1 = MakeNode("sum", n->attrs.name + "_grad_sub1", {sub0},
                              {{"axis", std::to_string(param.axis)}, {"keepdims", "true"}});
    NodeEntry sub2 = MakeNode("negative", n->attrs.name + "_grad_sub2", {sub1});
    NodeEntry sub3 = MakeNode("ones_like", n->attrs.name + "_grad_sub3", {sub2});
    NodeEntry sub4 = MakeNode("elemwise_add", n->attrs.name + "_grad_sub4", {sub2, sub3});
    return std::vector<NodeEntry> {
      MakeNode("expand_like", n->attrs.name + "_grad", {sub4, output},
               {{"axis", std::to_string(param.axis)}})
    };
})
.set_support_level(1);

// leaky_rlu
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
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
    return Array<Tensor>{ topi::leaky_relu<float>(inputs[0], 0.0, param.alpha) };
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
                              {n->inputs[0], zero}, {{"exclude", "true"}});
    NodeEntry sub1 = MakeNode("less", n->attrs.name + "_neg_grad",
                              {n->inputs[0], zero}, {{"exclude", "true"}});
    NodeEntry sub2 = MakeNode("__mul_scalar__", n->attrs.name + "_neg_mul_2",
                              {sub1},
                              {{"scalar", std::to_string(param.alpha)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_add", n->attrs.name + "_add_grad", {sub0, sub2})
    };
})
.set_support_level(1);


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
    return Array<Tensor>{ topi::pad(inputs[0], pad_before, pad_after, param.pad_value) };
})
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
