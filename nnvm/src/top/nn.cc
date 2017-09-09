/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./op_common.h"
#include "./elemwise_op_common.h"

namespace nnvm {
namespace top {

// dense
DMLC_REGISTER_PARAMETER(DenseParam);

inline std::vector<std::string> DenseListInputNames(const NodeAttrs& attrs) {
  const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

inline bool DenseInferShape(const nnvm::NodeAttrs& attrs,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape) {
  const DenseParam& param = nnvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[DenseParam::kData];
  TShape oshape = (*out_shape)[0];
  // require data to be known
  if (dshape.ndim() ==  0) return false;
  dim_t num_input;
  num_input = dshape.ProdShape(1, dshape.ndim());
  SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kWeight, TShape({param.units, num_input}));
  if (param.use_bias) {
    SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kBias, TShape({param.units}));
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({dshape[0], param.units}));
  if (oshape.ndim() != 0) {
    dshape[0] = oshape[0];
    SHAPE_ASSIGN_CHECK(*in_shape, DenseParam::kData, dshape);
  }
  return true;
}

NNVM_REGISTER_OP(dense)
.NNVM_DESCRIBE(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(DenseParam::__FIELDS__())
.set_attr_parser(ParamParser<DenseParam>)
.set_num_outputs(1)
.set_num_inputs([](const NodeAttrs& attrs) {
    const DenseParam& param = get<DenseParam>(attrs.parsed);
    return param.use_bias ? 3 : 2;
  })
.set_attr<FListInputNames>("FListInputNames", DenseListInputNames)
.set_attr<FInferShape>("FInferShape", DenseInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_support_level(1);

// relu
NNVM_REGISTER_ELEMWISE_UNARY_OP(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(input, 0)

)code" NNVM_ADD_FILELINE)
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
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DropoutParam>)
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
)" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input to which dropout will be applied")
.add_argument("gamma", "Tensor", "The gamma scale factor")
.add_argument("beta", "Tensor", "The beta offset factor")
.add_argument("moving_mean", "Tensor", "running mean of input")
.add_argument("moving_var", "Tensor", "running variance of input")
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr_parser(ParamParser<BatchNormParam>)
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

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(1);

// log_softmax
DMLC_REGISTER_PARAMETER(LogSoftmaxParam);

NNVM_REGISTER_OP(log_softmax)
.describe(R"code(Computes softmax.

.. math:: \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LogSoftmaxParam>)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(1);

}  // namespace top
}  // namespace nnvm
