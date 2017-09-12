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

template<typename ParamType>
inline std::vector<std::string> UseBiasListInputNames(const NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

template<typename ParamType>
inline uint32_t UseBiasNumInputs(const NodeAttrs& attrs) {
  const ParamType& param = get<ParamType>(attrs.parsed);
  return param.use_bias ? 3 : 2;
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
.set_num_inputs(UseBiasNumInputs<DenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<DenseParam>)
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


DMLC_REGISTER_PARAMETER(Conv2DParam);

inline bool Conv2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  const Conv2DParam& param = nnvm::get<Conv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  return true;
}

NNVM_REGISTER_OP(conv2d)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr<FInferShape>("FInferShape", Conv2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_support_level(2);


DMLC_REGISTER_PARAMETER(Conv2DTransposeParam);

inline bool Conv2DTransposeInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  const Conv2DTransposeParam& param = nnvm::get<Conv2DTransposeParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  return true;
}

NNVM_REGISTER_OP(conv2d_transpose)
.describe(R"code(Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

            out_height and out_width are calculated as::
                out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DTransposeParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DTransposeParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DTransposeParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DTransposeParam>)
.set_attr<FInferShape>("FInferShape", Conv2DTransposeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_support_level(2);


DMLC_REGISTER_PARAMETER(Pool2DParam);

inline bool Pool2DInferShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  const Pool2DParam& param = nnvm::get<Pool2DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  return true;
}

NNVM_REGISTER_OP(max_pool2d)
.describe(R"code(Max pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::
               out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
               out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1
           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(Pool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Pool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(2);


NNVM_REGISTER_OP(avg_pool2d)
.describe(R"code(Average pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::
               out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
               out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1
           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(Pool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Pool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(2);


DMLC_REGISTER_PARAMETER(GlobalPool2DParam);

inline bool GlobalPool2DInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape> *in_shape,
                                   std::vector<TShape> *out_shape) {
  const GlobalPool2DParam& param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  return true;
}

NNVM_REGISTER_OP(global_max_pool2d)
.describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(GlobalPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<GlobalPool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(2);


NNVM_REGISTER_OP(global_avg_pool2d)
.describe(R"code(Global average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(GlobalPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<GlobalPool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_support_level(2);


}  // namespace top
}  // namespace nnvm
