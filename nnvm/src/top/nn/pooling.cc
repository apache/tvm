/*!
 *  Copyright (c) 2017 by Contributors
 * \file pooling.cc
 * \brief Property def of pooling operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

DMLC_REGISTER_PARAMETER(Pool2DParam);

inline bool Pool2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  const Pool2DParam& param = nnvm::get<Pool2DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);

  TShape oshape = dshape;
  CHECK_EQ(dshape.ndim(), 4U)
      << "Pooling: Input data should be 4D";
  CHECK(param.pool_size[0] <= dshape[2] + 2 * param.padding[0])
      << "pool size (" << param.pool_size[0] << ") exceeds input (" << dshape[2]
      << " padded to " << (dshape[2] + 2*param.padding[0]) << ")";
  CHECK(param.pool_size[1] <= dshape[3] + 2 * param.padding[1])
      << "pool size (" << param.pool_size[1] << ") exceeds input (" << dshape[3]
      << " padded to " << (dshape[3] + 2*param.padding[1]) << ")";

  if (!param.ceil_mode) {
    oshape[2] = ((dshape[2] + 2 * param.padding[0] - param.pool_size[0]) /
                 param.strides[0]) + 1;
    oshape[3] = ((dshape[3] + 2 * param.padding[1] - param.pool_size[1]) /
                 param.strides[1]) + 1;
  } else {
    oshape[2] = ((dshape[2] + 2 * param.padding[0] - param.pool_size[0] +
                  param.strides[0] - 1) / param.strides[0]) + 1;
    oshape[3] = ((dshape[3] + 2 * param.padding[1] - param.pool_size[1] +
                  param.strides[1] - 1) / param.strides[1]) + 1;
  }
  oshape = ConvertLayout(oshape, kNCHW, param.layout);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
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
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Pool2DParam>)
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
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Pool2DParam>)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);


DMLC_REGISTER_PARAMETER(GlobalPool2DParam);

inline bool GlobalPool2DInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shape,
                                   std::vector<TShape>* out_shape) {
  const GlobalPool2DParam& param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);
  TShape oshape = dshape;
  oshape[2] = oshape[3] = 1;
  oshape = ConvertLayout(oshape, kNCHW, param.layout);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
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
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GlobalPool2DParam>)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_outputs(1)
.set_num_inputs(1)
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
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GlobalPool2DParam>)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);

}  // namespace top
}  // namespace nnvm
