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

DMLC_REGISTER_PARAMETER(UpSamplingParam);

inline bool UpSamplingInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shape,
                                   std::vector<TShape>* out_shape) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);
  TShape oshape = dshape;
  oshape[2] = oshape[2] * param.scale;
  oshape[3] = oshape[3] * param.scale;
  oshape = ConvertLayout(oshape, kNCHW, param.layout);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(upsampling)
.describe(R"(Perform nearest neighbor upsampling to input array.

- **data**: Input is 4D array of shape (batch_size, channels, in_height, in_width).
- **out**: Output is 4D array of shape (batch_size, channels, in_height*scale, in_width*scale).

)" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(UpSamplingParam::__FIELDS__())
.set_attr_parser(ParamParser<UpSamplingParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<UpSamplingParam>)
.set_attr<FInferShape>("FInferShape", UpSamplingInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);

}  // namespace top
}  // namespace nnvm
