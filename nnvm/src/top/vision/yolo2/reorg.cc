/*!
 *  Copyright (c) 2018 by Contributors
 * \file reorg.cc
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "../../op_common.h"
#include "../../elemwise_op_common.h"
#include "reorg.h"

namespace nnvm {
namespace top {

// reorg
DMLC_REGISTER_PARAMETER(ReorgParam);

inline bool ReorgInferShape(const nnvm::NodeAttrs &attrs,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape) {
  const ReorgParam &param = nnvm::get<ReorgParam>(attrs.parsed);
  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0)
    return false;
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 0, dshape);
  CHECK_EQ(dshape.ndim(), 4) << "Input data should be 4D";
  CHECK_GT(param.stride, 0U) << "Stride value cannot be 0";
  TShape oshape({dshape[0], 0, 0, 0});
  oshape[1] = dshape[1] * param.stride * param.stride;
  oshape[2] = dshape[2] / param.stride;
  oshape[3] = dshape[3] / param.stride;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(yolo2_reorg)
.describe(R"(Perform reorg operation on input array based on the stride value.
- **data**: Input is 4D array of shape (batch_size, channels, in_height, in_width).
- **out**: Output is 4D array of shape (batch_size, channels/(stride*stride), in_height*stride, in_width*stride).
)" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(5)
.add_argument("data", "Tensor", "Data input to reorganize")
.set_attr_parser(ParamParser<ReorgParam>)
.add_arguments(ReorgParam::__FIELDS__())
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReorgParam>)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FInferShape>("FInferShape", ReorgInferShape);
}  // namespace top
}  // namespace nnvm
