/*!
 *  Copyright (c) 2018 by Contributors
 * \file yolo.cc
 * \brief Property def of yolo operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "../../elemwise_op_common.h"

namespace nnvm {
namespace top {

NNVM_REGISTER_OP(yolov3_yolo)
.describe(R"code(Yolo layer
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(5)
.add_argument("data", "Tensor", "Input data")
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInplaceOption>(
    "FInplaceOption",
    [](const NodeAttrs &attrs) {
      return std::vector<std::pair<int, int>>{{0, 0}, {1, 0}};
    })
.set_attr<FGradient>("FGradient", [](const NodePtr &n,
                                     const std::vector<NodeEntry> &ograds) {
  return std::vector<NodeEntry>{ograds[0], ograds[0]};
});
}  // namespace top
}  // namespace nnvm
