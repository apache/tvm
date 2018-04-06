/*!
 *  Copyright (c) 2018 by Contributors
 * \file region.cc
 * \brief Property def of pooling operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "../../op_common.h"
#include "region.h"

namespace nnvm {
namespace top {

NNVM_REGISTER_OP(yolo2_region)
.describe(R"code(Region layer
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(5)
.add_argument("data", "Tensor", "Input data")
.set_attr<FInferType>("FInferType", RegionType<1, 1>)
.set_attr<FInferShape>("FInferShape", RegionShape<1, 1>)
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
