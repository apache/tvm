/*
 * Copyright (c) 2018 by Contributors
 * \file device_copy_op.h
 * \brief Register an operator to perform data copy across different devices.
 */
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>

#include "../top/elemwise_op_common.h"
#include "../top/op_common.h"

namespace nnvm {
namespace op {

NNVM_REGISTER_OP(device_copy_op)
.describe(R"code(
Copy data from one tensor to another. The source and destination might be
on different devices.
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", nnvm::top::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", nnvm::top::ElemwiseType<1, 1>)
.set_attr<FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FCorrectLayout>(
  "FCorrectLayout", nnvm::top::ElemwiseArbitraryLayout<1, 1>);

}  // namespace op
}  // namespace nnvm
