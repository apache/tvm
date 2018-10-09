/*
 * Copyright (c) 2018 by Contributors
 * \file device_copy_op.h
 * \brief Register an operator to perform data copy across different devices.
 */
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include <tvm/tensor.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../top/elemwise_op_common.h"
#include "../top/op_common.h"

namespace nnvm {
namespace op {

inline bool DeviceCopyOpInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shapes,
                                   std::vector<TShape>* out_shapes) {
  CHECK_EQ(in_shapes->size(), 1U)
      << "Cross device copy op can only have one input.";
  CHECK_EQ(out_shapes->size(), 1U)
      << "Cross device copy op can only have one output.";

  if (out_shapes->at(0).ndim() != 0) return true;
  SHAPE_ASSIGN(out_shapes->at(0), in_shapes->at(0));
  return true;
}

inline bool DeviceCopyOpInferType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_types,
                                  std::vector<int>* out_types) {
  CHECK_EQ(in_types->size(), 1U)
      << "Cross device copy op can only have one input.";
  CHECK_EQ(out_types->size(), 1U)
      << "Cross device copy op can only have one output.";

  out_types->back() = in_types->at(0);
  return true;
}

NNVM_REGISTER_OP(device_copy_op)
    .describe(
        R"code(Copy data from one tensor to antoher.
               The source and destination might be \
               one different devices.)code" NNVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FInferShape>("FInferShape", DeviceCopyOpInferShape)
    .set_attr<nnvm::FInferType>("FInferType", DeviceCopyOpInferType)
    .set_attr<nnvm::FCorrectLayout>(
        "FCorrectLayout", nnvm::top::ElemwiseFixedLayoutCopyToOut<1, 1>);

}  // namespace op
}  // namespace nnvm
