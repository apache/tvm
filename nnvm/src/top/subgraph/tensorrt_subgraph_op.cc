/*!
 *  Copyright (c) 2018 by Contributors
 * \file tensorrt_subgraph_op.cc
 * \brief Subgraph operator for wrapping TensorRT.
 */

#include <nnvm/compiler/op_attr_types.h>
#include "./common.h"

namespace nnvm {
namespace top {

NNVM_REGISTER_OP(_tensorrt_subgraph_op)
.set_num_inputs(DefaultSubgraphOpNumInputs)
.set_num_outputs(DefaultSubgraphOpNumOutputs)
.set_attr<FListInputNames>("FListInputNames", DefaultSubgraphOpListInputs)
.set_attr<FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
.set_attr<FInferShape>("FInferShape", DefaultSubgraphOpShape)
.set_attr<FInferType>("FInferType", DefaultSubgraphOpType)
.set_attr<FMutateInputs>("FMutateInputs", DefaultSubgraphOpMutableInputs)
.set_attr<compiler::TOpPattern>("TOpPattern", compiler::kOpaque)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "Tensor-or-Tensor[]", "input data list");

}  // namespace top
}  // namespace nnvm
