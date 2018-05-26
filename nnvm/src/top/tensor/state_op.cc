/*!
 *  Copyright (c) 2018 by Contributors
 * \file state_op.cc
 * \brief Experimental operators
 *   Currently we only support assign
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include <topi/elemwise.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

using namespace tvm;
using namespace nnvm::compiler;

NNVM_REGISTER_OP(_assign)
.describe(R"doc(Assign rhs to the lhs.

lhs must be a Variable.
This is an experimental operator.

)doc" NNVM_ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FMutateInputs>(
  "FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
})
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    // This implementation is needed for the special
    // logic handling assign in the compiler
    // It simply copies the result of rhs the output
    // The later decoration in compiler will change
    // the memory assignment of assign to tie
    // the lhs to the output.
    return Array<Tensor>{ topi::identity(inputs[1]) };
})
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>(
  "FInplaceOption", [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{1, 0}};
});

}  // namespace top
}  // namespace nnvm
