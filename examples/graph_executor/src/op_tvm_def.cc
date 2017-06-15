/*!
 *  Copyright (c) 2017 by Contributors
 * \file Operator defintions in TVM.
 */
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include "./op_attr_types.h"

namespace tvm {
namespace contrib {

using namespace nnvm;

Array<Tensor>
ComputeAdd(const NodeAttrs& attrs,
           const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.add");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor>
ComputeExp(const NodeAttrs& attrs,
           const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.exp");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Schedule ScheduleEWise(const NodeAttrs& attrs,
                       const Array<Tensor>& outs,
                       const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.ewise");
  return pf(outs, target);
}

NNVM_REGISTER_OP(__add_symbol__)
.set_attr<FTVMCompute>("FTVMCompute", ComputeAdd)
.set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise);

NNVM_REGISTER_OP(exp)
.set_attr<FTVMCompute>("FTVMCompute", ComputeExp)
.set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise);
}  // namespace contrib
}  // namespace tvm
