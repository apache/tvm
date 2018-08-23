/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.trunc")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.fabs")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.round")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.popcount")
.set_body(DispatchExtern<Direct>);

// There is no warp shuffle instruction in standard OpenCL
// When shuffle is used, we assume it is intel's shuffle extension
struct IntelShuffle {
  std::string operator()(Type t, std::string name) const {
    return "intel_sub_group_shuffle";
  }
};

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.tvm_warp_shuffle")
.set_body(DispatchExtern<IntelShuffle>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
