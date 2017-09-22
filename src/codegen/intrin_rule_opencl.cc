/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
#include "./intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.exp")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.log")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.tanh")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.sqrt")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opencl.pow")
.set_body(DispatchExtern<FloatDirect>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
