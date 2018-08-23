/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.popcount")
.set_body(DispatchExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
