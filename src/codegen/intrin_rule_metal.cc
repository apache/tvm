/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.trunc")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.fabs")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.round")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.popcount")
.set_body(DispatchExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
