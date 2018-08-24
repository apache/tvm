/*!
 *  Copyright (c) 2018 by Contributors
 * \file intrin_rule_aocl.cc
 * \brief AOCL intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.trunc")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.fabs")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.round")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.popcount")
.set_body(DispatchExtern<Direct>);


TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.trunc")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.fabs")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.round")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.popcount")
.set_body(DispatchExtern<Direct>);


}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
