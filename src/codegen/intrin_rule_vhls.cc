/*!
 *  Copyright (c) 2018 by Contributors
 * \file intrin_rule_vhls.cc
 * \brief VHLS intrinsic rules.
 */
#include "./intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.popcount")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.tvm_warp_shuffle")
.set_body(DispatchExtern<Direct>);


}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
