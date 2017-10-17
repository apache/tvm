/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic rules.
 */
#include "./intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.tanh")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sqrt")
.set_body(DispatchExtern<FloatDirect>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.pow")
.set_body(DispatchExtern<FloatDirect>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
