/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_default.cc
 * \brief Default intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.exp")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tanh")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sqrt")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.pow")
.set_body(DispatchExtern<FloatSuffix>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
