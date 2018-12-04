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

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sigmoid")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    Expr e = args[0];
    const Call* call = e.as<Call>();
    CHECK(call != nullptr);

    auto one = make_const(call->args[0].type(), 1);
    *rv = one / (one + exp(-call->args[0]));
  });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
