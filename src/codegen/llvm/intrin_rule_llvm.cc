/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include "./intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.prefetch")
.set_body(DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 0>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fma")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sqrt")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.tanh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  Expr one = make_const(x.type(), 1);
  Expr two = make_const(x.type(), 2);
  Expr neg_two = make_const(x.type(), -2);

  Expr exp_neg2x = ir::Call::make(
      x.type(), "exp", {neg_two * x}, ir::Call::PureIntrinsic);
  Expr exp_pos2x = ir::Call::make(
      x.type(), "exp", {two * x}, ir::Call::PureIntrinsic);

  Expr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
  Expr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
  *rv = ir::Select::make(
      x >= make_zero(x.type()), tanh_pos, tanh_neg);
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.pow")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.popcount")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
