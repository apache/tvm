/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/ir.h>
#include <tvm/api_registry.h>
#include <tvm/codegen.h>
#include <string>
#include "./llvm_common.h"

namespace tvm {
namespace codegen {
namespace llvm {

using namespace ir;

// num_signature means number of arguments used to query signature
template<unsigned id, int num_signature>
inline void DispatchLLVMPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(UIntImm::make(UInt(32), id));
  cargs.push_back(UIntImm::make(UInt(32), num_signature));

  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = Call::make(
      call->type, "llvm_intrin", cargs, Call::PureIntrinsic);
}

template<unsigned id, int num_signature>
inline void DispatchLLVMIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(UIntImm::make(UInt(32), id));
  cargs.push_back(UIntImm::make(UInt(32), num_signature));
  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = Call::make(
      call->type, "llvm_intrin", cargs, Call::Intrinsic);
}

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

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.pow")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
