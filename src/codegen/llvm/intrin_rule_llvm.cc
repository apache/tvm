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

template<unsigned id>
inline void DispatchLLVMBuildin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(UIntImm::make(UInt(32), id));
  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = Call::make(
      call->type, "llvm_buildin", cargs, Call::Intrinsic);
}

template<unsigned id>
inline void DispatchLLVMPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(UIntImm::make(UInt(32), id));
  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = Call::make(
      call->type, "llvm_intrin", cargs, Call::PureIntrinsic);
}

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.__buildin_prefetch")
.set_body(DispatchLLVMBuildin<::llvm::Intrinsic::prefetch>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sqrt")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
