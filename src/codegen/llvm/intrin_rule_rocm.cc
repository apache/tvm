/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include "./intrin_rule_llvm.h"
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/api_registry.h>
#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternOCML(const TVMArgs& args, TVMRetValue* rv) {
  Expr e = args[0];
  using namespace ir;
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << call->name << "_f" << call->type.bits();
  *rv = Call::make(call->type, intrinsic_name.str(), call->args,
                   Call::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp")
.set_body(DispatchExternOCML);

// TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fma")
// .set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sqrt")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.pow")
.set_body(DispatchExternOCML);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
