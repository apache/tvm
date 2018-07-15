/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_spirv.cc
 */
#include <tvm/packed_func_ext.h>
#include <tvm/ir.h>
#include <GLSL.std.450.h>

namespace tvm {
namespace codegen {
namespace spirv {

using namespace runtime;

// num_signature means number of arguments used to query signature
template<unsigned id>
inline void DispatchGLSLPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(ir::UIntImm::make(UInt(32), id));

  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = ir::Call::make(
      call->type, "spirv_glsl450", cargs, ir::Call::PureIntrinsic);
}

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.floor")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.ceil")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.round")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.trunc")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.fabs")
.set_body(DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.exp")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.log")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.sqrt")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.pow")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.tanh")
.set_body(DispatchGLSLPureIntrin<GLSLstd450Tanh>);

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
