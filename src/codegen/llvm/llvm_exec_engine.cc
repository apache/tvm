/*!
 *  Copyright (c) 2017 by Contributors
 * \file llvm_exec_engine.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/codegen.h>
#include "./llvm_common.h"
#include "./codegen_llvm.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

#ifdef TVM_LLVM_VERSION
// Environment to keep jit resources alive.
struct LLVMJITEnv {
  std::shared_ptr<llvm::LLVMContext> ctx;
  llvm::ExecutionEngine* ee{nullptr};
  // constructor
  LLVMJITEnv(std::shared_ptr<llvm::LLVMContext> ctx,
             llvm::ExecutionEngine* ee)
      : ctx(ctx), ee(ee) {
  }
  // destructor
  ~LLVMJITEnv() {
    if (ee != nullptr) {
      ee->runStaticConstructorsDestructors(true);
      delete ee;
    }
  }
};


PackedFunc JITCompile(std::unique_ptr<llvm::Module> module,
                      std::shared_ptr<llvm::LLVMContext> ctx,
                      const std::string& func_name) {
  llvm::EngineBuilder builder(std::move(module));
  builder.setEngineKind(llvm::EngineKind::JIT);
  builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
  std::shared_ptr<LLVMJITEnv> env = std::make_shared<LLVMJITEnv>(
      ctx, builder.create());
  CHECK(env->ee != nullptr);
  auto* faddr = reinterpret_cast<LLVMPackedCFunc>(
      env->ee->getFunctionAddress(func_name));
  env->ee->runStaticConstructorsDestructors(false);
  return PackedFunc([env, faddr](TVMArgs args, TVMRetValue* rv) {
      int ret = (*faddr)(
          (void*)args.values, // NOLINT(*)
          (int*)args.type_codes, // NOLINT(*)
          args.num_args);
      CHECK(ret == 0) << TVMGetLastError();
    });
}

PackedFunc BuildLLVM(LoweredFunc func) {
  InitializeLLVM();
  // use one context per function.
  std::shared_ptr<llvm::LLVMContext> ctx =
      std::make_shared<llvm::LLVMContext>();
  CodeGenLLVM cg;
  cg.Init(func->name, ctx.get());
  cg.AddFunction(func);
  std::unique_ptr<llvm::Module> m = cg.Finish();
  return JITCompile(std::move(m), ctx, func->name);
}

#else
PackedFunc BuildLLVM(LoweredFunc func) {
  LOG(FATAL) << "LLVM is not enabled";
  return PackedFunc();
}
#endif  // TVM_LLVM_VERSION
}  // namespace codegen
}  // namespace tvm
