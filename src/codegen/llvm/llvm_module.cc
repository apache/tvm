/*!
 *  Copyright (c) 2017 by Contributors
 * \file llvm_module.cc
 * \brief LLVM runtime module for TVM
 */
#ifdef TVM_LLVM_VERSION
#include <tvm/runtime/packed_func.h>
#include <tvm/codegen.h>
#include <mutex>
#include "./llvm_common.h"
#include "./codegen_llvm.h"
#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

class LLVMModuleNode final : public runtime::ModuleNode {
 public:
  ~LLVMModuleNode() {
    module_.reset();
    if (ee_ != nullptr) {
      ee_->runStaticConstructorsDestructors(true);
      delete ee_;
    }
  }

  const char* type_key() const {
    return "llvm";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (ee_ == nullptr) LazyInitJIT();
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string& fname = (name == runtime::symbol::tvm_module_main ?
                                entry_func_ : name);
    BackendPackedCFunc faddr =
        reinterpret_cast<BackendPackedCFunc>(ee_->getFunctionAddress(fname));
    if (faddr == nullptr) return PackedFunc();
    return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        int ret = (*faddr)(
            (void*)args.values, // NOLINT(*)
            (int*)args.type_codes, // NOLINT(*)
            args.num_args);
        CHECK_EQ(ret, 0) << TVMGetLastError();
      });
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    std::error_code ecode;
    llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::F_None);
    CHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_name
                               << " " << ecode.message();
    if (fmt == "o" || fmt == "obj") {
      llvm::legacy::PassManager pass;
      CHECK(tm_);
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, llvm::TargetMachine::CGFT_ObjectFile) == 0)
          << "Cannot emit target CGFT_ObjectFile";
      pass.run(*mptr_);
    } else if (fmt == "ll") {
      mptr_->print(dest, nullptr);
    } else if (fmt == "bc") {
      llvm::WriteBitcodeToFile(mptr_, dest);
    } else {
      LOG(FATAL) << "Do not know how to save file "
                 << file_name << " with format=\'"<< format << "\'";
    }
    dest.close();
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    LOG(FATAL) << "LLVMModule: SaveToBinary not supported";
  }

  std::string GetSource(const std::string& format) final {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    CHECK(mptr_ != nullptr);
    mptr_->print(rso, nullptr);
    return rso.str();
  }

  void Init(const Array<LoweredFunc>& funcs, std::string target) {
    InitializeLLVM();
    tm_ = GetLLVMTargetMachine(target);
    CHECK_NE(funcs.size(), 0U);
    ctx_ = std::make_shared<llvm::LLVMContext>();
    CodeGenLLVM cg;
    entry_func_ = funcs[0]->name;
    cg.Init(funcs[0]->name, tm_, ctx_.get());
    for (LoweredFunc f :  funcs) {
      cg.AddFunction(f);
    }
    cg.AddMainFunction(funcs[0]->name);
    module_ = cg.Finish();
    mptr_ = module_.get();
  }

  void LoadIR(const std::string& file_name) {
    InitializeLLVM();
    ctx_ = std::make_shared<llvm::LLVMContext>();
    llvm::SMDiagnostic err;
    module_ = llvm::parseIRFile(file_name, err, *ctx_);
    CHECK(module_.get() != nullptr)
        << "Fail to load ir file " << file_name;
    std::string target = module_->getTargetTriple();
    mptr_ = module_.get();
    std::ostringstream os;
    os << "llvm -target " << target;
    tm_ = GetLLVMTargetMachine(os.str());
  }

 private:
  void LazyInitJIT() {
    CHECK(ee_ == nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    llvm::EngineBuilder builder(std::move(module_));
    builder.setEngineKind(llvm::EngineKind::JIT);
    builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    llvm::TargetMachine *tm = builder.selectTarget();
    llvm::TargetMachine *tm_sys = GetLLVMTargetMachine("llvm");
    if (tm_sys->getTargetTriple().getArch() != tm->getTargetTriple().getArch()) {
      LOG(FATAL) << "Cannot run module, architecture mismatch "
                 << " module=" << tm->getTargetTriple().str()
                 << " system=" << tm_sys->getTargetTriple().str();
    }
    llvm::DataLayout layout(tm->createDataLayout());
    CHECK(layout == mptr_->getDataLayout())
        << "Data layout mismatch between module("
        << mptr_->getDataLayout().getStringRepresentation() << ")"
        << " and ExecutionEngine ("
        << layout.getStringRepresentation() << ")";
    ee_ = builder.create(tm);
    CHECK(ee_ != nullptr)
        << "Failed to initialize git engine for " << mptr_->getTargetTriple();
    ee_->runStaticConstructorsDestructors(false);
    // setup context address.
    void** ctx_addr =
        reinterpret_cast<void**>(
            ee_->getGlobalValueAddress(runtime::symbol::tvm_module_ctx));
    // setup context address.
    entry_func_ =
        reinterpret_cast<const char*>(
            ee_->getGlobalValueAddress(runtime::symbol::tvm_module_main));
    if (ctx_addr != nullptr) {
      *ctx_addr = this;
    }
  }
  // The target configuration string
  std::string target_;
  // Name of entry function.
  std::string entry_func_;
  // JIT lock
  std::mutex mutex_;
  // execution engine
  llvm::ExecutionEngine *ee_{nullptr};
  // The raw pointer to the module.
  llvm::Module* mptr_{nullptr};
  // The target machine
  llvm::TargetMachine* tm_{nullptr};
  // The module, can be moved to ee if JIT is enabled.
  std::unique_ptr<llvm::Module> module_;
  // the context.
  std::shared_ptr<llvm::LLVMContext> ctx_;
};

TVM_REGISTER_API("codegen.build_llvm")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<LLVMModuleNode> n = std::make_shared<LLVMModuleNode>();
    n->Init(args[0], args[1]);
    *rv = runtime::Module(n);
  });

TVM_REGISTER_API("module.loadfile_ll")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<LLVMModuleNode> n = std::make_shared<LLVMModuleNode>();
    n->LoadIR(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
