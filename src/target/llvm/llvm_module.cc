/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file llvm_module.cc
 * \brief LLVM runtime module for TVM
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/codegen.h>
#include <mutex>
#include "llvm_common.h"
#include "codegen_llvm.h"
#include "codegen_blob.h"
#include "../../runtime/file_util.h"
#include "../../runtime/library_module.h"

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
      const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "__tvm_is_system_module") {
      bool flag =
          (mptr_->getFunction("__tvm_module_startup") != nullptr);
      return PackedFunc([flag](TVMArgs args, TVMRetValue *rv) {
          * rv = flag;
        });
    } else if (name == "_get_target_triple") {
      std::string target_triple = tm_->getTargetTriple().str();
      return PackedFunc([target_triple](TVMArgs args, TVMRetValue *rv) {
        *rv = target_triple;
      });
    }
    if (ee_ == nullptr) LazyInitJIT();

    std::lock_guard<std::mutex> lock(mutex_);

    TVMBackendPackedCFunc faddr;
    if (name == runtime::symbol::tvm_module_main) {
      const char* entry_name = reinterpret_cast<const char*>(
          GetGlobalAddr(runtime::symbol::tvm_module_main));
      CHECK(entry_name != nullptr)
          << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(entry_name));
    } else {
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(name));
    }
    if (faddr == nullptr) return PackedFunc();
    return WrapPackedFunc(faddr, sptr_to_self);
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    std::error_code ecode;
    llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::F_None);
    CHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_name
                               << " " << ecode.message();
    if (fmt == "o" || fmt == "obj") {
#if TVM_LLVM_VERSION <= 60
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(mptr_);
#else
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(*mptr_);
#endif
      llvm::legacy::PassManager pass;
      CHECK(tm_);
#if TVM_LLVM_VERSION <= 60
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, llvm::TargetMachine::CGFT_ObjectFile) == 0)
          << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, nullptr, llvm::TargetMachine::CGFT_ObjectFile) == 0)
          << "Cannot emit target CGFT_ObjectFile";
#else
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, nullptr, llvm::CGFT_ObjectFile) == 0)
          << "Cannot emit target CGFT_ObjectFile";
#endif
      pass.run(*m);
    } else if (fmt == "s" || fmt == "asm") {
#if TVM_LLVM_VERSION <= 60
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(mptr_);
#else
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(*mptr_);
#endif
      llvm::legacy::PassManager pass;
      CHECK(tm_);
#if TVM_LLVM_VERSION <= 60
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#else
      CHECK(tm_->addPassesToEmitFile(
          pass, dest, nullptr, llvm::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#endif
      pass.run(*m);
    } else if (fmt == "ll") {
      mptr_->print(dest, nullptr);
    } else if (fmt == "bc") {
#if TVM_LLVM_VERSION <= 60
      llvm::WriteBitcodeToFile(mptr_, dest);
#else
      llvm::WriteBitcodeToFile(*mptr_, dest);
#endif
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
    std::string fmt = runtime::GetFileFormat("", format);
    std::string type_str;
    llvm::SmallString<256> str;
    llvm::raw_svector_ostream rso(str);

    if (fmt == "s" || fmt == "asm") {
    #if TVM_LLVM_VERSION <= 60
          std::unique_ptr<llvm::Module> m = llvm::CloneModule(mptr_);
    #else
          std::unique_ptr<llvm::Module> m = llvm::CloneModule(*mptr_);
    #endif
          llvm::legacy::PassManager pass;
          CHECK(tm_);
    #if TVM_LLVM_VERSION <= 60
          CHECK(tm_->addPassesToEmitFile(
              pass, rso, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
              << "Cannot emit target CGFT_AssemblyFile";
    #elif TVM_LLVM_VERSION <= 90
          CHECK(tm_->addPassesToEmitFile(
              pass, rso, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
              << "Cannot emit target CGFT_AssemblyFile";
    #else
          CHECK(tm_->addPassesToEmitFile(
              pass, rso, nullptr, llvm::CGFT_AssemblyFile) == 0)
              << "Cannot emit target CGFT_AssemblyFile";
    #endif
          pass.run(*m);
          return rso.str().str();
    } else if (fmt == "" || fmt == "ll") {
      std::string type_str;
      llvm::raw_string_ostream rso(type_str);
      CHECK(mptr_ != nullptr);
      mptr_->print(rso, nullptr);
      return rso.str();
    } else {
      LOG(FATAL) << "Do not know how to get source code with format: "
                 << format << "\'";
    }
    return "";
  }

  void Init(const IRModule& mod, std::string target) {
    InitializeLLVM();
    tm_ = GetLLVMTargetMachine(target);
    bool system_lib = (target.find("-system-lib") != std::string::npos);
    ctx_ = std::make_shared<llvm::LLVMContext>();
    std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(tm_.get());

    std::vector<PrimFunc> funcs;
    std::string entry_func;
    for (auto kv :  mod->functions) {
      CHECK(kv.second->IsInstance<PrimFuncNode>())
          << "Can only lower IR Module with PrimFuncs";
      auto f = Downcast<PrimFunc>(kv.second);
      if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
        CHECK(global_symbol.defined());
        entry_func = global_symbol.value();
      }
      funcs.push_back(f);
    }
    CHECK_NE(funcs.size(), 0U);
    // TODO(tqchen): remove the entry function behavior as it does not
    // makes sense when we start to use multiple modules.
    cg->Init("TVMMod", tm_.get(), ctx_.get(), system_lib, system_lib);

    for (const auto& f : funcs) {
      cg->AddFunction(f);
    }

    if (entry_func.length() != 0) {
      cg->AddMainFunction(entry_func);
    }

    module_ = cg->Finish();
    module_->addModuleFlag(llvm::Module::Warning, "tvm_target", llvm::MDString::get(*ctx_, target));
    module_->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                           llvm::DEBUG_METADATA_VERSION);

    if (tm_->getTargetTriple().isOSDarwin()) {
      module_->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
    }

    std::string verify_errors_storage;
    llvm::raw_string_ostream verify_errors(verify_errors_storage);
    LOG_IF(FATAL, llvm::verifyModule(*module_, &verify_errors))
        << "LLVM module verification failed with the following errors: \n"
        << verify_errors.str();
    target_ = target;
    mptr_ = module_.get();
  }

  void Init(std::unique_ptr<llvm::Module> module,
            std::shared_ptr<llvm::LLVMContext> ctx) {
    InitializeLLVM();
    ctx_ = ctx;
    llvm::SMDiagnostic err;
    module_ = std::move(module);
    if (module_ == nullptr) {
      std::string msg = std::string(err.getMessage());
      LOG(FATAL) << "Fail to load module: " << msg;
    }
    std::string target_;
    llvm::Metadata* mtarget = module_->getModuleFlag("tvm_target");
    if (mtarget != nullptr) {
      llvm::MDString* pstr = llvm::dyn_cast<llvm::MDString>(mtarget);
      CHECK(pstr != nullptr);
      target_ = pstr->getString().str();
    } else {
      std::ostringstream os;
      os << "llvm -target " << module_->getTargetTriple();
      target_ = os.str();
    }
    mptr_ = module_.get();
    tm_ = GetLLVMTargetMachine(target_);
  }

  void LoadIR(const std::string& file_name) {
    auto ctx = std::make_shared<llvm::LLVMContext>();
    llvm::SMDiagnostic err;
    auto module = llvm::parseIRFile(file_name, err, *ctx);
    if (module == nullptr) {
      std::string msg = std::string(err.getMessage());
      LOG(FATAL) << "Fail to load ir file " << file_name << "\n"
                 << "line " << err.getLineNo() << ":" << msg;
    }
    Init(std::move(module), ctx);
  }

 private:
  void LazyInitJIT() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ee_) {
      return;
    }
    llvm::EngineBuilder builder(std::move(module_));
    std::string triple, mcpu, mattr;
    llvm::TargetOptions opt;
    ParseLLVMTargetOptions(target_, &triple, &mcpu, &mattr, &opt);
    builder.setEngineKind(llvm::EngineKind::JIT);
    builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
    if (mcpu.length() != 0) {
      builder.setMCPU(mcpu);
    }
    if (mattr.length() != 0) {
      std::vector<std::string> mattrs{mattr};
      builder.setMAttrs(mattrs);
    }
    builder.setTargetOptions(opt);
    auto tm = std::unique_ptr<llvm::TargetMachine>(builder.selectTarget());
    std::unique_ptr<llvm::TargetMachine> tm_sys = GetLLVMTargetMachine("llvm");
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
    ee_ = builder.create(tm.release());
    CHECK(ee_ != nullptr)
        << "Failed to initialize jit engine for " << mptr_->getTargetTriple();
    ee_->runStaticConstructorsDestructors(false);

    if (void** ctx_addr = reinterpret_cast<void**>(
            GetGlobalAddr(runtime::symbol::tvm_module_ctx))) {
      *ctx_addr = this;
    }
    runtime::InitContextFunctions([this](const char *name) {
        return reinterpret_cast<void*>(GetGlobalAddr(name));
      });
  }
  // Get global address from execution engine.
  uint64_t GetGlobalAddr(const std::string& name) const {
    // first verifies if GV exists.
    if (mptr_->getGlobalVariable(name) != nullptr) {
      return ee_->getGlobalValueAddress(name);
    } else {
      return 0;
    }
  }
  uint64_t GetFunctionAddr(const std::string& name) const {
    // first verifies if GV exists.
    if (mptr_->getFunction(name) != nullptr) {
      return ee_->getFunctionAddress(name);
    } else {
      return 0;
    }
  }

  // The target configuration string
  std::string target_;
  // JIT lock
  std::mutex mutex_;
  // execution engine
  llvm::ExecutionEngine *ee_{nullptr};
  // The raw pointer to the module.
  llvm::Module* mptr_{nullptr};
  // The target machine
  std::unique_ptr<llvm::TargetMachine> tm_{nullptr};
  // The module, can be moved to ee if JIT is enabled.
  std::unique_ptr<llvm::Module> module_;
  // the context.
  std::shared_ptr<llvm::LLVMContext> ctx_;
};

unsigned LookupLLVMIntrinsic(const std::string& name) {
  return llvm::Function::lookupIntrinsicID(name);
}


TVM_REGISTER_GLOBAL("target.build.llvm")
.set_body_typed([](IRModule mod, std::string target) {
  auto n = make_object<LLVMModuleNode>();
  n->Init(mod, target);
  return runtime::Module(n);
});


TVM_REGISTER_GLOBAL("codegen.LLVMModuleCreate")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  auto n = make_object<LLVMModuleNode>();
  auto target = args[0].operator std::string();
  auto module_name = args[1].operator std::string();

  // Generate a LLVM module from an input target string
  InitializeLLVM();
  auto tm = GetLLVMTargetMachine(target);
  auto ctx = std::make_shared<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module(new llvm::Module(module_name, *ctx));

  // Use a default data layout and target triple
  auto triple = tm->getTargetTriple();
  module->setTargetTriple(triple.str());
  module->setDataLayout(tm->createDataLayout());

  n->Init(std::move(module), ctx);

  *rv = runtime::Module(n);
});

TVM_REGISTER_GLOBAL("target.llvm_lookup_intrinsic_id")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = static_cast<int64_t>(LookupLLVMIntrinsic(args[0]));
  });

TVM_REGISTER_GLOBAL("target.llvm_version_major")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    int major = TVM_LLVM_VERSION / 10;
    *rv = major;
  });

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ll")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    auto n = make_object<LLVMModuleNode>();
    n->LoadIR(args[0]);
    *rv = runtime::Module(n);
  });

TVM_REGISTER_GLOBAL("codegen.llvm_target_enabled")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    InitializeLLVM();
    *rv = (GetLLVMTargetMachine(args[0], true) != nullptr);
  });

TVM_REGISTER_GLOBAL("codegen.codegen_blob")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = make_object<LLVMModuleNode>();
  auto p = CodeGenBlob(args[0].operator std::string(),
                       args[1].operator bool(),
                       args[2].operator std::string());
  n->Init(std::move(p.first), p.second);
  *rv = runtime::Module(n);
});
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
