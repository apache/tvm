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

#include <tvm/ir/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>

#include <mutex>

#include "../../runtime/file_utils.h"
#include "../../runtime/library_module.h"
#include "../func_registry_generator.h"
#include "codegen_blob.h"
#include "codegen_cpu.h"
#include "codegen_llvm.h"
#include "llvm_common.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

class LLVMModuleNode final : public runtime::ModuleNode {
 public:
  ~LLVMModuleNode() {
    module_.reset();
    if (ee_ != nullptr) {
      ee_->runStaticConstructorsDestructors(true);
      delete ee_;
    }
  }

  const char* type_key() const { return "llvm"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "__tvm_is_system_module") {
      bool flag = (mptr_->getFunction("__tvm_module_startup") != nullptr);
      return PackedFunc([flag](TVMArgs args, TVMRetValue* rv) { *rv = flag; });
    } else if (name == "get_func_names") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->function_names_; });
    } else if (name == "get_symbol") {
      return PackedFunc(nullptr);
    } else if (name == "get_const_vars") {
      return PackedFunc(nullptr);
    } else if (name == "_get_target_triple") {
      std::ostringstream target_triple_ss;
      target_triple_ss << tm_->getTargetTriple().str();
      // getTargetTriple() doesn't include other flags besides the triple. Add back flags which are
      // important for ModulePackImportsToLLVM.
      if (tm_->Options.FloatABIType == llvm::FloatABI::ABIType::Soft) {
        target_triple_ss << " -mfloat-abi=soft";
      }
      std::string mabi = tm_->Options.MCOptions.ABIName;
      if (!mabi.empty()) {
        target_triple_ss << " -mabi=" << mabi;
      }
      llvm::StringRef mcpu = tm_->getTargetCPU();
      if (!mcpu.empty() && mcpu != "generic") {
        target_triple_ss << " -mcpu=" << mcpu.str();
      }
      std::string target_triple = target_triple_ss.str();
      return PackedFunc([target_triple](TVMArgs args, TVMRetValue* rv) { *rv = target_triple; });
    }
    if (ee_ == nullptr) LazyInitJIT();

    std::lock_guard<std::mutex> lock(mutex_);

    TVMBackendPackedCFunc faddr;
    if (name == runtime::symbol::tvm_module_main) {
      const char* entry_name =
          reinterpret_cast<const char*>(GetGlobalAddr(runtime::symbol::tvm_module_main));
      ICHECK(entry_name != nullptr)
          << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(entry_name));
    } else {
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(name));
    }
    if (faddr == nullptr) return PackedFunc();
    return WrapPackedFunc(faddr, sptr_to_self);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    std::error_code ecode;
#if TVM_LLVM_VERSION <= 70
    llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::F_None);
#else
    llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::OF_None);
#endif
    ICHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_name << " " << ecode.message();
    if (fmt == "o" || fmt == "obj") {
#if TVM_LLVM_VERSION <= 60
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(mptr_);
#else
      std::unique_ptr<llvm::Module> m = llvm::CloneModule(*mptr_);
#endif
      llvm::legacy::PassManager pass;
      ICHECK(tm_);
#if TVM_LLVM_VERSION <= 60
      ICHECK(tm_->addPassesToEmitFile(pass, dest, llvm::TargetMachine::CGFT_ObjectFile) == 0)
          << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
      ICHECK(tm_->addPassesToEmitFile(pass, dest, nullptr, llvm::TargetMachine::CGFT_ObjectFile) ==
             0)
          << "Cannot emit target CGFT_ObjectFile";
#else
      ICHECK(tm_->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile) == 0)
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
      ICHECK(tm_);
#if TVM_LLVM_VERSION <= 60
      ICHECK(tm_->addPassesToEmitFile(pass, dest, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
      ICHECK(tm_->addPassesToEmitFile(pass, dest, nullptr,
                                      llvm::TargetMachine::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#else
      ICHECK(tm_->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_AssemblyFile) == 0)
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
      LOG(FATAL) << "Do not know how to save file " << file_name << " with format=\'" << format
                 << "\'";
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
      ICHECK(tm_);
#if TVM_LLVM_VERSION <= 60
      ICHECK(tm_->addPassesToEmitFile(pass, rso, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
      ICHECK(tm_->addPassesToEmitFile(pass, rso, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) ==
             0)
          << "Cannot emit target CGFT_AssemblyFile";
#else
      ICHECK(tm_->addPassesToEmitFile(pass, rso, nullptr, llvm::CGFT_AssemblyFile) == 0)
          << "Cannot emit target CGFT_AssemblyFile";
#endif
      pass.run(*m);
      return rso.str().str();
    } else if (fmt == "" || fmt == "ll") {
      std::string type_str;
      llvm::raw_string_ostream rso(type_str);
      ICHECK(mptr_ != nullptr);
      mptr_->print(rso, nullptr);
      return rso.str();
    } else {
      LOG(FATAL) << "Do not know how to get source code with format: " << format << "\'";
    }
    return "";
  }

  void Init(const IRModule& mod, const Target& target) {
    InitializeLLVM();
    tm_ = GetLLVMTargetMachine(target);
    bool system_lib = target->GetAttr<Bool>("system-lib").value_or(Bool(false));
    bool target_c_runtime = (target->GetAttr<String>("runtime").value_or("") == kTvmRuntimeCrt);
    ctx_ = std::make_shared<llvm::LLVMContext>();
    std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(tm_.get());

    std::vector<PrimFunc> funcs;
    std::string entry_func;
    Map<String, LinkedParam> linked_params;
    bool found_linked_params = false;
    bool could_have_linked_params = target->GetAttr<Bool>("link-params").value_or(Bool(false));
    for (auto kv : mod->functions) {
      if (could_have_linked_params &&
          kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
        Map<String, ObjectRef> attrs_dict =
            Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
        CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
            << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";
        linked_params =
            Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
        found_linked_params = true;
        continue;
      }
      if (!kv.second->IsInstance<PrimFuncNode>()) {
        // (@jroesch): we relax constraints here, Relay functions will just be ignored.
        DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got "
                   << kv.second->GetTypeKey();
        continue;
      }
      auto f = Downcast<PrimFunc>(kv.second);
      auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(global_symbol.defined());
      function_names_.push_back(global_symbol.value());
      if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        entry_func = global_symbol.value();
      }
      funcs.push_back(f);
    }
    // TODO(@jroesch): follow up on this condition.
    // ICHECK(funcs.size() > 0 || (could_have_linked_params && found_linked_params));
    // TODO(tqchen): remove the entry function behavior as it does not
    // makes sense when we start to use multiple modules.
    cg->Init("TVMMod", tm_.get(), ctx_.get(), system_lib, system_lib, target_c_runtime);

    char* fast_math_flag_setting = getenv("TVM_FAST_MATH_FLAG");
    bool fast_math_flag;
    if (fast_math_flag_setting != nullptr) {
      fast_math_flag = atoi(fast_math_flag_setting);
    } else {
      fast_math_flag = false;
    }
    // For testing purposes only
    // cg->SetFastMathFlag(fast_math_flag);
    cg->SetFastMathFlag(true);

    cg->AddFunctionsOrdered(funcs.begin(), funcs.end());

    if (entry_func.length() != 0) {
      cg->AddMainFunction(entry_func);
    }

    if (found_linked_params) {
      cg->LinkParameters(linked_params);
    }
    module_ = cg->Finish();
    module_->addModuleFlag(llvm::Module::Warning, "tvm_target",
                           llvm::MDString::get(*ctx_, LLVMTargetToString(target)));
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

  void Init(std::unique_ptr<llvm::Module> module, std::shared_ptr<llvm::LLVMContext> ctx) {
    InitializeLLVM();
    ctx_ = ctx;
    llvm::SMDiagnostic err;
    module_ = std::move(module);
    if (module_ == nullptr) {
      std::string msg = std::string(err.getMessage());
      LOG(FATAL) << "Fail to load module: " << msg;
    }
    std::string target_metadata;
    llvm::Metadata* tvm_target = module_->getModuleFlag("tvm_target");
    if (tvm_target != nullptr) {
      llvm::MDString* pstr = llvm::dyn_cast<llvm::MDString>(tvm_target);
      ICHECK(pstr != nullptr);
      target_metadata = pstr->getString().str();
      if (!(target_metadata.length() >= 4 && target_metadata.substr(0, 4) == "llvm")) {
        target_metadata = "llvm " + target_metadata;
      }
    } else {
      std::ostringstream os;
      os << "llvm -mtriple " << module_->getTargetTriple();
      target_metadata = os.str();
    }
    mptr_ = module_.get();
    tm_ = GetLLVMTargetMachine(Target(target_metadata));
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
    if (!target_.defined()) {
      target_ = Target("llvm");
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
    std::unique_ptr<llvm::TargetMachine> tm_sys = GetLLVMTargetMachine(Target("llvm"));
    if (tm_sys->getTargetTriple().getArch() != tm->getTargetTriple().getArch()) {
      LOG(FATAL) << "Cannot run module, architecture mismatch "
                 << " module=" << tm->getTargetTriple().str()
                 << " system=" << tm_sys->getTargetTriple().str();
    }
    llvm::DataLayout layout(tm->createDataLayout());
    ICHECK(layout == mptr_->getDataLayout())
        << "Data layout mismatch between module("
        << mptr_->getDataLayout().getStringRepresentation() << ")"
        << " and ExecutionEngine (" << layout.getStringRepresentation() << ")";
    ee_ = builder.create(tm.release());
    ICHECK(ee_ != nullptr) << "Failed to initialize jit engine for " << mptr_->getTargetTriple();
    ee_->runStaticConstructorsDestructors(false);

    if (void** ctx_addr =
            reinterpret_cast<void**>(GetGlobalAddr(runtime::symbol::tvm_module_ctx))) {
      *ctx_addr = this;
    }
    runtime::InitContextFunctions(
        [this](const char* name) { return reinterpret_cast<void*>(GetGlobalAddr(name)); });
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
  Target target_;
  // JIT lock
  std::mutex mutex_;
  // execution engine
  llvm::ExecutionEngine* ee_{nullptr};
  // The raw pointer to the module.
  llvm::Module* mptr_{nullptr};
  // The target machine
  std::unique_ptr<llvm::TargetMachine> tm_{nullptr};
  // The module, can be moved to ee if JIT is enabled.
  std::unique_ptr<llvm::Module> module_;
  // the context.
  std::shared_ptr<llvm::LLVMContext> ctx_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
};

TVM_REGISTER_GLOBAL("target.build.llvm")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->Init(mod, target);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("codegen.LLVMModuleCreate")
    .set_body_typed([](std::string target_str, std::string module_name) -> runtime::Module {
      Target target = Target(target_str);
      auto n = make_object<LLVMModuleNode>();
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
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("target.llvm_lookup_intrinsic_id")
    .set_body_typed([](std::string name) -> int64_t {
      return static_cast<int64_t>(llvm::Function::lookupIntrinsicID(name));
    });

TVM_REGISTER_GLOBAL("target.llvm_version_major").set_body_typed([]() -> int {
  return TVM_LLVM_VERSION / 10;
});

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ll")
    .set_body_typed([](std::string filename, std::string fmt) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->LoadIR(filename);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("codegen.llvm_target_enabled")
    .set_body_typed([](std::string target_str) -> bool {
      InitializeLLVM();
      Target target = Target(target_str);
      return (GetLLVMTargetMachine(target, true) != nullptr);
    });

TVM_REGISTER_GLOBAL("codegen.codegen_blob")
    .set_body_typed([](std::string data, bool system_lib,
                       std::string target_triple) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      auto p = CodeGenBlob(data, system_lib, target_triple);
      n->Init(std::move(p.first), p.second);
      return runtime::Module(n);
    });

runtime::Module CreateLLVMCrtMetadataModule(const Array<runtime::Module>& modules, Target target) {
  Array<String> func_names;
  for (runtime::Module mod : modules) {
    auto pf_funcs = mod.GetFunction("get_func_names");
    if (pf_funcs != nullptr) {
      Array<String> func_names_ = pf_funcs();
      for (const auto& fname : func_names_) {
        func_names.push_back(fname);
      }
    }
  }

  InitializeLLVM();
  auto tm = GetLLVMTargetMachine(target);
  bool system_lib = target->GetAttr<Bool>("system-lib").value_or(Bool(false));
  bool target_c_runtime = (target->GetAttr<String>("runtime").value_or("") == kTvmRuntimeCrt);
  ICHECK(system_lib && target_c_runtime)
      << "For LLVM C-runtime metadata module, must include --system-lib and --runtime=c; "
      << "got target: " << target->str();
  auto ctx = std::make_shared<llvm::LLVMContext>();
  std::unique_ptr<CodeGenCPU> cg{new CodeGenCPU()};
  cg->Init("TVMMetadataMod", tm.get(), ctx.get(), system_lib, system_lib, target_c_runtime);

  cg->DefineFunctionRegistry(func_names);
  auto mod = cg->Finish();
  mod->addModuleFlag(llvm::Module::Warning, "tvm_target",
                     llvm::MDString::get(*ctx, LLVMTargetToString(target)));
  mod->addModuleFlag(llvm::Module::Override, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);

  if (tm->getTargetTriple().isOSDarwin()) {
    mod->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
  }

  std::string verify_errors_storage;
  llvm::raw_string_ostream verify_errors(verify_errors_storage);
  LOG_IF(FATAL, llvm::verifyModule(*mod, &verify_errors))
      << "LLVM module verification failed with the following errors: \n"
      << verify_errors.str();

  auto n = make_object<LLVMModuleNode>();
  n->Init(std::move(mod), ctx);
  for (auto m : modules) {
    n->Import(m);
  }
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CreateLLVMCrtMetadataModule")
    .set_body_typed([](const Array<runtime::Module>& modules, Target target) {
      return CreateLLVMCrtMetadataModule(modules, target);
    });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
