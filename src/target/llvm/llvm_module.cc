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

#include <dmlc/io.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <tvm/ffi/reflection/registry.h>
#if _WIN32
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#endif
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#if TVM_LLVM_VERSION >= 180
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/support/with.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../../runtime/file_utils.h"
#include "codegen_blob.h"
#include "codegen_cpu.h"
#include "codegen_llvm.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

class LLVMModuleNode final : public ffi::ModuleObj {
 public:
  ~LLVMModuleNode();

  const char* kind() const final { return "llvm"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  /*! \brief Get the property of the runtime module .*/
  // TODO(tvm-team): Make it serializable
  int GetPropertyMask() const override {
    return ffi::Module::kRunnable | ffi::Module::kCompilationExportable;
  }

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final;
  ffi::Bytes SaveToBytes() const final;
  ffi::String InspectSource(const ffi::String& format) const final;

  void Init(const IRModule& mod, const Target& target);
  void Init(std::unique_ptr<llvm::Module> module, std::unique_ptr<LLVMInstance> llvm_instance);
  void LoadIR(const std::string& file_name);

  bool ImplementsFunction(const ffi::String& name) final;

  void SetJITEngine(const std::string& jit_engine) { jit_engine_ = jit_engine; }

 private:
  void InitMCJIT();
  void InitORCJIT();
  bool IsCompatibleWithHost(const llvm::TargetMachine* tm) const;
  void* GetGlobalAddr(const std::string& name, const LLVMTarget& llvm_target) const;
  void* GetFunctionAddr(const std::string& name, const LLVMTarget& llvm_target) const;

  // The LLVM scope object.
  std::unique_ptr<LLVMInstance> llvm_instance_;
  // JIT lock
  std::mutex mutex_;
  // jit execution engines
  llvm::ExecutionEngine* mcjit_ee_{nullptr};
  std::unique_ptr<llvm::orc::LLJIT> orcjit_ee_{nullptr};
  // The raw pointer to the module.
  llvm::Module* module_{nullptr};
  // The unique_ptr owning the module. This becomes empty once JIT has been initialized
  // (EngineBuilder takes ownership of the module).
  std::unique_ptr<llvm::Module> module_owning_ptr_;
  /* \brief names of the external functions declared in this module */
  ffi::Array<ffi::String> function_names_;
  std::string jit_engine_;
};

LLVMModuleNode::~LLVMModuleNode() {
  if (mcjit_ee_ != nullptr) {
    mcjit_ee_->runStaticConstructorsDestructors(true);
    delete mcjit_ee_;
  }
  if (orcjit_ee_ != nullptr) {
    auto dtors = llvm::orc::getDestructors(*module_);
    auto dtorRunner = std::make_unique<llvm::orc::CtorDtorRunner>(orcjit_ee_->getMainJITDylib());
    dtorRunner->add(dtors);
    auto err = dtorRunner->run();
    ICHECK(!err) << llvm::toString(std::move(err));
    orcjit_ee_.reset();
  }
  module_owning_ptr_.reset();
}

ffi::Optional<ffi::Function> LLVMModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
  if (name == "__tvm_is_system_module") {
    bool flag = (module_->getFunction("__tvm_module_startup") != nullptr);
    return ffi::Function([flag](ffi::PackedArgs args, ffi::Any* rv) { *rv = flag; });
  } else if (name == "__tvm_get_system_lib_prefix") {
    return ffi::Function([this](ffi::PackedArgs args, ffi::Any* rv) {
      auto* md = module_->getModuleFlag("tvm_system_lib_prefix");
      if (md != nullptr) {
        *rv = llvm::cast<llvm::MDString>(md)->getString().str();
      } else {
        *rv = nullptr;
      }
    });
  } else if (name == "get_func_names") {
    return ffi::Function(
        [sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) { *rv = this->function_names_; });
  } else if (name == "get_symbol") {
    return std::nullopt;
  } else if (name == "get_const_vars") {
    return std::nullopt;
  } else if (name == "_get_target_string") {
    std::string target_string = LLVMTarget::GetTargetMetadata(*module_);
    return ffi::Function(
        [target_string](ffi::PackedArgs args, ffi::Any* rv) { *rv = target_string; });
  }
  ICHECK(jit_engine_.size()) << "JIT engine type is missing";
  if ((jit_engine_ == "mcjit") && (mcjit_ee_ == nullptr)) InitMCJIT();
  if ((jit_engine_ == "orcjit") && (orcjit_ee_ == nullptr)) InitORCJIT();

  std::lock_guard<std::mutex> lock(mutex_);

  TVMFFISafeCallType faddr;
  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  ffi::String name_with_prefix = ffi::symbol::tvm_ffi_symbol_prefix + name;
  faddr = reinterpret_cast<TVMFFISafeCallType>(GetFunctionAddr(name_with_prefix, *llvm_target));
  if (faddr == nullptr) return std::nullopt;
  ffi::Module self_strong_ref = ffi::GetRef<ffi::Module>(this);
  return ffi::Function::FromPacked([faddr, self_strong_ref](ffi::PackedArgs args, ffi::Any* rv) {
    TVM_FFI_ICHECK_LT(rv->type_index(), ffi::TypeIndex::kTVMFFIStaticObjectBegin);
    TVM_FFI_CHECK_SAFE_CALL((*faddr)(nullptr, reinterpret_cast<const TVMFFIAny*>(args.data()),
                                     args.size(), reinterpret_cast<TVMFFIAny*>(rv)));
  });
}

namespace {
#if TVM_LLVM_VERSION <= 70
constexpr auto llvm_open_output_flag = llvm::sys::fs::F_None;
#else
constexpr auto llvm_open_output_flag = llvm::sys::fs::OF_None;
#endif

#if TVM_LLVM_VERSION <= 60
std::unique_ptr<llvm::Module> CloneLLVMModule(llvm::Module* mod) { return llvm::CloneModule(mod); }
#else
std::unique_ptr<llvm::Module> CloneLLVMModule(llvm::Module* mod) { return llvm::CloneModule(*mod); }
#endif

#if TVM_LLVM_VERSION <= 90
constexpr auto llvm_object_file_target = llvm::TargetMachine::CGFT_ObjectFile;
constexpr auto llvm_assembly_file_target = llvm::TargetMachine::CGFT_AssemblyFile;
#elif TVM_LLVM_VERSION <= 170
constexpr auto llvm_object_file_target = llvm::CGFT_ObjectFile;
constexpr auto llvm_assembly_file_target = llvm::CGFT_AssemblyFile;
#else
constexpr auto llvm_object_file_target = llvm::CodeGenFileType::ObjectFile;
constexpr auto llvm_assembly_file_target = llvm::CodeGenFileType::AssemblyFile;
#endif

bool LLVMAddPassesToEmitFile(llvm::TargetMachine* tm, llvm::legacy::PassManager* pm,
                             llvm::raw_fd_ostream* dest,
                             decltype(llvm_object_file_target) llvm_file_target) {
#if TVM_LLVM_VERSION <= 60
  return tm->addPassesToEmitFile(*pm, *dest, llvm_file_target);
#else
  return tm->addPassesToEmitFile(*pm, *dest, nullptr, llvm_file_target);
#endif
}

}  // namespace

void LLVMModuleNode::WriteToFile(const ffi::String& file_name_str,
                                 const ffi::String& format) const {
  // CHECK(imports_.empty()) << "SaveToFile does not handle imported modules";
  std::string file_name = file_name_str;
  std::string fmt = runtime::GetFileFormat(file_name, format);
  std::error_code ecode;
  llvm::raw_fd_ostream dest(file_name, ecode, llvm_open_output_flag);
  ICHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_name << " " << ecode.message();
  bool is_obj_file = fmt == "o" || fmt == "obj";
  bool is_asm_file = fmt == "s" || fmt == "asm";
  if (is_obj_file || is_asm_file) {
    auto llvm_file_target = is_obj_file ? llvm_object_file_target : llvm_assembly_file_target;

    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();

    auto err = LLVMAddPassesToEmitFile(tm, &pass, &dest, llvm_file_target);
    ICHECK(!err) << "Cannot emit target CGFT_ObjectFile";

    pass.run(*CloneLLVMModule(module_));
  } else if (fmt == "ll") {
    module_->print(dest, nullptr);
  } else if (fmt == "bc") {
#if TVM_LLVM_VERSION <= 60
    llvm::WriteBitcodeToFile(module_, dest);
#else
    llvm::WriteBitcodeToFile(*module_, dest);
#endif
  } else {
    LOG(FATAL) << "Do not know how to save file " << file_name << " with format=\'" << format
               << "\'";
  }
  dest.close();
}

ffi::Bytes LLVMModuleNode::SaveToBytes() const {
  LOG(FATAL) << "LLVMModule: SaveToBytes not supported";
}

ffi::String LLVMModuleNode::InspectSource(const ffi::String& format) const {
  std::string fmt = runtime::GetFileFormat("", format);
  std::string type_str;
  llvm::SmallString<256> str;
  llvm::raw_svector_ostream rso(str);

  if (fmt == "s" || fmt == "asm") {
    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
#if TVM_LLVM_VERSION <= 60
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(module_);
#else
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(*module_);
#endif
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
#if TVM_LLVM_VERSION <= 60
    ICHECK(tm->addPassesToEmitFile(pass, rso, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
    ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 170
    ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#else
    ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::CodeGenFileType::AssemblyFile) == 0)
        << "Cannot emit target CodeGenFileType::AssemblyFile";
#endif
    pass.run(*m);
    return rso.str().str();
  } else if (fmt == "" || fmt == "ll") {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    ICHECK(module_ != nullptr);
    module_->print(rso, nullptr);
    return rso.str();
  } else {
    LOG(FATAL) << "Do not know how to get source code with format: " << format << "\'";
  }
  return "";
}

void LLVMModuleNode::Init(const IRModule& mod, const Target& target) {
  llvm_instance_ = std::make_unique<LLVMInstance>();
  With<LLVMTarget> llvm_target(*llvm_instance_, target);
  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(llvm_target.get());

  std::string entry_func;

  ffi::Optional<ffi::String> system_lib_prefix =
      mod->GetAttr<ffi::String>(tvm::attr::kSystemLibPrefix);

  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got " << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    bool is_entry_func = f->HasNonzeroAttr(tir::attr::kIsEntryFunc);

    ICHECK(global_symbol || !is_entry_func) << "The entry func must be exposed externally.";

    if (global_symbol) {
      function_names_.push_back(global_symbol.value());
      if (is_entry_func) {
        entry_func = global_symbol.value();
      }
    }
  }
  // TODO(@jroesch): follow up on this condition.
  // ICHECK(funcs.size() > 0);
  // TODO(tqchen): remove the entry function behavior as it does not
  // makes sense when we start to use multiple modules.
  cg->Init("TVMMod", llvm_target.get(), system_lib_prefix, system_lib_prefix.has_value(), false);
  cg->SetFastMathFlags(llvm_target->GetFastMathFlags());
  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  module_owning_ptr_ = cg->Finish();
  module_ = module_owning_ptr_.get();
  jit_engine_ = llvm_target->GetJITEngine();
  llvm_target->SetTargetMetadata(module_);
  module_->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                         llvm::DEBUG_METADATA_VERSION);

  if (system_lib_prefix) {
    std::string str_val = system_lib_prefix.value();
    module_->addModuleFlag(llvm::Module::Warning, "tvm_system_lib_prefix",
                           llvm::MDString::get(*(llvm_target->GetContext()), str_val));
  }

  module_->addModuleFlag(llvm::Module::Override, "Dwarf Version",
                         tm->getTargetTriple().isOSDarwin() ? 2 : 4);
}

void LLVMModuleNode::Init(std::unique_ptr<llvm::Module> module,
                          std::unique_ptr<LLVMInstance> llvm_instance) {
  module_owning_ptr_ = std::move(module);
  module_ = module_owning_ptr_.get();
  llvm_instance_ = std::move(llvm_instance);
}

void LLVMModuleNode::LoadIR(const std::string& file_name) {
  auto llvm_instance = std::make_unique<LLVMInstance>();
  std::unique_ptr<llvm::Module> module = llvm_instance->LoadIR(file_name);
  Init(std::move(module), std::move(llvm_instance));
}

bool LLVMModuleNode::ImplementsFunction(const ffi::String& name) {
  return std::find(function_names_.begin(), function_names_.end(),
                   ffi::symbol::tvm_ffi_symbol_prefix + name) != function_names_.end();
}

void LLVMModuleNode::InitMCJIT() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (mcjit_ee_) {
    return;
  }
  // MCJIT builder
  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  llvm::EngineBuilder builder(std::move(module_owning_ptr_));

  // set options
  builder.setEngineKind(llvm::EngineKind::JIT);
#if TVM_LLVM_VERSION <= 170
  builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
#else
  builder.setOptLevel(llvm::CodeGenOptLevel::Aggressive);
#endif
  builder.setMCPU(llvm_target->GetCPU());
  builder.setMAttrs(llvm_target->GetTargetFeatures());
  builder.setTargetOptions(llvm_target->GetTargetOptions());

  // create the taget machine
  auto tm = std::unique_ptr<llvm::TargetMachine>(builder.selectTarget());
  if (!IsCompatibleWithHost(tm.get())) {
    LOG(FATAL) << "Cannot run module, architecture mismatch";
  }

  // data layout
  llvm::DataLayout layout(tm->createDataLayout());
  ICHECK(layout == module_->getDataLayout())
      << "Data layout mismatch between module("
      << module_->getDataLayout().getStringRepresentation() << ")"
      << " and ExecutionEngine (" << layout.getStringRepresentation() << ")";

  // create MCJIT
  mcjit_ee_ = builder.create(tm.release());
  ICHECK(mcjit_ee_ != nullptr) << "Failed to initialize LLVM MCJIT engine for "
#if TVM_LLVM_VERSION >= 210
                               << module_->getTargetTriple().str();
#else
                               << module_->getTargetTriple();
#endif

  VLOG(2) << "LLVM MCJIT execute " << module_->getModuleIdentifier() << " for triple `"
          << llvm_target->GetTargetTriple() << "`"
          << " on cpu `" << llvm_target->GetCPU() << "`";

  // run ctors
  mcjit_ee_->runStaticConstructorsDestructors(false);

  if (void** ctx_addr =
          reinterpret_cast<void**>(GetGlobalAddr(ffi::symbol::tvm_ffi_library_ctx, *llvm_target))) {
    *ctx_addr = this;
  }

  ffi::Module::VisitContextSymbols([this, &llvm_target](const ffi::String& name, void* symbol) {
    if (void** ctx_addr = reinterpret_cast<void**>(GetGlobalAddr(name, *llvm_target))) {
      *ctx_addr = symbol;
    }
  });
  // There is a problem when a JITed function contains a call to a runtime function.
  // The runtime function (e.g. __truncsfhf2) may not be resolved, and calling it will
  // lead to a runtime crash.
  // Do name lookup on a symbol that doesn't exist. This will force MCJIT to finalize
  // all loaded objects, which will resolve symbols in JITed code.
  mcjit_ee_->getFunctionAddress(
      "__some_name_that_hopefully_doesnt_exist__b49f8aaade5877eaba7583b91");
}

void LLVMModuleNode::InitORCJIT() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orcjit_ee_) {
    return;
  }
  // ORCJIT builder
  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  llvm::orc::JITTargetMachineBuilder tm_builder(llvm::Triple(llvm_target->GetTargetTriple()));

  // set options
  tm_builder.setCPU(llvm_target->GetCPU());
  tm_builder.setFeatures(llvm_target->GetTargetFeatureString());
  tm_builder.setOptions(llvm_target->GetTargetOptions());
#if TVM_LLVM_VERSION <= 170
  tm_builder.setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);
#else
  tm_builder.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
#endif

  // Default is no explicit JIT code & reloc model
  // Propagate instance code & reloc for RISCV case.
  auto arch = tm_builder.getTargetTriple().getArch();
  if (arch == llvm::Triple::riscv32 || arch == llvm::Triple::riscv64) {
    tm_builder.setRelocationModel(llvm_target->GetTargetRelocModel());
    tm_builder.setCodeModel(llvm_target->GetTargetCodeModel());
  }

  // create the taget machine
  std::unique_ptr<llvm::TargetMachine> tm = llvm::cantFail(tm_builder.createTargetMachine());
  if (!IsCompatibleWithHost(tm.get())) {
    LOG(FATAL) << "Cannot run module, architecture mismatch";
  }

  // data layout
  ffi::String module_name = module_->getModuleIdentifier();
  llvm::DataLayout layout(tm->createDataLayout());
  ICHECK(layout == module_->getDataLayout())
      << "Data layout mismatch between module("
      << module_->getDataLayout().getStringRepresentation() << ")"
      << " and ExecutionEngine (" << layout.getStringRepresentation() << ")";

  // compiler
  const auto compilerBuilder = [&](const llvm::orc::JITTargetMachineBuilder&)
      -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(tm));
  };

#if TVM_LLVM_VERSION >= 130
  // linker
  const auto linkerBuilder =
#if TVM_LLVM_VERSION >= 210
      [&](llvm::orc::ExecutionSession& session)
      -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
#else
      [&](llvm::orc::ExecutionSession& session,
          const llvm::Triple& triple) -> std::unique_ptr<llvm::orc::ObjectLayer> {
#endif
#if _WIN32
#if TVM_LLVM_VERSION >= 210
    auto GetMemMgr = [](const llvm::MemoryBuffer&) {
      return std::make_unique<llvm::SectionMemoryManager>();
    };
#else
    auto GetMemMgr = []() { return std::make_unique<llvm::SectionMemoryManager>(); };
#endif
    auto ObjLinkingLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, std::move(GetMemMgr));
#else
    auto ObjLinkingLayer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);
#endif
#if TVM_LLVM_VERSION >= 210
    if (tm_builder.getTargetTriple().isOSBinFormatCOFF()) {
#else
    if (triple.isOSBinFormatCOFF()) {
#endif
      ObjLinkingLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      ObjLinkingLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }
#if TVM_LLVM_VERSION >= 210
    return llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>(std::move(ObjLinkingLayer));
#else
    return ObjLinkingLayer;
#endif
  };  // NOLINT(readability/braces)
#endif

  // create LLJIT
  orcjit_ee_ = llvm::cantFail(llvm::orc::LLJITBuilder()
#if TVM_LLVM_VERSION >= 110
                                  .setDataLayout(layout)
#endif
                                  .setCompileFunctionCreator(compilerBuilder)
#if TVM_LLVM_VERSION >= 130
                                  .setObjectLinkingLayerCreator(linkerBuilder)
#endif
                                  .create());

  ICHECK(orcjit_ee_ != nullptr) << "Failed to initialize LLVM ORCJIT engine for "
#if TVM_LLVM_VERSION >= 210
                                << module_->getTargetTriple().str();
#else
                                << module_->getTargetTriple();
#endif

  // store ctors
  auto ctors = llvm::orc::getConstructors(*module_);
  llvm::orc::CtorDtorRunner ctorRunner(orcjit_ee_->getMainJITDylib());
  ctorRunner.add(ctors);

  // resolve system symbols (like pthread, dl, m, etc.)
  auto gen =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(layout.getGlobalPrefix());
  ICHECK(gen) << llvm::toString(gen.takeError()) << "\n";
  orcjit_ee_->getMainJITDylib().addGenerator(std::move(gen.get()));

  // transfer module to a clone
  auto uctx = std::make_unique<llvm::LLVMContext>();
  auto umod = llvm::CloneModule(*(std::move(module_owning_ptr_)));

  // add the llvm module to run
  llvm::orc::ThreadSafeModule tsm(std::move(umod), std::move(uctx));
  auto err = orcjit_ee_->addIRModule(std::move(tsm));
  ICHECK(!err) << llvm::toString(std::move(err));

  VLOG(2) << "LLVM ORCJIT execute " << module_->getModuleIdentifier() << " for triple `"
          << llvm_target->GetTargetTriple() << "`"
          << " on cpu `" << llvm_target->GetCPU() << "`";

  // run ctors
  err = ctorRunner.run();
  ICHECK(!err) << llvm::toString(std::move(err));

  if (void** ctx_addr =
          reinterpret_cast<void**>(GetGlobalAddr(ffi::symbol::tvm_ffi_library_ctx, *llvm_target))) {
    *ctx_addr = this;
  }
  ffi::Module::VisitContextSymbols([this, &llvm_target](const ffi::String& name, void* symbol) {
    if (void** ctx_addr = reinterpret_cast<void**>(GetGlobalAddr(name, *llvm_target))) {
      *ctx_addr = symbol;
    }
  });
}

bool LLVMModuleNode::IsCompatibleWithHost(const llvm::TargetMachine* tm) const {
  LLVMTargetInfo host_target(*llvm_instance_, "llvm");
  auto tm_host = host_target.GetOrCreateTargetMachine();
  if (tm_host->getTargetTriple().getArch() != tm->getTargetTriple().getArch()) {
    LOG(INFO) << "Architecture mismatch: module=" << tm->getTargetTriple().str()
              << " host=" << tm_host->getTargetTriple().str();
    return false;
  }
  return true;
}

// Get global address from execution engine.
void* LLVMModuleNode::GetGlobalAddr(const std::string& name, const LLVMTarget& llvm_target) const {
  // first verifies if GV exists.
  if (module_->getGlobalVariable(name) != nullptr) {
    if (jit_engine_ == "mcjit") {
      return reinterpret_cast<void*>(mcjit_ee_->getGlobalValueAddress(name));
    } else if (jit_engine_ == "orcjit") {
#if TVM_LLVM_VERSION >= 150
      auto addr = llvm::cantFail(orcjit_ee_->lookup(name)).getValue();
#else
      auto addr = llvm::cantFail(orcjit_ee_->lookup(name)).getAddress();
#endif
      return reinterpret_cast<void*>(addr);
    } else {
      LOG(FATAL) << "Either `mcjit` or `orcjit` are not initialized.";
    }
  }
  return nullptr;
}

void* LLVMModuleNode::GetFunctionAddr(const std::string& name,
                                      const LLVMTarget& llvm_target) const {
  // first verifies if GV exists.
  if (module_->getFunction(name) != nullptr) {
    if (jit_engine_ == "mcjit") {
      return reinterpret_cast<void*>(mcjit_ee_->getFunctionAddress(name));
    } else if (jit_engine_ == "orcjit") {
#if TVM_LLVM_VERSION >= 150
      auto addr = llvm::cantFail(orcjit_ee_->lookup(name)).getValue();
#else
      auto addr = llvm::cantFail(orcjit_ee_->lookup(name)).getAddress();
#endif
      return reinterpret_cast<void*>(addr);
    } else {
      LOG(FATAL) << "Either `mcjit` or `orcjit` are not initialized.";
    }
  }
  return nullptr;
}

static void LLVMReflectionRegister() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.llvm",
           [](IRModule mod, Target target) -> ffi::Module {
             auto n = ffi::make_object<LLVMModuleNode>();
             n->Init(mod, target);
             return ffi::Module(n);
           })
      .def("codegen.LLVMModuleCreate",
           [](std::string target_str, std::string module_name) -> ffi::Module {
             auto llvm_instance = std::make_unique<LLVMInstance>();
             With<LLVMTarget> llvm_target(*llvm_instance, target_str);
             auto n = ffi::make_object<LLVMModuleNode>();
             // Generate a LLVM module from an input target string
             auto module = std::make_unique<llvm::Module>(module_name, *llvm_target->GetContext());
             llvm_target->SetTargetMetadata(module.get());
#if TVM_LLVM_VERSION >= 210
             module->setTargetTriple(llvm::Triple(llvm_target->GetTargetTriple()));
#else
             module->setTargetTriple(llvm_target->GetTargetTriple());
#endif
             module->setDataLayout(llvm_target->GetOrCreateTargetMachine()->createDataLayout());
             n->Init(std::move(module), std::move(llvm_instance));
             n->SetJITEngine(llvm_target->GetJITEngine());
             return ffi::Module(n);
           })
      .def("target.llvm_lookup_intrinsic_id",
           [](std::string name) -> int64_t {
#if TVM_LLVM_VERSION >= 200
             return static_cast<int64_t>(llvm::Intrinsic::lookupIntrinsicID(name));
#else
      return static_cast<int64_t>(llvm::Function::lookupIntrinsicID(name));
#endif
           })
      .def("target.llvm_get_intrinsic_name",
           [](int64_t id) -> ffi::String { return llvmGetIntrinName(id); })
      .def("target.llvm_get_system_x86_vendor",
           []() -> ffi::String {
#if TVM_LLVM_VERSION >= 120
#if defined(__i386__) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
             using namespace llvm::sys::detail::x86;
             const auto x86_sign = getVendorSignature();
             if (x86_sign == VendorSignatures::GENUINE_INTEL)
               return "intel";
             else if (x86_sign == VendorSignatures::AUTHENTIC_AMD)
               return "amd";
             else if (x86_sign == VendorSignatures::UNKNOWN)
               return "unknown";
#endif
#endif
             return "unimplemented";
           })
      .def("target.llvm_get_vector_width",
           [](const Target& target) -> int {
             auto use_target = target.defined() ? target : Target::Current(false);
             // ignore non "llvm" target
             if (target.defined()) {
               if (target->kind->name != "llvm") {
                 return -1;
               }
             }
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_backend(*llvm_instance, use_target);
             return llvm_backend.GetVectorWidth();
           })
      .def("target.llvm_get_system_triple",
           []() -> ffi::String { return llvm::sys::getDefaultTargetTriple(); })
      .def("target.llvm_get_system_cpu",
           []() -> ffi::String { return llvm::sys::getHostCPUName().str(); })
      .def("target.llvm_get_targets",
           []() -> ffi::Array<ffi::String> {
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_backend(*llvm_instance, "llvm");
             return llvm_backend.GetAllLLVMTargets();
           })
      .def("target.llvm_get_cpu_archlist",
           [](const Target& target) -> ffi::Array<ffi::String> {
             auto use_target = target.defined() ? target : Target::Current(false);
             // ignore non "llvm" target
             if (target.defined()) {
               if (target->kind->name != "llvm") {
                 return ffi::Array<ffi::String>{};
               }
             }
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_backend(*llvm_instance, use_target);
             return llvm_backend.GetAllLLVMTargetArches();
           })
      .def("target.llvm_get_cpu_features",
           [](const Target& target) -> ffi::Map<ffi::String, ffi::String> {
             auto use_target = target.defined() ? target : Target::Current(false);
             // ignore non "llvm" target
             if (target.defined()) {
               if (target->kind->name != "llvm") {
                 return {};
               }
             }
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_backend(*llvm_instance, use_target);
             return llvm_backend.GetAllLLVMCpuFeatures();
           })
      .def("target.llvm_cpu_has_feature",
           [](const ffi::String feature, const Target& target) -> bool {
             auto use_target = target.defined() ? target : Target::Current(false);
             // ignore non "llvm" target
             if (target.defined()) {
               if (target->kind->name != "llvm") {
                 return false;
               }
             }
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_backend(*llvm_instance, use_target);
             auto cpu_features = llvm_backend.GetAllLLVMCpuFeatures();
             bool has_feature = cpu_features.find(feature) != cpu_features.end();
             return has_feature;
           })
      .def("target.target_has_feature",
           [](const ffi::String feature, const Target& target) -> bool {
             auto use_target = target.defined() ? target : Target::Current(false);
             // ignore non "llvm" target
             if (target.defined()) {
               if (target->kind->name != "llvm") {
                 return false;
               }
             }
             auto llvm_instance = std::make_unique<LLVMInstance>();
             LLVMTargetInfo llvm_target(*llvm_instance, use_target);
             return llvm_target.TargetHasCPUFeature(feature);
           })
      .def("target.llvm_version_major", []() -> int { return TVM_LLVM_VERSION / 10; })
      .def("ffi.Module.load_from_file.ll",
           [](std::string filename, std::string fmt) -> ffi::Module {
             auto n = ffi::make_object<LLVMModuleNode>();
             n->SetJITEngine("orcjit");
             n->LoadIR(filename);
             return ffi::Module(n);
           })
      .def("codegen.llvm_target_enabled",
           [](std::string target_str) -> bool {
             LLVMInstance llvm_instance;
             auto* tm = With<LLVMTarget>(llvm_instance, target_str)
                            ->GetOrCreateTargetMachine(/*allow_missing=*/true);
             return tm != nullptr;
           })
      .def("codegen.codegen_blob",
           [](std::string data, bool system_lib, std::string llvm_target_string,
              std::string c_symbol_prefix) -> ffi::Module {
             auto n = ffi::make_object<LLVMModuleNode>();
             auto llvm_instance = std::make_unique<LLVMInstance>();
             With<LLVMTarget> llvm_target(*llvm_instance, llvm_target_string);
             std::unique_ptr<llvm::Module> blob =
                 CodeGenBlob(data, system_lib, llvm_target.get(), c_symbol_prefix);
             n->Init(std::move(blob), std::move(llvm_instance));
             n->SetJITEngine(llvm_target->GetJITEngine());
             return ffi::Module(n);
           });
}

TVM_FFI_STATIC_INIT_BLOCK() { LLVMReflectionRegister(); }

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
