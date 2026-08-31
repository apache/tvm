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

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/io.h>
#if TVM_LLVM_VERSION >= 180
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/ir/with_context.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <memory>
#include <string>
#include <system_error>
#include <utility>

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
  const char* kind() const final { return "llvm"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  void ImportModule(const ffi::Module& other) final;

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

 private:
  void EnsureOrcJITModule();

  // The LLVM scope object.
  std::unique_ptr<LLVMInstance> llvm_instance_;
  // The package-owned JITDylib, initialized lazily from the retained LLVM module.
  ffi::Optional<ffi::Module> tvm_ffi_orcjit_module_;
  // The retained module IR used for inspection and object emission.
  std::unique_ptr<llvm::Module> module_;
  /* \brief names of the external functions declared in this module */
  ffi::Array<ffi::String> function_names_;
};

ffi::Optional<ffi::Function> LLVMModuleNode::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
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
  EnsureOrcJITModule();
  return tvm_ffi_orcjit_module_.value()->GetFunction(name);
}

void LLVMModuleNode::ImportModule(const ffi::Module& other) {
  ffi::ModuleObj::ImportModule(other);
  if (tvm_ffi_orcjit_module_.has_value()) {
    tvm_ffi_orcjit_module_.value()->ImportModule(other);
  }
}

namespace {
constexpr auto llvm_open_output_flag = llvm::sys::fs::OF_None;

std::unique_ptr<llvm::Module> CloneLLVMModule(const llvm::Module* mod) {
  return llvm::CloneModule(*mod);
}

#if TVM_LLVM_VERSION <= 170
constexpr auto llvm_object_file_target = llvm::CGFT_ObjectFile;
constexpr auto llvm_assembly_file_target = llvm::CGFT_AssemblyFile;
#else
constexpr auto llvm_object_file_target = llvm::CodeGenFileType::ObjectFile;
constexpr auto llvm_assembly_file_target = llvm::CodeGenFileType::AssemblyFile;
#endif

bool LLVMAddPassesToEmitFile(llvm::TargetMachine* tm, llvm::legacy::PassManager* pm,
                             llvm::raw_pwrite_stream* dest,
                             decltype(llvm_object_file_target) llvm_file_target) {
  return tm->addPassesToEmitFile(*pm, *dest, nullptr, llvm_file_target);
}

}  // namespace

void LLVMModuleNode::WriteToFile(const ffi::String& file_name_str,
                                 const ffi::String& format) const {
  // TVM_FFI_ICHECK(imports_.empty()) << "SaveToFile does not handle imported modules";
  std::string file_name = file_name_str;
  std::string fmt = runtime::GetFileFormat(file_name, format);
  std::error_code ecode;
  llvm::raw_fd_ostream dest(file_name, ecode, llvm_open_output_flag);
  TVM_FFI_ICHECK_EQ(ecode.value(), 0)
      << "Cannot open file: " << file_name << " " << ecode.message();
  bool is_obj_file = fmt == "o" || fmt == "obj";
  bool is_asm_file = fmt == "s" || fmt == "asm";
  if (is_obj_file || is_asm_file) {
    auto llvm_file_target = is_obj_file ? llvm_object_file_target : llvm_assembly_file_target;

    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();

    auto err = LLVMAddPassesToEmitFile(tm, &pass, &dest, llvm_file_target);
    TVM_FFI_ICHECK(!err) << "Cannot emit target CGFT_ObjectFile";

    pass.run(*CloneLLVMModule(module_.get()));
  } else if (fmt == "ll") {
    module_->print(dest, nullptr);
  } else if (fmt == "bc") {
    llvm::WriteBitcodeToFile(*module_, dest);
  } else {
    TVM_FFI_THROW(InternalError) << "Do not know how to save file " << file_name
                                 << " with format=\'" << format << "\'";
  }
  dest.close();
}

ffi::Bytes LLVMModuleNode::SaveToBytes() const {
  TVM_FFI_THROW(InternalError) << "LLVMModule: SaveToBytes not supported";
}

ffi::String LLVMModuleNode::InspectSource(const ffi::String& format) const {
  std::string fmt = runtime::GetFileFormat("", format);
  std::string type_str;
  llvm::SmallString<256> str;
  llvm::raw_svector_ostream rso(str);

  if (fmt == "s" || fmt == "asm") {
    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(*module_);
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
#if TVM_LLVM_VERSION <= 170
    TVM_FFI_ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#else
    TVM_FFI_ICHECK(
        tm->addPassesToEmitFile(pass, rso, nullptr, llvm::CodeGenFileType::AssemblyFile) == 0)
        << "Cannot emit target CodeGenFileType::AssemblyFile";
#endif
    pass.run(*m);
    return rso.str().str();
  } else if (fmt == "" || fmt == "ll") {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    TVM_FFI_ICHECK(module_ != nullptr);
    module_->print(rso, nullptr);
    return rso.str();
  } else {
    TVM_FFI_THROW(InternalError) << "Do not know how to get source code with format: " << format
                                 << "\'";
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
    auto f = kv.second.as_or_throw<PrimFunc>();
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    bool is_entry_func = f->HasNonzeroAttr(tirx::attr::kIsEntryFunc);

    TVM_FFI_ICHECK(global_symbol || !is_entry_func) << "The entry func must be exposed externally.";

    if (global_symbol) {
      function_names_.push_back(global_symbol.value());
      if (is_entry_func) {
        entry_func = global_symbol.value();
      }
    }
  }
  // TODO(@jroesch): follow up on this condition.
  // TVM_FFI_ICHECK(funcs.size() > 0);
  // TODO(tqchen): remove the entry function behavior as it does not
  // makes sense when we start to use multiple modules.
  cg->Init("TVMMod", llvm_target.get(), system_lib_prefix, system_lib_prefix.has_value(), false);
  cg->SetFastMathFlags(llvm_target->GetFastMathFlags());
  cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  module_ = cg->Finish();
  llvm_target->SetTargetMetadata(module_.get());
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
  module_ = std::move(module);
  llvm_instance_ = std::move(llvm_instance);
}

void LLVMModuleNode::LoadIR(const std::string& file_name) {
  auto llvm_instance = std::make_unique<LLVMInstance>();
  std::unique_ptr<llvm::Module> module = llvm_instance->LoadIR(file_name);
  Init(std::move(module), std::move(llvm_instance));
}

void LLVMModuleNode::EnsureOrcJITModule() {
  if (tvm_ffi_orcjit_module_.has_value()) {
    return;
  }
  ffi::Optional<ffi::Function> get_default_session =
      ffi::Function::GetGlobal("tvm_ffi_orcjit.GlobalDefaultSession");
  ffi::Optional<ffi::Function> load_module =
      ffi::Function::GetGlobal("tvm_ffi_orcjit.SessionLoadModule");
  TVM_FFI_CHECK(get_default_session.has_value() && load_module.has_value(), InternalError)
      << "LLVMModule execution requires the separately installed apache-tvm-ffi-orcjit "
         "package and its global execution-session functions. "
         "Install it with `pip install 'apache-tvm-ffi-orcjit>=0.1.1'`.";

  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();

  llvm::DataLayout layout(tm->createDataLayout());
  TVM_FFI_ICHECK(layout == module_->getDataLayout())
      << "Data layout mismatch between module("
      << module_->getDataLayout().getStringRepresentation() << ")"
      << " and JIT target machine (" << layout.getStringRepresentation() << ")";

  std::unique_ptr<llvm::Module> object_module = CloneLLVMModule(module_.get());

  llvm::SmallString<0> object;
  llvm::raw_svector_ostream object_stream(object);
  llvm::legacy::PassManager pass;
  TVM_FFI_ICHECK(!LLVMAddPassesToEmitFile(tm, &pass, &object_stream, llvm_object_file_target))
      << "Cannot emit LLVM object for apache-tvm-ffi-orcjit";
  pass.run(*object_module);

  ffi::ObjectRef session = get_default_session.value()().cast<ffi::ObjectRef>();
  ffi::Array<ffi::Variant<ffi::String, ffi::Bytes>> objects = {
      ffi::Bytes(object.data(), object.size())};
  ffi::Module dylib = load_module.value()(session, objects, ffi::String("")).cast<ffi::Module>();
  for (const ffi::Any& imported_module : imports()) {
    dylib->ImportModule(imported_module.cast<ffi::Module>());
  }

  tvm_ffi_orcjit_module_ = dylib;

  VLOG(2) << "apache-tvm-ffi-orcjit execute " << module_->getModuleIdentifier() << " for triple `"
          << llvm_target->GetTargetTriple() << "` on cpu `" << llvm_target->GetCPU()
          << "` with features `" << llvm_target->GetTargetFeatureString() << "`";
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
      .def("target.llvm_is_valid_cpu",
           [](ffi::String cpu, ffi::String triple) -> bool {
             auto llvm_instance = std::make_unique<LLVMInstance>();
             ffi::Map<ffi::String, ffi::Any> target_map;
             target_map.Set("kind", ffi::String("llvm"));
             target_map.Set("mtriple", triple);
             LLVMTargetInfo llvm_backend(*llvm_instance, Target(target_map));
             return llvm_backend.IsValidCPU(std::string(cpu));
           })
      .def("target.llvm_version_major", []() -> int { return TVM_LLVM_VERSION / 10; })
      .def("ffi.Module.load_from_file.ll",
           [](std::string filename, std::string fmt) -> ffi::Module {
             auto n = ffi::make_object<LLVMModuleNode>();
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
             return ffi::Module(n);
           });
}

TVM_FFI_STATIC_INIT_BLOCK() { LLVMReflectionRegister(); }

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
