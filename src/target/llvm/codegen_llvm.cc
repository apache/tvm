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
 * \file codegen_llvm.cc
 */
#ifdef TVM_LLVM_VERSION
// Part of the code are adapted from Halide's CodeGen_LLVM
#include "codegen_llvm.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#if TVM_LLVM_VERSION >= 50
#include <llvm/BinaryFormat/Dwarf.h>
#else
#include <llvm/Support/Dwarf.h>
#endif
#if TVM_LLVM_VERSION >= 60
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#else
#include <llvm/Target/TargetSubtargetInfo.h>
#endif
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DerivedTypes.h>
#if TVM_LLVM_VERSION >= 150
#include <llvm/IR/FMF.h>
#else
#include <llvm/IR/Operator.h>
#endif
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Pass.h>
#if TVM_LLVM_VERSION >= 160
#include <llvm/IR/Verifier.h>  // For VerifierPass
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#else
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#endif
#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#include <llvm/Support/TypeSize.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../arith/pattern_match.h"
#include "../build_common.h"
#include "../func_registry_generator.h"
#include "codegen_params.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

// CodeGenLLVM has members of type std::unique_ptr<T>. These members will be
// instantiated in the constructor, which will requre that the type T is
// complete at that point. Put the constructor (and destructor) here, since
// all types should be complete here.
CodeGenLLVM::CodeGenLLVM() = default;
CodeGenLLVM::~CodeGenLLVM() = default;
CodeGenLLVM::DebugInfo::~DebugInfo() = default;

std::unique_ptr<CodeGenLLVM> CodeGenLLVM::Create(LLVMTarget* llvm_target) {
  std::string target = llvm_target->GetOrCreateTargetMachine()->getTarget().getName();
  std::string factory_template = "tvm.codegen.llvm.target_";
  void* handle = nullptr;
  if (const PackedFunc* f = runtime::Registry::Get(factory_template + target)) {
    handle = (*f)();
  } else if (const PackedFunc* f = runtime::Registry::Get(factory_template + "cpu")) {
    handle = (*f)();
  } else {
    LOG(FATAL) << "no factory function for codegen for target " << target;
  }
  if (handle) {
    return std::unique_ptr<CodeGenLLVM>(static_cast<CodeGenLLVM*>(handle));
  } else {
    LOG(FATAL) << "unable to create codegen for target " << target;
    return nullptr;  // unreachable
  }
}

void CodeGenLLVM::Init(const std::string& module_name, LLVMTarget* llvm_target, bool system_lib,
                       bool dynamic_lookup, bool target_c_runtime) {
  llvm_target_ = llvm_target;
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  builder_.reset(new IRBuilder(*ctx));
  module_.reset(new llvm::Module(module_name, *ctx));
  md_builder_.reset(new llvm::MDBuilder(*ctx));
  // types
  t_void_ = llvm::Type::getVoidTy(*ctx);
  t_void_p_ = llvm::Type::getInt8Ty(*ctx)->getPointerTo(GetGlobalAddressSpace());
  t_int_ = llvm::Type::getInt32Ty(*ctx);
  t_char_ = llvm::Type::getInt8Ty(*ctx);
  t_int8_ = llvm::Type::getInt8Ty(*ctx);
  t_int16_ = llvm::Type::getInt16Ty(*ctx);
  t_int32_ = llvm::Type::getInt32Ty(*ctx);
  t_int64_ = llvm::Type::getInt64Ty(*ctx);
  t_float64_ = llvm::Type::getDoubleTy(*ctx);
  // meta data
  md_very_likely_branch_ = md_builder_->createBranchWeights(1 << 20, 1);
  md_tbaa_root_ = md_builder_->createTBAARoot("tvm-tbaa");
  md_tbaa_alias_set_ = md_builder_->createTBAANode("tvm-alias", md_tbaa_root_);
  InitTarget();
}

void CodeGenLLVM::SetFastMathFlags(llvm::FastMathFlags fmf) { builder_->setFastMathFlags(fmf); }

void CodeGenLLVM::InitTarget() {
  llvm::TargetMachine* tm = llvm_target_->GetOrCreateTargetMachine();
  module_->setTargetTriple(tm->getTargetTriple().str());
  module_->setDataLayout(tm->createDataLayout());
  data_layout_.reset(new llvm::DataLayout(module_.get()));
  if (native_vector_bits_ == 0) {
    const auto& arch = tm->getTargetTriple().getArch();
    if (arch == llvm::Triple::x86_64) {
      // for avx512
      native_vector_bits_ = 512;
    } else if (arch == llvm::Triple::x86) {
      native_vector_bits_ = 256;
    } else if (arch == llvm::Triple::arm || arch == llvm::Triple::aarch64) {
      native_vector_bits_ = 128;
    } else {
      native_vector_bits_ = 128;
      std::string arch_name = std::string(tm->getTargetTriple().getArchName());
      LOG(WARNING) << "Set native vector bits to be 128 for " << arch_name;
    }
  }

#if TVM_LLVM_VERSION >= 60
  bool use_float16_abi = false;
#if TVM_LLVM_VERSION >= 150
  // For conversions between _Float16 and float, LLVM uses runtime functions
  // __extendhfsf2 and __truncsfhf2.  On X86 up until version 14, LLVM used
  // "uint16_t" for representing _Float16. Starting with LLVM 15, half-precision
  // values can be passed in XMM registers (i.e. as floating-point). This happens
  // when the compilation target has SSE2 enabled (either directly, or by enabling
  // a feature that implies SSE2).
  // Because the names of the conversion functions remain unchanged, it is impossible
  // for TVM to provide them in the runtime, and have them work in both cases.
  // To alleviate this issue, emit these functions directly into the target module
  // after detecting whether or not to use floating-point ABI. To allow the linker
  // to remove potential duplicates (or if they are unused), they are weak and
  // reside in a separate section (ELF).
  llvm::Triple::ArchType arch_type = tm->getTargetTriple().getArch();
  if (arch_type == llvm::Triple::x86 || arch_type == llvm::Triple::x86_64) {
    // Detect if SSE2 is enabled. This determines whether float16 ABI is used.
    std::stringstream os;
    const char fname[] = "test_sse2";
    os << "target triple = \"" << llvm_target_->GetTargetTriple() << "\"\n"
       << "define void @" << fname << "() #0 { ret void } attributes #0 = { \"target-cpu\"=\""
       << llvm_target_->GetCPU() << "\" ";
    if (auto&& fs = llvm_target_->GetTargetFeatureString(); !fs.empty()) {
      os << "\"target-features\"=\"" << fs << "\" ";
    }
    os << "}\n";
    auto mod = llvm_target_->GetInstance().ParseIR(os.str());
    auto* test_sse2 = mod->getFunction(fname);
    ICHECK(test_sse2 != nullptr) << "Module creation error";
    use_float16_abi = tm->getSubtargetImpl(*test_sse2)->checkFeatures("+sse2");
  }
#endif  // TVM_LLVM_VERSION >= 150

  // Call this function only with LLVM >= 6.0. The code it emits uses "dso_local"
  // which was introduced in LLVM 6.
  EmitFloat16ConversionBuiltins(use_float16_abi);
#endif  // TVM_LLVM_VERSION >= 60
}

void CodeGenLLVM::AddFunction(const PrimFunc& f) { this->AddFunctionInternal(f, false); }

void CodeGenLLVM::InitFuncState() {
  var_map_.clear();
  alias_var_set_.clear();
  alloc_storage_info_.clear();
  volatile_buf_.clear();
  analyzer_.reset(new arith::Analyzer());
}

void CodeGenLLVM::AddFunctionInternal(const PrimFunc& f, bool ret_void) {
  this->InitFuncState();

  ICHECK_EQ(f->buffer_map.size(), 0U)
      << "Cannot codegen function with buffer_map, please lower them first";

  std::vector<llvm::Type*> param_types;
  is_restricted_ = f->HasNonzeroAttr(tir::attr::kNoAlias);
  for (Var param : f->params) {
    param_types.push_back(GetLLVMType(param));
    if (!is_restricted_ && param.dtype().is_handle()) {
      alias_var_set_.insert(param.get());
    }
  }
  // TODO(tvm-team):
  // Update the function type to respect the ret_type field of f.
  // Once we allow more flexibility in the PrimFunc.
  llvm::FunctionType* ftype =
      llvm::FunctionType::get(ret_void ? t_void_ : t_int_, param_types, false);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
  function_ = module_->getFunction(MakeStringRef(global_symbol.value()));
  if (function_ == nullptr) {
    function_ = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                       MakeStringRef(global_symbol.value()), module_.get());
  }
  function_->setCallingConv(llvm::CallingConv::C);
  function_->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  SetTargetAttributes(function_);

  // set var map and align information
  auto arg_it = function_->arg_begin();
  for (size_t i = 0; i < f->params.size(); ++i, ++arg_it) {
    llvm::Argument* v = &(*arg_it);
    const Var& var = f->params[i];
    var_map_[var.get()] = v;
    v->setName(std::string(var->name_hint));
    if (is_restricted_) {
      if (var.dtype().is_handle() && !alias_var_set_.count(var.get())) {
        // set non alias.
#if TVM_LLVM_VERSION >= 50
        function_->addParamAttr(i, llvm::Attribute::NoAlias);
#else
        function_->setDoesNotAlias(i + 1);
#endif
      }
    }
  }
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*ctx, "entry", function_);
  builder_->SetInsertPoint(entry);
  this->VisitStmt(f->body);

  // Add alignment attribute if needed.
#if TVM_LLVM_VERSION >= 50
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var& var = f->params[i];
    auto f = alloc_storage_info_.find(var.get());
    if (f != alloc_storage_info_.end()) {
      unsigned align = f->second.alignment;
      if (align > 1) {
        auto attr = llvm::Attribute::get(*ctx, llvm::Attribute::Alignment, align);
        function_->addParamAttr(i, attr);
      }
    }
  }
#endif

  if (ret_void) {
    builder_->CreateRetVoid();
  } else {
    builder_->CreateRet(ConstInt32(0));
  }
}

std::unique_ptr<llvm::Module> CodeGenLLVM::Finish() {
  this->AddStartupFunction();
  for (size_t i = 0; i < link_modules_.size(); ++i) {
    ICHECK(!llvm::Linker::linkModules(*module_, std::move(link_modules_[i])))
        << "Failed to link modules";
  }
  link_modules_.clear();
  // optimize
  this->Optimize();
  return std::move(module_);
}

void CodeGenLLVM::HandleImport(const std::string& code) {
  llvm::StringRef code_str(code);
  std::unique_ptr<llvm::Module> mlib;
  if (code_str.endswith(".ll") || code_str.endswith(".bc")) {
    mlib = llvm_target_->GetInstance().LoadIR(code);
  } else {
    mlib = llvm_target_->GetInstance().ParseIR(code);
  }

  mlib->setTargetTriple(llvm_target_->GetTargetTriple());
  mlib->setDataLayout(llvm_target_->GetOrCreateTargetMachine()->createDataLayout());
  // mark all the functions as force inline
  for (llvm::Function& f : mlib->functions()) {
    f.removeFnAttr(llvm::Attribute::NoInline);
    f.addFnAttr(llvm::Attribute::AlwaysInline);
    f.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
  }
  // add to linker libraries.
  this->AddLinkModule(std::move(mlib));
}

void CodeGenLLVM::AddLinkModule(std::unique_ptr<llvm::Module>&& mod) {
  link_modules_.emplace_back(std::move(mod));
}

void CodeGenLLVM::AddMainFunction(const std::string& entry_func_name) {
  LOG(FATAL) << "not implemented";
}

llvm::Value* CodeGenLLVM::GetThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

llvm::Value* CodeGenLLVM::CreateStorageSync(const CallNode* op) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

#if TVM_LLVM_VERSION >= 160

// Use new pass manager

void CodeGenLLVM::Optimize() {
  llvm::TargetMachine* tm = llvm_target_->GetOrCreateTargetMachine();

  bool debug_logging = false;
  bool verify_each = false;

  llvm::PipelineTuningOptions pto = llvm::PipelineTuningOptions();
  llvm::PassInstrumentationCallbacks pic;
  llvm::PassBuilder builder(tm, pto, llvm::None, &pic);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  builder.registerLoopAnalyses(lam);
  builder.registerFunctionAnalyses(fam);
  builder.registerCGSCCAnalyses(cgam);
  builder.registerModuleAnalyses(mam);
  builder.crossRegisterProxies(lam, fam, cgam, mam);

  // Construct the default pass pipeline depending on the opt level.
  std::string pipeline;
  switch (llvm_target_->GetOptLevel()) {
    case llvm::CodeGenOpt::Level::None:
      pipeline = "default<O0>";
      break;
    case llvm::CodeGenOpt::Level::Less:
      pipeline = "default<O1>";
      break;
    case llvm::CodeGenOpt::Level::Default:
      pipeline = "default<O2>";
      break;
    default:
      // CodeGenOpt::Level::Aggressive
      pipeline = "default<O3>";
      break;
  }

  llvm::StandardInstrumentations si(*llvm_target_->GetContext(), debug_logging, verify_each);
  si.registerCallbacks(pic, &fam);
  llvm::ModulePassManager mpass;
  if (verify_each) {
    mpass.addPass(llvm::VerifierPass());
  }
  if (auto err = builder.parsePassPipeline(mpass, pipeline)) {
    LOG(FATAL) << "error parsing pass pipeline '" << pipeline
               << "':" << llvm::toString(std::move(err)) << '\n';
  }

  mpass.run(*module_, mam);
}

#else  // TVM_LLVM_VERSION

class FPassManager : public llvm::legacy::FunctionPassManager {
 public:
  explicit FPassManager(llvm::Module* m) : llvm::legacy::FunctionPassManager(m) {}
  // override add to allow messaging
  void add(llvm::Pass* p) final { llvm::legacy::FunctionPassManager::add(p); }
};

class MPassManager : public llvm::legacy::PassManager {
 public:
  // override add to allow messaging
  void add(llvm::Pass* p) final { llvm::legacy::PassManager::add(p); }
};

void CodeGenLLVM::InitPassManagerBuilder(llvm::PassManagerBuilder* builder) {}

void CodeGenLLVM::Optimize() {
  // pass manager
  FPassManager fpass(module_.get());
  MPassManager mpass;
  llvm::TargetMachine* tm = llvm_target_->GetOrCreateTargetMachine();
  mpass.add(llvm::createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));
  fpass.add(llvm::createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));

  // place optimization pass
  llvm::PassManagerBuilder builder;

  // Use the same opt-level as specified in TargetMachine for running passes
  llvm::CodeGenOpt::Level opt_level = llvm_target_->GetOptLevel();

  switch (opt_level) {
    case llvm::CodeGenOpt::Level::None:
      builder.OptLevel = 0;
      break;
    case llvm::CodeGenOpt::Level::Less:
      builder.OptLevel = 1;
      break;

    case llvm::CodeGenOpt::Level::Default:
      builder.OptLevel = 2;
      break;

    default:
      // CodeGenOpt::Level::Aggressive
      builder.OptLevel = 3;
  }

#if TVM_LLVM_VERSION >= 50
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
#else
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0);
#endif
  builder.LoopVectorize = true;
  builder.SLPVectorize = true;
  this->InitPassManagerBuilder(&builder);

#if TVM_LLVM_VERSION >= 50
  tm->adjustPassManager(builder);
#endif

  builder.populateFunctionPassManager(fpass);
  builder.populateModulePassManager(mpass);

  fpass.doInitialization();
  for (auto it = module_->begin(); it != module_->end(); ++it) {
    fpass.run(*it);
  }
  fpass.doFinalization();
  mpass.run(*module_);
}
#endif  // TVM_LLVM_VERSION

int CodeGenLLVM::NativeVectorBits(const runtime::StorageScope& storage_scope) const {
  return native_vector_bits_;
}

unsigned CodeGenLLVM::GetGlobalAddressSpace() const { return 0; }

llvm::Type* CodeGenLLVM::DTypeToLLVMType(const DataType& dtype) const {
  if (dtype.is_handle()) {
    ICHECK_EQ(dtype.lanes(), 1);
    return t_void_p_;
  }
  if (dtype.is_void()) {
    return t_void_;
  }
  llvm::Type* etype = nullptr;
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  if (dtype.is_int() || dtype.is_uint()) {
    etype = llvm::Type::getIntNTy(*ctx, dtype.bits());
  } else if (dtype.is_float()) {
    switch (dtype.bits()) {
      case 16:
        etype = llvm::Type::getHalfTy(*ctx);
        break;
      case 32:
        etype = llvm::Type::getFloatTy(*ctx);
        break;
      case 64:
        etype = llvm::Type::getDoubleTy(*ctx);
        break;
      default:
        LOG(FATAL) << "do not support " << dtype;
    }
  }
  if (dtype.lanes() != 1) {
#if TVM_LLVM_VERSION >= 110
    return llvm::FixedVectorType::get(etype, dtype.lanes());
#else
    return llvm::VectorType::get(etype, dtype.lanes());
#endif
  } else {
    return etype;
  }
}  // namespace codegen

llvm::Type* CodeGenLLVM::GetLLVMType(const Type& type) const {
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return DTypeToLLVMType(ptr->dtype);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    // LLVM IR doesn't allow void*, so we need to recognize this
    // pattern explicitly.
    if (auto* primtype = ptr->element_type.as<PrimTypeNode>()) {
      if (primtype->dtype.is_void()) {
        return t_void_p_;
      }
    }
    // TODO(tvm-team) consider put storage scope into the pointer type.
    return GetLLVMType(ptr->element_type)->getPointerTo(GetGlobalAddressSpace());
  } else if (IsVoidType(type)) {
    return t_void_;
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding LLVM Type";
    return t_void_;
  }
}

llvm::Type* CodeGenLLVM::GetLLVMType(const PrimExpr& expr) const {
  return GetLLVMType(GetType(expr));
}

// Add tbaa alias information for load
//
// use a binary tree typed system to declare information
// and allow alias to be distinguished across nodes.
//
// This trick comes from Halide's CodeGen_LLVM
//
void CodeGenLLVM::AddAliasInfo(llvm::Instruction* inst, const VarNode* buffer_var, PrimExpr index,
                               DataType access_dtype) {
  if (alias_var_set_.count(buffer_var) != 0) {
    // Mark all possibly aliased pointer as same type.
    llvm::MDNode* meta = md_tbaa_alias_set_;
    inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
    return;
  }

  int64_t base = 0, width = 0;
  arith::PVar<IntImm> pbase, pstride;
  arith::PVar<int> planes;
  // create meta-data for alias analysis
  // Use a group of binary tree ranges of memory banks.
  int64_t xwith = 0;
  if (arith::ramp(pbase, pstride, planes).Match(index)) {
    base = pbase.Eval()->value;
    xwith = planes.Eval() * pstride.Eval()->value;
  } else if (auto* ptr = index.as<tir::IntImmNode>()) {
    base = ptr->value;
    xwith = 1;
  }
  // adjust address index unit to byte
  const int64_t unit_bit_width = 8;
  const int64_t access_elem_bits = access_dtype.bits() * access_dtype.lanes();
  base = base * access_elem_bits / unit_bit_width;
  xwith = (xwith * access_elem_bits + unit_bit_width - 1) / unit_bit_width;
  if (xwith > 0) {
    width = 1;
    while (width < xwith) {
      width *= 2;
    }
    while (base % width) {
      base -= base % width;
      width *= 2;
    }
  }

  llvm::MDNode* meta = md_tbaa_root_;
  std::ostringstream buffer_addr;
  buffer_addr << buffer_var;
  meta = md_builder_->createTBAAScalarTypeNode(buffer_addr.str(), meta);

  // create a tree-shape access structure.
  if (width != 0) {
    for (int64_t w = 1024; w >= width; w /= 2) {
      int64_t b = (base / w) * w;
      std::stringstream os;
      os << buffer_var << ".w" << w << ".b" << b;
      meta = md_builder_->createTBAAScalarTypeNode(os.str(), meta);
    }
  }
  inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta, meta, 0));
}

void CodeGenLLVM::GetAlignment(DataType t, const VarNode* buf_var, const PrimExpr& index,
                               int* p_alignment, int* p_native_bits) {
  int max_align_bits = t.bits();
  auto it = alloc_storage_info_.find(buf_var);
  if (it != alloc_storage_info_.end()) {
    const StorageInfo& info = it->second;
    *p_native_bits =
        NativeVectorBits(runtime::StorageScope::Create(GetPtrStorageScope(GetRef<Var>(buf_var))));
    max_align_bits = info.alignment * 8;
  } else {
    *p_native_bits = native_vector_bits_;
  }

  arith::ModularSet me = analyzer_->modular_set(index);
  int64_t base = me->base;
  int64_t coeff = me->coeff;

  int align_bits = t.bits();
  while (align_bits < max_align_bits && base % 2 == 0 && coeff % 2 == 0) {
    base = base / 2;
    coeff = coeff / 2;
    align_bits *= 2;
  }
  if (align_bits < 8) {
    align_bits = 8;
  }
  *p_alignment = align_bits / 8;
}

llvm::GlobalVariable* CodeGenLLVM::AllocateSharedMemory(DataType dtype, size_t size,
                                                        unsigned int shared_address_space,
                                                        int alignment,
                                                        llvm::GlobalValue::LinkageTypes linkage) {
  llvm::Type* type = llvm::ArrayType::get(DTypeToLLVMType(dtype), size);
  llvm::GlobalVariable* global =
      new llvm::GlobalVariable(*module_, type, false, linkage, nullptr, "shmem", nullptr,
                               llvm::GlobalValue::NotThreadLocal, shared_address_space);
#if TVM_LLVM_VERSION >= 100
  global->setAlignment(llvm::Align(alignment));
#else
  global->setAlignment(alignment);
#endif
  return global;
}

std::unique_ptr<CodeGenLLVM::DebugInfo> CodeGenLLVM::CreateDebugInfo(llvm::Module* module) {
#if TVM_LLVM_VERSION >= 100
  auto debug_info = std::make_unique<CodeGenLLVM::DebugInfo>();
  debug_info->di_builder_ = std::make_unique<llvm::DIBuilder>(*module);
#else
  auto debug_info = llvm::make_unique<CodeGenLLVM::DebugInfo>();
  debug_info->di_builder_ = llvm::make_unique<llvm::DIBuilder>(*module);
#endif
  // TODO(tulloch): pass this information through relay::Span classes to the IRModule instance?
  debug_info->file_ = debug_info->di_builder_->createFile("model.tvm", "/tmp/");
  debug_info->compilation_unit_ = debug_info->di_builder_->createCompileUnit(
      llvm::dwarf::DW_LANG_C, debug_info->file_, "TVM", 0, "", 0, "",
      llvm::DICompileUnit::DebugEmissionKind::FullDebug,
      /* SplitDebugInlining */ true,
      /* DebugInfoForProfiling */ true);
  return debug_info;
}

llvm::Value* CodeGenLLVM::CreateBroadcast(llvm::Value* value, int lanes) {
#if TVM_LLVM_VERSION >= 110
  llvm::Type* type = llvm::FixedVectorType::get(value->getType(), lanes);
#else
  llvm::Type* type = llvm::VectorType::get(value->getType(), lanes);
#endif
  llvm::Constant* undef = llvm::UndefValue::get(type);
  llvm::Constant* zero = ConstInt32(0);
  value = builder_->CreateInsertElement(undef, value, zero);
#if TVM_LLVM_VERSION >= 120
  llvm::Constant* mask = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(lanes), zero);
#elif TVM_LLVM_VERSION >= 110
  llvm::Constant* mask =
      llvm::ConstantVector::getSplat(llvm::ElementCount(lanes, /*Scalable=*/false), zero);
#else
  llvm::Constant* mask = llvm::ConstantVector::getSplat(lanes, zero);
#endif
  return builder_->CreateShuffleVector(value, undef, mask);
}

llvm::Value* CodeGenLLVM::CreateVecSlice(llvm::Value* vec, int begin, int extent) {
  int num_elems = GetVectorNumElements(vec);
  if (extent == num_elems && begin == 0) return vec;
  ICHECK(begin >= 0 && extent <= num_elems) << "Slicing out of bound!\n";
  std::vector<llvm::Constant*> indices;
  indices.reserve(extent);
  for (int i = 0; i < extent; ++i) {
    if (begin + i >= 0 && begin + i < num_elems) {
      indices.push_back(llvm::ConstantInt::get(t_int32_, begin + i));
    } else {
      indices.push_back(llvm::UndefValue::get(t_int32_));
    }
  }
  return builder_->CreateShuffleVector(vec, vec, llvm::ConstantVector::get(indices));
}

llvm::Value* CodeGenLLVM::CreateVecFlip(llvm::Value* vec) {
  int num_elems = GetVectorNumElements(vec);
#if TVM_LLVM_VERSION >= 110
  std::vector<int> indices;
#else
  std::vector<unsigned> indices;
#endif
  for (int i = 0; i < num_elems; ++i) {
    indices.push_back(num_elems - i - 1);
  }
  return builder_->CreateShuffleVector(vec, vec, indices);
}

llvm::Value* CodeGenLLVM::CreateVecPad(llvm::Value* vec, int target_lanes) {
  llvm::Value* mask = llvm::UndefValue::get(DTypeToLLVMType(DataType::Int(32, target_lanes)));
  int num_elems = GetVectorNumElements(vec);
  if (num_elems == target_lanes) return vec;
  ICHECK_LT(num_elems, target_lanes);
  for (int i = 0; i < num_elems; ++i) {
    mask = builder_->CreateInsertElement(mask, ConstInt32(i), ConstInt32(i));
  }
  return builder_->CreateShuffleVector(vec, vec, mask);
}

llvm::Value* CodeGenLLVM::CreateVecConcat(std::vector<llvm::Value*> vecs) {
  // To allow creating vectors from scalars, convert any scalars in "vecs" to single-lane
  // LLVM vector types.
  for (size_t i = 0, e = vecs.size(); i != e; ++i) {
    llvm::Value* v = vecs[i];
    if (!v->getType()->isVectorTy()) {
#if TVM_LLVM_VERSION >= 110
      llvm::Type* vec_ty = llvm::FixedVectorType::get(v->getType(), 1);
#else
      llvm::Type* vec_ty = llvm::VectorType::get(v->getType(), 1);
#endif
      vecs[i] = builder_->CreateInsertElement(llvm::UndefValue::get(vec_ty), v, ConstInt32(0));
    }
  }

  // concat vector, tree shape reduction
  int total_lanes = 0;

  for (llvm::Value* v : vecs) {
    total_lanes += GetVectorNumElements(v);
  }
  while (vecs.size() > 1) {
    std::vector<llvm::Value*> new_vecs;
    for (size_t i = 0; i < vecs.size() - 1; i += 2) {
      llvm::Value* lhs = vecs[i];
      llvm::Value* rhs = vecs[i + 1];
      const size_t lhs_lanes = GetVectorNumElements(lhs);
      const size_t rhs_lanes = GetVectorNumElements(rhs);
      if (lhs_lanes < rhs_lanes) {
        lhs = CreateVecPad(lhs, rhs_lanes);
      } else if (rhs_lanes < lhs_lanes) {
        rhs = CreateVecPad(rhs, lhs_lanes);
      }
      const size_t shared_lanes = std::max(lhs_lanes, rhs_lanes);
#if TVM_LLVM_VERSION >= 110
      std::vector<int> mask;
#else
      std::vector<unsigned> mask;
#endif
      for (size_t i = 0; i < lhs_lanes; ++i) {
        mask.push_back(i);
      }
      for (size_t i = 0; i < rhs_lanes; ++i) {
        mask.push_back(shared_lanes + i);
      }
      new_vecs.push_back(builder_->CreateShuffleVector(lhs, rhs, mask));
    }
    if (vecs.size() % 2 != 0) {
      new_vecs.push_back(vecs.back());
    }
    vecs.swap(new_vecs);
  }
  return CreateVecSlice(vecs[0], 0, total_lanes);
}

void CodeGenLLVM::CreateSerialFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride,
                                  const Var& loop_var, const Stmt& body) {
  llvm::BasicBlock* pre_block = builder_->GetInsertBlock();
  std::string loop_var_name = loop_var->name_hint;
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  auto* for_begin = llvm::BasicBlock::Create(*ctx, "for_begin_" + loop_var_name, function_);
  auto* for_body = llvm::BasicBlock::Create(*ctx, "for_body_" + loop_var_name, function_);
  auto* for_end = llvm::BasicBlock::Create(*ctx, "for_end_" + loop_var_name, function_);
  builder_->CreateBr(for_begin);
  builder_->SetInsertPoint(for_begin);
  llvm::PHINode* loop_value = builder_->CreatePHI(begin->getType(), 2);
  loop_value->setName(loop_var->name_hint.c_str());
  loop_value->addIncoming(begin, pre_block);
  ICHECK(!var_map_.count(loop_var.get()));
  var_map_[loop_var.get()] = loop_value;
  builder_->CreateCondBr(CreateLT(loop_var.dtype(), loop_value, end), for_body, for_end,
                         md_very_likely_branch_);
  builder_->SetInsertPoint(for_body);
  this->VisitStmt(body);
  var_map_.erase(loop_var.get());
  llvm::Value* loop_next = CreateAdd(loop_var.dtype(), loop_value, stride);
  loop_value->addIncoming(loop_next, builder_->GetInsertBlock());
  builder_->CreateBr(for_begin);
  builder_->SetInsertPoint(for_end);
}

// cast operatpr
llvm::Value* CodeGenLLVM::CreateCast(DataType from, DataType to, llvm::Value* value) {
  llvm::Type* target = DTypeToLLVMType(to);
  if (value->getType() == target) return value;
  if (to.is_handle()) {
    return builder_->CreateBitCast(value, target);
  } else if (to.is_uint() && to.bits() == 1) {
    if (from.is_float()) {
      llvm::Constant* zero = llvm::ConstantFP::get(DTypeToLLVMType(from), 0.);
      return builder_->CreateFCmpONE(value, zero);
    } else {
      llvm::Constant* zero = llvm::ConstantInt::get(DTypeToLLVMType(from), 0);
      return builder_->CreateICmpNE(value, zero);
    }
  } else if (!from.is_float() && !to.is_float()) {
    return builder_->CreateIntCast(value, target, from.is_int());
  } else if (from.is_float() && to.is_int()) {
    return builder_->CreateFPToSI(value, target);
  } else if (from.is_float() && to.is_uint()) {
    if (to.bits() < 8) {
      value = builder_->CreateFPToUI(value, DTypeToLLVMType(to.with_bits(8)));
      return builder_->CreateIntCast(value, target, false);
    } else {
      return builder_->CreateFPToUI(value, target);
    }
  } else if (from.is_int() && to.is_float()) {
    return builder_->CreateSIToFP(value, target);
  } else if (from.is_uint() && to.is_float()) {
    return builder_->CreateUIToFP(value, target);
  } else {
    ICHECK(from.is_float() && to.is_float());
    return builder_->CreateFPCast(value, target);
  }
}

llvm::Constant* CodeGenLLVM::GetGlobalConstant(llvm::Constant* const_data, const std::string& name,
                                               llvm::GlobalValue::LinkageTypes linkage_type) {
  llvm::Type* ty = const_data->getType();
  llvm::GlobalVariable* global =
      new llvm::GlobalVariable(*module_, ty, true, linkage_type, const_data, name);
#if TVM_LLVM_VERSION >= 100
  global->setAlignment(llvm::Align(1));
#else
  global->setAlignment(1);
#endif
  llvm::Constant* zero = ConstInt32(0);
  llvm::Constant* indices[] = {zero, zero};
  llvm::Constant* ptr = llvm::ConstantExpr::getGetElementPtr(ty, global, indices);
  return ptr;
}

llvm::Constant* CodeGenLLVM::GetConstString(const std::string& str) {
  auto it = str_map_.find(str);
  if (it != str_map_.end()) return it->second;
  auto llvm_str = llvm::ConstantDataArray::getString(*llvm_target_->GetContext(), str);
  auto ptr = GetGlobalConstant(llvm_str, ".str", llvm::GlobalValue::PrivateLinkage);
  str_map_[str] = ptr;
  return ptr;
}

CodeGenLLVM::TypedPointer CodeGenLLVM::CreateBufferPtr(llvm::Value* buffer_ptr,
                                                       DataType buffer_element_dtype,
                                                       llvm::ArrayRef<llvm::Value*> indices,
                                                       DataType value_dtype) {
  ICHECK_EQ(indices.size(), 1) << "CodeGenLLVM requires all buffers to be flat 1-d buffers.";
  llvm::Value* index = indices[0];

  llvm::PointerType* buffer_ptr_type = llvm::dyn_cast<llvm::PointerType>(buffer_ptr->getType());
  ICHECK(buffer_ptr_type != nullptr);
  auto address_space = buffer_ptr_type->getAddressSpace();

  llvm::Type* element_type = DTypeToLLVMType(buffer_element_dtype);
  llvm::PointerType* element_ptr_type =
      DTypeToLLVMType(buffer_element_dtype)->getPointerTo(address_space);
  llvm::Type* value_type = DTypeToLLVMType(value_dtype);
  llvm::PointerType* value_ptr_type = value_type->getPointerTo(address_space);

  ICHECK(index->getType()->isIntegerTy()) << "Expected buffer index to be an integer";

  if (buffer_ptr_type != element_ptr_type) {
    buffer_ptr = builder_->CreatePointerCast(buffer_ptr, element_ptr_type);
  }
  ICHECK(!HasAlignmentPadding(buffer_element_dtype))
      << "DType " << buffer_element_dtype
      << " has padding for alignment.  TVM data arrays are expected to be densely packed, with no "
         "padding for alignment.";
  llvm::Value* value_ptr = builder_->CreateInBoundsGEP(element_type, buffer_ptr, index);

  if (element_ptr_type != value_ptr_type) {
    value_ptr = builder_->CreatePointerCast(value_ptr, value_ptr_type);
  }

  return TypedPointer(value_type, value_ptr);
}

llvm::Value* CodeGenLLVM::GetVarValue(const VarNode* v) const {
  auto it = var_map_.find(v);
  ICHECK(it != var_map_.end()) << "cannot find variable " << v->name_hint;
  return it->second;
}

void CodeGenLLVM::CreatePrintf(const std::string& format,
                               llvm::ArrayRef<llvm::Value*> format_args) {
  llvm::Function* func_printf = module_->getFunction("printf");
  if (func_printf == nullptr) {
    llvm::FunctionType* ftype = llvm::FunctionType::get(t_int32_, true);
    func_printf =
        llvm::Function::Create(ftype, llvm::Function::ExternalLinkage, "printf", module_.get());
  }

  llvm::Function* func_fflush = module_->getFunction("fflush");
  if (!func_fflush) {
    llvm::FunctionType* ftype = llvm::FunctionType::get(t_int32_, {t_void_p_}, false);
    func_fflush =
        llvm::Function::Create(ftype, llvm::Function::ExternalLinkage, "fflush", module_.get());
  }

  llvm::Value* str = builder_->CreateGlobalStringPtr(format);
  str->setName("printf_format_str");

  std::vector<llvm::Value*> printf_args = {str};
  printf_args.insert(printf_args.end(), format_args.begin(), format_args.end());
  builder_->CreateCall(func_printf, printf_args);

  // Call fflush() immediately, as this utility is intended for debug
  // purposes.  A segfault occurring within the generated LLVM code
  // would otherwise leave the stdout buffer unflushed.
  llvm::Value* null_stream = llvm::ConstantPointerNull::get(t_void_p_);
  null_stream->setName("null_stream");
  builder_->CreateCall(func_fflush, {null_stream});
}

llvm::Value* CodeGenLLVM::CreateLookupReturnAddress(unsigned int level) {
  llvm::Value* level_val = llvm::ConstantInt::get(t_int32_, level);
  llvm::Function* builtin =
      llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::returnaddress);
  llvm::Value* call = builder_->CreateCall(builtin, level_val);
  call->setName("return_addr");

  return call;
}

llvm::Value* CodeGenLLVM::CreateCallExtern(Type ret_type, String global_symbol,
                                           const Array<PrimExpr>& args, bool skip_first_arg) {
  std::vector<llvm::Value*> arg_value;
  std::vector<llvm::Type*> arg_type;
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    arg_value.push_back(MakeValue(args[i]));
    arg_type.push_back(arg_value.back()->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(GetLLVMType(ret_type), arg_type, false);
  llvm::Function* f = module_->getFunction(MakeStringRef(global_symbol));
  if (f == nullptr) {
    f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage, MakeStringRef(global_symbol),
                               module_.get());
  }
  llvm::CallInst* call = builder_->CreateCall(f, arg_value);
  return call;
}

llvm::Function* CodeGenLLVM::GetIntrinsicDecl(llvm::Intrinsic::ID id, llvm::Type* ret_type,
                                              llvm::ArrayRef<llvm::Type*> arg_types) {
  llvm::Module* module = module_.get();

  if (!llvm::Intrinsic::isOverloaded(id)) {
    return llvm::Intrinsic::getDeclaration(module, id, {});
  }

  llvm::SmallVector<llvm::Intrinsic::IITDescriptor, 4> infos;
  llvm::Intrinsic::getIntrinsicInfoTableEntries(id, infos);
  llvm::SmallVector<llvm::Type*, 4> overload_types;

#if TVM_LLVM_VERSION >= 90
  auto try_match = [&](llvm::FunctionType* f_ty, bool var_arg) {
    overload_types.clear();
    llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> ref(infos);
    auto match = llvm::Intrinsic::matchIntrinsicSignature(f_ty, ref, overload_types);
    if (match == llvm::Intrinsic::MatchIntrinsicTypes_Match) {
      bool error = llvm::Intrinsic::matchIntrinsicVarArg(var_arg, ref);
      if (error) {
        return llvm::Intrinsic::MatchIntrinsicTypes_NoMatchArg;
      }
    }
    return match;
  };

  // First, try matching the signature assuming non-vararg case.
  auto* fn_ty = llvm::FunctionType::get(ret_type, arg_types, false);
  switch (try_match(fn_ty, false)) {
    case llvm::Intrinsic::MatchIntrinsicTypes_NoMatchRet:
      // The return type doesn't match, there is nothing else to do.
      return nullptr;
    case llvm::Intrinsic::MatchIntrinsicTypes_Match:
      return llvm::Intrinsic::getDeclaration(module, id, overload_types);
    case llvm::Intrinsic::MatchIntrinsicTypes_NoMatchArg:
      break;
  }

  // Keep adding one type at a time (starting from empty list), and
  // try matching the vararg signature.
  llvm::SmallVector<llvm::Type*, 4> var_types;
  for (int i = 0, e = arg_types.size(); i <= e; ++i) {
    if (i > 0) var_types.push_back(arg_types[i - 1]);
    auto* ft = llvm::FunctionType::get(ret_type, var_types, true);
    if (try_match(ft, true) == llvm::Intrinsic::MatchIntrinsicTypes_Match) {
      return llvm::Intrinsic::getDeclaration(module, id, overload_types);
    }
  }
  // Failed to identify the type.
  return nullptr;

#else   // TVM_LLVM_VERSION
  llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> ref(infos);
  // matchIntrinsicType returns true on error.
  if (llvm::Intrinsic::matchIntrinsicType(ret_type, ref, overload_types)) {
    return nullptr;
  }
  for (llvm::Type* t : arg_types) {
    if (llvm::Intrinsic::matchIntrinsicType(t, ref, overload_types)) {
      return nullptr;
    }
  }
  return llvm::Intrinsic::getDeclaration(module, id, overload_types);
#endif  // TVM_LLVM_VERSION
}

void CodeGenLLVM::SetTargetAttributes(llvm::Function* func) {
  const std::string& cpu = llvm_target_->GetCPU();
  if (!cpu.empty()) {
    func->addFnAttr("target-cpu", cpu);
  }
  const std::string& features = llvm_target_->GetTargetFeatureString();
  if (!features.empty()) {
    func->addFnAttr("target-features", features);
  }
}

void CodeGenLLVM::EmitFloat16ConversionBuiltins(bool use_float16_abi) {
  // The LLVM IR for these function was obtained by compiling
  //
  // For integer ABI:
  // __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
  // __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
  // For floating-point ABI:
  // __truncXfYf2__<float, uint32_t, 23, _Float16, uint16_t, 10>(a);
  // __extendXfYf2__<_Float16, uint16_t, 10, float, uint32_t, 23>(a);

  static const char trunc_body[] =  // __truncsfhf2
      "  %v0 = bitcast float %a0 to i32\n"
      "  %v1 = and i32 %v0, 2147483647\n"
      "  %v2 = add nsw i32 %v1, -947912704\n"
      "  %v3 = add nsw i32 %v1, -1199570944\n"
      "  %v4 = icmp ult i32 %v2, %v3\n"
      "  br i1 %v4, label %b1, label %b5\n"
      "b1:\n"
      "  %v5 = lshr i32 %v0, 13\n"
      "  %v6 = and i32 %v5, 65535\n"
      "  %v7 = add nuw nsw i32 %v6, -114688\n"
      "  %v8 = and i32 %v0, 8191\n"
      "  %v9 = icmp ugt i32 %v8, 4096\n"
      "  br i1 %v9, label %b2, label %b3\n"
      "b2:\n"
      "  %v10 = add nuw nsw i32 %v6, -114687\n"
      "  br label %b13\n"
      "b3:\n"
      "  %v11 = icmp eq i32 %v8, 4096\n"
      "  br i1 %v11, label %b4, label %b13\n"
      "b4:\n"
      "  %v12 = and i32 %v7, 65535\n"
      "  %v13 = and i32 %v5, 1\n"
      "  %v14 = add nuw nsw i32 %v12, %v13\n"
      "  br label %b13\n"
      "b5:\n"
      "  %v15 = icmp ugt i32 %v1, 2139095040\n"
      "  br i1 %v15, label %b6, label %b7\n"
      "b6:\n"
      "  %v16 = lshr i32 %v0, 13\n"
      "  %v17 = and i32 %v16, 511\n"
      "  %v18 = or i32 %v17, 32256\n"
      "  br label %b13\n"
      "b7:\n"
      "  %v19 = icmp ugt i32 %v1, 1199570943\n"
      "  br i1 %v19, label %b13, label %b8\n"
      "b8:\n"
      "  %v20 = icmp ult i32 %v1, 754974720\n"
      "  br i1 %v20, label %b13, label %b9\n"
      "b9:\n"
      "  %v21 = lshr i32 %v1, 23\n"
      "  %v22 = sub nsw i32 113, %v21\n"
      "  %v23 = and i32 %v0, 8388607\n"
      "  %v24 = or i32 %v23, 8388608\n"
      "  %v25 = add nsw i32 %v21, -81\n"
      "  %v26 = shl i32 %v24, %v25\n"
      "  %v27 = icmp ne i32 %v26, 0\n"
      "  %v28 = lshr i32 %v24, %v22\n"
      "  %v29 = zext i1 %v27 to i32\n"
      "  %v30 = lshr i32 %v28, 13\n"
      "  %v31 = and i32 %v28, 8191\n"
      "  %v32 = or i32 %v31, %v29\n"
      "  %v33 = icmp ugt i32 %v32, 4096\n"
      "  br i1 %v33, label %b10, label %b11\n"
      "b10:\n"
      "  %v34 = add nuw nsw i32 %v30, 1\n"
      "  br label %b13\n"
      "b11:\n"
      "  %v35 = icmp eq i32 %v32, 4096\n"
      "  br i1 %v35, label %b12, label %b13\n"
      "b12:\n"
      "  %v36 = and i32 %v30, 1\n"
      "  %v37 = add nuw nsw i32 %v36, %v30\n"
      "  br label %b13\n"
      "b13:\n"
      "  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ],\n"
      "                 [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ],\n"
      "                 [ %v30, %b11 ]\n"
      "  %v39 = lshr i32 %v0, 16\n"
      "  %v40 = and i32 %v39, 32768\n"
      "  %v41 = or i32 %v38, %v40\n"
      "  %vlast = trunc i32 %v41 to i16\n";

  static const char extend_body[] =  // __extendhfsf2
      "  %v1 = and i16 %vinp, 32767\n"
      "  %v2 = zext i16 %v1 to i32\n"
      "  %v3 = add nsw i16 %v1, -1024\n"
      "  %v4 = icmp ult i16 %v3, 30720\n"
      "  br i1 %v4, label %b1, label %b2\n"
      "b1:\n"
      "  %v5 = shl nuw nsw i32 %v2, 13\n"
      "  %v6 = add nuw nsw i32 %v5, 939524096\n"
      "  br label %b6\n"
      "b2:\n"
      "  %v7 = icmp ugt i16 %v1, 31743\n"
      "  br i1 %v7, label %b3, label %b4\n"
      "b3:\n"
      "  %v8 = shl nuw nsw i32 %v2, 13\n"
      "  %v9 = or i32 %v8, 2139095040\n"
      "  br label %b6\n"
      "b4:\n"
      "  %v10 = icmp eq i16 %v1, 0\n"
      "  br i1 %v10, label %b6, label %b5\n"
      "b5:\n"
      "  %v11 = icmp ult i16 %v1, 256\n"
      "  %v12 = lshr i32 %v2, 8\n"
      "  %v13 = select i1 %v11, i32 %v2, i32 %v12\n"
      "  %v14 = select i1 %v11, i32 32, i32 24\n"
      "  %v15 = icmp ult i32 %v13, 16\n"
      "  %v16 = lshr i32 %v13, 4\n"
      "  %v17 = add nsw i32 %v14, -4\n"
      "  %v18 = select i1 %v15, i32 %v13, i32 %v16\n"
      "  %v19 = select i1 %v15, i32 %v14, i32 %v17\n"
      "  %v20 = icmp ult i32 %v18, 4\n"
      "  %v21 = lshr i32 %v18, 2\n"
      "  %v22 = add nsw i32 %v19, -2\n"
      "  %v23 = select i1 %v20, i32 %v18, i32 %v21\n"
      "  %v24 = select i1 %v20, i32 %v19, i32 %v22\n"
      "  %v25 = icmp ult i32 %v23, 2\n"
      "  %v26 = sub nsw i32 0, %v23\n"
      "  %v27 = select i1 %v25, i32 %v26, i32 -2\n"
      "  %v28 = add nsw i32 %v27, %v24\n"
      "  %v29 = add nsw i32 %v28, -8\n"
      "  %v30 = shl i32 %v2, %v29\n"
      "  %v31 = xor i32 %v30, 8388608\n"
      "  %v32 = shl i32 %v28, 23\n"
      "  %v33 = sub i32 1124073472, %v32\n"
      "  %v34 = or i32 %v31, %v33\n"
      "  br label %b6\n"
      "b6:\n"
      "  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]\n"
      "  %v36 = and i16 %vinp, -32768\n"
      "  %v37 = zext i16 %v36 to i32\n"
      "  %v38 = shl nuw i32 %v37, 16\n"
      "  %v39 = or i32 %v35, %v38\n"
      "  %v40 = bitcast i32 %v39 to float\n"
      "  ret float %v40\n"
      "}\n";

  std::string short_type = use_float16_abi ? "half" : "i16";

  std::string short_cast_in, short_cast_out;
  if (use_float16_abi) {
    short_cast_in = "  %vinp = bitcast half %a0 to i16\n";
    short_cast_out = "  %vres = bitcast i16 %vlast to half\n";
  } else {
    // No-ops that preserve the i16 values.
    short_cast_in = "  %vinp = add i16 %a0, 0\n";
    short_cast_out = "  %vres = add i16 %vlast, 0\n";
  }

  llvm::Triple triple(llvm_target_->GetTargetTriple());

  static const char elf_section_name[] = ".text.tvm.fp16.conv";
  std::string section = triple.getObjectFormat() == llvm::Triple::ELF
                            ? std::string("section \"") + elf_section_name + "\" "
                            : "";

  std::string trunc_header = "define weak dso_local " + short_type +
                             " @__truncsfhf2(float %a0) local_unnamed_addr #0 " + section +
                             "{\nb0:\n";
  std::string trunc_return = "  ret " + short_type + " %vres\n}\n";

  std::string extend_header = "define weak dso_local float @__extendhfsf2(" + short_type +
                              " %a0) local_unnamed_addr #0 " + section + "{\nb0:\n";

  // truncate = trunc_header + trunc_body + short_cast_out + trunc_return
  // extend   = extend_header + short_cast_in + extend_body

  std::string attributes = "attributes #0 = { nounwind readnone \"target-cpu\"=\"" +
                           llvm_target_->GetCPU() + "\" \"target-features\"=\"" +
                           llvm_target_->GetTargetFeatureString() + "\" }\n";

  auto data_layout = llvm_target_->GetOrCreateTargetMachine()->createDataLayout();
  std::string module_ir = "target triple = \"" + llvm_target_->GetTargetTriple() + "\"\n" +
                          "target datalayout = \"" + data_layout.getStringRepresentation() +
                          "\"\n" + trunc_header + trunc_body + short_cast_out + trunc_return +
                          extend_header + short_cast_in + extend_body + attributes;

  auto builtins_module = llvm_target_->GetInstance().ParseIR(module_ir);
  link_modules_.push_back(std::move(builtins_module));
}

llvm::Value* CodeGenLLVM::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin_call_llvm_intrin_) || op->op.same_as(builtin_call_llvm_pure_intrin_)) {
    ICHECK_GE(op->args.size(), 2U);
    llvm::Intrinsic::ID id = static_cast<llvm::Intrinsic::ID>(Downcast<IntImm>(op->args[0])->value);
    int64_t num_signature = Downcast<IntImm>(op->args[1])->value;
    std::vector<llvm::Value*> arg_value;
    std::vector<llvm::Type*> arg_type;
    for (size_t i = 2; i < op->args.size(); ++i) {
      arg_value.push_back(MakeValue(op->args[i]));
      if (i - 2 < static_cast<size_t>(num_signature)) {
        arg_type.push_back(arg_value.back()->getType());
      }
    }
    // LLVM's prefetch intrinsic returns "void", while TVM's prefetch
    // returns int32. This causes problems because prefetch is one of
    // those intrinsics that is generated automatically via the
    // tvm.intrin.rule mechanism. Any other intrinsic with a type
    // mismatch will have to be treated specially here.
    // TODO(kparzysz-quic): fix this once TVM prefetch uses the same
    // type as LLVM.
    llvm::Type* return_type =
        (id != llvm::Intrinsic::prefetch) ? GetLLVMType(GetRef<PrimExpr>(op)) : t_void_;
    llvm::Function* f = GetIntrinsicDecl(id, return_type, arg_type);
    ICHECK(f) << "Cannot find intrinsic declaration, possible type mismatch: "
#if TVM_LLVM_VERSION >= 130
              << llvm::Intrinsic::getBaseName(id).str();
#else
              << llvm::Intrinsic::getName(id, {});
#endif
    return builder_->CreateCall(f, arg_value);
  } else if (op->op.same_as(builtin::bitwise_and())) {
    return builder_->CreateAnd(MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->op.same_as(builtin::bitwise_or())) {
    return builder_->CreateOr(MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->op.same_as(builtin::bitwise_not())) {
    return builder_->CreateNot(MakeValue(op->args[0]));
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    return builder_->CreateXor(MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->op.same_as(builtin::shift_left())) {
    return builder_->CreateShl(MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->op.same_as(builtin::shift_right())) {
    if (op->args[0].dtype().is_int()) {
      return builder_->CreateAShr(MakeValue(op->args[0]), MakeValue(op->args[1]));
    } else {
      return builder_->CreateLShr(MakeValue(op->args[0]), MakeValue(op->args[1]));
    }
  } else if (op->op.same_as(builtin::tvm_storage_sync())) {
    return CreateStorageSync(op);
  } else if (op->op.same_as(builtin::address_of())) {
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);

    Array<PrimExpr> indices = load->indices;
    if (const RampNode* r = indices[indices.size() - 1].as<RampNode>()) {
      indices.Set(indices.size() - 1, r->base);
    }

    std::vector<llvm::Value*> indices_val;
    for (const auto& index : indices) {
      indices_val.push_back(MakeValue(index));
    }

    TypedPointer buffer_ptr = CreateBufferPtr(MakeValue(load->buffer->data), load->buffer->dtype,
                                              indices_val, load->dtype);
    unsigned addrspace =
        llvm::dyn_cast<llvm::PointerType>(buffer_ptr.addr->getType())->getAddressSpace();
    return builder_->CreatePointerCast(buffer_ptr.addr, t_char_->getPointerTo(addrspace));
  } else if (op->op.same_as(builtin::reinterpret()) && is_zero(op->args[0])) {
    return llvm::Constant::getNullValue(t_void_p_);
  } else if (op->op.same_as(builtin::isnullptr())) {
    return builder_->CreateIsNull(MakeValue(op->args[0]));
  } else if (op->op.same_as(builtin::large_uint_imm())) {
    ICHECK_EQ(op->args.size(), 2U);
    uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
    uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
    uint64_t val = (high << 32U) | low;
    return llvm::ConstantInt::get(DTypeToLLVMType(op->dtype), val);
  } else if (op->op.same_as(builtin::if_then_else())) {
    ICHECK_EQ(op->args[0].dtype().lanes(), 1) << "if_then_else can only take scalar condition";
    llvm::LLVMContext* ctx = llvm_target_->GetContext();
    auto* then_block = llvm::BasicBlock::Create(*ctx, "if_then", function_);
    auto* else_block = llvm::BasicBlock::Create(*ctx, "if_else", function_);
    auto* end_block = llvm::BasicBlock::Create(*ctx, "if_end", function_);
    builder_->CreateCondBr(MakeValue(op->args[0]), then_block, else_block);
    builder_->SetInsertPoint(then_block);
    llvm::Value* then_value = MakeValue(op->args[1]);
    llvm::BasicBlock* then_value_block = builder_->GetInsertBlock();
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(else_block);
    llvm::Value* else_value = MakeValue(op->args[2]);
    llvm::BasicBlock* else_value_block = builder_->GetInsertBlock();
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(end_block);
    llvm::PHINode* value = builder_->CreatePHI(then_value->getType(), 2);
    value->addIncoming(then_value, then_value_block);
    value->addIncoming(else_value, else_value_block);
    return value;
  } else if (op->op.same_as(builtin::ret())) {
    auto const* val = op->args[0].as<IntImmNode>();
    ICHECK(val) << "the tir.ret should be transformed to return zero "
                << "before the llvm code generation.";
    ICHECK_EQ(val->value, 0) << "the tir.ret should be transformed to "
                             << "return zero before the llvm code generation.";
    builder_->CreateRet(ConstInt32(0));
    // LLVM allows exactly one terminator in a single basic block
    // append a new dummy basic block to avoid error.
    llvm::BasicBlock* ret_dummy =
        llvm::BasicBlock::Create(*llvm_target_->GetContext(), "ret_dummy", function_);
    builder_->SetInsertPoint(ret_dummy);
    return ret_dummy;
  } else if (op->op.same_as(builtin::reinterpret())) {
    llvm::Type* target = DTypeToLLVMType(op->dtype);
    return builder_->CreateBitCast(MakeValue(op->args[0]), target);
  } else if (op->op.same_as(builtin::isnan())) {
    // TODO(hgt312): set fast math flag
    llvm::Value* a = MakeValue(op->args[0]);
    return builder_->CreateFCmpUNO(a, a);
  } else if (op->op.same_as(builtin::vectorlow())) {
    llvm::Value* v = MakeValue(op->args[0]);
    int l = GetVectorNumElements(v);
    return CreateVecSlice(v, 0, l / 2);
  } else if (op->op.same_as(builtin::vectorhigh())) {
    llvm::Value* v = MakeValue(op->args[0]);
    int l = GetVectorNumElements(v);
    return CreateVecSlice(v, l / 2, l / 2);
  } else if (op->op.same_as(builtin::vectorcombine())) {
    llvm::Value* v0 = MakeValue(op->args[0]);
    llvm::Value* v1 = MakeValue(op->args[1]);
    int num_elems = GetVectorNumElements(v0) * 2;
#if TVM_LLVM_VERSION >= 110
    std::vector<int> indices;
#else
    std::vector<unsigned> indices;
#endif
    for (int i = 0; i < num_elems; ++i) {
      indices.push_back(i);
    }
    return builder_->CreateShuffleVector(v0, v1, indices);
  } else if (op->op.same_as(builtin::atomic_add())) {
    // TODO(masahi): Support atomic for CPU backend
    LOG(FATAL) << "CPU backend does not support atomic add yet.";
    return nullptr;
  } else if (op->op.same_as(builtin::start_profile_intrinsic()) ||
             op->op.same_as(builtin::end_profile_intrinsic())) {
    LOG(INFO) << "Ignoring profile_intrinsic ... " << op->op;
    return nullptr;
  } else {
    LOG(FATAL) << "unknown intrinsic " << op->op;
    return nullptr;
  }
}

void CodeGenLLVM::Scalarize(const PrimExpr& e, std::function<void(int i, llvm::Value* v)> f) {
  if (const RampNode* ramp = e.as<RampNode>()) {
    for (int i = 0; i < ramp->dtype.lanes(); ++i) {
      PrimExpr offset = ramp->base + (ramp->stride * i);
      f(i, MakeValue(offset));
    }
  } else {
    llvm::Value* value = MakeValue(e);
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      f(i, builder_->CreateExtractElement(value, i));
    }
  }
}

// Visitors
llvm::Value* CodeGenLLVM::VisitExpr_(const VarNode* op) { return GetVarValue(op); }

llvm::Value* CodeGenLLVM::VisitExpr_(const CastNode* op) {
  return CreateCast(op->value.dtype(), op->dtype, MakeValue(op->value));
}
llvm::Value* CodeGenLLVM::VisitExpr_(const IntImmNode* op) {
  return llvm::ConstantInt::getSigned(DTypeToLLVMType(op->dtype), op->value);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const FloatImmNode* op) {
  return llvm::ConstantFP::get(DTypeToLLVMType(op->dtype), op->value);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const StringImmNode* op) { return GetConstString(op->value); }

#define DEFINE_CODEGEN_BINARY_OP(Op)                                                 \
  llvm::Value* CodeGenLLVM::Create##Op(DataType t, llvm::Value* a, llvm::Value* b) { \
    if (t.is_int()) {                                                                \
      if (t.bits() >= 32) {                                                          \
        return builder_->CreateNSW##Op(a, b);                                        \
      } else {                                                                       \
        return builder_->Create##Op(a, b);                                           \
      }                                                                              \
    } else if (t.is_uint()) {                                                        \
      if (t.bits() >= 32) {                                                          \
        return builder_->CreateNUW##Op(a, b);                                        \
      } else {                                                                       \
        return builder_->Create##Op(a, b);                                           \
      }                                                                              \
    } else {                                                                         \
      ICHECK(t.is_float());                                                          \
      return builder_->CreateF##Op(a, b);                                            \
    }                                                                                \
  }                                                                                  \
  llvm::Value* CodeGenLLVM::VisitExpr_(const Op##Node* op) {                         \
    return Create##Op(op->dtype, MakeValue(op->a), MakeValue(op->b));                \
  }

DEFINE_CODEGEN_BINARY_OP(Add);
DEFINE_CODEGEN_BINARY_OP(Sub);
DEFINE_CODEGEN_BINARY_OP(Mul);

#define DEFINE_CODEGEN_CMP_OP(Op)                                                    \
  llvm::Value* CodeGenLLVM::Create##Op(DataType t, llvm::Value* a, llvm::Value* b) { \
    if (t.is_int()) {                                                                \
      return builder_->CreateICmpS##Op(a, b);                                        \
    } else if (t.is_uint()) {                                                        \
      return builder_->CreateICmpU##Op(a, b);                                        \
    } else {                                                                         \
      ICHECK(t.is_float());                                                          \
      return builder_->CreateFCmpO##Op(a, b);                                        \
    }                                                                                \
  }                                                                                  \
  llvm::Value* CodeGenLLVM::VisitExpr_(const Op##Node* op) {                         \
    return Create##Op(op->a.dtype(), MakeValue(op->a), MakeValue(op->b));            \
  }

DEFINE_CODEGEN_CMP_OP(LT);
DEFINE_CODEGEN_CMP_OP(LE);
DEFINE_CODEGEN_CMP_OP(GT);
DEFINE_CODEGEN_CMP_OP(GE);

llvm::Value* CodeGenLLVM::VisitExpr_(const DivNode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->CreateSDiv(a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->CreateUDiv(a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->CreateFDiv(a, b);
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const ModNode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->CreateSRem(a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->CreateURem(a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->CreateFRem(a, b);
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const MinNode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  return builder_->CreateSelect(CreateLT(op->a.dtype(), a, b), a, b);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const MaxNode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  return builder_->CreateSelect(CreateGT(op->a.dtype(), a, b), a, b);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const EQNode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  if (op->a.dtype().is_int() || op->a.dtype().is_uint()) {
    return builder_->CreateICmpEQ(a, b);
  } else {
    return builder_->CreateFCmpOEQ(a, b);
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const NENode* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  if (op->a.dtype().is_int() || op->a.dtype().is_uint()) {
    return builder_->CreateICmpNE(a, b);
  } else {
    return builder_->CreateFCmpONE(a, b);
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const AndNode* op) {
  return builder_->CreateAnd(MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const OrNode* op) {
  return builder_->CreateOr(MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const NotNode* op) {
  return builder_->CreateNot(MakeValue(op->a));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const SelectNode* op) {
  return builder_->CreateSelect(MakeValue(op->condition), MakeValue(op->true_value),
                                MakeValue(op->false_value));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const LetNode* op) {
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  auto var_value = MakeValue(op->value);
  var_map_[op->var.get()] = var_value;
  var_value->setName(op->var->name_hint.c_str());
  analyzer_->Bind(op->var, op->value);
  return MakeValue(op->body);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const LoadNode* op) {
  LOG(FATAL) << "Unexpected deprecated LoadNode.  Use BufferLoadNode instead.";
  return nullptr;
}

bool CodeGenLLVM::HasAlignmentPadding(DataType dtype) {
  const llvm::DataLayout& data_layout = module_->getDataLayout();
  int bytes = data_layout.getTypeAllocSize(DTypeToLLVMType(dtype));
  int bytes_scalar = data_layout.getTypeAllocSize(DTypeToLLVMType(dtype.element_of()));
  return bytes != bytes_scalar * dtype.lanes();
}

void CodeGenLLVM::BufferAccessHelper(
    Buffer buffer, Array<PrimExpr> indices, DataType value_dtype,
    std::function<llvm::Instruction*(TypedPointer buffer_ptr, int subelement_i, int alignment,
                                     bool is_volatile)>
        make_instruction) {
  DataType buffer_element_dtype = buffer->dtype;

  ICHECK_GE(indices.size(), 1)
      << "Buffer " << buffer->name << " is accessed with no indices.  "
      << "0-d scalar buffers are expected to be flattened to 1-d buffers prior to codegen.";

  // Only the last index is allowed to be multi-lane.  All earlier
  // indices must be scalar.  This only matters for subclasses of
  // CodeGenLLVM, because the default implementation of GetBufferPtr
  // requires 1-d indices.
  std::vector<llvm::Value*> earlier_index_values;
  for (size_t i = 0; i < indices.size() - 1; i++) {
    ICHECK_EQ(indices[i].dtype().lanes(), 1)
        << "Buffer " << buffer->name << " is accessed with a multi-lane index at position " << i
        << ".  Multi-lane indices are only supported as the last index.";
    earlier_index_values.push_back(MakeValue(indices[i]));
  }

  PrimExpr last_index = indices[indices.size() - 1];
  ICHECK_EQ(value_dtype.lanes(), last_index.dtype().lanes() * buffer_element_dtype.lanes());

  // Record index and elemtype in original form used for alias info
  PrimExpr last_index_origin = last_index;
  DataType buffer_element_dtype_origin = buffer_element_dtype;

  bool is_volatile = volatile_buf_.count(buffer->data.get());

  // If the buffer index is a contiguous ramp node, we only need to
  // access the first element, then cast to the value type.
  if (const RampNode* ramp_index = last_index.as<RampNode>()) {
    if (is_one(ramp_index->stride)) {
      last_index = ramp_index->base;
    }
  }

  // All TVM arrays are densely packed.  If the vectorized LLVM type
  // contains padding for alignment, we need to index based on the
  // size of the scalar type to avoid introducing that padding.
  if (last_index.dtype().lanes() == 1 && HasAlignmentPadding(buffer_element_dtype)) {
    last_index = buffer_element_dtype.lanes() * last_index;
    buffer_element_dtype = buffer_element_dtype.element_of();
  }

  int alignment;
  if (last_index.dtype().lanes() == 1) {
    // If we are accessing with a single index, then the vectorized
    // element being accessed may require more alignment than the
    // underlying data type.
    int native_bits;
    GetAlignment(value_dtype, buffer->data.get(), last_index, &alignment, &native_bits);
  } else {
    // Otherwise, alignment is based on the return value's scalar
    // type.
    ICHECK_GE(value_dtype.bits(), 8);
    alignment = value_dtype.bits() / 8;
  }

  llvm::Value* cached_vector_index = nullptr;
  for (int i = 0; i < last_index.dtype().lanes(); ++i) {
    llvm::Value* last_index_value;
    int subelement_i = i;
    if (const RampNode* ramp = last_index.as<RampNode>()) {
      PrimExpr offset = ramp->base + (ramp->stride * i);
      last_index_value = MakeValue(offset);
    } else if (last_index.dtype().lanes() > 1) {
      if (i == 0) {
        cached_vector_index = MakeValue(last_index);
      }
      last_index_value = builder_->CreateExtractElement(cached_vector_index, i);
    } else {
      last_index_value = MakeValue(last_index);
      subelement_i = -1;
    }

    std::vector<llvm::Value*> all_index_values = earlier_index_values;
    all_index_values.push_back(last_index_value);

    TypedPointer buffer_ptr =
        CreateBufferPtr(MakeValue(buffer->data), buffer_element_dtype, all_index_values,
                        value_dtype.with_lanes(value_dtype.lanes() / last_index.dtype().lanes()));
    auto instruction = make_instruction(buffer_ptr, subelement_i, alignment, is_volatile);
    AddAliasInfo(instruction, buffer->data.get(), last_index_origin, buffer_element_dtype_origin);
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const BufferLoadNode* op) {
  DataType value_dtype = op->dtype;

  std::vector<llvm::Value*> loads;

  auto make_load = [this, &loads](TypedPointer buffer_ptr, int /* subelement_i */, int alignment,
                                  bool is_volatile) {
#if TVM_LLVM_VERSION >= 110
    auto load = builder_->CreateAlignedLoad(buffer_ptr.type, buffer_ptr.addr,
                                            llvm::Align(alignment), is_volatile);
#elif TVM_LLVM_VERSION >= 80
    auto load =
        builder_->CreateAlignedLoad(buffer_ptr.type, buffer_ptr.addr, alignment, is_volatile);
#else
    auto load = builder_->CreateAlignedLoad(buffer_ptr.addr, alignment, is_volatile);
#endif

    loads.push_back(load);
    return load;
  };

  // Pass all indices into BufferAccessHelper.  In CodeGenLLVM,
  // non-flat indices will result in an error in CreateBufferPtr, but
  // a subclass may override CreateBufferPtr.
  BufferAccessHelper(op->buffer, op->indices, value_dtype, make_load);

  if (loads.size() == 1) {
    return loads[0];
  } else {
    llvm::Value* ret = llvm::UndefValue::get(DTypeToLLVMType(value_dtype));
    for (size_t i = 0; i < loads.size(); i++) {
      ret = builder_->CreateInsertElement(ret, loads[i], ConstInt32(i));
    }
    return ret;
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const CallNode* op) {
  if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);
    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      // call extern intrinsic
      ICHECK_GE(op->args.size(), 1U);
      auto global_symbol = Downcast<StringImm>(op->args[0]);
      return this->CreateCallExtern(GetType(GetRef<PrimExpr>(op)), global_symbol->value, op->args,
                                    true);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      return this->CreateCallExtern(GetType(GetRef<PrimExpr>(op)), op_attr_global_symbol_[call_op],
                                    op->args, false);
    } else {
      VLOG(2) << "CreateIntrinsic: " << GetRef<Call>(op);
      auto x = CreateIntrinsic(op);
      VLOG(2) << "CreateIntrinsic done";
      return x;
    }
  } else {
    ICHECK(op->op.as<GlobalVarNode>());
    LOG(FATAL) << "Do not yet support cross function call";
    return nullptr;
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const RampNode* op) {
  llvm::Value* vec = llvm::UndefValue::get(DTypeToLLVMType(op->dtype));
  for (int i = 0; i < op->lanes; ++i) {
    vec = builder_->CreateInsertElement(
        vec, MakeValue(op->base + op->stride * make_const(op->stride.dtype(), i)), ConstInt32(i));
  }
  return vec;
}

llvm::Value* CodeGenLLVM::VisitExpr_(const ShuffleNode* op) {
  std::vector<llvm::Value*> vecs(op->vectors.size());
  int total_lanes = 0;
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    vecs[i] = VisitExpr(op->vectors[i]);
    total_lanes += op->vectors[i].dtype().lanes();
  }
  llvm::Value* v0 = CreateVecConcat(vecs);
  std::vector<uint32_t> idx(op->indices.size());
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t* val = as_const_int(op->indices[i]);
    ICHECK(val && *val >= 0 && *val < total_lanes) << "Shuffled indeces are suppose to be int, "
                                                   << "but get " << op->indices[i] << "\n";
    idx[i] = *val;
  }
  llvm::Value* mask = llvm::ConstantDataVector::get(builder_->getContext(), idx);
  auto res = builder_->CreateShuffleVector(v0, llvm::UndefValue::get(v0->getType()), mask);
  // If the output is a single-element vector, convert it back to a scalar.
  if (idx.size() == 1) {
    res = builder_->CreateExtractElement(res, ConstInt32(0));
  }
  return res;
}

llvm::Value* CodeGenLLVM::VisitExpr_(const BroadcastNode* op) {
  return CreateBroadcast(MakeValue(op->value), op->lanes);
}

void CodeGenLLVM::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected deprecated StoreNode.  Use BufferStoreNode instead.";
}

void CodeGenLLVM::VisitStmt_(const BufferStoreNode* op) {
  DataType value_dtype = op->value.dtype();
  Var buffer_var = op->buffer->data;

  llvm::Value* value = MakeValue(op->value);

  auto make_store = [this, value](TypedPointer buffer_ptr, int subelement_i, int alignment,
                                  bool is_volatile) {
    llvm::Value* to_store = value;
    if (subelement_i != -1) {
      to_store = builder_->CreateExtractElement(value, subelement_i);
    }
#if TVM_LLVM_VERSION >= 110
    return builder_->CreateAlignedStore(to_store, buffer_ptr.addr, llvm::Align(alignment),
                                        is_volatile);
#else
    return builder_->CreateAlignedStore(to_store, buffer_ptr.addr, alignment, is_volatile);
#endif
  };

  // Pass all indices into BufferAccessHelper.  In CodeGenLLVM,
  // non-flat indices will result in an error in CreateBufferPtr, but
  // a subclass may override CreateBufferPtr.
  BufferAccessHelper(op->buffer, op->indices, value_dtype, make_store);
}

void CodeGenLLVM::VisitStmt_(const ForNode* op) {
  ICHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  if (op->kind == ForKind::kUnrolled) {
    LOG(WARNING) << "Unroll hint get ignore at CodeGenLLVM backend, "
                 << " consider set unroll_explicit=True";
  } else {
    ICHECK(op->kind == ForKind::kSerial);
  }
  CreateSerialFor(MakeValue(op->min), MakeValue(op->extent),
                  llvm::ConstantInt::getSigned(GetLLVMType(op->extent), 1), op->loop_var, op->body);
}

void CodeGenLLVM::VisitStmt_(const WhileNode* op) {
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  auto* while_cond = llvm::BasicBlock::Create(*ctx, "while_cond", function_);
  auto* while_body = llvm::BasicBlock::Create(*ctx, "while_body", function_);
  auto* while_merge = llvm::BasicBlock::Create(*ctx, "while_merge", function_);
  builder_->CreateBr(while_cond);
  builder_->SetInsertPoint(while_cond);
  builder_->CreateCondBr(MakeValue(op->condition), while_body, while_merge);
  builder_->SetInsertPoint(while_body);
  this->VisitStmt(op->body);
  builder_->CreateBr(while_cond);
  builder_->SetInsertPoint(while_merge);
}

void CodeGenLLVM::VisitStmt_(const IfThenElseNode* op) {
  llvm::Value* cond = MakeValue(op->condition);
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  auto* then_block = llvm::BasicBlock::Create(*ctx, "if_then", function_);
  auto* end_block = llvm::BasicBlock::Create(*ctx, "if_end", function_);
  if (op->else_case) {
    auto* else_block = llvm::BasicBlock::Create(*ctx, "if_else", function_);
    builder_->CreateCondBr(cond, then_block, else_block);
    builder_->SetInsertPoint(then_block);
    this->VisitStmt(op->then_case);
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(else_block);
    this->VisitStmt(op->else_case.value());
    builder_->CreateBr(end_block);
  } else {
    builder_->CreateCondBr(cond, then_block, end_block, md_very_likely_branch_);
    builder_->SetInsertPoint(then_block);
    this->VisitStmt(op->then_case);
    builder_->CreateBr(end_block);
  }
  builder_->SetInsertPoint(end_block);
}

void CodeGenLLVM::VisitStmt_(const AllocateConstNode* op) {
  auto data = op->data.value();
  auto array = NDArrayToLLVMArray(llvm_target_->GetContext(), data);
  std::string symbol_name = op->buffer_var->name_hint;
  llvm::GlobalVariable* param_symbol = new llvm::GlobalVariable(
      *module_, array->getType(), true, llvm::GlobalValue::InternalLinkage, array, symbol_name);

  var_map_[op->buffer_var.operator->()] = param_symbol;
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const AllocateNode* op) {
  ICHECK_EQ(op->extents.size(), 1)
      << "LLVM codegen only supports flat 1-d buffer allocation, but allocation of "
      << op->buffer_var->name_hint << " is " << op->extents << "-d";

  ICHECK(!is_zero(op->condition));
  llvm::Value* buf = nullptr;

  int32_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation";
  StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
  if (constant_size % 4 == 0 && info.alignment == 0) {
    info.alignment = GetTempAllocaAlignment(op->dtype, constant_size);
  }
  // maximum necessary alignment in the NV devices
  if (info.alignment > 16) {
    info.alignment = 16;
  }
  llvm::AllocaInst* alloca = WithFunctionEntry([&]() {
    return builder_->CreateAlloca(DTypeToLLVMType(op->dtype), ConstInt32(constant_size));
  });
#if TVM_LLVM_VERSION >= 110
  auto alignment = static_cast<unsigned>(alloca->getAlign().value());
#else
  unsigned alignment = alloca->getAlignment();
#endif
  if (alignment < static_cast<unsigned>(info.alignment)) {
#if TVM_LLVM_VERSION >= 100
    alloca->setAlignment(llvm::Align(info.alignment));
#else
    alloca->setAlignment(info.alignment);
#endif
  }
#if TVM_LLVM_VERSION >= 110
  info.alignment = static_cast<unsigned>(alloca->getAlign().value());
#else
  info.alignment = alloca->getAlignment();
#endif

  buf = alloca;

  buf = builder_->CreatePointerCast(
      buf, DTypeToLLVMType(op->dtype)->getPointerTo(buf->getType()->getPointerAddressSpace()));
  buf->setName(op->buffer_var->name_hint.c_str());

  ICHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_map_.count(iv->var.get())) {
        var_map_[iv->var.get()] = GetThreadIndex(iv);
        analyzer_->Bind(iv->var, Range::FromMinExtent(0, op->value));
      }
    }
  } else if (op->attr_key == tir::attr::storage_alignment) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    alloc_storage_info_[v].alignment = static_cast<int>(op->value.as<IntImmNode>()->value);
    if (var_map_.count(v) && alloc_storage_info_[v].alignment > 1) {
      builder_->CreateAlignmentAssumption(*data_layout_, GetVarValue(v),
                                          alloc_storage_info_[v].alignment);
    }
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    volatile_buf_.insert(v);
  }
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const AssertStmtNode* op) {
  With<arith::ConstraintContext> cctx(analyzer_.get(), op->condition);
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const LetStmtNode* op) {
  const VarNode* v = op->var.get();
  ICHECK(!var_map_.count(v));
  if (v->dtype.is_handle()) {
    if (!is_restricted_) {
      alias_var_set_.insert(v);
    }
  }
  llvm::Value* value = MakeValue(op->value);
  value->setName(v->name_hint.c_str());
  var_map_[v] = value;
  analyzer_->Bind(op->var, op->value);
  if (alloc_storage_info_.count(v) && alloc_storage_info_[v].alignment > 1) {
    builder_->CreateAlignmentAssumption(*data_layout_, GetVarValue(v),
                                        alloc_storage_info_[v].alignment);
  }
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenLLVM::VisitStmt_(const EvaluateNode* op) { MakeValue(op->value); }

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
