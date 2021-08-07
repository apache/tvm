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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/op.h>

#include <algorithm>

#include "../../arith/pattern_match.h"
#include "../build_common.h"
#include "../func_registry_generator.h"
#include "codegen_cpu.h"
#include "codegen_params.h"
#include "llvm/Support/raw_os_ostream.h"
namespace tvm {
namespace codegen {

std::unique_ptr<CodeGenLLVM> CodeGenLLVM::Create(llvm::TargetMachine* tm) {
  std::string target = tm->getTarget().getName();
  std::string factory_name = "tvm.codegen.llvm.target_" + target;
  const PackedFunc* f = runtime::Registry::Get(factory_name);
  if (f != nullptr) {
    void* handle = (*f)();
    return std::unique_ptr<CodeGenLLVM>(static_cast<CodeGenLLVM*>(handle));
  } else {
    return std::unique_ptr<CodeGenLLVM>(new CodeGenCPU());
  }
}

void CodeGenLLVM::Init(const std::string& module_name, llvm::TargetMachine* tm,
                       llvm::LLVMContext* ctx, bool system_lib, bool dynamic_lookup,
                       bool target_c_runtime) {
  InitializeLLVM();
  ctx_ = ctx;
  builder_.reset(new IRBuilder(*ctx_));
  module_.reset(new llvm::Module(module_name, *ctx_));
  md_builder_.reset(new llvm::MDBuilder(*ctx_));
  // types
  t_void_ = llvm::Type::getVoidTy(*ctx_);
  t_void_p_ = llvm::Type::getInt8Ty(*ctx_)->getPointerTo(GetGlobalAddressSpace());
  t_int_ = llvm::Type::getInt32Ty(*ctx_);
  t_char_ = llvm::Type::getInt8Ty(*ctx_);
  t_int8_ = llvm::Type::getInt8Ty(*ctx_);
  t_int16_ = llvm::Type::getInt16Ty(*ctx_);
  t_int32_ = llvm::Type::getInt32Ty(*ctx_);
  t_int64_ = llvm::Type::getInt64Ty(*ctx_);
  t_float64_ = llvm::Type::getDoubleTy(*ctx_);
  // meta data
  md_very_likely_branch_ = md_builder_->createBranchWeights(1 << 20, 1);
  md_tbaa_root_ = md_builder_->createTBAARoot("tvm-tbaa");
  md_tbaa_alias_set_ = md_builder_->createTBAANode("tvm-alias", md_tbaa_root_);
  this->InitTarget(tm);
}

void CodeGenLLVM::InitTarget(llvm::TargetMachine* tm) {
  module_->setTargetTriple(tm->getTargetTriple().str());
  module_->setDataLayout(tm->createDataLayout());
  data_layout_.reset(new llvm::DataLayout(module_.get()));
  target_machine_ = tm;
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
  ICHECK(module_->getFunction(static_cast<std::string>(global_symbol.value())) == nullptr)
      << "Function " << global_symbol << " already exist in module";

  function_ = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                     global_symbol.value().operator std::string(), module_.get());
  function_->setCallingConv(llvm::CallingConv::C);
  function_->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

  // set var map and align information
  auto arg_it = function_->arg_begin();
  for (size_t i = 0; i < f->params.size(); ++i, ++arg_it) {
    llvm::Argument* v = &(*arg_it);
    const Var& var = f->params[i];
    var_map_[var.get()] = v;
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
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*ctx_, "entry", function_);
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
        auto attr = llvm::Attribute::get(*ctx_, llvm::Attribute::Alignment, align);
        function_->addParamAttr(i, attr);
      }
    }
  }
#endif

  llvm::StringRef fs = target_machine_->getTargetFeatureString();
  if (!fs.empty()) {
    function_->addFnAttr("target-features", fs);
  }

  if (ret_void) {
    builder_->CreateRetVoid();
  } else {
    builder_->CreateRet(ConstInt32(0));
  }
}

void CodeGenLLVM::LinkParameters(const Map<String, LinkedParam> params) {
  // It would be nice to de-dupe these declarations frm src/tir/transforms/make_packed_api.cc,
  // but they are at a different layer in the compiler...
  llvm::Type* t_int_p = t_int_->getPointerTo(GetGlobalAddressSpace());

  // args, tcodes, num_args, ret_value, ret_tcode, resource_handle
  std::vector<llvm::Type*> param_types{t_void_p_, t_int_p, t_int_, t_void_p_, t_int_p, t_void_p_};
  llvm::FunctionType* ftype = llvm::FunctionType::get(t_int_, param_types, false);

  llvm::Function* function =
      llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                             ::tvm::runtime::symbol::tvm_lookup_linked_param, module_.get());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*ctx_, "entry", function);
  builder_->SetInsertPoint(entry);

  auto getArg = [function](int i) -> llvm::Argument* {
#if TVM_LLVM_VERSION >= 100
    return function->getArg(i);
#elif TVM_LLVM_VERSION >= 50
    return &function->arg_begin()[i];
#else
    return &*std::next(function->arg_begin(), i);
#endif
  };

  llvm::Type* t_int64_p = t_int64_->getPointerTo(GetGlobalAddressSpace());
  llvm::Value* sid = builder_->CreateLoad(t_int64_, builder_->CreateBitCast(getArg(0), t_int64_p));

  auto ret_tcode = builder_->CreateBitCast(getArg(4), t_int_p);
  auto ret_value =
      builder_->CreateBitCast(getArg(3), t_void_p_->getPointerTo(GetGlobalAddressSpace()));

  llvm::BasicBlock* default_block = llvm::BasicBlock::Create(*ctx_, "default_block", function);
  llvm::SwitchInst* switch_inst = builder_->CreateSwitch(sid, default_block, params.size() + 1);

  builder_->SetInsertPoint(default_block);
  builder_->CreateStore(llvm::ConstantInt::get(t_int_, kTVMNullptr), ret_tcode);
  builder_->CreateRet(ConstInt32(kTvmErrorNoError));

  // Add data to the global section.
  for (auto kv : params) {
    auto array = NDArrayToLLVMArray(ctx_, kv.second->param);
    std::string symbol_name = std::string(::tvm::runtime::symbol::tvm_param_prefix) + kv.first;
    llvm::GlobalVariable* param_symbol = new llvm::GlobalVariable(
        *module_, array->getType(), true, llvm::GlobalValue::InternalLinkage, array, symbol_name);
    auto dtype = tvm::runtime::DataType(kv.second->param->dtype);
    size_t align = std::max(tvm::runtime::GetVectorBytes(dtype), tvm::runtime::kAllocAlignment);
#if TVM_LLVM_VERSION >= 100
    param_symbol->setAlignment(llvm::Align(align));
#else
    param_symbol->setAlignment(align);
#endif

    llvm::BasicBlock* case_block = llvm::BasicBlock::Create(*ctx_, "case_" + symbol_name, function);
    switch_inst->addCase(
        llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(t_int64_, kv.second->id)), case_block);
    builder_->SetInsertPoint(case_block);
    builder_->CreateStore(builder_->CreatePointerCast(param_symbol, t_void_p_), ret_value);
    builder_->CreateStore(llvm::ConstantInt::get(t_int_, kTVMOpaqueHandle), ret_tcode);
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
  std::unique_ptr<llvm::Module> mlib;
  llvm::SMDiagnostic err;
  if (code.length() >= 3 &&
      (code.substr(code.length() - 3) == ".ll" || code.substr(code.length() - 3) == ".bc")) {
    mlib = llvm::parseIRFile(code, err, *ctx_);
    if (mlib.get() == nullptr) {
      std::string msg = std::string(err.getMessage());
      LOG(FATAL) << "Fail to load bitcode file " << code << "\n"
                 << "line " << err.getLineNo() << ":" << msg;
    }
  } else {
    std::unique_ptr<llvm::MemoryBuffer> buf = llvm::MemoryBuffer::getMemBuffer(code);
    mlib = llvm::parseIR(*buf, err, *ctx_);
    if (mlib.get() == nullptr) {
      std::string msg = std::string(err.getMessage());
      LOG(FATAL) << "Fail to load llvm ir "
                 << "line " << err.getLineNo() << ":" << msg << "\ncontent:\n"
                 << code;
    }
  }
  mlib->setTargetTriple(target_machine_->getTargetTriple().str());
  mlib->setDataLayout(target_machine_->createDataLayout());
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
  mpass.add(llvm::createTargetTransformInfoWrapperPass(
      target_machine_ ? target_machine_->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));
  fpass.add(llvm::createTargetTransformInfoWrapperPass(
      target_machine_ ? target_machine_->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));

  // place optimization pass
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;

#if TVM_LLVM_VERSION >= 50
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
#else
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0);
#endif
  builder.LoopVectorize = true;
  builder.SLPVectorize = true;
  this->InitPassManagerBuilder(&builder);

#if TVM_LLVM_VERSION >= 50
  target_machine_->adjustPassManager(builder);
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
  if (dtype.is_int() || dtype.is_uint()) {
    etype = llvm::Type::getIntNTy(*ctx_, dtype.bits());
  } else if (dtype.is_float()) {
    switch (dtype.bits()) {
      case 16:
        etype = llvm::Type::getHalfTy(*ctx_);
        break;
      case 32:
        etype = llvm::Type::getFloatTy(*ctx_);
        break;
      case 64:
        etype = llvm::Type::getDoubleTy(*ctx_);
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
}

llvm::Type* CodeGenLLVM::GetLLVMType(const Type& type) const {
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return DTypeToLLVMType(ptr->dtype);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
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
void CodeGenLLVM::AddAliasInfo(llvm::Instruction* inst, const VarNode* buffer, PrimExpr index) {
  if (alias_var_set_.count(buffer) != 0) {
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
  if (index.defined()) {
    if (arith::ramp(pbase, pstride, planes).Match(index)) {
      base = pbase.Eval()->value;
      int64_t xwith = planes.Eval() * pstride.Eval()->value;
      width = 1;
      while (width < xwith) {
        width *= 2;
      }
      while (base % width) {
        base -= base % width;
        width *= 2;
      }
    } else if (auto* ptr = index.as<tir::IntImmNode>()) {
      width = 1;
      base = ptr->value;
    }
  }
  llvm::MDNode* meta = md_tbaa_root_;
  std::ostringstream buffer_addr;
  buffer_addr << buffer;
  meta = md_builder_->createTBAAScalarTypeNode(buffer_addr.str(), meta);

  // Extract the underlying type of the allocated buffer.
  llvm::Type* buf_type = GetVarValue(buffer)->getType()->getScalarType();
  if (buf_type->isPointerTy()) {
    buf_type = buf_type->getPointerElementType();
  }

  std::string tmp;
  llvm::raw_string_ostream buffer_type(tmp);
  buffer_type << *buf_type;
  meta = md_builder_->createTBAAScalarTypeNode(buffer_type.str(), meta);

  // create a tree-shape access structure.
  if (width != 0) {
    for (int64_t w = 1024; w >= width; w /= 2) {
      int64_t b = (base / w) * w;
      std::stringstream os;
      os << buffer << ".w" << w << ".b" << b;
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
  using llvm::BasicBlock;
  BasicBlock* pre_block = builder_->GetInsertBlock();
  BasicBlock* for_begin = BasicBlock::Create(*ctx_, "for_begin", function_);
  BasicBlock* for_body = BasicBlock::Create(*ctx_, "for_body", function_);
  BasicBlock* for_end = BasicBlock::Create(*ctx_, "for_end", function_);
  builder_->CreateBr(for_begin);
  builder_->SetInsertPoint(for_begin);
  llvm::PHINode* loop_value = builder_->CreatePHI(begin->getType(), 2);
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

llvm::Constant* CodeGenLLVM::GetConstString(const std::string& str) {
  auto it = str_map_.find(str);
  if (it != str_map_.end()) return it->second;
  llvm::Type* type = llvm::ArrayType::get(t_char_, str.length() + 1);
  llvm::GlobalVariable* global = new llvm::GlobalVariable(
      *module_, type, true, llvm::GlobalValue::PrivateLinkage, nullptr, ".str");
#if TVM_LLVM_VERSION >= 100
  global->setAlignment(llvm::Align(1));
#else
  global->setAlignment(1);
#endif
  global->setInitializer(llvm::ConstantDataArray::getString(*ctx_, str));
  llvm::Constant* zero = ConstInt32(0);
  llvm::Constant* indices[] = {zero, zero};
  llvm::Constant* ptr = llvm::ConstantExpr::getGetElementPtr(type, global, indices);
  str_map_[str] = ptr;
  return ptr;
}

llvm::Value* CodeGenLLVM::CreateBufferPtr(DataType t, llvm::Value* buffer, llvm::Value* index) {
  llvm::PointerType* btype = llvm::dyn_cast<llvm::PointerType>(buffer->getType());
  ICHECK(btype != nullptr);
  llvm::PointerType* ptype = DTypeToLLVMType(t)->getPointerTo(btype->getAddressSpace());
  if (btype != ptype) {
    buffer = builder_->CreatePointerCast(buffer, ptype);
  }
  return builder_->CreateInBoundsGEP(buffer, index);
}

llvm::Value* CodeGenLLVM::GetVarValue(const VarNode* v) const {
  auto it = var_map_.find(v);
  ICHECK(it != var_map_.end()) << "cannot find variable " << v->name_hint;
  return it->second;
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
  llvm::Function* f = module_->getFunction(global_symbol);
  if (f == nullptr) {
    f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                               global_symbol.operator llvm::StringRef(), module_.get());
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
    llvm::Type* return_type = (id != llvm::Intrinsic::prefetch) ? GetLLVMType(GetRef<PrimExpr>(op))
                                                                : llvm::Type::getVoidTy(*ctx_);
    llvm::Function* f = GetIntrinsicDecl(id, return_type, arg_type);
    ICHECK(f) << "Cannot find intrinsic declaration, possible type mismatch: "
              << llvm::Intrinsic::getName(id, return_type, {});
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
    const LoadNode* l = op->args[0].as<LoadNode>();
    ICHECK(op->args.size() == 1 && l);
    const RampNode* r = l->index.as<RampNode>();
    llvm::Value* ptr;
    unsigned addrspace;
    if (!r) {
      ptr = CreateBufferPtr(l->dtype, MakeValue(l->buffer_var), MakeValue(l->index));
      addrspace = llvm::dyn_cast<llvm::PointerType>(ptr->getType())->getAddressSpace();
    } else {
      PrimExpr index = r->base / make_const(DataType::Int(32), r->lanes);
      ptr = CreateBufferPtr(l->dtype, MakeValue(l->buffer_var), MakeValue(index));
      addrspace = llvm::dyn_cast<llvm::PointerType>(ptr->getType())->getAddressSpace();
    }
    return builder_->CreatePointerCast(ptr, t_char_->getPointerTo(addrspace));
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
    using llvm::BasicBlock;
    BasicBlock* then_block = BasicBlock::Create(*ctx_, "if_then", function_);
    BasicBlock* else_block = BasicBlock::Create(*ctx_, "if_else", function_);
    BasicBlock* end_block = BasicBlock::Create(*ctx_, "if_end", function_);
    builder_->CreateCondBr(MakeValue(op->args[0]), then_block, else_block);
    builder_->SetInsertPoint(then_block);
    llvm::Value* then_value = MakeValue(op->args[1]);
    BasicBlock* then_value_block = builder_->GetInsertBlock();
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(else_block);
    llvm::Value* else_value = MakeValue(op->args[2]);
    BasicBlock* else_value_block = builder_->GetInsertBlock();
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
    llvm::BasicBlock* ret_dummy = llvm::BasicBlock::Create(*ctx_, "ret_dummy", function_);
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
  var_map_[op->var.get()] = MakeValue(op->value);
  analyzer_->Bind(op->var, op->value);
  return MakeValue(op->body);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const LoadNode* op) {
  DataType t = op->dtype;
  bool is_volatile = volatile_buf_.count(op->buffer_var.get());
  llvm::Value* buffer = MakeValue(op->buffer_var);
  llvm::Value* index = MakeValue(op->index);

  if (t.lanes() == 1) {
    int alignment, native_bits;
    GetAlignment(t, op->buffer_var.get(), op->index, &alignment, &native_bits);
    llvm::Value* ptr = CreateBufferPtr(t, buffer, index);
#if TVM_LLVM_VERSION >= 110
    llvm::LoadInst* load = builder_->CreateAlignedLoad(ptr, llvm::Align(alignment), is_volatile);
#else
    llvm::LoadInst* load = builder_->CreateAlignedLoad(ptr, alignment, is_volatile);
#endif
    AddAliasInfo(load, op->buffer_var.get(), op->index);
    return load;
  } else {
    // vector load
    unsigned addrspace = llvm::dyn_cast<llvm::PointerType>(buffer->getType())->getAddressSpace();
    if (const RampNode* ramp = op->index.as<RampNode>()) {
      if (is_one(ramp->stride)) {
        int alignment, native_bits;
        GetAlignment(t, op->buffer_var.get(), ramp->base, &alignment, &native_bits);
        ICHECK_EQ(ramp->lanes, t.lanes());
        llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, MakeValue(ramp->base));
        ptr = builder_->CreatePointerCast(ptr, DTypeToLLVMType(t)->getPointerTo(addrspace));
#if TVM_LLVM_VERSION >= 110
        llvm::LoadInst* load =
            builder_->CreateAlignedLoad(ptr, llvm::Align(alignment), is_volatile);
#else
        llvm::LoadInst* load = builder_->CreateAlignedLoad(ptr, alignment, is_volatile);
#endif
        AddAliasInfo(load, op->buffer_var.get(), op->index);
        return load;
      }
    }
  }
  // scalarized load.
  int basic_align = t.bits() / 8;
  llvm::Value* ret = llvm::UndefValue::get(DTypeToLLVMType(t));
  auto f = [&](int i, llvm::Value* index) {
    llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, index);
#if TVM_LLVM_VERSION >= 110
    llvm::LoadInst* load = builder_->CreateAlignedLoad(ptr, llvm::Align(basic_align), is_volatile);
#else
    llvm::LoadInst* load = builder_->CreateAlignedLoad(ptr, basic_align, is_volatile);
#endif
    ret = builder_->CreateInsertElement(ret, load, ConstInt32(i));
    AddAliasInfo(load, op->buffer_var.get(), PrimExpr());
  };
  this->Scalarize(op->index, f);
  return ret;
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
      return CreateIntrinsic(op);
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
  ICHECK(is_one(op->predicate)) << op->predicate;
  DataType t = op->value.dtype();
  bool is_volatile = volatile_buf_.count(op->buffer_var.get());
  llvm::Value* buffer = MakeValue(op->buffer_var);
  llvm::Value* index = MakeValue(op->index);
  llvm::Value* value = MakeValue(op->value);

  if (t.lanes() == 1) {
    int alignment, native_bits;
    GetAlignment(t, op->buffer_var.get(), op->index, &alignment, &native_bits);
    llvm::Value* ptr = CreateBufferPtr(t, buffer, index);
#if TVM_LLVM_VERSION >= 110
    llvm::StoreInst* store =
        builder_->CreateAlignedStore(value, ptr, llvm::Align(alignment), is_volatile);
#else
    llvm::StoreInst* store = builder_->CreateAlignedStore(value, ptr, alignment, is_volatile);
#endif
    AddAliasInfo(store, op->buffer_var.get(), op->index);
    return;
  } else {
    // vector store
    unsigned addrspace = llvm::dyn_cast<llvm::PointerType>(buffer->getType())->getAddressSpace();
    if (const RampNode* ramp = op->index.as<RampNode>()) {
      if (is_one(ramp->stride)) {
        int alignment, native_bits;
        GetAlignment(t, op->buffer_var.get(), ramp->base, &alignment, &native_bits);
        ICHECK_EQ(ramp->lanes, t.lanes());
        llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, MakeValue(ramp->base));
        ptr = builder_->CreatePointerCast(ptr, DTypeToLLVMType(t)->getPointerTo(addrspace));
#if TVM_LLVM_VERSION >= 110
        llvm::StoreInst* store =
            builder_->CreateAlignedStore(value, ptr, llvm::Align(alignment), is_volatile);
#else
        llvm::StoreInst* store = builder_->CreateAlignedStore(value, ptr, alignment, is_volatile);
#endif
        AddAliasInfo(store, op->buffer_var.get(), op->index);
        return;
      }
    }
  }
  ICHECK_GE(t.bits(), 8);
  // scalarized store.
  int basic_align = t.bits() / 8;
  auto f = [&](int i, llvm::Value* index) {
    llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, index);
#if TVM_LLVM_VERSION >= 110
    llvm::StoreInst* store = builder_->CreateAlignedStore(
        builder_->CreateExtractElement(value, i), ptr, llvm::Align(basic_align), is_volatile);
#else
    llvm::StoreInst* store = builder_->CreateAlignedStore(builder_->CreateExtractElement(value, i),
                                                          ptr, basic_align, is_volatile);
#endif
    AddAliasInfo(store, op->buffer_var.get(), PrimExpr());
  };
  this->Scalarize(op->index, f);
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
  using llvm::BasicBlock;
  BasicBlock* while_cond = BasicBlock::Create(*ctx_, "while_cond", function_);
  BasicBlock* while_body = BasicBlock::Create(*ctx_, "while_body", function_);
  BasicBlock* while_merge = BasicBlock::Create(*ctx_, "while_merge", function_);
  builder_->CreateBr(while_cond);
  builder_->SetInsertPoint(while_cond);
  builder_->CreateCondBr(MakeValue(op->condition), while_body, while_merge);
  builder_->SetInsertPoint(while_body);
  this->VisitStmt(op->body);
  builder_->CreateBr(while_cond);
  builder_->SetInsertPoint(while_merge);
}

void CodeGenLLVM::VisitStmt_(const IfThenElseNode* op) {
  using llvm::BasicBlock;
  llvm::Value* cond = MakeValue(op->condition);
  BasicBlock* then_block = BasicBlock::Create(*ctx_, "if_then", function_);
  BasicBlock* end_block = BasicBlock::Create(*ctx_, "if_end", function_);
  if (op->else_case.defined()) {
    BasicBlock* else_block = BasicBlock::Create(*ctx_, "if_else", function_);
    builder_->CreateCondBr(cond, then_block, else_block);
    builder_->SetInsertPoint(then_block);
    this->VisitStmt(op->then_case);
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(else_block);
    this->VisitStmt(op->else_case);
    builder_->CreateBr(end_block);
  } else {
    builder_->CreateCondBr(cond, then_block, end_block, md_very_likely_branch_);
    builder_->SetInsertPoint(then_block);
    this->VisitStmt(op->then_case);
    builder_->CreateBr(end_block);
  }
  builder_->SetInsertPoint(end_block);
}

void CodeGenLLVM::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  llvm::Value* buf = nullptr;

  int32_t constant_size = op->constant_allocation_size();
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
  if (alloca->getAlignment() < static_cast<uint32_t>(info.alignment)) {
#if TVM_LLVM_VERSION >= 100
    alloca->setAlignment(llvm::Align(info.alignment));
#else
    alloca->setAlignment(info.alignment);
#endif
  }
  info.alignment = alloca->getAlignment();
  buf = alloca;

  buf = builder_->CreatePointerCast(
      buf, DTypeToLLVMType(op->dtype)->getPointerTo(buf->getType()->getPointerAddressSpace()));
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
  var_map_[v] = MakeValue(op->value);
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
