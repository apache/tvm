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
 * \file codegen_cpu.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/tir/analysis.h>
#include <memory>
#include <unordered_map>
#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

void CodeGenCPU::Init(const std::string& module_name,
                      llvm::TargetMachine* tm,
                      llvm::LLVMContext* ctx,
                      bool system_lib,
                      bool dynamic_lookup) {
  CodeGenLLVM::Init(module_name, tm, ctx, system_lib, dynamic_lookup);
  dbg_info_ = CreateDebugInfo(module_.get());
  static_assert(sizeof(TVMValue) == sizeof(double), "invariant");
  func_handle_map_.clear();
  export_system_symbols_.clear();
  // TVM runtime types
  t_tvm_shape_index_ = llvm::Type::getIntNTy(*ctx, DataType::ShapeIndex().bits());
  t_tvm_context_ = llvm::StructType::create({t_int_, t_int_});
  t_tvm_type_ = llvm::StructType::create({t_int8_, t_int8_, t_int16_});
  t_tvm_func_handle_ = t_void_p_;
  t_tvm_array_ = llvm::StructType::create(
      {t_void_p_,
       t_tvm_context_,
       t_int_,
       t_tvm_type_,
       t_tvm_shape_index_->getPointerTo(),
       t_tvm_shape_index_->getPointerTo(),
       t_int64_});
  t_tvm_value_ = llvm::StructType::create({t_float64_});
  t_tvm_parallel_group_env_ = llvm::StructType::create({
      t_int32_->getPointerTo(), t_int32_});
  ftype_tvm_parallel_lambda_ = llvm::FunctionType::get(
      t_int_,
      {t_int_,
       t_tvm_parallel_group_env_->getPointerTo(),
       t_void_p_}, false);
  md_tbaa_ctx_ptr_ = md_builder_->createTBAAScalarTypeNode("ctx_ptr", md_tbaa_root_);
  // Runtime functions.
  ftype_tvm_func_call_ = llvm::FunctionType::get(t_int_, {
      t_tvm_func_handle_,
      t_tvm_value_->getPointerTo(),
      t_int_->getPointerTo(),
      t_int_,
      t_tvm_value_->getPointerTo(),
      t_int_->getPointerTo()}, false);
  ftype_tvm_get_func_from_env_ = llvm::FunctionType::get(t_int_, {
      t_void_p_,
      t_char_->getPointerTo(),
      t_tvm_func_handle_->getPointerTo()}, false);
  ftype_tvm_api_set_last_error_ = llvm::FunctionType::get(
      t_void_, {t_char_->getPointerTo()}, false);
  ftype_tvm_parallel_launch_ =
      llvm::FunctionType::get(t_int_, {
          ftype_tvm_parallel_lambda_->getPointerTo(), t_void_p_, t_int_}
        , false);
  ftype_tvm_parallel_barrier_ =
      llvm::FunctionType::get(t_int_, {
          t_int_, t_tvm_parallel_group_env_->getPointerTo()}
        , false);
  ftype_tvm_static_init_callback_ =
      llvm::FunctionType::get(t_int_, {t_void_p_}, false);
  ftype_tvm_static_init_ =
      llvm::FunctionType::get(t_int_, {
          t_void_p_->getPointerTo(),
          ftype_tvm_static_init_callback_->getPointerTo(),
          t_void_p_, t_int_}
        , false);
  // initialize TVM runtime API
  if (system_lib) {
    // We will need this in environment for backward registration.
    f_tvm_register_system_symbol_ = llvm::Function::Create(
        llvm::FunctionType::get(t_int_, {t_char_->getPointerTo(), t_void_p_}, false),
        llvm::Function::ExternalLinkage, "TVMBackendRegisterSystemLibSymbol", module_.get());
  } else {
    f_tvm_register_system_symbol_ = nullptr;
  }
  if (dynamic_lookup || system_lib) {
    f_tvm_func_call_ = llvm::Function::Create(
        ftype_tvm_func_call_,
        llvm::Function::ExternalLinkage, "TVMFuncCall", module_.get());
    f_tvm_get_func_from_env_ = llvm::Function::Create(
        ftype_tvm_get_func_from_env_,
        llvm::Function::ExternalLinkage, "TVMBackendGetFuncFromEnv", module_.get());
    f_tvm_api_set_last_error_ = llvm::Function::Create(
        ftype_tvm_api_set_last_error_,
        llvm::Function::ExternalLinkage, "TVMAPISetLastError", module_.get());
    f_tvm_parallel_launch_ = llvm::Function::Create(
        ftype_tvm_parallel_launch_,
        llvm::Function::ExternalLinkage, "TVMBackendParallelLaunch", module_.get());
    f_tvm_parallel_barrier_ = llvm::Function::Create(
        ftype_tvm_parallel_barrier_,
        llvm::Function::ExternalLinkage, "TVMBackendParallelBarrier", module_.get());
  }
  this->InitGlobalContext(dynamic_lookup);
}

void CodeGenCPU::AddFunction(const PrimFunc& f) {
  CodeGenLLVM::AddFunction(f);
  if (f_tvm_register_system_symbol_ != nullptr) {
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(global_symbol.defined())
        << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
    export_system_symbols_.emplace_back(
        std::make_pair(global_symbol.value().operator std::string(),
                       builder_->CreatePointerCast(function_, t_void_p_)));
  }
  AddDebugInformation(function_);
}

// Following Glow |DebugInfo::generateFunctionDebugInfo|, https://git.io/fjadv
void CodeGenCPU::AddDebugInformation(llvm::Function* function) {
#if TVM_LLVM_VERSION >= 50 && TVM_LLVM_VERSION < 70
  CHECK(!function->getSubprogram());
  llvm::SmallVector<llvm::Metadata*, 4> paramTys;
  llvm::DIType* returnTy =
      getDebugType(builder_.get(), dbg_info_->di_builder_.get(), function->getReturnType());
  paramTys.push_back(returnTy);
  for (size_t i = 0; i < function->arg_size(); ++i) {
    paramTys.push_back(getDebugType(builder_.get(), dbg_info_->di_builder_.get(),
                                    function->getFunctionType()->getParamType(i)));
  }
  auto* DIFunctionTy = dbg_info_->di_builder_->createSubroutineType(
      dbg_info_->di_builder_->getOrCreateTypeArray(paramTys));

#if TVM_LLVM_VERSION >= 80
  auto* DIFunction = dbg_info_->di_builder_->createFunction(
      dbg_info_->file_, function->getName(), "",
      dbg_info_->file_,
      0 /* line number */,
      DIFunctionTy,
      false /* internal linkage */);
#else
  auto* DIFunction = dbg_info_->di_builder_->createFunction(
      dbg_info_->file_, function->getName(), "",
      dbg_info_->file_,
      0 /* line number */,
      DIFunctionTy,
      false, /* internal linkage */
      true,
      0 /* line number */,
      llvm::DINode::FlagPrototyped,
      true /* isOptimized */);
#endif

  CHECK(DIFunction);
  function->setSubprogram(DIFunction);
  CHECK_EQ(function->getSubprogram(), DIFunction);

  IRBuilder builder(&function->getEntryBlock());
  if (!function->getEntryBlock().empty()) {
    builder.SetInsertPoint(&function->getEntryBlock().front());
  }
  llvm::DebugLoc DL;
  builder.SetCurrentDebugLocation(DL);
  for (size_t i = 0; i < function->arg_size(); ++i) {
    auto* paramAlloca = builder.CreateAlloca(function->getFunctionType()->getParamType(i));
    std::string paramName = "arg" + std::to_string(i + 1);
    auto param = dbg_info_->di_builder_->createParameterVariable(
        DIFunction, paramName, i + 1, dbg_info_->file_, 0,
        getDebugType(builder_.get(), dbg_info_->di_builder_.get(),
                     function->getFunctionType()->getParamType(i)),
        /* alwaysPreserve */ true);
    auto* store = builder.CreateStore(function->arg_begin() + i, paramAlloca);
    dbg_info_->di_builder_->insertDeclare(paramAlloca, param,
                                          dbg_info_->di_builder_->createExpression(),
                                          llvm::DebugLoc::get(0, 0, DIFunction), store);
  }
  dbg_info_->di_builder_->finalizeSubprogram(function->getSubprogram());
  auto* scope = function->getSubprogram();
  if (!scope) {
    return;
  }
  for (auto& BB : *function) {
    for (auto& I : BB) {
      if (I.getDebugLoc()) {
        continue;
      }
      I.setDebugLoc(llvm::DebugLoc::get(0, 0, scope));
    }
  }
#endif
}

llvm::DIType* CodeGenCPU::getDebugType(IRBuilder* builder, llvm::DIBuilder* di_builder,
                                       llvm::Type* ty) {
  if (ty == builder->getVoidTy()) {
    return nullptr;
  } else if (ty == builder->getFloatTy()) {
    return di_builder->createBasicType("float", 32, llvm::dwarf::DW_ATE_float);
  } else if (ty == builder->getInt8Ty()) {
    return di_builder->createBasicType("int8", 8, llvm::dwarf::DW_ATE_signed);
  } else if (ty == builder->getInt32Ty()) {
    return di_builder->createBasicType("int32", 32, llvm::dwarf::DW_ATE_signed);
  } else if (ty->isPointerTy()) {
    return di_builder->createPointerType(
        getDebugType(builder, di_builder, ty->getPointerElementType()),
        ty->getPrimitiveSizeInBits());
  } else {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    ty->print(rso);
    LOG(FATAL) << "Unknown LLVM type:" << rso.str();
  }
  return nullptr;
}

void CodeGenCPU::AddMainFunction(const std::string& entry_func_name) {
  llvm::Function* f = module_->getFunction(entry_func_name);
  CHECK(f) << "Function " << entry_func_name << "does not in module";
  llvm::Type* type = llvm::ArrayType::get(t_char_, entry_func_name.length() + 1);
  llvm::GlobalVariable *global = new llvm::GlobalVariable(
      *module_, type, true, llvm::GlobalValue::WeakAnyLinkage, 0,
      runtime::symbol::tvm_module_main);
#if TVM_LLVM_VERSION >= 100
  global->setAlignment(llvm::Align(1));
#else
  global->setAlignment(1);
#endif
  global->setInitializer(llvm::ConstantDataArray::getString(*ctx_, entry_func_name));
}

std::unique_ptr<llvm::Module> CodeGenCPU::Finish() {
  // link modules
  if (dbg_info_ != nullptr) {
    dbg_info_->di_builder_->finalize();
  }
  return CodeGenLLVM::Finish();
}
llvm::Value* CodeGenCPU::CreateStructRefPtr(
    DataType t, llvm::Value* buf, llvm::Value* index, int kind) {
  if (kind < intrinsic::kArrKindBound_) {
    if (buf->getType() == t_void_p_) {
      buf = builder_->CreatePointerCast(buf, t_tvm_array_->getPointerTo());
    } else {
      CHECK_EQ(buf->getType(), t_tvm_array_->getPointerTo());
    }
  }
  switch (kind) {
    case intrinsic::kArrAddr: {
      return builder_->CreateInBoundsGEP(buf, index);
    }
    case intrinsic::kArrData: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(0)});
    }
    case intrinsic::kArrShape: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(4)});
    }
    case intrinsic::kArrStrides: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(5)});
    }
    case intrinsic::kArrNDim: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(2)});
    }
    case intrinsic::kArrTypeCode: {
      return builder_->CreateInBoundsGEP(
          buf, {index, ConstInt32(3), ConstInt32(0)});
    }
    case intrinsic::kArrTypeBits: {
      return builder_->CreateInBoundsGEP(
          buf, {index, ConstInt32(3), ConstInt32(1)});
    }
    case intrinsic::kArrTypeLanes: {
      return builder_->CreateInBoundsGEP(
          buf, {index, ConstInt32(3), ConstInt32(2)});
    }
    case intrinsic::kArrByteOffset: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(6)});
    }
    case intrinsic::kArrDeviceId: {
      return builder_->CreateInBoundsGEP(
          buf, {index, ConstInt32(1), ConstInt32(1)});
    }
    case intrinsic::kArrDeviceType: {
      return builder_->CreateInBoundsGEP(
          buf, {index, ConstInt32(1), ConstInt32(0)});
    }
    case intrinsic::kTVMValueContent: {
      CHECK_EQ(t.lanes(), 1);
      CHECK(t.is_handle() || t.bits() == 64);
      if (t.is_int()) {
        buf = builder_->CreatePointerCast(buf, t_int64_->getPointerTo());
        return builder_->CreateInBoundsGEP(buf, index);
      } else if (t.is_float()) {
        buf = builder_->CreatePointerCast(buf, t_float64_->getPointerTo());
        return builder_->CreateInBoundsGEP(buf, index);
      } else {
        CHECK(t.is_handle());
        buf = builder_->CreatePointerCast(buf, t_tvm_value_->getPointerTo());
        buf = builder_->CreateInBoundsGEP(buf, index);
        return builder_->CreatePointerCast(buf, t_void_p_->getPointerTo());
      }
    }
    default: LOG(FATAL) << "unknown field code"; return nullptr;
  }
}

llvm::Value* CodeGenCPU::CreateCallExtern(const CallNode* op) {
  std::vector<llvm::Value*> arg_values(op->args.size());
  for (size_t i = 0; i < op->args.size(); ++i) {
    arg_values[i] = MakeValue(op->args[i]);
  }
  std::vector<llvm::Type*> arg_types;
  for (llvm::Value* v : arg_values) {
    arg_types.push_back(v->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      GetLLVMType(GetRef<PrimExpr>(op)), arg_types, false);
  // Check if it is available in global function table as injected function.
  auto it = gv_func_map_.find(op->name);
  if (it != gv_func_map_.end()) {
    if (it->second == nullptr) {
      gv_func_map_[op->name] = InitContextPtr(ftype->getPointerTo(), "__" + op->name);
      it = gv_func_map_.find(op->name);
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(ftype, GetContextPtr(it->second));
#else
    auto ext_callee = GetContextPtr(it->second);
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  } else {
    llvm::Function* f = module_->getFunction(op->name);
    if (f == nullptr) {
      f = llvm::Function::Create(
          ftype, llvm::Function::ExternalLinkage, op->name, module_.get());
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(f);
#else
    auto ext_callee = f;
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  }
}

llvm::GlobalVariable* CodeGenCPU::InitContextPtr(
    llvm::Type* p_type, std::string name) {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, p_type, false,
      llvm::GlobalValue::LinkOnceAnyLinkage, 0,
      name);
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(p_type)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(p_type));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(p_type));
  gv->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  return gv;
}

llvm::Value* CodeGenCPU::GetContextPtr(llvm::GlobalVariable* gv) {
  CHECK(gv != nullptr);
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, llvm::Align(gv->getAlignment()));
#else
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, gv->getAlignment());
#endif
  faddr->setMetadata(
      "tbaa",
      md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
  return faddr;
}

void CodeGenCPU::InitGlobalContext(bool dynamic_lookup) {
  // Module context
  gv_mod_ctx_ = InitContextPtr(t_void_p_, tvm::runtime::symbol::tvm_module_ctx);
  // Register back the locations.
  if (f_tvm_register_system_symbol_ != nullptr) {
    export_system_symbols_.emplace_back(
        std::make_pair(tvm::runtime::symbol::tvm_module_ctx, gv_mod_ctx_));
  } else {
    if (!dynamic_lookup) {
      gv_tvm_func_call_ = InitContextPtr(
          ftype_tvm_func_call_->getPointerTo(), "__TVMFuncCall");
      gv_tvm_get_func_from_env_ = InitContextPtr(
          ftype_tvm_get_func_from_env_->getPointerTo(), "__TVMBackendGetFuncFromEnv");
      gv_tvm_api_set_last_error_ = InitContextPtr(
          ftype_tvm_api_set_last_error_->getPointerTo(), "__TVMAPISetLastError");
      gv_tvm_parallel_launch_ = InitContextPtr(
          ftype_tvm_parallel_launch_->getPointerTo(), "__TVMBackendParallelLaunch");
      gv_tvm_parallel_barrier_ = InitContextPtr(
          ftype_tvm_parallel_barrier_->getPointerTo(), "__TVMBackendParallelBarrier");
      // Mark as context functions
      gv_func_map_["TVMBackendAllocWorkspace"] = nullptr;
      gv_func_map_["TVMBackendFreeWorkspace"] = nullptr;
    }
  }
}

llvm::BasicBlock* CodeGenCPU::CheckCallSuccess(llvm::Value* retcode) {
  // create emit codes that checks and load the function.
  using llvm::BasicBlock;
  BasicBlock* fail_block = BasicBlock::Create(
      *ctx_, "call_fail", function_);
  BasicBlock* end_block = BasicBlock::Create(
      *ctx_, "call_end", function_);
  llvm::Value* succ = builder_->CreateICmpEQ(
      retcode, llvm::ConstantInt::get(t_int_, 0));
  builder_->CreateCondBr(succ, end_block, fail_block, md_very_likely_branch_);
  builder_->SetInsertPoint(fail_block);
  // return the code.
  builder_->CreateRet(retcode);
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  return end_block;
}

void CodeGenCPU::CreateComputeScope(const AttrStmtNode* op) {
  // There are two reasons why we create another function for compute_scope
  // - Make sure the generated compute function is clearly separately(though it can get inlined)
  // - Set noalias on all the pointer arguments, some of them are loaded from TVMArgs.
  //   This is easier than set the alias scope manually.
  using llvm::BasicBlock;
  Array<Var> vargs = tir::UndefinedVars(op->body, {});
  std::vector<llvm::Value*> arg_values;
  std::vector<llvm::Type*> arg_types;
  for (Var v : vargs) {
    llvm::Value* value = MakeValue(v);
    arg_values.push_back(value);
    arg_types.push_back(value->getType());
  }
  llvm::FunctionType* ftype =
      llvm::FunctionType::get(t_int_, arg_types, false);
  llvm::Function* fcompute =
      llvm::Function::Create(ftype,
                             llvm::Function::PrivateLinkage,
                             op->value.as<StringImmNode>()->value,
                             module_.get());
  BasicBlock* compute_call_end = CheckCallSuccess(
      builder_->CreateCall(fcompute, arg_values));
  // setup compute fuinction.
  std::unordered_map<const VarNode*, llvm::Value*> new_vmap;
  size_t idx = 0;
  for (auto it = fcompute->arg_begin();
       it != fcompute->arg_end(); ++it, ++idx) {
    llvm::Argument* v = &(*it);
    const Var& var = vargs[idx];
    new_vmap[var.get()] = v;
    if (var.dtype().is_handle() && !alias_var_set_.count(var.get())) {
      // set non alias.
#if TVM_LLVM_VERSION >= 50
      fcompute->addParamAttr(idx, llvm::Attribute::NoAlias);
      // always not inline compute function to make the code structure clean
#else
      fcompute->setDoesNotAlias(idx + 1);
#endif
      fcompute->addFnAttr(llvm::Attribute::NoInline);
    }
  }
  std::swap(function_, fcompute);
  std::swap(new_vmap, var_map_);
  BasicBlock *compute_entry = BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(compute_entry);
  this->VisitStmt(op->body);
  builder_->CreateRet(ConstInt32(0));
  // swap the var map back, now we are back on track.
  std::swap(new_vmap, var_map_);
  std::swap(function_, fcompute);
  builder_->SetInsertPoint(compute_call_end);
}

llvm::Value* CodeGenCPU::PackClosureData(const Array<Var>& vfields, uint64_t* num_bytes) {
  if (vfields.size() == 0) {
    *num_bytes = 0U;
    return llvm::Constant::getNullValue(t_void_p_);
  }
  std::vector<llvm::Type*> fields;
  for (Var v : vfields) {
    auto it = var_map_.find(v.get());
    CHECK(it != var_map_.end());
    fields.push_back(it->second->getType());
  }
  llvm::StructType* tcdata = llvm::StructType::create(fields);
  llvm::Value* cdata = builder_->CreateAlloca(tcdata, ConstInt32(1));
  llvm::Value* zero = ConstInt32(0);
  for (size_t i = 0; i < vfields.size(); ++i) {
    builder_->CreateStore(
        var_map_.at(vfields[i].get()),
        builder_->CreateInBoundsGEP(cdata, {zero, ConstInt32(i)}));
  }
  *num_bytes = data_layout_->getTypeAllocSize(
      llvm::cast<llvm::PointerType>(cdata->getType())->getElementType());
  return cdata;
}

void CodeGenCPU::UnpackClosureData(llvm::Value* cdata,
                                   const Array<Var>& vfields,
                                   std::unordered_map<const VarNode*, llvm::Value*>* vmap) {
  for (size_t i = 0; i < vfields.size(); ++i) {
    (*vmap)[vfields[i].get()] =
        builder_->CreateLoad(builder_->CreateInBoundsGEP(
            cdata, {ConstInt32(0), ConstInt32(i)}));
  }
}

void CodeGenCPU::CreateParallelLaunch(const Stmt& body, int num_task) {
  using llvm::BasicBlock;
  // closure data
  llvm::Function* f = llvm::Function::Create(
      ftype_tvm_parallel_lambda_,
      llvm::Function::PrivateLinkage,
      "__tvm_parallel_lambda", module_.get());
  // allocate and setup the closure, call the closure.
  Array<Var> vfields = tir::UndefinedVars(body, {});
  uint64_t nbytes;
  llvm::Value* cdata = PackClosureData(vfields, &nbytes);
#if TVM_LLVM_VERSION >= 90
  auto launch_callee = llvm::FunctionCallee(
      ftype_tvm_parallel_launch_, RuntimeTVMParallelLaunch());
#else
  auto launch_callee = RuntimeTVMParallelLaunch();
#endif
  BasicBlock* par_launch_end = CheckCallSuccess(
      builder_->CreateCall(
          launch_callee,
          {f, builder_->CreatePointerCast(cdata, t_void_p_), ConstInt32(num_task)}));
  // Setup the closure function.
  BasicBlock *lambda_entry = BasicBlock::Create(*ctx_, "entry", f);
  builder_->SetInsertPoint(lambda_entry);
  auto it = f->arg_begin();
  llvm::Value* task_id = &(*it++);
  llvm::Value* penv = &(*it++);
  cdata = builder_->CreatePointerCast(&(*it++), cdata->getType());
  // setup new variable map, swap it with current var context.
  std::unordered_map<const VarNode*, llvm::Value*> new_vmap;
  UnpackClosureData(cdata, vfields, &new_vmap);
  // setup parallel env
  ParallelEnv par_env;
  par_env.task_id = Var("task_id", DataType::Int(32));
  par_env.num_task = Var("num_task", DataType::Int(32));
  new_vmap[par_env.task_id.get()] = task_id;
  new_vmap[par_env.num_task.get()] = builder_->CreateLoad(
      builder_->CreateInBoundsGEP(
          penv, {ConstInt32(0), ConstInt32(1)}));
  par_env.penv = penv;
  std::swap(function_, f);
  std::swap(parallel_env_, par_env);
  std::swap(var_map_, new_vmap);
  this->VisitStmt(body);
  builder_->CreateRet(ConstInt32(0));
  // swap the var map back, now we are back on track.
  std::swap(var_map_, new_vmap);
  std::swap(parallel_env_, par_env);
  std::swap(function_, f);
  CHECK_NE(par_env.parallel_loop_count, 0)
      << "Cannot find parallel loop within parallel launch";
  builder_->SetInsertPoint(par_launch_end);
}

llvm::Value* CodeGenCPU::CreateStaticHandle() {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, t_void_p_, false,
      llvm::GlobalValue::PrivateLinkage, 0,
      "__tvm_static_handle");
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(t_void_p_)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(t_void_p_));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(t_void_p_));
  return gv;
}

void CodeGenCPU::CreateStaticInit(const std::string& init_fname, const Stmt& body) {
  using llvm::BasicBlock;
  // closure data
  llvm::Function* f = llvm::Function::Create(
      ftype_tvm_static_init_callback_,
      llvm::Function::PrivateLinkage,
      "__tvm_static_init_lambda", module_.get());
  llvm::Value* gv = CreateStaticHandle();
  llvm::Function* finit = module_->getFunction(init_fname);
  if (finit == nullptr) {
    finit = llvm::Function::Create(
        ftype_tvm_static_init_, llvm::Function::ExternalLinkage, init_fname, module_.get());
  }
  // allocate and setup the closure, call the closure.
  uint64_t nbytes;
  Array<Var> vfields = tir::UndefinedVars(body, {});
  llvm::Value* cdata = PackClosureData(vfields, &nbytes);
  BasicBlock* init_end = CheckCallSuccess(
      builder_->CreateCall(
          finit,
          {gv, f, builder_->CreatePointerCast(cdata, t_void_p_), ConstInt32(nbytes)}));
  // Setup the closure function.
  BasicBlock *lambda_entry = BasicBlock::Create(*ctx_, "entry", f);
  builder_->SetInsertPoint(lambda_entry);
  auto it = f->arg_begin();
  cdata = builder_->CreatePointerCast(&(*it++), cdata->getType());
  // setup new variable map, swap it with current var context.
  std::unordered_map<const VarNode*, llvm::Value*> new_vmap;
  UnpackClosureData(cdata, vfields, &new_vmap);
  CHECK(parallel_env_.penv == nullptr);
  std::swap(function_, f);
  std::swap(var_map_, new_vmap);
  this->VisitStmt(body);
  builder_->CreateRet(ConstInt32(0));
  // swap the var map back, now we are back on track.
  std::swap(var_map_, new_vmap);
  std::swap(function_, f);
  builder_->SetInsertPoint(init_end);
}

llvm::Value* CodeGenCPU::GetPackedFuncHandle(const std::string& fname) {
  using llvm::BasicBlock;
  // We will store the packed function handle in global space.
  // Initialize it during the first call.
  llvm::DataLayout layout(module_.get());
  uint64_t align = layout.getTypeAllocSize(t_tvm_func_handle_);
  auto it = func_handle_map_.find(fname);

  llvm::GlobalVariable* hptr;
  if (it == func_handle_map_.end()) {
    // create global location for the handle
    // create the function handle
    hptr = new llvm::GlobalVariable(
        *module_, t_tvm_func_handle_, false,
        llvm::GlobalValue::InternalLinkage, nullptr, ".tvm_func." + fname);
#if TVM_LLVM_VERSION >= 100
    hptr->setAlignment(llvm::Align(align));
#else
    hptr->setAlignment(align);
#endif
    hptr->setInitializer(llvm::Constant::getNullValue(t_tvm_func_handle_));
    func_handle_map_[fname] = hptr;
  } else {
    hptr = it->second;
  }
  // create emit codes that checks and load the function.
  BasicBlock* pre_block = builder_->GetInsertBlock();
  BasicBlock* init_block = BasicBlock::Create(
      *ctx_, "handle_init", function_);
  BasicBlock* end_block = BasicBlock::Create(
      *ctx_, "handle_init_end", function_);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, llvm::Align(align));
#else
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, align);
#endif
  llvm::Value* handle_not_null =  builder_->CreateICmpNE(
      handle, llvm::Constant::getNullValue(t_tvm_func_handle_));
  builder_->CreateCondBr(
      handle_not_null, end_block, init_block, md_very_likely_branch_);
  // Initialize the handle if needed.
  builder_->SetInsertPoint(init_block);
  llvm::Value* out = WithFunctionEntry([&]() {
      return builder_->CreateAlloca(t_tvm_func_handle_);
    });
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* ctx = builder_->CreateAlignedLoad(
      gv_mod_ctx_, llvm::Align(gv_mod_ctx_->getAlignment()));
#else
  llvm::LoadInst* ctx = builder_->CreateAlignedLoad(
      gv_mod_ctx_, gv_mod_ctx_->getAlignment());
#endif
  ctx->setMetadata(
      "tbaa",
      md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
#if TVM_LLVM_VERSION >= 90
  auto env_callee = llvm::FunctionCallee(
      ftype_tvm_get_func_from_env_, RuntimeTVMGetFuncFromEnv());
#else
  auto env_callee = RuntimeTVMGetFuncFromEnv();
#endif
  llvm::Value* retcode = builder_->CreateCall(
      env_callee, {ctx, GetConstString(fname), out});
  init_block = CheckCallSuccess(retcode);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(out, llvm::Align(align));
#else
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(out, align);
#endif
  // Store the handle
  builder_->CreateStore(loaded_handle, hptr);
  builder_->CreateBr(end_block);
  // end block
  builder_->SetInsertPoint(end_block);
  llvm::PHINode* phi = builder_->CreatePHI(t_tvm_func_handle_, 2);
  phi->addIncoming(handle, pre_block);
  phi->addIncoming(loaded_handle, init_block);
  return phi;
}

llvm::BasicBlock *
CodeGenCPU::MakeCallPacked(const Array<PrimExpr> &args, llvm::Value **rvalue,
                           llvm::Value **ret_tcode, const DataType &r_type,
                           const int64_t begin, const int64_t end) {
  using llvm::BasicBlock;
  std::string func_name = args[0].as<StringImmNode>()->value;
  llvm::Value *handle = GetPackedFuncHandle(func_name);
  // call the function
  int64_t nargs = end - begin;
  CHECK_GE(nargs, 0);
  llvm::Value *stack_value = MakeValue(args[1]);
  llvm::Value *stack_tcode = MakeValue(args[2]);
  llvm::Value *arg_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(stack_value, t_tvm_value_->getPointerTo()),
      ConstInt32(begin));
  llvm::Value *arg_tcode =
      CreateBufferPtr(DataType::Int(32), stack_tcode, ConstInt32(begin));
  llvm::Value *ret_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(stack_value, t_tvm_value_->getPointerTo()),
      ConstInt32(end));
  *ret_tcode = CreateBufferPtr(DataType::Int(32), stack_tcode, ConstInt32(end));
#if TVM_LLVM_VERSION >= 90
  auto call_callee = llvm::FunctionCallee(ftype_tvm_func_call_, RuntimeTVMFuncCall());
#else
  auto call_callee = RuntimeTVMFuncCall();
#endif
  BasicBlock *end_block = CheckCallSuccess(builder_->CreateCall(
      call_callee, {handle, arg_value, arg_tcode, ConstInt32(nargs),
                    ret_value, *ret_tcode}));
  DataType r_api_type = tir::APIType(r_type);
  llvm::Value* load_ptr = builder_->CreatePointerCast(
      ret_value, DTypeToLLVMType(r_api_type)->getPointerTo());
#if TVM_LLVM_VERSION >= 110
  *rvalue = builder_->CreateAlignedLoad(load_ptr, llvm::Align(8));
#else
  *rvalue = builder_->CreateAlignedLoad(load_ptr, 8);
#endif
  *rvalue = CreateCast(r_api_type, r_type, *rvalue);
  return end_block;
}

llvm::Value *CodeGenCPU::CreateCallPacked(const CallNode *op) {
  CHECK_EQ(op->args.size(), 5U);
  llvm::Value *rvalue = nullptr;
  llvm::Value *ret_tcode = nullptr;
  MakeCallPacked(op->args, &rvalue, &ret_tcode, op->dtype,
                 op->args[3].as<IntImmNode>()->value,
                 op->args[4].as<IntImmNode>()->value);
  return rvalue;
}

llvm::Value *CodeGenCPU::CreateCallTracePacked(const CallNode *op) {
  using llvm::BasicBlock;
  CHECK_EQ(op->args.size(), 6U);
  llvm::Value *rvalue = nullptr;
  llvm::Value *ret_tcode = nullptr;
  BasicBlock *end_block = MakeCallPacked(
      op->args, &rvalue, &ret_tcode, op->dtype, op->args[3].as<IntImmNode>()->value,
      op->args[4].as<IntImmNode>()->value);
  // Get traced value.
  llvm::Value *traced_value = MakeValue(op->args[5]);
  // The update_block handles case when we need to update the return value.
  BasicBlock *update_block =
      BasicBlock::Create(*ctx_, "update_block", function_);
  // The continue_block handles case when we need to return original
  // traced value.
  BasicBlock *continue_block =
      BasicBlock::Create(*ctx_, "continue_block", function_);
#if TVM_LLVM_VERSION >= 110
  llvm::Value *ret_tcode_value = builder_->CreateAlignedLoad(ret_tcode, llvm::Align(8));
#else
  llvm::Value *ret_tcode_value = builder_->CreateAlignedLoad(ret_tcode, 8);
#endif
  // Check the ret_type_code and create cmp instruction.
  llvm::Value *cmp = builder_->CreateICmpNE(
      ret_tcode_value, llvm::ConstantInt::get(t_int_, kTVMNullptr));
  builder_->CreateCondBr(cmp, update_block, continue_block);
  builder_->SetInsertPoint(update_block);
  builder_->CreateBr(continue_block);
  builder_->SetInsertPoint(continue_block);
  // The return value depends on from what bb we come from.
  llvm::PHINode *phi_rvalue = builder_->CreatePHI(traced_value->getType(), 2);
  phi_rvalue->addIncoming(rvalue, update_block);
  phi_rvalue->addIncoming(traced_value, end_block);
  return phi_rvalue;
}

llvm::Value* CodeGenCPU::RuntimeTVMFuncCall() {
  if (f_tvm_func_call_ != nullptr) return f_tvm_func_call_;
  return GetContextPtr(gv_tvm_func_call_);
}

llvm::Value* CodeGenCPU::RuntimeTVMGetFuncFromEnv() {
  if (f_tvm_get_func_from_env_ != nullptr) return f_tvm_get_func_from_env_;
  return GetContextPtr(gv_tvm_get_func_from_env_);
}
llvm::Value* CodeGenCPU::RuntimeTVMAPISetLastError() {
  if (f_tvm_api_set_last_error_ != nullptr) return f_tvm_api_set_last_error_;
  return GetContextPtr(gv_tvm_api_set_last_error_);
}
llvm::Value* CodeGenCPU::RuntimeTVMParallelLaunch() {
  if (f_tvm_parallel_launch_ != nullptr) return f_tvm_parallel_launch_;
  return GetContextPtr(gv_tvm_parallel_launch_);
}

llvm::Value* CodeGenCPU::RuntimeTVMParallelBarrier() {
  if (f_tvm_parallel_barrier_ != nullptr) return f_tvm_parallel_barrier_;
  return GetContextPtr(gv_tvm_parallel_barrier_);
}

void CodeGenCPU::AddStartupFunction() {
  if (export_system_symbols_.size() != 0) {
    llvm::FunctionType* ftype = llvm::FunctionType::get(t_void_, {}, false);
    function_ = llvm::Function::Create(
        ftype,
        llvm::Function::InternalLinkage,
        "__tvm_module_startup", module_.get());
    llvm::BasicBlock* startup_entry = llvm::BasicBlock::Create(*ctx_, "entry", function_);
    builder_->SetInsertPoint(startup_entry);
    for (const auto& kv : export_system_symbols_) {
      llvm::Value* name = GetConstString(kv.first);
      builder_->CreateCall(
          f_tvm_register_system_symbol_, {
            name, builder_->CreateBitCast(kv.second, t_void_p_)});
    }
    llvm::appendToGlobalCtors(*module_, function_, 65535);
    builder_->CreateRet(nullptr);
  }
}

llvm::Value* CodeGenCPU::CreateIntrinsic(const CallNode* op) {
  if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    return CreateCallPacked(op);
  } else if (op->is_intrinsic(intrinsic::tvm_call_trace_packed_lowered)) {
    return CreateCallTracePacked(op);
  } else if (op->is_intrinsic(intrinsic::tvm_static_handle)) {
    return CreateStaticHandle();
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    builder_->CreateRet(ConstInt32(-1));
    return ConstInt32(-1);
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImmNode>()->value;
    llvm::Value* ref = this->CreateStructRefPtr(
        op->dtype, MakeValue(op->args[0]),
        MakeValue(op->args[1]), kind);
    if (kind == intrinsic::kArrAddr) {
      return builder_->CreatePointerCast(ref, t_void_p_);
    } else {
      return builder_->CreateLoad(ref);
    }
  } else if (op->is_intrinsic(intrinsic::tvm_struct_set)) {
    CHECK_EQ(op->args.size(), 4U);
    int kind = op->args[2].as<IntImmNode>()->value;
    llvm::Value* value = MakeValue(op->args[3]);
    llvm::Value* ref = this->CreateStructRefPtr(
        op->args[3].dtype(), MakeValue(op->args[0]),
        MakeValue(op->args[1]), kind);
    CHECK(kind != intrinsic::kArrAddr);
    if (value->getType()->isPointerTy()) {
      value = builder_->CreatePointerCast(
          value, ref->getType()->getPointerElementType());
    }
    builder_->CreateStore(value, ref);
    return ConstInt32(0);
  } else if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
    CHECK_EQ(op->args.size(), 2U);
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    return WithFunctionEntry([&]() -> llvm::AllocaInst* {
        const int64_t* pval = as_const_int(op->args[1]);
        CHECK(pval) << "require stack alloca to contain constant value";
        llvm::Value* num = ConstInt32(pval[0]);
        if (type == "shape") {
          return builder_->CreateAlloca(t_tvm_shape_index_, num);
        } else if (type == "arg_value") {
          return builder_->CreateAlloca(t_tvm_value_, num);
        } else if (type == "arg_tcode") {
          return builder_->CreateAlloca(t_int_, num);
        } else if (type == "array") {
          return builder_->CreateAlloca(t_tvm_array_, num);
        } else {
          LOG(FATAL) << "Unknown stack alloca type " << type;
          return nullptr;
        }
      });
  } else {
    return CodeGenLLVM::CreateIntrinsic(op);
  }
}

void CodeGenCPU::VisitStmt_(const AssertStmtNode* op) {
  using llvm::BasicBlock;
  llvm::Value* cond = MakeValue(op->condition);
  std::ostringstream os;
  os << "Assert fail: " << op->condition;
  if (op->message.as<StringImmNode>()) {
    os << ", " << op->message.as<StringImmNode>()->value;
  }
  llvm::Value* msg = GetConstString(os.str());
  BasicBlock* fail_block = BasicBlock::Create(
      *ctx_, "assert_fail", function_);
  BasicBlock* end_block = BasicBlock::Create(
      *ctx_, "assert_end", function_);
  builder_->CreateCondBr(cond, end_block, fail_block, md_very_likely_branch_);
  // fail condition.
  builder_->SetInsertPoint(fail_block);
#if TVM_LLVM_VERSION >= 90
  auto err_callee = llvm::FunctionCallee(
      ftype_tvm_api_set_last_error_, RuntimeTVMAPISetLastError());
#else
  auto err_callee = RuntimeTVMAPISetLastError();
#endif
  builder_->CreateCall(err_callee, {msg});
  builder_->CreateRet(ConstInt32(-1));
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  CodeGenLLVM::VisitStmt_(op);
}

void CodeGenCPU::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::coproc_uop_scope) {
    this->CreateStaticInit(op->value.as<StringImmNode>()->value, op->body);
  } else  if (op->attr_key == tir::attr::compute_scope) {
    this->CreateComputeScope(op);
  } else if (tir::attr::IsPragmaKey(op->attr_key)) {
    if (op->attr_key == "pragma_parallel_stride_pattern") {
      CHECK(parallel_env_.penv != nullptr)
          << "Pragma parallel_stride_pattern only valid in parallel launch";
      parallel_env_.stride_pattern = true;
      this->VisitStmt(op->body);
    } else if (op->attr_key == "pragma_parallel_launch_point") {
      CreateParallelLaunch(op->body, 0);
    } else if (op->attr_key == "pragma_parallel_barrier_when_finish") {
      CHECK(parallel_env_.penv != nullptr)
          << "Cannot run barrier without parallel environment";
      CHECK(!parallel_env_.in_parallel_loop)
          << "Cannot not place within parallel loop as the workload may differ, "
          << " place it between parallel and parallel_launch_point";
      this->VisitStmt(op->body);
#if TVM_LLVM_VERSION >= 90
      auto bar_callee = llvm::FunctionCallee(
          ftype_tvm_parallel_barrier_, RuntimeTVMParallelBarrier());
#else
      auto bar_callee = RuntimeTVMParallelBarrier();
#endif
      builder_->CreateCall(
          bar_callee, {MakeValue(parallel_env_.task_id),  parallel_env_.penv});
    } else if (op->attr_key == tir::attr::pragma_import_llvm) {
      const StringImmNode* value = op->value.as<StringImmNode>();
      CHECK(value != nullptr);
      this->HandleImport(value->value);
      this->VisitStmt(op->body);
    } else {
      LOG(WARNING) << "Unknown pragma " << op->attr_key;
      this->VisitStmt(op->body);
    }
  } else {
    CodeGenLLVM::VisitStmt_(op);
  }
}

void CodeGenCPU::VisitStmt_(const ForNode* op) {
  CHECK(is_zero(op->min));
  if (op->for_type == ForType::Serial ||
      op->for_type == ForType::Unrolled) {
    CodeGenLLVM::VisitStmt_(op);
  } else if (op->for_type == ForType::Parallel) {
    if (parallel_env_.penv == nullptr) {
      CreateParallelLaunch(
          ForNode::make(
              op->loop_var, op->min, op->extent,
              op->for_type, op->device_api, op->body), 0);
    } else {
      // already in parallel env.
      CHECK(parallel_env_.task_id.defined());
      CHECK(parallel_env_.num_task.defined());
      CHECK(parallel_env_.penv != nullptr);
      DataType t = op->extent.dtype();
      PrimExpr num_task = cast(t, parallel_env_.num_task);
      PrimExpr task_id = cast(t, parallel_env_.task_id);
      CHECK(!parallel_env_.in_parallel_loop)
          << "Nested parallel loop is not supported by threadpool, try fuse them instead";
      parallel_env_.in_parallel_loop = true;
      if (parallel_env_.stride_pattern) {
        CreateSerialFor(MakeValue(task_id),
                        MakeValue(op->extent),
                        MakeValue(num_task),
                        op->loop_var,
                        op->body);
      } else {
        PrimExpr step = (op->extent + num_task - make_const(t, 1)) / num_task;
        PrimExpr begin = MinNode::make(task_id * step, op->extent);
        PrimExpr end = MinNode::make((task_id + make_const(t, 1)) * step, op->extent);
        CreateSerialFor(MakeValue(begin),
                        MakeValue(end),
                        llvm::ConstantInt::getSigned(GetLLVMType(end), 1),
                        op->loop_var,
                        op->body);
      }
      parallel_env_.in_parallel_loop = false;
      ++parallel_env_.parallel_loop_count;
    }
  } else {
    LOG(FATAL) << "cannot handle for type " << op->for_type;
  }
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
