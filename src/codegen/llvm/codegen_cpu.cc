/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cpu.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/ir_pass.h>
#include "codegen_cpu.h"
#include "../../pass/ir_util.h"

namespace tvm {
namespace codegen {

void CodeGenCPU::Init(const std::string& module_name,
                          llvm::TargetMachine* tm,
                          llvm::LLVMContext* ctx,
                          bool system_lib,
                          bool dynamic_lookup) {
  CodeGenLLVM::Init(module_name, tm, ctx, system_lib, dynamic_lookup);
  static_assert(sizeof(TVMValue) == sizeof(double), "invariant");
  func_handle_map_.clear();
  export_system_symbols_.clear();
  // TVM runtime types
  t_tvm_shape_index_ = llvm::Type::getIntNTy(*ctx, TVMShapeIndexType().bits());
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

void CodeGenCPU::AddFunction(const LoweredFunc& f) {
  CodeGenLLVM::AddFunction(f);
  if (f_tvm_register_system_symbol_ != nullptr) {
    export_system_symbols_.emplace_back(
        std::make_pair(f->name, builder_->CreatePointerCast(function_, t_void_p_)));
  }
}

void CodeGenCPU::AddMainFunction(const std::string& entry_func_name) {
  llvm::Function* f = module_->getFunction(entry_func_name);
  CHECK(f) << "Function " << entry_func_name << "does not in module";
  llvm::Type* type = llvm::ArrayType::get(t_char_, entry_func_name.length() + 1);
  llvm::GlobalVariable *global = new llvm::GlobalVariable(
      *module_, type, true, llvm::GlobalValue::WeakAnyLinkage, 0,
      runtime::symbol::tvm_module_main);
  global->setAlignment(1);
  global->setInitializer(llvm::ConstantDataArray::getString(*ctx_, entry_func_name));
}

llvm::Value* CodeGenCPU::CreateStructRefPtr(
    Type t, llvm::Value* buf, llvm::Value* index, int kind) {
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

llvm::Value* CodeGenCPU::CreateCallExtern(const Call* op) {
  std::vector<llvm::Value*> arg_values(op->args.size());
  for (size_t i = 0; i < op->args.size(); ++i) {
    arg_values[i] = MakeValue(op->args[i]);
  }
  std::vector<llvm::Type*> arg_types;
  for (llvm::Value* v : arg_values) {
    arg_types.push_back(v->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(
      LLVMType(op->type), arg_types, false);
  // Check if it is available in global function table as injected function.
  auto it = gv_func_map_.find(op->name);
  if (it != gv_func_map_.end()) {
    if (it->second == nullptr) {
      gv_func_map_[op->name] = InitContextPtr(ftype->getPointerTo(), "__" + op->name);
      it = gv_func_map_.find(op->name);
    }
    return builder_->CreateCall(GetContextPtr(it->second), arg_values);
  } else {
    llvm::Function* f = module_->getFunction(op->name);
    if (f == nullptr) {
      f = llvm::Function::Create(
          ftype, llvm::Function::ExternalLinkage, op->name, module_.get());
    }
    return builder_->CreateCall(f, arg_values);
  }
}

llvm::GlobalVariable* CodeGenCPU::InitContextPtr(
    llvm::Type* p_type, std::string name) {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, p_type, false,
      llvm::GlobalValue::LinkOnceAnyLinkage, 0,
      name);
  gv->setAlignment(data_layout_->getTypeAllocSize(p_type));
  gv->setInitializer(llvm::Constant::getNullValue(p_type));
  gv->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  return gv;
}

llvm::Value* CodeGenCPU::GetContextPtr(llvm::GlobalVariable* gv) {
  CHECK(gv != nullptr);
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, gv->getAlignment());
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

void CodeGenCPU::CreateComputeScope(const AttrStmt* op) {
  // There are two reasons why we create another function for compute_scope
  // - Make sure the generated compute function is clearly separately(though it can get inlined)
  // - Set noalias on all the pointer arguments, some of them are loaded from TVMArgs.
  //   This is easier than set the alias scope manually.
  using llvm::BasicBlock;
  Array<Var> vargs = ir::UndefinedVars(op->body, {});
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
                             op->value.as<StringImm>()->value,
                             module_.get());
  BasicBlock* compute_call_end = CheckCallSuccess(
      builder_->CreateCall(fcompute, arg_values));
  // setup compute fuinction.
  std::unordered_map<const Variable*, llvm::Value*> new_vmap;
  size_t idx = 0;
  for (auto it = fcompute->arg_begin();
       it != fcompute->arg_end(); ++it, ++idx) {
    llvm::Argument* v = &(*it);
    const Var& var = vargs[idx];
    new_vmap[var.get()] = v;
    if (var.type().is_handle() && !alias_var_set_.count(var.get())) {
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
                                   std::unordered_map<const Variable*, llvm::Value*>* vmap) {
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
  Array<Var> vfields = ir::UndefinedVars(body, {});
  uint64_t nbytes;
  llvm::Value* cdata = PackClosureData(vfields, &nbytes);
  BasicBlock* par_launch_end = CheckCallSuccess(
      builder_->CreateCall(
          RuntimeTVMParallelLaunch(),
          {f, builder_->CreatePointerCast(cdata, t_void_p_), ConstInt32(num_task)}));
  // Setup the closure function.
  BasicBlock *lambda_entry = BasicBlock::Create(*ctx_, "entry", f);
  builder_->SetInsertPoint(lambda_entry);
  auto it = f->arg_begin();
  llvm::Value* task_id = &(*it++);
  llvm::Value* penv = &(*it++);
  cdata = builder_->CreatePointerCast(&(*it++), cdata->getType());
  // setup new variable map, swap it with current var context.
  std::unordered_map<const Variable*, llvm::Value*> new_vmap;
  UnpackClosureData(cdata, vfields, &new_vmap);
  // setup parallel env
  ParallelEnv par_env;
  par_env.task_id = Var("task_id", Int(32));
  par_env.num_task = Var("num_task", Int(32));
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
  gv->setAlignment(data_layout_->getTypeAllocSize(t_void_p_));
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
  Array<Var> vfields = ir::UndefinedVars(body, {});
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
  std::unordered_map<const Variable*, llvm::Value*> new_vmap;
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
    hptr->setAlignment(align);
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
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, align);
  llvm::Value* handle_not_null =  builder_->CreateICmpNE(
      handle, llvm::Constant::getNullValue(t_tvm_func_handle_));
  builder_->CreateCondBr(
      handle_not_null, end_block, init_block, md_very_likely_branch_);
  // Initialize the handle if needed.
  builder_->SetInsertPoint(init_block);
  llvm::Value* out = builder_->CreateAlloca(t_tvm_func_handle_);
  llvm::LoadInst* ctx = builder_->CreateAlignedLoad(
      gv_mod_ctx_, gv_mod_ctx_->getAlignment());
  ctx->setMetadata(
      "tbaa",
      md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
  llvm::Value* retcode = builder_->CreateCall(
      RuntimeTVMGetFuncFromEnv(), {ctx, GetConstString(fname), out});
  init_block = CheckCallSuccess(retcode);
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(out, align);
  builder_->CreateBr(end_block);
  // end block
  builder_->SetInsertPoint(end_block);
  llvm::PHINode* phi = builder_->CreatePHI(t_tvm_func_handle_, 2);
  phi->addIncoming(handle, pre_block);
  phi->addIncoming(loaded_handle, init_block);
  return phi;
}

llvm::Value* CodeGenCPU::CreateCallPacked(const Call* op) {
  CHECK_EQ(op->args.size(), 5U);
  std::string func_name = op->args[0].as<StringImm>()->value;
  llvm::Value* handle = GetPackedFuncHandle(func_name);
  // call the function
  int64_t begin = op->args[3].as<IntImm>()->value;
  int64_t end = op->args[4].as<IntImm>()->value;
  int64_t nargs = end - begin;
  CHECK_GE(nargs, 0);
  llvm::Value* stack_value = MakeValue(op->args[1]);
  llvm::Value* stack_tcode = MakeValue(op->args[2]);
  llvm::Value* arg_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(
          stack_value, t_tvm_value_->getPointerTo()), ConstInt32(begin));
  llvm::Value* arg_tcode = CreateBufferPtr(
      Int(32), stack_tcode, ConstInt32(begin));
  llvm::Value* ret_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(
          stack_value, t_tvm_value_->getPointerTo()), ConstInt32(end));
  llvm::Value* ret_tcode = CreateBufferPtr(
      Int(32), stack_tcode, ConstInt32(end));
  CheckCallSuccess(
      builder_->CreateCall(
          RuntimeTVMFuncCall(),
          {handle, arg_value, arg_tcode, ConstInt32(nargs),
                ret_value, ret_tcode}));
  Type r_type = op->type;
  Type r_api_type = ir::APIType(r_type);
  llvm::Value* rvalue =
      builder_->CreateAlignedLoad(
          builder_->CreatePointerCast(
              ret_value, LLVMType(r_api_type)->getPointerTo()), 8);
  rvalue = CreateCast(r_api_type, r_type, rvalue);
  return rvalue;
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

llvm::Value* CodeGenCPU::CreateIntrinsic(const Call* op) {
  if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    return CreateCallPacked(op);
  } else if (op->is_intrinsic(intrinsic::tvm_static_handle)) {
    return CreateStaticHandle();
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    builder_->CreateRet(ConstInt32(-1));
    return ConstInt32(-1);
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImm>()->value;
    llvm::Value* ref = this->CreateStructRefPtr(
        op->type, MakeValue(op->args[0]),
        MakeValue(op->args[1]), kind);
    if (kind == intrinsic::kArrAddr) {
      return builder_->CreatePointerCast(ref, t_void_p_);
    } else {
      return builder_->CreateLoad(ref);
    }
  } else if (op->is_intrinsic(intrinsic::tvm_struct_set)) {
    CHECK_EQ(op->args.size(), 4U);
    int kind = op->args[2].as<IntImm>()->value;
    llvm::Value* value = MakeValue(op->args[3]);
    llvm::Value* ref = this->CreateStructRefPtr(
        op->args[3].type(), MakeValue(op->args[0]),
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
    const std::string& type = op->args[0].as<StringImm>()->value;
    llvm::Value* num = MakeValue(op->args[1]);
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
  } else {
    return CodeGenLLVM::CreateIntrinsic(op);
  }
}

void CodeGenCPU::VisitStmt_(const AssertStmt* op) {
  using llvm::BasicBlock;
  llvm::Value* cond = MakeValue(op->condition);
  std::ostringstream os;
  os << "Assert fail: " << op->condition;
  if (op->message.as<StringImm>()) {
    os << ", " << op->message.as<StringImm>()->value;
  }
  llvm::Value* msg = GetConstString(os.str());
  BasicBlock* fail_block = BasicBlock::Create(
      *ctx_, "assert_fail", function_);
  BasicBlock* end_block = BasicBlock::Create(
      *ctx_, "assert_end", function_);
  builder_->CreateCondBr(cond, end_block, fail_block, md_very_likely_branch_);
  // fail condition.
  builder_->SetInsertPoint(fail_block);
  builder_->CreateCall(RuntimeTVMAPISetLastError(), {msg});
  builder_->CreateRet(ConstInt32(-1));
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  CodeGenLLVM::VisitStmt_(op);
}

void CodeGenCPU::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::coproc_uop_scope) {
    this->CreateStaticInit(op->value.as<StringImm>()->value, op->body);
  } else  if (op->attr_key == ir::attr::compute_scope) {
    this->CreateComputeScope(op);
  } else if (attr::IsPragmaKey(op->attr_key)) {
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
      builder_->CreateCall(
          RuntimeTVMParallelBarrier(),
          {MakeValue(parallel_env_.task_id),  parallel_env_.penv});
    } else if (op->attr_key == ir::attr::pragma_import_llvm) {
      const StringImm* value = op->value.as<StringImm>();
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

void CodeGenCPU::VisitStmt_(const For* op) {
  CHECK(is_zero(op->min));
  if (op->for_type == ForType::Serial ||
      op->for_type == ForType::Unrolled) {
    CodeGenLLVM::VisitStmt_(op);
  } else if (op->for_type == ForType::Parallel) {
    if (parallel_env_.penv == nullptr) {
      CreateParallelLaunch(
          For::make(
              op->loop_var, op->min, op->extent,
              op->for_type, op->device_api, op->body), 0);
    } else {
      // already in parallel env.
      CHECK(parallel_env_.task_id.defined());
      CHECK(parallel_env_.num_task.defined());
      CHECK(parallel_env_.penv != nullptr);
      Type t = op->extent.type();
      Expr num_task = cast(t, parallel_env_.num_task);
      Expr task_id = cast(t, parallel_env_.task_id);
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
        Expr step = (op->extent + num_task - make_const(t, 1)) / num_task;
        Expr begin = Min::make(task_id * step, op->extent);
        Expr end = Min::make((task_id + make_const(t, 1)) * step, op->extent);
        CreateSerialFor(MakeValue(begin),
                        MakeValue(end),
                        ConstInt32(1),
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
