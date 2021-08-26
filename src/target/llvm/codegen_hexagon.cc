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

#if defined(TVM_LLVM_VERSION) && TVM_LLVM_VERSION >= 70

#include <llvm/Bitcode/BitcodeWriter.h>
#if TVM_LLVM_VERSION <= 90
#include <llvm/IR/Intrinsics.h>
#else
#include <llvm/IR/IntrinsicsHexagon.h>
#endif
#include <llvm/Support/CommandLine.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../runtime/hexagon/hexagon_module.h"
#include "../build_common.h"
#include "codegen_llvm.h"

namespace tvm {
namespace codegen {

static std::string get_name(const PrimFunc& f) {
  auto global_symbol = f->GetAttr<runtime::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
  return std::string(global_symbol.value());
}

// Hexagon code generation
class CodeGenHexagon final : public CodeGenLLVM {
 public:
  void InitTarget(llvm::TargetMachine* tm) final;
  void Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
            bool system_lib, bool dynamic_lookup, bool target_c_runtime) final;

  void VisitStmt_(const AssertStmtNode* op) override;

  llvm::Value* CreateIntrinsic(const CallNode* op) override;
  llvm::Value* CreateCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                bool skip_first_arg) override;
  llvm::Module* GetModulePtr() const { return module_.get(); }

 protected:
  // meta data
  llvm::MDNode* md_tbaa_ctx_ptr_{nullptr};
  llvm::FunctionType* ftype_tvm_func_call_{nullptr};
  llvm::FunctionType* ftype_tvm_get_func_from_env_{nullptr};
  llvm::FunctionType* ftype_tvm_api_set_last_error_{nullptr};

 private:
  llvm::Value* CreateStructRefPtr(DataType t, llvm::Value* buf, llvm::Value* index, int kind);

  // Check if the call to packed function is successful
  // if not directly finalize function and pass on return code.
  // return the end block after the check
  llvm::BasicBlock* CheckCallSuccess(llvm::Value* retcode);

  // Get runtime functions
  llvm::Value* RuntimeTVMFuncCall();
  llvm::Value* RuntimeTVMGetFuncFromEnv();
  llvm::Value* RuntimeTVMAPISetLastError();

  void InitGlobalContext(bool dynamic_lookup);
  llvm::GlobalVariable* InitContextPtr(llvm::Type* type, std::string name);
  llvm::Value* GetContextPtr(llvm::GlobalVariable* gv);
  std::vector<std::pair<std::string, llvm::Value*>> export_system_symbols_;
  llvm::Value* GetPackedFuncHandle(const std::string& str);

  // global to packed function handle
  std::unordered_map<std::string, llvm::GlobalVariable*> func_handle_map_;

  // Make packed call.
  llvm::BasicBlock* MakeCallPacked(const Array<PrimExpr>& args, llvm::Value** rvalue,
                                   llvm::Value** ret_tcode, const DataType& r_type,
                                   const int64_t begin, const int64_t end);
  // create call into tvm packed function.
  llvm::Value* CreateCallPacked(const CallNode* op);
  // Create trace call into tvm packed function.
  llvm::Value* CreateCallTracePacked(const CallNode* op);

  std::map<std::string, llvm::Type*> types_for_alloca_;

  // Type definitions.
  llvm::Type* t_tvm_func_handle_{nullptr};
  llvm::Type* t_tvm_value_{nullptr};
  llvm::Type* t_tvm_shape_index_{nullptr};
  llvm::Type* t_tvm_context_{nullptr};
  llvm::Type* t_tvm_type_{nullptr};
  llvm::Type* t_tvm_array_{nullptr};

  // Context for injection lookup
  llvm::GlobalVariable* gv_mod_ctx_{nullptr};
  llvm::GlobalVariable* gv_tvm_func_call_{nullptr};
  llvm::GlobalVariable* gv_tvm_get_func_from_env_{nullptr};
  llvm::GlobalVariable* gv_tvm_api_set_last_error_{nullptr};
  std::unordered_map<std::string, llvm::GlobalVariable*> gv_func_map_;

  // context for direct dynamic lookup
  llvm::Function* f_tvm_func_call_{nullptr};
  llvm::Function* f_tvm_get_func_from_env_{nullptr};
  llvm::Function* f_tvm_api_set_last_error_{nullptr};
  llvm::Function* f_tvm_register_system_symbol_{nullptr};
};

void CodeGenHexagon::InitTarget(llvm::TargetMachine* tm) {
  native_vector_bits_ = 64;  // Assume "scalar" vectors at first.
  llvm::StringRef fs = tm->getTargetFeatureString();
  size_t npos = llvm::StringRef::npos;
  const auto hvx_length_feature = "+hvx-length";  // +hvx-length{64|128}b
  size_t len_begin = fs.find(hvx_length_feature);
  size_t len_end = len_begin != npos ? fs.find('b', len_begin) : npos;
  if (len_end != npos) {
    int hvx_bytes = 0;
    len_begin += std::strlen(hvx_length_feature);
    ICHECK(!fs.substr(len_begin, len_end - len_begin).getAsInteger(10, hvx_bytes))
        << "invalid HVX length in feature string: " << fs.str();
    ICHECK(hvx_bytes == 64 || hvx_bytes == 128)
        << "invalid HVX vector length: " << hvx_bytes << ", should be 64 or 128";
    native_vector_bits_ = hvx_bytes * 8;
  }
  CodeGenLLVM::InitTarget(tm);
}

void CodeGenHexagon::Init(const std::string& module_name, llvm::TargetMachine* tm,
                          llvm::LLVMContext* ctx, bool system_lib, bool dynamic_lookup,
                          bool target_c_runtime) {
  CodeGenLLVM::Init(module_name, tm, ctx, system_lib, dynamic_lookup, false);

  func_handle_map_.clear();
  t_tvm_value_ = llvm::StructType::create({t_float64_}, "t_tvm_value");
  t_tvm_shape_index_ = llvm::Type::getIntNTy(*ctx, DataType::ShapeIndex().bits());
  t_tvm_context_ = llvm::StructType::create({t_int_, t_int_}, "t_tvm_context");
  t_tvm_type_ = llvm::StructType::create({t_int8_, t_int8_, t_int16_}, "t_tvm_type");
  t_tvm_func_handle_ = t_void_p_;
  // DLTensor
  t_tvm_array_ = llvm::StructType::create(
      {t_void_p_, t_tvm_context_, t_int_, t_tvm_type_, t_tvm_shape_index_->getPointerTo(),
       t_tvm_shape_index_->getPointerTo(), t_int64_},
      "t_tvm_array");

  types_for_alloca_ = {
      {"shape", t_tvm_shape_index_},
      {"arg_value", t_tvm_value_},
      {"arg_tcode", t_int_},
      {"array", t_tvm_array_},
  };

  // Runtime functions.
  ftype_tvm_func_call_ = llvm::FunctionType::get(
      t_int_,
      {t_tvm_func_handle_, t_tvm_value_->getPointerTo(), t_int_->getPointerTo(), t_int_,
       t_tvm_value_->getPointerTo(), t_int_->getPointerTo()},
      false);
  ftype_tvm_get_func_from_env_ = llvm::FunctionType::get(
      t_int_, {t_void_p_, t_char_->getPointerTo(), t_tvm_func_handle_->getPointerTo()}, false);
  ftype_tvm_api_set_last_error_ =
      llvm::FunctionType::get(t_void_, {t_char_->getPointerTo()}, false);
  md_tbaa_ctx_ptr_ = md_builder_->createTBAAScalarTypeNode("ctx_ptr", md_tbaa_root_);

  // initialize TVM runtime API
  if (system_lib) {
    // We will need this in environment for backward registration.
    f_tvm_register_system_symbol_ = llvm::Function::Create(
        llvm::FunctionType::get(t_int_, {t_char_->getPointerTo(), t_void_p_}, false),
        llvm::Function::ExternalLinkage, "TVMBackendRegisterSystemLibSymbol", module_.get());
  } else {
    f_tvm_register_system_symbol_ = nullptr;
  }
  this->InitGlobalContext(dynamic_lookup);
}

llvm::Value* CodeGenHexagon::CreateCallExtern(Type ret_type, String global_symbol,
                                              const Array<PrimExpr>& args, bool skip_first_arg) {
  std::vector<llvm::Value*> arg_values;
  for (size_t i = skip_first_arg; i < args.size(); ++i) {
    arg_values.push_back(MakeValue(args[i]));
  }
  std::vector<llvm::Type*> arg_types;
  for (llvm::Value* v : arg_values) {
    arg_types.push_back(v->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(GetLLVMType(ret_type), arg_types, false);
  // Check if it is available in global function table as injected function.
  auto it = gv_func_map_.find(global_symbol);
  if (it != gv_func_map_.end()) {
    if (it->second == nullptr) {
      gv_func_map_[global_symbol] = InitContextPtr(ftype->getPointerTo(), "__" + global_symbol);
      it = gv_func_map_.find(global_symbol);
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(ftype, GetContextPtr(it->second));
#else
    auto ext_callee = GetContextPtr(it->second);
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  } else {
    llvm::Function* f = module_->getFunction(global_symbol);
    if (f == nullptr) {
      f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                 global_symbol.operator llvm::StringRef(), module_.get());
    }
#if TVM_LLVM_VERSION >= 90
    auto ext_callee = llvm::FunctionCallee(f);
#else
    auto ext_callee = f;
#endif
    return builder_->CreateCall(ext_callee, arg_values);
  }
}

llvm::GlobalVariable* CodeGenHexagon::InitContextPtr(llvm::Type* p_type, std::string name) {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, p_type, false, llvm::GlobalValue::LinkOnceAnyLinkage, nullptr, name);
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(p_type)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(p_type));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(p_type));
  gv->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  return gv;
}

llvm::Value* CodeGenHexagon::GetContextPtr(llvm::GlobalVariable* gv) {
  ICHECK(gv != nullptr);
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, llvm::Align(gv->getAlignment()));
#else
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, gv->getAlignment());
#endif
  faddr->setMetadata("tbaa",
                     md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
  return faddr;
}

void CodeGenHexagon::InitGlobalContext(bool dynamic_lookup) {
  // Module context
  gv_mod_ctx_ = InitContextPtr(t_void_p_, tvm::runtime::symbol::tvm_module_ctx);
  // Register back the locations.
  if (f_tvm_register_system_symbol_ != nullptr) {
    export_system_symbols_.emplace_back(
        std::make_pair(tvm::runtime::symbol::tvm_module_ctx, gv_mod_ctx_));
  } else {
    if (!dynamic_lookup) {
      gv_tvm_func_call_ = InitContextPtr(ftype_tvm_func_call_->getPointerTo(), "__TVMFuncCall");
      gv_tvm_get_func_from_env_ = InitContextPtr(ftype_tvm_get_func_from_env_->getPointerTo(),
                                                 "__TVMBackendGetFuncFromEnv");
      gv_tvm_api_set_last_error_ =
          InitContextPtr(ftype_tvm_api_set_last_error_->getPointerTo(), "__TVMAPISetLastError");
      // Mark as context functions
      gv_func_map_["TVMBackendAllocWorkspace"] = nullptr;
      gv_func_map_["TVMBackendFreeWorkspace"] = nullptr;
    }
  }
}

llvm::Value* CodeGenHexagon::RuntimeTVMFuncCall() {
  if (f_tvm_func_call_ != nullptr) return f_tvm_func_call_;
  return GetContextPtr(gv_tvm_func_call_);
}

llvm::Value* CodeGenHexagon::RuntimeTVMGetFuncFromEnv() {
  if (f_tvm_get_func_from_env_ != nullptr) return f_tvm_get_func_from_env_;
  return GetContextPtr(gv_tvm_get_func_from_env_);
}

llvm::Value* CodeGenHexagon::RuntimeTVMAPISetLastError() {
  if (f_tvm_api_set_last_error_ != nullptr) return f_tvm_api_set_last_error_;
  return GetContextPtr(gv_tvm_api_set_last_error_);
}

llvm::BasicBlock* CodeGenHexagon::MakeCallPacked(const Array<PrimExpr>& args, llvm::Value** rvalue,
                                                 llvm::Value** ret_tcode, const DataType& r_type,
                                                 const int64_t begin, const int64_t end) {
  using llvm::BasicBlock;
  // using namespace tir;
  std::string func_name = args[0].as<StringImmNode>()->value;
  llvm::Value* handle = GetPackedFuncHandle(func_name);
  // call the function
  int64_t nargs = end - begin;
  ICHECK_GE(nargs, 0);
  llvm::Value* stack_value = MakeValue(args[1]);
  llvm::Value* stack_tcode = MakeValue(args[2]);
  llvm::Value* arg_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(stack_value, t_tvm_value_->getPointerTo()), ConstInt32(begin));
  llvm::Value* arg_tcode = CreateBufferPtr(DataType::Int(32), stack_tcode, ConstInt32(begin));
  llvm::Value* ret_value = builder_->CreateInBoundsGEP(
      builder_->CreatePointerCast(stack_value, t_tvm_value_->getPointerTo()), ConstInt32(end));
  *ret_tcode = CreateBufferPtr(DataType::Int(32), stack_tcode, ConstInt32(end));
#if TVM_LLVM_VERSION >= 90
  auto call_callee = llvm::FunctionCallee(ftype_tvm_func_call_, RuntimeTVMFuncCall());
#else
  auto call_callee = RuntimeTVMFuncCall();
#endif
  BasicBlock* end_block = CheckCallSuccess(builder_->CreateCall(
      call_callee, {handle, arg_value, arg_tcode, ConstInt32(nargs), ret_value, *ret_tcode}));
  DataType r_api_type = tir::APIType(r_type);
#if TVM_LLVM_VERSION >= 110
  *rvalue = builder_->CreateAlignedLoad(
      builder_->CreatePointerCast(ret_value, DTypeToLLVMType(r_api_type)->getPointerTo()),
      llvm::Align(8));
#else
  *rvalue = builder_->CreateAlignedLoad(
      builder_->CreatePointerCast(ret_value, DTypeToLLVMType(r_api_type)->getPointerTo()), 8);
#endif
  *rvalue = CreateCast(r_api_type, r_type, *rvalue);
  return end_block;
}

llvm::Value* CodeGenHexagon::GetPackedFuncHandle(const std::string& fname) {
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
    hptr =
        new llvm::GlobalVariable(*module_, t_tvm_func_handle_, false,
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
  BasicBlock* init_block = BasicBlock::Create(*ctx_, "handle_init", function_);
  BasicBlock* end_block = BasicBlock::Create(*ctx_, "handle_init_end", function_);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, llvm::Align(align));
#else
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, align);
#endif
  llvm::Value* handle_not_null =
      builder_->CreateICmpNE(handle, llvm::Constant::getNullValue(t_tvm_func_handle_));
  builder_->CreateCondBr(handle_not_null, end_block, init_block, md_very_likely_branch_);
  // Initialize the handle if needed.
  builder_->SetInsertPoint(init_block);
  llvm::Value* out =
      WithFunctionEntry([&]() { return builder_->CreateAlloca(t_tvm_func_handle_); });
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* ctx =
      builder_->CreateAlignedLoad(gv_mod_ctx_, llvm::Align(gv_mod_ctx_->getAlignment()));
#else
  llvm::LoadInst* ctx = builder_->CreateAlignedLoad(gv_mod_ctx_, gv_mod_ctx_->getAlignment());
#endif
  ctx->setMetadata("tbaa",
                   md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
#if TVM_LLVM_VERSION >= 90
  auto env_callee = llvm::FunctionCallee(ftype_tvm_get_func_from_env_, RuntimeTVMGetFuncFromEnv());
#else
  auto env_callee = RuntimeTVMGetFuncFromEnv();
#endif
  llvm::Value* retcode = builder_->CreateCall(env_callee, {ctx, GetConstString(fname), out});
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

llvm::Value* CodeGenHexagon::CreateCallPacked(const CallNode* op) {
  // There is always a call to __tvm_set_device in a standalone op,
  // and we can't have calls to packed functions, because they need
  // a Module object to work (or at least TVMBackendGetFuncFromEnv
  // function).
  const std::string& name = op->args[0].as<StringImmNode>()->value;
  if (name == "__tvm_set_device") {
    return ConstInt32(0);
  }

  ICHECK_EQ(op->args.size(), 5U);
  llvm::Value* rvalue = nullptr;
  llvm::Value* ret_tcode = nullptr;
  MakeCallPacked(op->args, &rvalue, &ret_tcode, op->dtype, op->args[3].as<IntImmNode>()->value,
                 op->args[4].as<IntImmNode>()->value);
  return rvalue;
}

llvm::Value* CodeGenHexagon::CreateCallTracePacked(const CallNode* op) {
  using llvm::BasicBlock;
  ICHECK_EQ(op->args.size(), 6U);
  llvm::Value* rvalue = nullptr;
  llvm::Value* ret_tcode = nullptr;
  BasicBlock* end_block =
      MakeCallPacked(op->args, &rvalue, &ret_tcode, op->dtype, op->args[3].as<IntImmNode>()->value,
                     op->args[4].as<IntImmNode>()->value);
  // Get traced value.
  llvm::Value* traced_value = MakeValue(op->args[5]);
  // The update_block handles case when we need to update the return value.
  BasicBlock* update_block = BasicBlock::Create(*ctx_, "update_block", function_);
  // The continue_block handles case when we need to return original
  // traced value.
  BasicBlock* continue_block = BasicBlock::Create(*ctx_, "continue_block", function_);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* ret_tcode_value = builder_->CreateAlignedLoad(ret_tcode, llvm::Align(8));
#else
  llvm::Value* ret_tcode_value = builder_->CreateAlignedLoad(ret_tcode, 8);
#endif
  // Check the ret_type_code and create cmp instruction.
  llvm::Value* cmp =
      builder_->CreateICmpNE(ret_tcode_value, llvm::ConstantInt::get(t_int_, kTVMNullptr));
  builder_->CreateCondBr(cmp, update_block, continue_block);
  builder_->SetInsertPoint(update_block);
  builder_->CreateBr(continue_block);
  builder_->SetInsertPoint(continue_block);
  // The return value depends on from what bb we come from.
  llvm::PHINode* phi_rvalue = builder_->CreatePHI(traced_value->getType(), 2);
  phi_rvalue->addIncoming(rvalue, update_block);
  phi_rvalue->addIncoming(traced_value, end_block);
  return phi_rvalue;
}

llvm::BasicBlock* CodeGenHexagon::CheckCallSuccess(llvm::Value* retcode) {
  // create emit codes that checks and load the function.
  using llvm::BasicBlock;
  BasicBlock* fail_block = BasicBlock::Create(*ctx_, "call_fail", function_);
  BasicBlock* end_block = BasicBlock::Create(*ctx_, "call_end", function_);
  llvm::Value* succ = builder_->CreateICmpEQ(retcode, llvm::ConstantInt::get(t_int_, 0));
  builder_->CreateCondBr(succ, end_block, fail_block, md_very_likely_branch_);
  builder_->SetInsertPoint(fail_block);
  // return the code.
  builder_->CreateRet(retcode);
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  return end_block;
}

void CodeGenHexagon::VisitStmt_(const AssertStmtNode* op) {
  using llvm::BasicBlock;
  llvm::Value* cond = MakeValue(op->condition);
  std::ostringstream os;
  os << "Assert fail: " << op->condition;
  if (op->message.as<StringImmNode>()) {
    os << ", " << op->message.as<StringImmNode>()->value;
  }
  llvm::Value* msg = GetConstString(os.str());
  BasicBlock* fail_block = BasicBlock::Create(*ctx_, "assert_fail", function_);
  BasicBlock* end_block = BasicBlock::Create(*ctx_, "assert_end", function_);
  builder_->CreateCondBr(cond, end_block, fail_block, md_very_likely_branch_);
  // fail condition.
  builder_->SetInsertPoint(fail_block);
#if TVM_LLVM_VERSION >= 90
  auto err_callee =
      llvm::FunctionCallee(ftype_tvm_api_set_last_error_, RuntimeTVMAPISetLastError());
#else
  auto err_callee = RuntimeTVMAPISetLastError();
#endif
  builder_->CreateCall(err_callee, {msg});
  builder_->CreateRet(ConstInt32(-1));
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  CodeGenLLVM::VisitStmt_(op);
}

llvm::Value* CodeGenHexagon::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    return CreateCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_call_trace_packed_lowered())) {
    return CreateCallTracePacked(op);
  } else if (op->op.same_as(builtin::tvm_struct_get())) {
    ICHECK_EQ(op->args.size(), 3);
    int kind = op->args[2].as<IntImmNode>()->value;
    llvm::Value* ref =
        CreateStructRefPtr(op->dtype, MakeValue(op->args[0]), MakeValue(op->args[1]), kind);
    if (kind == builtin::kArrAddr) {
      return builder_->CreatePointerCast(ref, t_void_p_);
    }
    return builder_->CreateLoad(ref);
  } else if (op->op.same_as(builtin::tvm_struct_set())) {
    ICHECK_EQ(op->args.size(), 4);
    int kind = op->args[2].as<IntImmNode>()->value;
    ICHECK(kind != builtin::kArrAddr);
    llvm::Value* ref = CreateStructRefPtr(op->args[3].dtype(), MakeValue(op->args[0]),
                                          MakeValue(op->args[1]), kind);
    llvm::Value* value = MakeValue(op->args[3]);
    if (value->getType()->isPointerTy()) {
      value = builder_->CreatePointerCast(value, ref->getType()->getPointerElementType());
    }
    builder_->CreateStore(value, ref);
    return ConstInt32(0);
  } else if (op->op.same_as(builtin::tvm_stack_alloca())) {
    ICHECK_EQ(op->args.size(), 2);
    const std::string& name = op->args[0].as<StringImmNode>()->value;
    llvm::Value* size = ConstInt32(op->args[1].as<IntImmNode>()->value);
    return builder_->CreateAlloca(types_for_alloca_.at(name), size);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    llvm::Value* neg_1 = ConstInt32(-1);
    builder_->CreateRet(neg_1);
    auto next_block = std::next(builder_->GetInsertBlock()->getIterator());
    llvm::BasicBlock* new_bb = llvm::BasicBlock::Create(*ctx_, "cont", function_, &*next_block);
    builder_->SetInsertPoint(new_bb);
    return neg_1;
  }

  return CodeGenLLVM::CreateIntrinsic(op);
}

llvm::Value* CodeGenHexagon::CreateStructRefPtr(DataType t, llvm::Value* buf, llvm::Value* index,
                                                int kind) {
  static const std::map<int, int> field_index = {
      {builtin::kArrData, 0},      {builtin::kArrDeviceType, 1}, {builtin::kArrDeviceId, 1},
      {builtin::kArrNDim, 2},      {builtin::kArrTypeCode, 3},   {builtin::kArrTypeBits, 3},
      {builtin::kArrTypeLanes, 3}, {builtin::kArrShape, 4},      {builtin::kArrStrides, 5},
      {builtin::kArrByteOffset, 6}};
  static const std::map<int, int> subfield_index = {
      {builtin::kArrDeviceType, 0}, {builtin::kArrDeviceId, 1},  {builtin::kArrTypeCode, 0},
      {builtin::kArrTypeBits, 1},   {builtin::kArrTypeLanes, 2},
  };

  if (kind < builtin::kArrKindBound_) {
    if (buf->getType() == t_void_p_) {
      buf = builder_->CreatePointerCast(buf, t_tvm_array_->getPointerTo());
    } else {
      ICHECK_EQ(buf->getType(), t_tvm_array_->getPointerTo());
    }
    /* The following "kinds" are accessing the members of DLTensor:
       typedef struct {
         void* data;            kArrData
         DLDevice device;       kArrDeviceType (device.device_type)
                                kArrDeviceId (device.device_id)
         int ndim;              kArrNDim
         DLDataType dtype;      kArrTypeCode (dtype.code)
                                kArrTypeBits (dtype.bits)
                                kArrTypeLanes (dtype.lanes)
         int64_t* shape;        kArrShape
         int64_t* strides;      kArrStrides
         uint64_t byte_offset;  kArrByteOffset
       } DLTensor;
    */
    llvm::Value* base_gep = builder_->CreateInBoundsGEP(buf, index, "base_gep");
    if (kind == builtin::kArrAddr) {
      return base_gep;
    }
    llvm::Value* field_gep = builder_->CreateInBoundsGEP(
        base_gep, {ConstInt32(0), ConstInt32(field_index.at(kind))}, "field_gep");
    switch (kind) {
      // These fields have no sub-fields.
      case builtin::kArrData:
      case builtin::kArrNDim:
      case builtin::kArrShape:
      case builtin::kArrStrides:
      case builtin::kArrByteOffset:
        return field_gep;
    }
    return builder_->CreateInBoundsGEP(
        field_gep, {ConstInt32(0), ConstInt32(subfield_index.at(kind))}, "subfield_gep");
  }

  if (kind == builtin::kTVMValueContent) {
    /* TVMValue is a union:
       typedef union {
         int64_t v_int64;
         double v_float64;
         void* v_handle;
         const char* v_str;
         TVMType v_type;
         DLDevice v_device;
       } TVMValue;
    */
    ICHECK_EQ(t.lanes(), 1);
    ICHECK(t.is_handle() || t.bits() == 64);
    if (t.is_int()) {
      buf = builder_->CreatePointerCast(buf, t_int64_->getPointerTo());
      return builder_->CreateInBoundsGEP(buf, index);
    } else if (t.is_float()) {
      buf = builder_->CreatePointerCast(buf, t_float64_->getPointerTo());
      return builder_->CreateInBoundsGEP(buf, index);
    } else {
      ICHECK(t.is_handle());
      buf = builder_->CreatePointerCast(buf, t_tvm_value_->getPointerTo());
      buf = builder_->CreateInBoundsGEP(buf, index);
      return builder_->CreatePointerCast(buf, t_void_p_->getPointerTo());
    }
  }

  assert(!"Unknown kind");
  return nullptr;
}

namespace {
// Check if the function matches the TVMBackendPackedCFunc prototype.
bool UsesExportABI(const PrimFunc& f) {
  if (f->attrs.defined()) {
    auto it = f->attrs->dict.find("calling_conv");
    return it != f->attrs->dict.end() &&
           Downcast<Integer>((*it).second) == CallingConv::kCPackedFunc;
  }
  return false;
}

DMLC_ATTRIBUTE_UNUSED std::ostream& operator<<(std::ostream& os, const llvm::Module& m) {
  std::string ms;
  llvm::raw_string_ostream sos(ms);
  sos << m;
  os << sos.str();
  return os;
}

void ProcessLLVMOptions(const std::vector<std::string>& llvm_vec) {
  if (llvm_vec.empty()) return;

  // LLVM options.
  std::vector<const char*> starts;
  std::transform(llvm_vec.begin(), llvm_vec.end(), std::back_inserter(starts),
                 std::mem_fn(&std::string::c_str));
  const char** args = &starts.front();

  llvm::cl::ParseCommandLineOptions(llvm_vec.size(), args);
}

}  // namespace

runtime::Module BuildHexagon(IRModule mod, Target target) {
  // Make sure all targets are registered. InitializeLLVM can be called
  // multiple times, after the first call all subsequent calls are no-ops.
  InitializeLLVM();

  auto split = [](const std::string& str, char delim = ' ') {
    std::vector<std::string> vec;
    std::string tmp;
    for (std::istringstream iss(str); std::getline(iss, tmp, delim);) {
      vec.push_back(tmp);
    }
    return vec;
  };
  std::string llvm_options_str;
  if (const Optional<String> llvm_options = target->GetAttr<String>("llvm-options")) {
    llvm_options_str = "llvm," + llvm_options.value();
  } else {
    llvm_options_str = "llvm";
  }
  // Postprocess the LLVM options string: replace '@' with '=', and ',' with ' '.
  for (int i = 0, e = llvm_options_str.size(); i != e; ++i) {
    switch (llvm_options_str[i]) {
      case '@':
        llvm_options_str[i] = '=';
        break;
      case ',':
        llvm_options_str[i] = ' ';
        break;
    }
  }

  // The vector of LLVM options is treated at "argv" from "main(argc, argv)". The entry at
  // position 0 is the name of the executable, and is ignored by the LLVM cl::option parser.
  // Make sure it's set to "llvm" (tvm.target.hexagon does that).
  std::vector<std::string> llvm_options_vec = split(llvm_options_str);
  assert(llvm_options_vec.size() >= 1 && llvm_options_vec[0] == "llvm");
  llvm_options_vec.insert(std::next(llvm_options_vec.begin()),
                          {"-hexagon-small-data-threshold=0",
                           "-force-target-max-vector-interleave=1", "-hexagon-autohvx=1"});

  // Process extra command line options for LLVM. Make sure it's only
  // done once.
  static bool CallOnce = (ProcessLLVMOptions(llvm_options_vec), true);
  (void)CallOnce;

  std::unique_ptr<llvm::TargetMachine> tm = GetLLVMTargetMachine(target);
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  std::unique_ptr<CodeGenHexagon> cg(new CodeGenHexagon());

  std::vector<PrimFunc> funcs;
  std::string entry_func;
  Map<String, LinkedParam> linked_params;
  bool found_linked_params = false;
  bool could_have_linked_params = target->GetAttr<Bool>("link-params").value_or(Bool(false));

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "Can only lower IR Module with PrimFuncs";
    if (could_have_linked_params &&
        kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
      Map<String, ObjectRef> attrs_dict = Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
      CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
          << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";
      linked_params =
          Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
      found_linked_params = true;
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
      auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(global_symbol.defined());
      entry_func = global_symbol.value();
    }
    funcs.emplace_back(f);
  }

  cg->Init("TVMHexagonModule", tm.get(), ctx.get(), false, false, false);
  for (const PrimFunc& f : funcs) {
    cg->AddFunction(f);
  }
  if (found_linked_params) {
    cg->LinkParameters(linked_params);
  }

  // Uncomment to get the LLVM module right out of codegen, before optimizations.
  // std::cerr << "HexagonModule.0 {\n" << *cg->GetModulePtr() << "}\n";
  std::unique_ptr<llvm::Module> module = cg->Finish();

  enum CodeGenFileType { Asm, Obj, IR, BC };

  auto EmitToString = [&tm](const llvm::Module& m, CodeGenFileType cgft) {
    std::string out;

    if (cgft == IR || cgft == BC) {
      llvm::raw_string_ostream os(out);
      if (cgft == IR)
        m.print(os, nullptr);
      else
        llvm::WriteBitcodeToFile(m, os);
    } else if (cgft == Asm || cgft == Obj) {
      using namespace llvm;
#if TVM_LLVM_VERSION <= 90
      auto ft = cgft == Asm ? TargetMachine::CodeGenFileType::CGFT_AssemblyFile
                            : TargetMachine::CodeGenFileType::CGFT_ObjectFile;
#else
      auto ft = cgft == Asm ? llvm::CGFT_AssemblyFile : llvm::CGFT_ObjectFile;
#endif

      SmallString<16384> ss;  // Will grow on demand.
      llvm::raw_svector_ostream os(ss);
      std::unique_ptr<llvm::Module> cm = CloneModule(m);
      legacy::PassManager pass;
      ICHECK(tm->addPassesToEmitFile(pass, os, nullptr, ft) == 0) << "Cannot emit target code";
      pass.run(*cm.get());
      out.assign(ss.c_str(), ss.size());
    }

    return out;
  };

  auto SaveToFile = [](const std::string& data, const std::string& suffix) {
    llvm::SmallString<64> file_name;
    int fd;
    std::error_code ec = llvm::sys::fs::createTemporaryFile("tvm", suffix, fd, file_name);
    ICHECK_EQ(static_cast<bool>(ec), false) << ec.message();
    llvm::raw_fd_ostream file(fd, true);
    file << data;
    ICHECK(!file.has_error()) << file.error().message();
    // If there is an error, execution will never get here, but return
    // {ec, name} anyway to allow caller to handle error conditions.
    // This way the "ICHECK" above can be removed with minimal effort.
    return std::make_pair(file.error(), std::string(file_name.c_str()));
  };

  std::string asm_str = EmitToString(*module.get(), Asm);
  std::string obj_str = EmitToString(*module.get(), Obj);
  std::string ir_str = EmitToString(*module.get(), IR);
  std::string bc_str = EmitToString(*module.get(), BC);

  std::string o_name = SaveToFile(obj_str, "o").second;
  std::string so_name(o_name, 0, o_name.size() - 1);
  so_name += "so";

  const auto* f = tvm::runtime::Registry::Get("tvm.contrib.hexagon.link_shared");
  ICHECK(f != nullptr) << "tvm.contrib.hexagon.link_shared does not to exist, "
                          "do import tvm.contrib.hexagon";

  Array<PrimExpr> o_names = {StringImm(o_name)};
  int rc = (*f)(so_name, o_names);
  ICHECK(rc == 0) << "Failed to link " << so_name;

  // Move it to ExtractFuncInfo?
  std::set<std::string> export_abi;
  for (auto kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    if (UsesExportABI(f)) export_abi.insert(get_name(f));
  }
  return HexagonModuleCreate(so_name, "so", ExtractFuncInfo(mod), asm_str, obj_str, ir_str, bc_str,
                             export_abi);
}

TVM_REGISTER_GLOBAL("target.build.hexagon").set_body_typed(BuildHexagon);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
