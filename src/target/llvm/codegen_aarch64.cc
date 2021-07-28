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
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>

#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

// -------------------------
// Utility functions to remove
void print_LLVM_type(llvm::Type* type) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  type->print(rso);
  std::cout << rso.str() << std::endl;
  ;
}

void print_LLVM_val(llvm::Value* val) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  val->print(rso);
  std::cout << rso.str() << std::endl;
  ;
}
// -------------------------

// AArch64 code generation
class CodeGenAArch64 final : public CodeGenCPU {
 public:
  void InitTarget(llvm::TargetMachine* tm) final {
    // set native vector bits.
    native_vector_bits_ = 16 * 8;
    CodeGenCPU::InitTarget(tm);
  }
  llvm::Value* VisitExpr_(const LoadNode* op);
  void VisitStmt_(const ForNode* op);
  void VisitStmt_(const StoreNode* op);

 private:
  // SVE LLVM intrinsics
  llvm::Value* sve_stride(int min_lanes);
  llvm::Value* sve_whilelt(llvm::Value* a, llvm::Value* b, int min_lanes);
  llvm::Value* sve_store(llvm::Value* ptr, llvm::Value* val, DataType t);
  llvm::Value* sve_load(llvm::Value* ptr, DataType t);
  void CreateSVEFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride, const Var& loop_var,
                    const Stmt& body, int min_lanes);

  // Predicate
  llvm::Value* mask_;
};

llvm::Value* CodeGenAArch64::sve_stride(int min_lanes) {
  llvm::Intrinsic::ID cnt_id;

  switch (min_lanes) {
    case 16:
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntb");
      break;
    case 8:  // half
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cnth");
      break;
    case 4:  // float
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntw");
      break;
    default:  // double
      cnt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.cntd");
  }

  // All pattern
  int all_pattern = 31;

  llvm::Value* in_param = llvm::ConstantInt::get(*ctx_, llvm::APInt(32, all_pattern));
  std::vector<llvm::Value*> arg_value{in_param};
  std::vector<llvm::Type*> arg_type{builder_->getInt32Ty()};
  llvm::Type* return_type = builder_->getInt64Ty();
  llvm::Function* func_cnt = GetIntrinsicDecl(cnt_id, return_type, arg_type);
  llvm::Value* vec_stride = builder_->CreateCall(func_cnt, arg_value);
  llvm::Value* vec_stride_int32 =
      builder_->CreateTruncOrBitCast(vec_stride, builder_->getInt32Ty());
  return vec_stride_int32;
}

llvm::Value* CodeGenAArch64::sve_whilelt(llvm::Value* a, llvm::Value* b, int min_lanes) {
  llvm::Intrinsic::ID whilelt_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.whilelt");
  std::vector<llvm::Value*> arg_value{a, b};
  std::vector<llvm::Type*> arg_type{builder_->getInt32Ty(), builder_->getInt32Ty()};

  // Needs to be a vector type
  llvm::Type* bool_type = llvm::Type::getIntNTy(*ctx_, 1);
  llvm::Type* return_type = llvm::ScalableVectorType::get(bool_type, min_lanes);

  llvm::Function* func_whilelt = GetIntrinsicDecl(whilelt_id, return_type, arg_type);
  llvm::Value* whilelt = builder_->CreateCall(func_whilelt, arg_value);
  return whilelt;
}

llvm::Value* CodeGenAArch64::sve_store(llvm::Value* ptr, llvm::Value* val, DataType t) {
  // Get the intrinsic
  llvm::Intrinsic::ID st1_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.st1");
  std::vector<llvm::Value*> arg_value{val, mask_, ptr};

  // Get the pointer type
  llvm::PointerType* ptr_type = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
  ICHECK(ptr_type != nullptr);

  // Input types
  llvm::Type* mask_type = mask_->getType();
  llvm::Type* scalar_type = ptr_type->getElementType();
  llvm::Type* store_type = llvm::ScalableVectorType::get(scalar_type, t.lanes());
  std::vector<llvm::Type*> arg_type{store_type, mask_type, ptr_type};

  // Return type (void)
  llvm::Type* return_type = llvm::Type::getVoidTy(*ctx_);
  llvm::Function* func_st1 = GetIntrinsicDecl(st1_id, return_type, arg_type);

  // Create the call
  llvm::Value* st1 = builder_->CreateCall(func_st1, arg_value);
  return st1;
}

llvm::Value* CodeGenAArch64::sve_load(llvm::Value* ptr, DataType t) {
  llvm::Intrinsic::ID ld1_id = llvm::Function::lookupIntrinsicID("llvm.aarch64.sve.ld1");
  std::vector<llvm::Value*> arg_value{mask_, ptr};
  llvm::Type* ptr_type = ptr->getType();
  llvm::Type* mask_type = mask_->getType();

  std::vector<llvm::Type*> arg_type{mask_type, ptr_type};
  llvm::PointerType* ptype = llvm::dyn_cast<llvm::PointerType>(ptr_type);
  ICHECK(ptype != nullptr);

  llvm::Type* scalar_type = ptype->getElementType();
  llvm::Type* return_type = llvm::ScalableVectorType::get(scalar_type, t.lanes());
  llvm::Function* func_ld1 = GetIntrinsicDecl(ld1_id, return_type, arg_type);

  llvm::Value* ld1 = builder_->CreateCall(func_ld1, arg_value);
  return ld1;
}

void CodeGenAArch64::CreateSVEFor(llvm::Value* begin, llvm::Value* end, llvm::Value* stride,
                                  const Var& loop_var, const Stmt& body, int min_lanes) {
  using llvm::BasicBlock;
  BasicBlock* for_begin = builder_->GetInsertBlock();
  BasicBlock* for_body = BasicBlock::Create(*ctx_, "for_body", function_);
  BasicBlock* for_end = BasicBlock::Create(*ctx_, "for_end", function_);

  // for_begin block
  builder_->SetInsertPoint(for_begin);
  llvm::Value* vec_stride = sve_stride(min_lanes);
  builder_->CreateBr(for_body);

  // for_body
  builder_->SetInsertPoint(for_body);
  llvm::PHINode* loop_value = builder_->CreatePHI(begin->getType(), 2);
  mask_ = sve_whilelt(loop_value, end, min_lanes);
  loop_value->addIncoming(begin, for_begin);
  ICHECK(!var_map_.count(loop_var.get()));
  var_map_[loop_var.get()] = loop_value;

  this->VisitStmt(body);
  var_map_.erase(loop_var.get());
  llvm::Value* loop_next = CreateAdd(loop_var.dtype(), loop_value, vec_stride);
  loop_value->addIncoming(loop_next, builder_->GetInsertBlock());
  builder_->CreateCondBr(CreateLT(loop_var.dtype(), loop_value, end), for_body, for_end,
                         md_very_likely_branch_);
  builder_->SetInsertPoint(for_end);
  function_->print(llvm::errs());
}

llvm::Value* CodeGenAArch64::VisitExpr_(const LoadNode* op) {
  DataType t = op->dtype;
  if (!t.is_scalable()) return CodeGenCPU::VisitExpr_(op);
  llvm::Value* buffer = MakeValue(op->buffer_var);

  // scalable vector load
  const RampNode* ramp = op->index.as<RampNode>();
  ICHECK(ramp);
  // TODO(giuseros): use gather to address a load-with-stride-greater-than-1
  ICHECK(is_one(ramp->stride));

  int alignment, native_bits;
  GetAlignment(t, op->buffer_var.get(), ramp->base, &alignment, &native_bits);
  ICHECK_EQ(ramp->lanes, t.lanes());
  llvm::Value* ptr = CreateBufferPtr(t.element_of(), buffer, MakeValue(ramp->base));

  llvm::Value* load = sve_load(ptr, t);
  return load;
}

void CodeGenAArch64::VisitStmt_(const StoreNode* op) {
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
        if (!t.is_scalable()) {
          ptr = builder_->CreatePointerCast(ptr, DTypeToLLVMType(t)->getPointerTo(addrspace));
        }
#if TVM_LLVM_VERSION >= 110
        if (t.is_scalable()) {
          sve_store(ptr, value, t);
          return;
        }
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

void CodeGenAArch64::VisitStmt_(const ForNode* op) {
  ICHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  if (op->is_vla) {
    CreateSVEFor(MakeValue(op->min), MakeValue(op->extent),
                 llvm::ConstantInt::getSigned(GetLLVMType(op->extent), 1), op->loop_var, op->body,
                 op->stride);
  } else {
    CodeGenCPU::VisitStmt_(op);
  }
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_aarch64")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      CodeGenLLVM* cg = new CodeGenAArch64();
      *rv = static_cast<void*>(cg);
    });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
