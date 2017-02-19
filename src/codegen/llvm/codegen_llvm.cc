/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/c_runtime_api.h>
#include "./codegen_llvm.h"
#include "../../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

void CodeGenLLVM::Init(const std::string& module_name,
                       const std::string& target_triple,
                       llvm::LLVMContext* ctx) {
  InitializeLLVM();
  static_assert(sizeof(TVMValue) == sizeof(double), "invariant");
  static_assert(alignof(TVMValue) == alignof(double), "invariant");
  // clear maps
  var_map_.clear();
  str_map_.clear();
  func_handle_map_.clear();
  // initialize types.
  if (ctx_ != ctx) {
    t_void_ = llvm::Type::getVoidTy(*ctx);
    t_void_p_ = llvm::Type::getInt8Ty(*ctx)->getPointerTo();
    t_int_ = llvm::Type::getIntNTy(*ctx, sizeof(int) * 8);
    t_char_ = llvm::Type::getInt8Ty(*ctx);
    t_int8_ = llvm::Type::getInt8Ty(*ctx);
    t_int16_ = llvm::Type::getInt16Ty(*ctx);
    t_int32_ = llvm::Type::getInt32Ty(*ctx);
    t_float64_ = llvm::Type::getDoubleTy(*ctx);
    t_tvm_index_ = llvm::Type::getIntNTy(*ctx, sizeof(tvm_index_t) * 8);
    t_tvm_context_ = llvm::StructType::create({t_int_, t_int_});
    t_tvm_type_ = llvm::StructType::create({t_int8_, t_int8_, t_int16_});
    t_tvm_func_handle_ = t_void_p_;
    t_tvm_array_ = llvm::StructType::create(
        {t_void_p_,
         t_tvm_index_->getPointerTo(),
         t_tvm_index_->getPointerTo(),
         t_tvm_index_,
         t_tvm_type_,
         t_tvm_context_});
    t_tvm_value_ = llvm::StructType::create({t_float64_});
    md_builder_.reset(new llvm::MDBuilder(*ctx));
    md_very_likely_branch_ =
        md_builder_->createBranchWeights(1 << 30, 0);
    md_tbaa_root_ = md_builder_->createTBAARoot("tvmtbaa");
  }
  ctx_ = ctx;
  // initialize modules
  module_.reset(new llvm::Module(module_name, *ctx));
  // initialize TVM runtime API
  f_tvm_func_call_ = llvm::Function::Create(
      llvm::FunctionType::get(t_int_, {
          t_tvm_func_handle_,
          t_tvm_value_->getPointerTo(),
          t_int_->getPointerTo(),
          t_int_,
          t_tvm_value_->getPointerTo(),
          t_int_->getPointerTo()}, false),
      llvm::Function::ExternalLinkage, "TVMFuncCall", module_.get());
  f_tvm_get_func_from_env_ = llvm::Function::Create(
      llvm::FunctionType::get(t_int_, {
          t_void_p_,
          t_char_->getPointerTo(),
          t_tvm_func_handle_->getPointerTo()}, false),
      llvm::Function::ExternalLinkage, "TVMBackendGetFuncFromEnv", module_.get());
  f_tvm_api_set_last_error_ = llvm::Function::Create(
      llvm::FunctionType::get(t_void_, {t_char_->getPointerTo()}, false),
      llvm::Function::ExternalLinkage, "TVMAPISetLastError", module_.get());

  this->InitTarget(target_triple);
  // initialize builder
  builder_.reset(new IRBuilder(*ctx));
  this->InitGlobalContext();
}

void CodeGenLLVM::InitTarget(const std::string& target) {
  llvm::TargetMachine* tm;
  std::string target_triple;
  std::tie(tm, target_triple) = LLVMGetTarget(target);
  module_->setTargetTriple(target_triple);
  module_->setDataLayout(tm->createDataLayout());
  data_layout_.reset(new llvm::DataLayout(module_.get()));
}

void CodeGenLLVM::InitGlobalContext() {
  gv_mod_ctx_ = new llvm::GlobalVariable(
      *module_, t_void_p_, false,
      llvm::GlobalValue::LinkOnceODRLinkage, 0, "__tvm_module_ctx");
  gv_mod_ctx_->setAlignment(data_layout_->getTypeAllocSize(t_void_p_));
  gv_mod_ctx_->setInitializer(llvm::Constant::getNullValue(t_void_p_));
}

void CodeGenLLVM::AddFunction(const LoweredFunc& f) {
  var_map_.clear();
  CHECK(!module_->getFunction(f->name))
      << "Function " << f->name << "already exists in module";
  std::vector<llvm::Type*> arg_type;
  for (Var arg : f->args) {
    Type t = arg.type();
    if (t.is_handle() && f->handle_data_type.count(arg)) {
      arg_type.push_back(
          LLVMType(f->handle_data_type[arg].type())->getPointerTo());
    } else {
      arg_type.push_back(LLVMType(t));
    }
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(t_int_, arg_type, false);
  // setup the function.
  function_ = llvm::cast<llvm::Function>(module_->getOrInsertFunction(f->name, ftype));
  function_->setCallingConv(llvm::CallingConv::C);
  size_t idx = 0;

  for (auto it = function_->arg_begin();
      it != function_->arg_end(); ++it, ++idx) {
    llvm::Argument* v = &(*it);
    var_map_[f->args[idx].get()] = v;
  }

  llvm::BasicBlock* block = llvm::BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(block);
  this->Visit(f->body);
  builder_->CreateRet(ConstInt32(0));
}

void CodeGenLLVM::AddMainFunction(const std::string& entry_func_name) {
  llvm::Function* f = module_->getFunction(entry_func_name);
  CHECK(f) << "Function " << entry_func_name << "does not in module";
  CHECK(!module_->getFunction(runtime::symbol::tvm_module_main));
  llvm::FunctionType* ftype = f->getFunctionType();
  function_ = llvm::cast<llvm::Function>(
      module_->getOrInsertFunction(runtime::symbol::tvm_module_main, ftype));
  function_->setCallingConv(llvm::CallingConv::C);
  std::vector<llvm::Value*> args;
  for (auto it = function_->arg_begin();
       it != function_->arg_end(); ++it) {
    args.push_back(&(*it));
  }
  llvm::BasicBlock* block = llvm::BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(block);
  builder_->CreateRet(builder_->CreateCall(f, args));
}

class FPassManager : public llvm::legacy::FunctionPassManager {
 public:
  explicit FPassManager(llvm::Module* m)
      : llvm::legacy::FunctionPassManager(m) {}
  // override add to allow messaging
  void add(llvm::Pass* p) final {
    llvm::legacy::FunctionPassManager::add(p);
  }
};
class MPassManager : public llvm::legacy::PassManager {
 public:
  // override add to allow messaging
  void add(llvm::Pass* p) final {
    llvm::legacy::PassManager::add(p);
  }
};


void CodeGenLLVM::Optimize() {
  // place optimization pass
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0);
  builder.LoopVectorize = true;
  builder.SLPVectorize = true;
  // pass manager
  FPassManager fpass(module_.get());
  MPassManager mpass;
  builder.populateFunctionPassManager(fpass);
  builder.populateModulePassManager(mpass);

  fpass.doInitialization();
  for (auto it = module_->begin(); it != module_->end(); ++it) {
    fpass.run(*it);
  }
  fpass.doFinalization();
  mpass.run(*module_);
}

std::unique_ptr<llvm::Module> CodeGenLLVM::Finish() {
  this->Optimize();
  var_map_.clear();
  str_map_.clear();
  func_handle_map_.clear();
  return std::move(module_);
}

llvm::Type* CodeGenLLVM::LLVMType(const Type& t) const {
  llvm::Type* ret = nullptr;
  if (t.is_uint() || t.is_int()) {
    ret = llvm::Type::getIntNTy(*ctx_, t.bits());
  } else if (t.is_float()) {
    switch (t.bits()) {
      case 16: ret = llvm::Type::getHalfTy(*ctx_); break;
      case 32: ret = llvm::Type::getFloatTy(*ctx_); break;
      case 64: ret = llvm::Type::getDoubleTy(*ctx_); break;
      default: LOG(FATAL) << "cannot handle " << t;
    }
  } else {
    CHECK(t.is_handle());
    ret = t_void_p_;
  }
  if (t.lanes() != 1) {
    ret = llvm::VectorType::get(ret, t.lanes());
  }
  return ret;
}

void CodeGenLLVM::Visit_(const Variable* op) {
  value_ = GetVarValue(op);
}

void CodeGenLLVM::Visit_(const Cast* op) {
  value_ = CreateCast(op->value.type(), op->type, MakeValue(op->value));
}

void CodeGenLLVM::Visit_(const IntImm* op) {
  value_ = llvm::ConstantInt::getSigned(LLVMType(op->type), op->value);
}

void CodeGenLLVM::Visit_(const UIntImm* op) {
  value_ = llvm::ConstantInt::get(LLVMType(op->type), op->value);
}

void CodeGenLLVM::Visit_(const FloatImm* op) {
  value_ = llvm::ConstantFP::get(LLVMType(op->type), op->value);
}

void CodeGenLLVM::Visit_(const StringImm* op) {
  value_ = GetConstString(op->value);
}

#define DEFINE_CODEGEN_BINARY_OP(OP)                                    \
  llvm::Value* CodeGenLLVM::Create ## OP(                               \
      Type t, llvm::Value* a, llvm::Value *b) {                         \
    if (t.is_float()) {                                                 \
      return builder_->CreateF ## OP (a, b);                            \
    } else if (t.is_int() && t.bits() >= 32) {                          \
      return builder_->CreateNSW ## OP (a, b);                          \
    } else {                                                            \
      return builder_->Create ## OP (a, b);                             \
    }                                                                   \
  }                                                                     \

DEFINE_CODEGEN_BINARY_OP(Add);
DEFINE_CODEGEN_BINARY_OP(Sub);
DEFINE_CODEGEN_BINARY_OP(Mul);

void CodeGenLLVM::Visit_(const Add* op) {
  value_ = CreateAdd(op->type, MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const Sub* op) {
  value_ = CreateSub(op->type, MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const Mul* op) {
  value_ = CreateMul(op->type, MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const Div* op) {
  llvm::Value* a = MakeValue(op->a);
  int shift;
  if (op->type.is_float()) {
    value_ = builder_->CreateFDiv(a, MakeValue(op->b));
  } else if ((op->type.is_int() || op->type.is_uint()) &&
             is_const_power_of_two_integer(op->b, &shift)) {
    value_ = builder_->CreateAShr(a, shift);
  } else {
    llvm::Value* b = MakeValue(op->b);
    if (op->type.is_int()) {
      value_ = builder_->CreateSDiv(a, b);
    } else {
      CHECK(op->type.is_uint());
      value_ = builder_->CreateUDiv(a, b);
    }
  }
}

void CodeGenLLVM::Visit_(const Mod* op) {
  CHECK(!op->type.is_float())
      << "Cannot do mod for float";
  if (op->type.is_int()) {
    value_ = builder_->CreateSRem(MakeValue(op->a), MakeValue(op->b));
  } else {
    CHECK(op->type.is_uint());
    value_ = builder_->CreateURem(MakeValue(op->a), MakeValue(op->b));
  }
}

void CodeGenLLVM::Visit_(const Min* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  llvm::Value* cond = CreateLT(op->a.type(), a, b);
  value_ = builder_->CreateSelect(cond, a, b);
}

void CodeGenLLVM::Visit_(const Max* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  llvm::Value* cond = CreateGT(op->a.type(), a, b);
  value_ = builder_->CreateSelect(cond, a, b);
}

#define DEFINE_CODEGEN_CMP_OP(OP)                                       \
  llvm::Value* CodeGenLLVM::Create ## OP(                               \
      Type t, llvm::Value* a, llvm::Value* b) {                         \
    if (t.is_float()) {                                                 \
      return builder_->CreateFCmpO ## OP (a, b);                        \
    } else if (t.is_int()) {                                            \
      return  builder_->CreateICmpS ## OP (a, b);                       \
    } else {                                                            \
      return builder_->CreateICmpU ## OP (a, b);                        \
    }                                                                   \
  }                                                                     \

DEFINE_CODEGEN_CMP_OP(LT);
DEFINE_CODEGEN_CMP_OP(LE);
DEFINE_CODEGEN_CMP_OP(GT);
DEFINE_CODEGEN_CMP_OP(GE);

void CodeGenLLVM::Visit_(const LT* op) {
  value_ = CreateLT(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
void CodeGenLLVM::Visit_(const LE* op) {
  value_ = CreateLE(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
void CodeGenLLVM::Visit_(const GT* op) {
  value_ = CreateGT(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
void CodeGenLLVM::Visit_(const GE* op) {
  value_ = CreateGE(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const EQ* op) {
  if (op->a.type().is_float()) {
    value_ = builder_->CreateFCmpOEQ(MakeValue(op->a), MakeValue(op->b));
  } else {
    value_ = builder_->CreateICmpEQ(MakeValue(op->a), MakeValue(op->b));
  }
}

void CodeGenLLVM::Visit_(const NE* op) {
  if (op->a.type().is_float()) {
    value_ = builder_->CreateFCmpONE(MakeValue(op->a), MakeValue(op->b));
  } else {
    value_ = builder_->CreateICmpNE(MakeValue(op->a), MakeValue(op->b));
  }
}

void CodeGenLLVM::Visit_(const And* op) {
  value_ = builder_->CreateAnd(MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const Or* op) {
  value_ = builder_->CreateOr(MakeValue(op->a), MakeValue(op->b));
}

void CodeGenLLVM::Visit_(const Not* op) {
  value_ = builder_->CreateNot(MakeValue(op->a));
}

void CodeGenLLVM::Visit_(const Select* op) {
  value_ = builder_->CreateSelect(
      MakeValue(op->condition),
      MakeValue(op->true_value),
      MakeValue(op->false_value));
}

void CodeGenLLVM::Visit_(const Let* op) {
  llvm::Value* v = MakeValue(op->value);
  CHECK(!var_map_.count(op->var.get()));
  var_map_[op->var.get()] = v;
  value_ = MakeValue(op->body);
}

void CodeGenLLVM::Visit_(const Broadcast* op) {
  value_ = CreateBroadcast(MakeValue(op->value), op->lanes);
}

void CodeGenLLVM::Visit_(const Ramp* op) {
  Type t = op->type;
  llvm::Value* base = MakeValue(op->base);
  llvm::Value* stride = MakeValue(op->stride);
  llvm::Value* value = llvm::UndefValue::get(LLVMType(t));
  for (int i = 0; i < t.lanes(); ++i) {
    if (i != 0) {
      base = CreateAdd(t, base, stride);
    }
    value = builder_->CreateInsertElement(
        value, base, llvm::ConstantInt::get(t_int32_, i));
  }
  value_ = value;
}

void CodeGenLLVM::Visit_(const Load* op) {
  Type t = op->type;
  CHECK(!t.is_vector());

  if (t.is_scalar()) {
    llvm::LoadInst* inst = builder_->CreateAlignedLoad(
        CreateBufferPtr(
            t,
            GetVarValue(op->buffer_var.get()),
            MakeValue(op->index)),
        data_layout_->getTypeAllocSize(LLVMType(t)));
    AddAliasInfo(inst, op->buffer_var.get(), op->index);
    value_ = inst;
  } else {
    LOG(FATAL) << "not yet supported";
  }
}

void CodeGenLLVM::Visit_(const Store* op) {
  llvm::Value* value = MakeValue(op->value);
  Type t = op->value.type();
  CHECK(!t.is_vector());
  if (t.is_scalar()) {
    llvm::StoreInst* inst = builder_->CreateAlignedStore(
        value,
        CreateBufferPtr(
            t,
            GetVarValue(op->buffer_var.get()),
            MakeValue(op->index)),
        data_layout_->getTypeAllocSize(value->getType()));
    AddAliasInfo(inst, op->buffer_var.get(), op->index);
  } else {
    LOG(FATAL) << "not yet supported";
  }
}

void CodeGenLLVM::Visit_(const Call* op) {
  if (op->is_intrinsic(intrinsic::tvm_call_packed)) {
    value_ = CreateCallPacked(op);
  } else if (op->call_type == Call::Intrinsic ||
             op->call_type == Call::PureIntrinsic) {
    value_ = CreateIntrinstic(op);
  } else {
    CHECK(op->call_type == Call::Extern ||
          op->call_type == Call::PureExtern);
    value_ = CreateCallExtern(op);
  }
}

llvm::Value* CodeGenLLVM::CreateIntrinstic(const Call* op) {
  if (op->is_intrinsic(Call::bitwise_and)) {
    CHECK_EQ(op->args.size(), 2U);
    return builder_->CreateAnd(
        MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    CHECK_EQ(op->args.size(), 2U);
    return builder_->CreateXor(
        MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    CHECK_EQ(op->args.size(), 2U);
    return builder_->CreateOr(
        MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    return builder_->CreateNot(MakeValue(op->args[0]));
  } else if (op->is_intrinsic(Call::shift_left)) {
    CHECK_EQ(op->args.size(), 2U);
    return builder_->CreateShl(
        MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->is_intrinsic(Call::shift_right)) {
    CHECK_EQ(op->args.size(), 2U);
    if (op->type.is_int()) {
      return builder_->CreateAShr(
          MakeValue(op->args[0]), MakeValue(op->args[1]));
    } else {
      return builder_->CreateLShr(
          MakeValue(op->args[0]), MakeValue(op->args[1]));
    }
  } else if (op->is_intrinsic(Call::address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    return CreateBufferPtr(
        l->type, GetVarValue(l->buffer_var.get()), MakeValue(l->index));
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    llvm::Value* ptr = MakeValue(op->args[0]);
    return builder_->CreateICmpEQ(
        ptr, llvm::Constant::getNullValue(ptr->getType()));
  } else if (op->is_intrinsic(intrinsic::tvm_api_load_arg)) {
    CHECK_EQ(op->args.size(), 3U);
    CHECK_EQ(op->type.lanes(), 1);
    llvm::Value* args = builder_->CreatePointerCast(
        MakeValue(op->args[0]), t_tvm_value_->getPointerTo());
    llvm::Value* ptr = builder_->CreateInBoundsGEP(
        args, MakeValue(op->args[2]));
    // always pass via 64 bit pointers
    // For handle type, Handle(64) will simply become 32 bit void*
    Type value_type = op->type.with_bits(64);
    ptr = builder_->CreatePointerCast(
        ptr, LLVMType(value_type)->getPointerTo());
    llvm::Value* value = builder_->CreateAlignedLoad(ptr, 8);
    // cast to the desired type
    if (value_type != op->type) {
      value = CreateCast(value_type, op->type, value);
    }
    return value;
  } else if (op->is_intrinsic(intrinsic::tvm_array_get_field)) {
    CHECK_EQ(op->args.size(), 2U);
    llvm::Value* arr = builder_->CreatePointerCast(
        MakeValue(op->args[0]), t_tvm_array_->getPointerTo());
    llvm::Constant* zero = ConstInt32(0);
    llvm::Value* ret = nullptr;
    switch (op->args[1].as<IntImm>()->value) {
      case intrinsic::kData: {
        ret = builder_->CreateInBoundsGEP(arr, {zero, ConstInt32(0)}); break;
      }
      case intrinsic::kShape: {
        ret = builder_->CreateInBoundsGEP(arr, {zero, ConstInt32(1)}); break;
      }
      case intrinsic::kStrides: {
        ret = builder_->CreateInBoundsGEP(arr, {zero, ConstInt32(2)}); break;
      }
      case intrinsic::kNDim: {
        ret = builder_->CreateInBoundsGEP(arr, {zero, ConstInt32(3)}); break;
      }
      case intrinsic::kTypeCode: {
        ret = builder_->CreateInBoundsGEP(
            arr, {zero, ConstInt32(4), ConstInt32(0)}); break;
      }
      case intrinsic::kTypeBits: {
        ret = builder_->CreateInBoundsGEP(
            arr, {zero, ConstInt32(4), ConstInt32(1)}); break;
      }
      case intrinsic::kTypeLanes: {
        ret = builder_->CreateInBoundsGEP(
            arr, {zero, ConstInt32(4), ConstInt32(2)}); break;
      }
      default: LOG(FATAL) << "unknown field code";
    }
    return builder_->CreateLoad(ret);
  } else {
    LOG(FATAL) << "Unknown intrinstic " << op->name;
  }
  return nullptr;
}

llvm::BasicBlock* CodeGenLLVM::CheckPackedCallSuccess(llvm::Value* retcode) {
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
void CodeGenLLVM::Visit_(const For* op) {
  using llvm::BasicBlock;
  BasicBlock* for_head = BasicBlock::Create(
      *ctx_, "for_head", function_);
  BasicBlock* for_body = BasicBlock::Create(
      *ctx_, "for_body", function_);
  BasicBlock* for_end = BasicBlock::Create(
      *ctx_, "for_end", function_);
  BasicBlock* pre_block = builder_->GetInsertBlock();
  CHECK(is_zero(op->min));
  Type t = op->min.type();
  llvm::Value* init = ConstInt32(0);
  llvm::Value* extent = MakeValue(op->extent);
  builder_->CreateBr(for_head);

  builder_->SetInsertPoint(for_head);
  llvm::PHINode* index = builder_->CreatePHI(LLVMType(t), 2);
  index->addIncoming(init, pre_block);
  llvm::Value* cond = CreateLT(t, index, extent);
  builder_->CreateCondBr(cond, for_body, for_end, md_very_likely_branch_);
  // body of for
  builder_->SetInsertPoint(for_body);
  var_map_[op->loop_var.get()] = index;
  this->Visit(op->body);
  llvm::Value* next_index = CreateAdd(t, index, ConstInt32(1));
  index->addIncoming(next_index, builder_->GetInsertBlock());
  builder_->CreateBr(for_head);
  // end of for
  builder_->SetInsertPoint(for_end);
}

void CodeGenLLVM::Visit_(const IfThenElse* op) {
  using llvm::BasicBlock;
  BasicBlock* then_block = BasicBlock::Create(
      *ctx_, "if_then", function_);
  BasicBlock* else_block = BasicBlock::Create(
      *ctx_, "if_else", function_);
  BasicBlock* end_block = BasicBlock::Create(
      *ctx_, "if_end", function_);
  if (!op->else_case.defined()) {
    else_block  = end_block;
  }
  // condition.
  llvm::Value* cond = MakeValue(op->condition);
  bool likely = true;
  if (likely) {
    builder_->CreateCondBr(cond, then_block, else_block, md_very_likely_branch_);
  } else {
    builder_->CreateCondBr(cond, then_block, else_block);
  }
  // then case.
  builder_->SetInsertPoint(then_block);
  this->Visit(op->then_case);
  builder_->CreateBr(end_block);
  // else case.
  if (op->else_case.defined()) {
    builder_->SetInsertPoint(else_block);
    this->Visit(op->else_case);
    builder_->CreateBr(end_block);
  }
  builder_->SetInsertPoint(end_block);
}

void CodeGenLLVM::Visit_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  llvm::Value* buf = nullptr;
  if (op->new_expr.defined()) {
    CHECK_EQ(op->free_function, "nop");
    buf = MakeValue(op->new_expr);
  } else {
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    buf = builder_->CreateAlloca(
        LLVMType(op->type), ConstInt32(constant_size));
  }
  buf = builder_->CreatePointerCast(buf, LLVMType(op->type)->getPointerTo());
  CHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;
}

void CodeGenLLVM::Visit_(const AttrStmt* op) {
  this->Visit(op->body);
}

void CodeGenLLVM::Visit_(const AssertStmt* op) {
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
  builder_->CreateCall(f_tvm_api_set_last_error_, {msg});
  builder_->CreateRet(llvm::ConstantInt::getSigned(t_int32_, -1));
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
}

void CodeGenLLVM::Visit_(const LetStmt* op) {
  llvm::Value* v = MakeValue(op->value);
  CHECK(!var_map_.count(op->var.get()));
  var_map_[op->var.get()] = v;
  this->Visit(op->body);
}

void CodeGenLLVM::AddAliasInfo(
    llvm::Instruction* inst, const Variable* buffer, Expr index) {
  int base = 0, width = 0;
  // create meta-data for alias analysis
  // Use a group of binary tree ranges.
  const Ramp* ramp = index.as<Ramp>();
  if (ramp) {
    int base, stride;
    if (arith::GetConstInt(ramp->base, &base) &&
        arith::GetConstInt(ramp->stride, &stride)) {
      int xwith = ramp->lanes * stride;
      width = 1;
      while (width < xwith) {
        width *= 2;
      }
      while (base % width) {
        base -= base % width;
        width *= 2;
      }
    }
  } else {
    if (arith::GetConstInt(index, &base)) width = 1;
  }

  llvm::MDNode* meta = md_tbaa_root_;
  std::ostringstream buffer_addr;
  buffer_addr << buffer;
  meta = md_builder_->createTBAAScalarTypeNode(buffer_addr.str(), meta);
  // create a tree-shape access structure.
  if (width != 0) {
    for (int w = 1024; w >= width; w /= 2) {
      int b = (base / w) * w;
      std::stringstream os;
      os << buffer << ".w" << w << ".b" << b;
      meta = md_builder_->createTBAAScalarTypeNode(os.str(), meta);
    }
  }
  inst->setMetadata(
      "tbaa",
      md_builder_->createTBAAStructTagNode(meta, meta, 0));
}

llvm::Value* CodeGenLLVM::CreateBroadcast(llvm::Value* value, int lanes) {
  llvm::Constant* init = llvm::UndefValue::get(
      llvm::VectorType::get(value->getType(), lanes));
  llvm::Constant* zero = ConstInt32(0);
  value = builder_->CreateInsertElement(init, value, zero);
  llvm::Constant* mask = llvm::ConstantVector::getSplat(lanes, zero);
  return builder_->CreateShuffleVector(value, init, mask);
}

llvm::Value* CodeGenLLVM::CreateBufferPtr(
    Type t, llvm::Value* buffer, llvm::Value* index) {
  llvm::Type* elem_type = buffer->getType();
  unsigned address_space = elem_type->getPointerAddressSpace();
  llvm::Type* load_type = LLVMType(t)->getPointerTo(address_space);

  if (load_type != elem_type) {
    buffer = builder_->CreatePointerCast(buffer, load_type);
  }
  llvm::Constant* cindex = llvm::dyn_cast<llvm::Constant>(index);
  if (cindex && cindex->isZeroValue()) {
    return buffer;
  }
  return builder_->CreateInBoundsGEP(buffer, index);
}

llvm::Value* CodeGenLLVM::CreateCast(Type from, Type to, llvm::Value* value) {
  llvm::Type * target = LLVMType(to);
  if (value->getType() == target) return value;
  if (from.is_handle() && from.is_handle()) {
    return builder_->CreateBitCast(value, target);
  } else if (!from.is_float() && !to.is_float()) {
    return builder_->CreateIntCast(value, target, from.is_int());
  } else if (from.is_float() && to.is_int()) {
    return builder_->CreateFPToSI(value, target);
  } else if (from.is_float() && to.is_uint()) {
    if (to.bits() < 8) {
      value = builder_->CreateFPToUI(value, LLVMType(to.with_bits(8)));
      return builder_->CreateIntCast(value, target, false);
    } else {
      return builder_->CreateFPToUI(value, target);
    }
  } else if (from.is_int() && to.is_float()) {
    return builder_->CreateSIToFP(value, target);
  } else if (from.is_uint() && to.is_float()) {
    return builder_->CreateUIToFP(value, target);
  } else {
    CHECK(from.is_float() && to.is_float());
    return builder_->CreateFPCast(value, target);
  }
}

llvm::Value* CodeGenLLVM::GetPackedFuncHandle(const std::string& fname) {
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
        llvm::GlobalValue::PrivateLinkage, 0, ".tvm_func");
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
  llvm::Value* ctx = builder_->CreateLoad(gv_mod_ctx_);
  llvm::Value* retcode = builder_->CreateCall(
      f_tvm_get_func_from_env_, {ctx, GetConstString(fname), out});
  init_block = CheckPackedCallSuccess(retcode);
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(out, align);
  builder_->CreateBr(end_block);
  // end block
  builder_->SetInsertPoint(end_block);
  llvm::PHINode* phi = builder_->CreatePHI(t_tvm_func_handle_, 2);
  phi->addIncoming(handle, pre_block);
  phi->addIncoming(loaded_handle, init_block);
  return phi;
}

llvm::Value* CodeGenLLVM::CreateCallPacked(const Call* op) {
  CHECK_GE(op->args.size(), 1U);
  std::string func_name = op->args[0].as<StringImm>()->value;
  llvm::Value* handle = GetPackedFuncHandle(func_name);
  // call the function
  unsigned nargs = static_cast<unsigned>(op->args.size() - 1);
  llvm::Value* targs = builder_->CreateAlloca(
      t_tvm_value_, ConstInt32(nargs));
  llvm::Value* tcodes = builder_->CreateAlloca(
      t_int_, ConstInt32(nargs));
  for (unsigned i = 0; i < nargs; ++i) {
    Expr expr = op->args[i + 1];
    Type t = expr.type();
    CHECK_EQ(t.lanes(), 1);
    // Always pass via 64 bit value.
    // For handle type, Handle(64) maps to 32 bit void* in 32bit platform.
    Type api_type = t.with_bits(64);
    llvm::Value* value = CreateCast(t, api_type, MakeValue(expr));
    llvm::Value* store_ptr = builder_->CreatePointerCast(
        builder_->CreateInBoundsGEP(targs, ConstInt32(i)),
        LLVMType(api_type)->getPointerTo());
    builder_->CreateAlignedStore(value,  store_ptr, 8);
    builder_->CreateAlignedStore(
        ConstInt32(t.code()),
        builder_->CreateInBoundsGEP(tcodes, ConstInt32(i)), 4);
  }
  llvm::Value* ret_value = builder_->CreateAlloca(t_tvm_value_);
  llvm::Value* ret_tcode = builder_->CreateAlloca(t_int_);
  CheckPackedCallSuccess(
      builder_->CreateCall(
          f_tvm_func_call_,
          {handle, targs, tcodes, ConstInt32(nargs), ret_value, ret_tcode}));
  Type r_type = op->type;
  Type r_api_type = op->type.with_bits(64);
  llvm::Value* rvalue =
      builder_->CreateAlignedLoad(
          builder_->CreatePointerCast(
              ret_value, LLVMType(r_api_type)->getPointerTo()), 8);
  rvalue = CreateCast(r_api_type, r_type, rvalue);
  return rvalue;
}

llvm::Value* CodeGenLLVM::CreateCallExtern(const Call* op) {
  std::vector<llvm::Value*> arg_values(op->args.size());
  for (size_t i = 0; i < op->args.size(); ++i) {
    arg_values[i] = MakeValue(op->args[i]);
  }
  if (op->type.is_scalar()) {
    llvm::Function* f = module_->getFunction(op->name);
    if (f) {
      return builder_->CreateCall(f, arg_values);
    } else {
      LOG(FATAL) << "cannot find function " << op->name;
    }
  } else {
    llvm::Function* f = module_->getFunction(op->name);
    if (f) {
      return CreateScalarizedCall(op, f, arg_values);
    } else {
      LOG(FATAL) << "cannot find function " << op->name;
    }
  }
  return nullptr;
}

llvm::Value* CodeGenLLVM::CreateScalarizedCall(
    const Call* op, llvm::Function* f, const std::vector<llvm::Value*>& args) {
  llvm::Value* value = llvm::UndefValue::get(LLVMType(op->type));
  for (int i = 0; i < op->type.lanes(); ++i) {
    std::vector<llvm::Value*> sargs(args.size());
    for (size_t j = 0; j < args.size(); ++j) {
      if (args[j]->getType()->isVectorTy()) {
        sargs[j] = builder_->CreateExtractElement(args[j], ConstInt32(i));
      } else {
        sargs[j] = args[j];
      }
    }
    llvm::CallInst* call = builder_->CreateCall(f, sargs);
    if (op->is_pure()) {
      call->setDoesNotAccessMemory();
    }
    call->setDoesNotThrow();
    if (!call->getType()->isVoidTy()) {
      value = builder_->CreateInsertElement(value, call, ConstInt32(i));
    }
  }
  return value;
}

llvm::Value* CodeGenLLVM::GetVarValue(const Variable* v) const {
  auto it = var_map_.find(v);
  CHECK(it != var_map_.end())
      << "Cannot find " << v->name_hint << " in the var map";
  return it->second;
}

llvm::Value* CodeGenLLVM::GetConstString(const std::string& str) {
  auto it = str_map_.find(str);
  if (it == str_map_.end()) {
    llvm::Type* type = llvm::ArrayType::get(t_char_, str.length() + 1);
    llvm::GlobalVariable *global = new llvm::GlobalVariable(
        *module_, type, true, llvm::GlobalValue::PrivateLinkage, 0, ".str");
    global->setAlignment(1);
    global->setInitializer(llvm::ConstantDataArray::getString(*ctx_, str));
    // useful constant value
    llvm::Constant* zero = ConstInt32(0);
    llvm::Constant* indices[] = {zero, zero};
    llvm::Constant* sptr = llvm::ConstantExpr::getGetElementPtr(
        type, global, indices);
    str_map_[str] = sptr;
    return sptr;
  } else {
    return it->second;
  }
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
