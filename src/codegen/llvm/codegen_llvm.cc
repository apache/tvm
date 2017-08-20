/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include "./codegen_llvm.h"
#include "./codegen_cpu.h"
#include "../../pass/ir_util.h"
#include "../../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

std::unique_ptr<CodeGenLLVM> CodeGenLLVM::Create(llvm::TargetMachine *tm) {
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

void CodeGenLLVM::Init(const std::string& module_name,
                       llvm::TargetMachine* tm,
                       llvm::LLVMContext* ctx,
                       bool system_lib,
                       bool dynamic_lookup) {
  InitializeLLVM();
  // clear maps
  var_map_.clear();
  str_map_.clear();
  ctx_ = ctx;
  t_void_ = llvm::Type::getVoidTy(*ctx);
  t_void_p_ = llvm::Type::getInt8Ty(*ctx)->getPointerTo();
  t_int_ = llvm::Type::getIntNTy(*ctx, sizeof(int) * 8);
  t_char_ = llvm::Type::getInt8Ty(*ctx);
  t_int8_ = llvm::Type::getInt8Ty(*ctx);
  t_int16_ = llvm::Type::getInt16Ty(*ctx);
  t_int32_ = llvm::Type::getInt32Ty(*ctx);
  t_int64_ = llvm::Type::getInt64Ty(*ctx);
  t_float64_ = llvm::Type::getDoubleTy(*ctx);
  md_builder_.reset(new llvm::MDBuilder(*ctx));
  md_very_likely_branch_ =
      md_builder_->createBranchWeights(1 << 30, 0);
  md_tbaa_root_ = md_builder_->createTBAARoot("tvmtbaa");
  md_tbaa_alias_set_ = md_builder_->createTBAAScalarTypeNode(
      "alias_set", md_tbaa_root_);
  // initialize Modules and function type
  module_.reset(new llvm::Module(module_name, *ctx));
  // initialize builder
  builder_.reset(new IRBuilder(*ctx));
  this->InitTarget(tm);
}

void CodeGenLLVM::InitTarget(llvm::TargetMachine* tm) {
  module_->setTargetTriple(tm->getTargetTriple().str());
  module_->setDataLayout(tm->createDataLayout());
  data_layout_.reset(new llvm::DataLayout(module_.get()));
  // initialize native vector bits
  std::string target = tm->getTarget().getName();
  if (target == "x86-64") {
    // for avx512
    native_vector_bits_ = 64 * 8;
  } else if (target == "x86") {
    native_vector_bits_ = 32 * 8;
  } else {
    if (native_vector_bits_ == 0) {
      native_vector_bits_ = 32 * 8;
      LOG(WARNING) << "set native vector to be " << native_vector_bits_ / 8
                   << " for target " << target;
    }
  }
}

void CodeGenLLVM::InitFuncState() {
  var_map_.clear();
  align_map_.clear();
  alloc_storage_info_.clear();
  alias_var_set_.clear();
}

void CodeGenLLVM::AddFunction(const LoweredFunc& f) {
  this->InitFuncState();
  is_restricted_ = f->is_restricted;
  CHECK(!module_->getFunction(f->name))
      << "Function " << f->name << "already exists in module";
  std::vector<llvm::Type*> arg_type;
  for (Var arg : f->args) {
    Type t = arg.type();
    if (t.is_handle() && f->handle_data_type.count(arg)) {
      arg_type.push_back(
          LLVMType(f->handle_data_type[arg].type())->getPointerTo());
      if (!is_restricted_) {
        alias_var_set_.insert(arg.get());
      }
    } else {
      arg_type.push_back(LLVMType(t));
    }
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(t_int_, arg_type, false);
  // setup the function.
  function_ = llvm::cast<llvm::Function>(module_->getOrInsertFunction(f->name, ftype));
  function_->setCallingConv(llvm::CallingConv::C);
  // set handle argument to be non alias.
  if (is_restricted_) {
    for (size_t i = 0; i < f->args.size(); ++i) {
      if (f->args[i].type().is_handle()) {
#if TVM_LLVM_VERSION >= 50
        function_->addParamAttr(i, llvm::Attribute::NoAlias);
#else
        function_->setDoesNotAlias(i + 1);
#endif
      }
    }
  }

  size_t idx = 0;
  for (auto it = function_->arg_begin();
      it != function_->arg_end(); ++it, ++idx) {
    llvm::Argument* v = &(*it);
    var_map_[f->args[idx].get()] = v;
  }

  llvm::BasicBlock* block = llvm::BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(block);
  this->VisitStmt(f->body);
  builder_->CreateRet(ConstInt32(0));
}

void CodeGenLLVM::AddMainFunction(const std::string& entry_func_name) {
  LOG(FATAL) << "Donot support add main function";
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

#if TVM_LLVM_VERSION >= 50
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
#else
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0);
#endif
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
  this->AddStartupFunction();
  this->Optimize();
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


void CodeGenLLVM::AddAliasInfo(
    llvm::Instruction* inst, const Variable* buffer, Expr index, Type t) {
  if (alias_var_set_.count(buffer) != 0) {
    // Mark all possibly aliased pointer as same type.
    llvm::MDNode* meta = md_tbaa_alias_set_;
    inst->setMetadata(
        "tbaa",
        md_builder_->createTBAAStructTagNode(meta, meta, 0));
    return;
  }
  int base = 0, width = 0;
  // create meta-data for alias analysis
  // Use a group of binary tree ranges.
  if (index.defined()) {
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
  }
  llvm::MDNode* meta = md_tbaa_root_;
  std::ostringstream buffer_addr, buffer_type;
  buffer_addr << buffer;
  meta = md_builder_->createTBAAScalarTypeNode(buffer_addr.str(), meta);
  buffer_type << t.element_of();
  meta = md_builder_->createTBAAScalarTypeNode(buffer_type.str(), meta);
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
  llvm::Constant* undef = llvm::UndefValue::get(
      llvm::VectorType::get(value->getType(), lanes));
  llvm::Constant* zero = ConstInt32(0);
  value = builder_->CreateInsertElement(undef, value, zero);
  llvm::Constant* mask = llvm::ConstantVector::getSplat(lanes, zero);
  return builder_->CreateShuffleVector(value, undef, mask);
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

llvm::Value* CodeGenLLVM::CreateCallExtern(const Call* op) {
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
  llvm::Function* f = module_->getFunction(op->name);
  if (f == nullptr) {
    f = llvm::Function::Create(
        ftype, llvm::Function::ExternalLinkage, op->name, module_.get());
  }
  return builder_->CreateCall(f, arg_values);
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

void CodeGenLLVM::CreateSerialFor(llvm::Value* begin,
                                  llvm::Value* end,
                                  llvm::Value* stride,
                                  const VarExpr& loop_var, const Stmt& body) {
  using llvm::BasicBlock;
  Type t = loop_var.type();
  BasicBlock* for_head = BasicBlock::Create(
      *ctx_, "for_head", function_);
  BasicBlock* for_body = BasicBlock::Create(
      *ctx_, "for_body", function_);
  BasicBlock* for_end = BasicBlock::Create(
      *ctx_, "for_end", function_);
  BasicBlock* pre_block = builder_->GetInsertBlock();
  builder_->CreateBr(for_head);
  builder_->SetInsertPoint(for_head);
  llvm::PHINode* index = builder_->CreatePHI(begin->getType(), 2);
  index->addIncoming(begin, pre_block);
  llvm::Value* cond = CreateLT(t, index, end);
  builder_->CreateCondBr(cond, for_body, for_end, md_very_likely_branch_);
  // body of for
  builder_->SetInsertPoint(for_body);
  var_map_[loop_var.get()] = index;
  this->VisitStmt(body);
  llvm::Value* next_index = CreateAdd(t, index, stride);
  index->addIncoming(next_index, builder_->GetInsertBlock());
  builder_->CreateBr(for_head);
  // end of for
  builder_->SetInsertPoint(for_end);
}

llvm::Value* CodeGenLLVM::CreateIntrinsic(const Call* op) {
  if (op->is_intrinsic("llvm_intrin")) {
    CHECK_GE(op->args.size(), 1U);
    std::vector<llvm::Value*> arg_values;
    std::vector<llvm::Type*> arg_types;
    for (size_t i = 1; i < op->args.size(); ++i) {
      llvm::Value* v = MakeValue(op->args[i]);
      arg_values.push_back(v);
      arg_types.push_back(v->getType());
    }
    auto id = static_cast<llvm::Intrinsic::ID>(op->args[0].as<UIntImm>()->value);
    llvm::Function* f = llvm::Intrinsic::getDeclaration(
        module_.get(), id, arg_types);
    return builder_->CreateCall(f, arg_values);
  } else if (op->is_intrinsic("llvm_builtin")) {
    CHECK_GE(op->args.size(), 1U);
    std::vector<llvm::Value*> arg_values;
    for (size_t i = 1; i < op->args.size(); ++i) {
      llvm::Value* v = MakeValue(op->args[i]);
      arg_values.push_back(v);
    }
    auto id = static_cast<llvm::Intrinsic::ID>(op->args[0].as<UIntImm>()->value);
    llvm::Function* f = llvm::Intrinsic::getDeclaration(module_.get(), id);
    return builder_->CreateCall(f, arg_values);
  } else if (op->is_intrinsic(Call::bitwise_and)) {
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
  } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    return builder_->CreatePointerCast(
        CreateBufferPtr(
            l->type, GetVarValue(l->buffer_var.get()), MakeValue(l->index)),
        t_void_p_);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    llvm::Value* ptr = MakeValue(op->args[0]);
    return builder_->CreateICmpEQ(
        ptr, llvm::Constant::getNullValue(ptr->getType()));
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    using llvm::BasicBlock;
    CHECK_EQ(op->args.size(), 3U);
    llvm::Value* cond = MakeValue(op->args[0]);
    BasicBlock* then_block = BasicBlock::Create(
        *ctx_, "if_then", function_);
    BasicBlock* else_block = BasicBlock::Create(
        *ctx_, "if_else", function_);
    BasicBlock* end_block = BasicBlock::Create(
        *ctx_, "if_end", function_);
    builder_->CreateCondBr(cond, then_block, else_block);
    // Then
    builder_->SetInsertPoint(then_block);
    llvm::Value* then_value = MakeValue(op->args[1]);
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(else_block);
    // else
    llvm::Value* else_value = MakeValue(op->args[2]);
    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(end_block);
    // phi
    llvm::PHINode* phi = builder_->CreatePHI(then_value->getType(), 2);
    phi->addIncoming(then_value, then_block);
    phi->addIncoming(else_value, else_block);
    return phi;
  } else if (op->is_intrinsic(Call::reinterpret) && is_zero(op->args[0])) {
    return llvm::Constant::getNullValue(t_void_p_);
  } else {
    LOG(FATAL) << "Unknown intrinstic " << op->name;
  }
  return nullptr;
}

int CodeGenLLVM::NativeVectorBits(const std::string& storage_scope) const {
  // By default, we ask the buffer to be aligned to 64 bytes
  return native_vector_bits_;
}

void CodeGenLLVM::GetAlignment(
    Type t, const Variable* buf_var, const Expr& index,
    int* p_alignment, int* p_native_bits) {
  int& alignment = *p_alignment;
  int& native_bits = *p_native_bits;
  // The storage scope.
  StorageInfo info;
  auto it = alloc_storage_info_.find(buf_var);
  if (it != alloc_storage_info_.end()) {
    info = it->second;
  }
  arith::ModularEntry m = EvalModular(index, align_map_);
  native_bits = NativeVectorBits(info.scope);
  alignment = t.element_of().bits();
  // find alignment, cannot exceed allocated alignment
  int max_align_bits = std::min(
      info.alignment * 8, alignment * t.lanes());
  while ((m.coeff & 1) == 0 &&
         (m.base & 1) == 0 &&
         alignment < max_align_bits &&
         alignment < native_bits) {
    m.coeff /= 2;
    m.base /= 2;
    alignment *= 2;
  }
  CHECK_EQ(alignment % 8, 0)
      << "Load from memory that does not align to 8 bits";
  alignment /= 8;
}

// visitor overrides
llvm::Value* CodeGenLLVM::VisitExpr_(const Variable* op) {
  return GetVarValue(op);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Cast* op) {
  return CreateCast(op->value.type(), op->type, MakeValue(op->value));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const IntImm* op) {
  return llvm::ConstantInt::getSigned(LLVMType(op->type), op->value);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const UIntImm* op) {
  return llvm::ConstantInt::get(LLVMType(op->type), op->value);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const FloatImm* op) {
  return llvm::ConstantFP::get(LLVMType(op->type), op->value);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const StringImm* op) {
  return GetConstString(op->value);
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

llvm::Value* CodeGenLLVM::VisitExpr_(const Add* op) {
  return CreateAdd(op->type, MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Sub* op) {
  return CreateSub(op->type, MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Mul* op) {
  return CreateMul(op->type, MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Div* op) {
  llvm::Value* a = MakeValue(op->a);
  int shift;
  if (op->type.is_float()) {
    return builder_->CreateFDiv(a, MakeValue(op->b));
  } else if ((op->type.is_int() || op->type.is_uint()) &&
             is_const_power_of_two_integer(op->b, &shift)) {
    return builder_->CreateAShr(a, shift);
  } else {
    llvm::Value* b = MakeValue(op->b);
    if (op->type.is_int()) {
      return builder_->CreateSDiv(a, b);
    } else {
      CHECK(op->type.is_uint());
      return builder_->CreateUDiv(a, b);
    }
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Mod* op) {
  CHECK(!op->type.is_float())
      << "Cannot do mod for float";
  if (op->type.is_int()) {
    return builder_->CreateSRem(MakeValue(op->a), MakeValue(op->b));
  } else {
    CHECK(op->type.is_uint());
    return builder_->CreateURem(MakeValue(op->a), MakeValue(op->b));
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Min* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  llvm::Value* cond = CreateLT(op->a.type(), a, b);
  return builder_->CreateSelect(cond, a, b);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Max* op) {
  llvm::Value* a = MakeValue(op->a);
  llvm::Value* b = MakeValue(op->b);
  llvm::Value* cond = CreateGT(op->a.type(), a, b);
  return builder_->CreateSelect(cond, a, b);
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

llvm::Value* CodeGenLLVM::VisitExpr_(const LT* op) {
  return CreateLT(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
llvm::Value* CodeGenLLVM::VisitExpr_(const LE* op) {
  return CreateLE(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
llvm::Value* CodeGenLLVM::VisitExpr_(const GT* op) {
  return CreateGT(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}
llvm::Value* CodeGenLLVM::VisitExpr_(const GE* op) {
  return CreateGE(op->a.type(), MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const EQ* op) {
  if (op->a.type().is_float()) {
    return builder_->CreateFCmpOEQ(MakeValue(op->a), MakeValue(op->b));
  } else {
    return builder_->CreateICmpEQ(MakeValue(op->a), MakeValue(op->b));
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const NE* op) {
  if (op->a.type().is_float()) {
    return builder_->CreateFCmpONE(MakeValue(op->a), MakeValue(op->b));
  } else {
    return builder_->CreateICmpNE(MakeValue(op->a), MakeValue(op->b));
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const And* op) {
  return builder_->CreateAnd(MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Or* op) {
  return builder_->CreateOr(MakeValue(op->a), MakeValue(op->b));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Not* op) {
  return builder_->CreateNot(MakeValue(op->a));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Select* op) {
  return builder_->CreateSelect(
      MakeValue(op->condition),
      MakeValue(op->true_value),
      MakeValue(op->false_value));
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Let* op) {
  llvm::Value* v = MakeValue(op->value);
  CHECK(!var_map_.count(op->var.get()));
  CHECK(!align_map_.count(op->var.get()));
  var_map_[op->var.get()] = v;
  align_map_[op->var.get()] = arith::EvalModular(op->value, align_map_);
  return MakeValue(op->body);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Broadcast* op) {
  return CreateBroadcast(MakeValue(op->value), op->lanes);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Ramp* op) {
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
  return value;
}

void CodeGenLLVM::Scalarize(
    const Expr& e,
    std::function<void(int i, llvm::Value* v)> f) {
  const Ramp* ramp = e.as<Ramp>();
  Type t = e.type();
  if (ramp) {
    for (int i = 0; i < t.lanes(); ++i) {
      Expr offset = arith::ComputeExpr<Add>(
          ramp->base,
          arith::ComputeExpr<Mul>(ramp->stride, i));
      f(i, MakeValue(offset));
    }
  } else {
    llvm::Value* index = MakeValue(e);
    for (int i = 0; i < t.lanes(); ++i) {
      f(i, builder_->CreateExtractElement(index, ConstInt32(i)));
    }
  }
}

llvm::Value* CodeGenLLVM::CreateVecFlip(llvm::Value* vec) {
  int lanes = static_cast<int>(vec->getType()->getVectorNumElements());
  std::vector<llvm::Constant*> indices;
  for (int i = lanes; i != 0; --i) {
    indices.push_back(ConstInt32(i - 1));
  }
  llvm::Constant* undef = llvm::UndefValue::get(vec->getType());
  return builder_->CreateShuffleVector(
      vec, undef, llvm::ConstantVector::get(indices));
}

llvm::Value* CodeGenLLVM::CreateVecSlice(
    llvm::Value* vec, int begin, int lanes) {
  int total_lanes = static_cast<int>(vec->getType()->getVectorNumElements());
  CHECK_LE(begin + lanes, total_lanes);
  if (lanes == total_lanes && begin == 0) return vec;
  std::vector<llvm::Constant*> indices;
  for (int i = 0; i < lanes; ++i) {
    indices.push_back(ConstInt32(begin + i));
  }
  llvm::Constant* undef = llvm::UndefValue::get(vec->getType());
  return builder_->CreateShuffleVector(
      vec, undef, llvm::ConstantVector::get(indices));
}

llvm::Value* CodeGenLLVM::CreateVecPad(llvm::Value* vec, int target_lanes) {
  int lanes = static_cast<int>(vec->getType()->getVectorNumElements());
  if (target_lanes == lanes) return vec;
  CHECK_GT(target_lanes, lanes);
  int pad_lanes = target_lanes - lanes;
  llvm::Constant* undef = llvm::UndefValue::get(
      llvm::VectorType::get(vec->getType()->getVectorElementType(), pad_lanes));
  std::vector<llvm::Constant*> indices;
  for (int i = 0; i < target_lanes; ++i) {
    indices.push_back(ConstInt32(i));
  }
  return builder_->CreateShuffleVector(
      vec, undef, llvm::ConstantVector::get(indices));
}

llvm::Value* CodeGenLLVM::CreateVecConcat(
    std::vector<llvm::Value*> vec) {
  CHECK_NE(vec.size(), 0U);
  int target_lanes = 0;
  for (llvm::Value* v : vec) {
    target_lanes += static_cast<int>(v->getType()->getVectorNumElements());
  }
  // tree shape merging
  while (vec.size() != 1) {
    std::vector<llvm::Value*> merged;
    for (size_t i = 0; i < vec.size() - 1; i += 2) {
      llvm::Value* v1 = vec[i];
      llvm::Value* v2 = vec[i + 1];
      int w1 = static_cast<int>(v1->getType()->getVectorNumElements());
      int w2 = static_cast<int>(v2->getType()->getVectorNumElements());
      int w = std::max(w1, w2);
      v1 = CreateVecPad(v1, w);
      v2 = CreateVecPad(v2, w);
      std::vector<llvm::Constant*> indices;
      for (int i = 0; i < w * 2; ++i) {
        indices.push_back(ConstInt32(i));
      }
      merged.push_back(
          builder_->CreateShuffleVector(
              v1, v2, llvm::ConstantVector::get(indices)));
    }
    if (vec.size() % 2 == 1) {
      merged.push_back(vec.back());
    }
    vec = merged;
  }
  return CreateVecSlice(vec[0], 0, target_lanes);
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Load* op) {
  CHECK(is_one(op->predicate))
      << "Predicated Load is not supported";
  Type t = op->type;
  const Ramp* ramp = op->index.as<Ramp>();
  llvm::Value* buf = GetVarValue(op->buffer_var.get());
  if (t.is_scalar()) {
    llvm::LoadInst* inst = builder_->CreateAlignedLoad(
        CreateBufferPtr(t, buf, MakeValue(op->index)),
        data_layout_->getTypeAllocSize(LLVMType(t)));
    AddAliasInfo(inst, op->buffer_var.get(), op->index, op->type);
    return inst;
  } else if (ramp && is_one(ramp->stride)) {
    int alignment, native_bits;
    GetAlignment(t, op->buffer_var.get(), ramp->base,
                 &alignment, &native_bits);
    int total_lanes = t.lanes();
    int step = native_bits / t.bits();

    std::vector<llvm::Value*> loads;
    for (int offset = 0; offset < total_lanes; offset += step) {
      int lanes = std::min(step, total_lanes - offset);
      Expr base = arith::ComputeExpr<Add>(
          ramp->base, make_const(ramp->base.type(), offset));
      llvm::Value* ptr = CreateBufferPtr(t.element_of(), buf, MakeValue(base));
      llvm::Type* vtype = llvm::VectorType::get(
          LLVMType(t.element_of()), lanes)->getPointerTo();
      llvm::LoadInst* inst = builder_->CreateAlignedLoad(
          builder_->CreatePointerCast(ptr, vtype), alignment);
      AddAliasInfo(inst, op->buffer_var.get(),
                   Ramp::make(base, make_const(base.type(), 1), lanes), op->type);
      loads.push_back(inst);
    }
    return CreateVecConcat(loads);
  } else if (ramp && is_const(ramp->stride, 2)) {
    int alignment, native_bits;
    GetAlignment(t, op->buffer_var.get(), ramp->base,
                 &alignment, &native_bits);
    arith::ModularEntry e = arith::EvalModular(ramp->base, align_map_);
    Type bt = ramp->base.type();
    int first_shift, next_shift;
    // If it is even base, and native alignments is bigger than twice
    // of the type, to ensure safe loading.
    if (e.coeff % 2  == 0 &&
        e.base % 2 == 0 &&
        native_bits >= t.bits() * 2) {
      first_shift = 0;
      next_shift = 0;
    } else if (e.coeff % 2  == 0 && e.base % 2 == 1) {
      // odd base, shift both to left.
      first_shift = -1;
      next_shift = -1;
    } else {
      // save option, right part, safe option.
      first_shift = 0;
      next_shift = -1;
    }
    llvm::Value* first = MakeValue(Load::make(
        t, op->buffer_var,
        Ramp::make(arith::ComputeExpr<Add>(
            ramp->base, make_const(bt, first_shift)),
                   make_const(bt, 1), ramp->lanes),
        const_true(t.lanes())));
    llvm::Value* next = MakeValue(Load::make(
        t, op->buffer_var,
        Ramp::make(arith::ComputeExpr<Add>(
            ramp->base, make_const(bt, ramp->lanes + next_shift)),
                   make_const(bt, 1), ramp->lanes),
        const_true(t.lanes())));
    // shuffle
    std::vector<llvm::Constant*> indices;
    int target_index = 0;
    for (int i = 0; i < ramp->lanes; ++i) {
      int idx = first_shift + i;
      if (idx == target_index) {
        indices.push_back(ConstInt32(i));
        target_index += 2;
      }
    }
    for (int i = 0; i < ramp->lanes; ++i) {
      int idx = ramp->lanes + next_shift + i;
      if (idx == target_index) {
        indices.push_back(ConstInt32(i + ramp->lanes));
        target_index += 2;
      }
    }
    CHECK_EQ(indices.size(), static_cast<size_t>(ramp->lanes));
    return builder_->CreateShuffleVector(
        first, next, llvm::ConstantVector::get(indices));
  } else if (ramp && is_const(ramp->stride, -1)) {
      int lanes = ramp->type.lanes();
      Expr neg_ramp =  Ramp::make(
          arith::ComputeExpr<Sub>(
              ramp->base,
              make_const(ramp->base.type(), lanes - 1)),
          make_const(ramp->base.type(), 1),
          lanes);
    // load value then flip
    llvm::Value* v = MakeValue(
        Load::make(t, op->buffer_var, neg_ramp, const_true(t.lanes())));
    return CreateVecFlip(v);
  } else {
    llvm::Value* ret = llvm::UndefValue::get(LLVMType(t));
    Scalarize(op->index, [&](int i, llvm::Value* offset) {
        llvm::Value* ptr = CreateBufferPtr(t.element_of(), buf, offset);
        llvm::LoadInst* inst = builder_->CreateAlignedLoad(
            ptr, data_layout_->getTypeAllocSize(LLVMType(t)));
        AddAliasInfo(inst, op->buffer_var.get(), Expr(), op->type);
        ret = builder_->CreateInsertElement(ret, inst, ConstInt32(i));
      });
    return ret;
  }
}

// stmts
void CodeGenLLVM::VisitStmt_(const Store* op) {
  CHECK(is_one(op->predicate))
      << "Predicated Load is not supported";
  llvm::Value* value = MakeValue(op->value);
  Type t = op->value.type();
  const Ramp* ramp = op->index.as<Ramp>();
  llvm::Value* buf = GetVarValue(op->buffer_var.get());

  if (t.is_scalar()) {
    llvm::StoreInst* inst = builder_->CreateAlignedStore(
        value,
        CreateBufferPtr(t, buf, MakeValue(op->index)),
        data_layout_->getTypeAllocSize(value->getType()));
    AddAliasInfo(inst, op->buffer_var.get(), op->index, op->value.type());
  } else if (ramp && is_one(ramp->stride)) {
    int alignment, native_bits;
    GetAlignment(t, op->buffer_var.get(), ramp->base,
                 &alignment, &native_bits);
    int total_lanes = t.lanes();
    int step = native_bits / t.bits();
    // vector store.
    for (int offset = 0; offset < total_lanes; offset += step) {
      int lanes = std::min(step, total_lanes - offset);
      Expr base = arith::ComputeExpr<Add>(
          ramp->base, make_const(ramp->base.type(), offset));
      llvm::Value* ptr = CreateBufferPtr(t.element_of(), buf, MakeValue(base));
      llvm::Type* vtype = llvm::VectorType::get(
          LLVMType(t.element_of()), lanes)->getPointerTo();
      llvm::StoreInst* inst = builder_->CreateAlignedStore(
          CreateVecSlice(value, offset, lanes),
          builder_->CreatePointerCast(ptr, vtype), alignment);
      AddAliasInfo(inst, op->buffer_var.get(),
                   Ramp::make(base, make_const(base.type(), 1), lanes), op->value.type());
    }
  } else {
    Scalarize(op->index, [&](int i, llvm::Value* offset) {
        llvm::Value* ptr = CreateBufferPtr(t.element_of(), buf, offset);
        llvm::StoreInst* inst = builder_->CreateAlignedStore(
            builder_->CreateExtractElement(value, ConstInt32(i)),
            ptr, data_layout_->getTypeAllocSize(LLVMType(t)));
        AddAliasInfo(inst, op->buffer_var.get(), Expr(), op->value.type());
      });
  }
}

llvm::Value* CodeGenLLVM::VisitExpr_(const Call* op) {
  if (op->call_type == Call::Intrinsic ||
             op->call_type == Call::PureIntrinsic) {
    return CreateIntrinsic(op);
  } else {
    CHECK(op->call_type == Call::Extern ||
          op->call_type == Call::PureExtern);
    return CreateCallExtern(op);
  }
}

void CodeGenLLVM::VisitStmt_(const For* op) {
  CHECK(is_zero(op->min));
  if (op->for_type == ForType::Serial) {
    CreateSerialFor(ConstInt32(0),
                    MakeValue(op->extent),
                    ConstInt32(1),
                    op->loop_var,
                    op->body);
  } else {
    LOG(FATAL) << "cannot handle for type " << op->for_type;
  }
}

void CodeGenLLVM::VisitStmt_(const IfThenElse* op) {
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
  this->VisitStmt(op->then_case);
  builder_->CreateBr(end_block);
  // else case.
  if (op->else_case.defined()) {
    builder_->SetInsertPoint(else_block);
    this->VisitStmt(op->else_case);
    builder_->CreateBr(end_block);
  }
  builder_->SetInsertPoint(end_block);
}

void CodeGenLLVM::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  llvm::Value* buf = nullptr;
  if (op->new_expr.defined()) {
    CHECK_EQ(op->free_function, "nop");
    buf = MakeValue(op->new_expr);
  } else {
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    llvm::AllocaInst* alloca = builder_->CreateAlloca(
        LLVMType(op->type), ConstInt32(constant_size));
    buf = alloca;
    StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];
    // Align stack to be TempAllocaAlignment.
    // TODO(tqchen) have pass to detect vector access and pre-set alignment
    if (constant_size % 4 == 0 && info.alignment == 0) {
      info.alignment = GetTempAllocaAlignment(op->type, constant_size);
    }
    if (alloca->getAlignment() < static_cast<uint32_t>(info.alignment)) {
      alloca->setAlignment(info.alignment);
    }
    info.alignment = alloca->getAlignment();
  }
  buf = builder_->CreatePointerCast(buf, LLVMType(op->type)->getPointerTo());
  CHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::storage_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_info_[v].scope = op->value.as<StringImm>()->value;
    this->VisitStmt(op->body);
  } else if (op->attr_key == ir::attr::storage_alignment) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_info_[v].alignment =
        static_cast<int>(op->value.as<IntImm>()->value);
    this->VisitStmt(op->body);
  } else {
    this->VisitStmt(op->body);
  }
}

void CodeGenLLVM::VisitStmt_(const AssertStmt* op) {
  // Detect useful invariant pattern and use them to visit child.
  // Pattern: Var % const  == 0
  // TODO(tqchen) move these pattern to a generic scope info visitor.
  if (const EQ* eq = op->condition.as<EQ>()) {
    const Mod* mod = eq->a.as<Mod>();
    int64_t factor, offset;
    if (mod && arith::GetConst(eq->b, &offset)) {
      const Variable *var = mod->a.as<Variable>();
      if (var && arith::GetConst(mod->b, &factor)) {
        arith::ModularEntry old = align_map_[var];
        if (factor > old.coeff) {
          arith::ModularEntry e;
          e.coeff = static_cast<int>(factor);
          e.base = static_cast<int>(offset);
          // new alignment info,
          align_map_[var] = e;
          this->VisitStmt(op->body);
          // restore old info
          align_map_[var] = old;
          return;
        }
      }
    }
  }
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const LetStmt* op) {
  llvm::Value* v = MakeValue(op->value);
  CHECK(!var_map_.count(op->var.get()));
  CHECK(!align_map_.count(op->var.get()));
  if (op->var.type().is_handle()) {
    if (!is_restricted_) {
      alias_var_set_.insert(op->var.get());
    }
  }
  var_map_[op->var.get()] = v;
  align_map_[op->var.get()] = arith::EvalModular(op->value, align_map_);
  this->VisitStmt(op->body);
}

void CodeGenLLVM::VisitStmt_(const Block* op) {
  VisitStmt(op->first);
  if (op->rest.defined()) VisitStmt(op->rest);
}

void CodeGenLLVM::VisitStmt_(const Evaluate *op) {
  MakeValue(op->value);
}

void CodeGenLLVM::VisitStmt_(const ProducerConsumer* op) {
  VisitStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
