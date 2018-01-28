/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_stack_vm.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <limits>
#include "./codegen_stack_vm.h"

namespace tvm {
namespace codegen {

using namespace ir;

StackVM CodeGenStackVM::Compile(LoweredFunc f) {
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    int vid = AllocVarID(v.get());
    CHECK_EQ(static_cast<size_t>(vid), i);
  }
  this->Push(f->body);
  return std::move(vm_);
}

void CodeGenStackVM::Push(const Stmt& n) {
  VisitStmt(n);
  if (debug_) {
    this->PushOp(StackVM::ASSERT_SP, 0);
  }
}

void CodeGenStackVM::PushOp(StackVM::OpCode opcode) {
  StackVM::Code code;
  code.op_code = opcode;
  vm_.code.push_back(code);
}

void CodeGenStackVM::SetOperand(int64_t operand_index, int64_t operand) {
  CHECK(operand >= std::numeric_limits<int>::min() &&
        operand <= std::numeric_limits<int>::max());
  vm_.code.at(operand_index).v_int = static_cast<int>(operand);
}

int64_t CodeGenStackVM::PushOp(StackVM::OpCode opcode, int operand) {
  int64_t pc = static_cast<int64_t>(vm_.code.size());
  StackVM::Code code;
  code.op_code = opcode;
  vm_.code.push_back(code);
  code.v_int = operand;
  vm_.code.push_back(code);
  return pc + 1;
}

int CodeGenStackVM::GetStrID(const std::string& key) {
  auto it = str_idmap_.find(key);
  if (it != str_idmap_.end()) return it->second;
  int sid = static_cast<int>(vm_.str_data.size());
  vm_.str_data.push_back(key);
  str_idmap_[key] = sid;
  return sid;
}

int CodeGenStackVM::AllocVarID(const Variable* v) {
  CHECK(!var_idmap_.count(v));
  int vid = static_cast<int>(vm_.heap_size);
  CHECK_EQ(vm_.heap_size, var_idmap_.size());
  vm_.heap_id_name.push_back(v->name_hint);
  ++vm_.heap_size;
  var_idmap_[v] = vid;
  return vid;
}

int CodeGenStackVM::GetVarID(const Variable* v) const {
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
      << "Find undefined Variable " << v->name_hint;
  return it->second;
}

void CodeGenStackVM::VisitExpr_(const Load* op) {
  this->Push(op->buffer_var);
  StackVM::OpCode code = StackVM::GetLoad(Type2TVMType(op->type));
  if (const IntImm* index = op->index.as<IntImm>()) {
    this->PushOp(code, index->value);
  } else {
    this->Push(op->index);
    this->PushOp(StackVM::PUSH_I64, op->type.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
    this->PushOp(code, 0);
  }
}

void CodeGenStackVM::VisitStmt_(const Store* op) {
  this->Push(op->buffer_var);
  StackVM::OpCode code = StackVM::GetStore(Type2TVMType(op->value.type()));
  if (const IntImm* index = op->index.as<IntImm>()) {
    this->Push(op->value);
    this->PushOp(code, index->value);
  } else {
    this->Push(op->index);
    this->PushOp(StackVM::PUSH_I64, op->value.type().element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
    this->Push(op->value);
    this->PushOp(code, 0);
  }
}

void CodeGenStackVM::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  int vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    this->Push(op->new_expr);
    this->PushOp(StackVM::STORE_HEAP, vid);
  } else {
    LOG(FATAL) << "Dynamic allocation not supported";
  }
}

void CodeGenStackVM::VisitExpr_(const Call* op) {
  if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    this->PushOp(StackVM::LOAD_HEAP, GetVarID(l->buffer_var.get()));
    this->Push(l->index);
    this->PushOp(StackVM::PUSH_I64, l->type.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
  } else if (op->is_intrinsic(Call::reinterpret)) {
    this->Push(op->args[0]);
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImm>()->value;
    this->Push(op->args[0]);
    const IntImm* index = op->args[1].as<IntImm>();
    CHECK(index != nullptr);
    StackVM::Code code;
    code.op_code = StackVM::TVM_STRUCT_GET;
    vm_.code.push_back(code);
    code.v_int = index->value;
    vm_.code.push_back(code);
    code.v_int = kind;
    vm_.code.push_back(code);
  } else if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    CHECK_GE(op->args.size(), 5U);
    const StringImm* s = op->args[0].as<StringImm>();
    CHECK(s != nullptr) << "tvm_call_global expect first argument as function name";
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    int begin = op->args[3].as<IntImm>()->value;
    int end = op->args[4].as<IntImm>()->value;
    // find the fuction id.
    const std::string& func_name = s->value;
    auto it = extern_fun_idmap_.find(func_name);
    int fid;
    if (it != extern_fun_idmap_.end()) {
      fid = it->second;
    } else {
      fid = static_cast<int>(vm_.extern_func_name.size());
      vm_.extern_func_name.push_back(func_name);
      extern_fun_idmap_[func_name] = fid;
    }
    // CALL_PACKED_FUNC
    StackVM::Code code;
    code.op_code = StackVM::CALL_PACKED_LOWERED;
    vm_.code.push_back(code);
    code.v_int = fid;
    vm_.code.push_back(code);
    code.v_int = begin;
    vm_.code.push_back(code);
    code.v_int = end;
    vm_.code.push_back(code);
  } else if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
    CHECK_EQ(op->args.size(), 2U);
    const std::string& type = op->args[0].as<StringImm>()->value;
    const IntImm* num = op->args[1].as<IntImm>();
    CHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(TVMArray) == 0, "invariant");
    // static_assert(alignof(TVMValue) % alignof(tvm_index_t) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(TVMArray) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    // add stack size to be safe.
    vm_.stack_size += size;
    this->PushOp(StackVM::TVM_STACK_ALLOCA_BY_8BYTE, static_cast<int>(size));
  } else if (op->name == "TVMBackendAllocWorkspace") {
    CHECK_EQ(op->args.size(), 5U);
    this->Push(op->args[0]);
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    this->Push(op->args[3]);
    this->Push(op->args[4]);
    this->PushOp(StackVM::TVM_DEVICE_ALLOCA);
  } else if (op->name == "TVMBackendFreeWorkspace") {
    CHECK_EQ(op->args.size(), 3U);
    this->Push(op->args[0]);
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    this->PushOp(StackVM::TVM_DEVICE_FREE);
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    this->PushOp(StackVM::TVM_THROW_LAST_ERROR);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    this->Push(op->args[0]);
    this->PushOp(StackVM::PUSH_I64, 0);
    this->PushOp(StackVM::EQ_HANDLE);
  } else {
    LOG(FATAL) << "unknown function call " << op->name;
  }
}

void CodeGenStackVM::PushBinary(StackVM::OpCode op_int64,
                                const Expr& a,
                                const Expr& b) {
  this->Push(a);
  this->Push(b);
  Type t = a.type();
  if (t.is_int()) {
    this->PushOp(op_int64);
  } else if (t.is_uint()) {
    this->PushOp(op_int64);
  } else {
    this->PushOp(StackVM::CodeI64ToF64(op_int64));
  }
}

void CodeGenStackVM::PushCast(Type dst, Type src) {
  if (dst.is_int()) {
    if (src.is_int() || src.is_uint()) return;
  } else if (dst.is_uint()) {
    if (src.is_int() || src.is_uint()) return;
  } else if (dst.is_float()) {
    if (src.is_float()) return;
  }
}

void CodeGenStackVM::VisitExpr_(const StringImm *op) {
  int sid = this->GetStrID(op->value);
  this->PushOp(StackVM::PUSH_I64, sid);
}

void CodeGenStackVM::VisitExpr_(const IntImm *op) {
  CHECK(op->value >= std::numeric_limits<int>::min() &&
        op->value <= std::numeric_limits<int>::max())
      << "Int constant exceed bound";
    this->PushOp(StackVM::PUSH_I64, static_cast<int>(op->value));
}

void CodeGenStackVM::VisitExpr_(const UIntImm *op) {
  CHECK(op->value <= std::numeric_limits<int>::max())
      << "Int constant exceed bound";
  this->PushOp(StackVM::PUSH_I64, static_cast<int>(op->value));
}

void CodeGenStackVM::VisitExpr_(const FloatImm *op) {
  LOG(FATAL) << "Float Imm is not supported";
}

void CodeGenStackVM::VisitExpr_(const Variable *op) {
  int vid = this->GetVarID(op);
  this->PushOp(StackVM::LOAD_HEAP, vid);
}

void CodeGenStackVM::VisitExpr_(const Cast *op) {
  this->Push(op->value);
  PushCast(op->type, op->value.type());
}

void CodeGenStackVM::VisitExpr_(const Add *op) {
  PushBinary(StackVM::ADD_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const Sub *op) {
  PushBinary(StackVM::SUB_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const Mul *op) {
  PushBinary(StackVM::MUL_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const Div *op) {
  PushBinary(StackVM::DIV_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const Mod *op) {
  PushBinary(StackVM::MOD_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const Min *op) {
  this->Push(op->a);
  this->Push(op->b);
  this->PushOp(StackVM::PUSH_VALUE, -1);
  this->PushOp(StackVM::PUSH_VALUE, -1);
  this->PushOp(StackVM::LT_I64);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitExpr_(const Max *op) {
  this->Push(op->a);
  this->Push(op->b);
  this->PushOp(StackVM::PUSH_VALUE, 0);
  this->PushOp(StackVM::PUSH_VALUE, -2);
  this->PushOp(StackVM::LT_I64);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitExpr_(const EQ *op) {
  PushBinary(StackVM::EQ_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const LE *op) {
  PushBinary(StackVM::LE_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const NE *op) {
  PushBinary(StackVM::EQ_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const LT *op) {
  PushBinary(StackVM::LT_I64, op->a, op->b);
}

void CodeGenStackVM::VisitExpr_(const GE *op) {
  PushBinary(StackVM::LT_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const GT *op) {
  PushBinary(StackVM::LE_I64, op->a, op->b);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitExpr_(const And *op) {
  this->Push(op->a);
  int64_t pc_jump = this->GetPC();
  int64_t opr_index = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->b);
  int64_t diff = this->GetPC() - pc_jump;
  this->SetOperand(opr_index, diff);
}

void CodeGenStackVM::VisitExpr_(const Or *op) {
  this->Push(op->a);
  int64_t pc_jump = this->GetPC();
  int64_t opr_index = this->PushOp(StackVM::RJUMP_IF_TRUE, 0);
  this->Push(op->b);
  int64_t diff = this->GetPC() - pc_jump;
  this->SetOperand(opr_index, diff);
}

void CodeGenStackVM::VisitExpr_(const Not* op) {
  this->Push(op->a);
  this->PushOp(StackVM::NOT);
}

void CodeGenStackVM::VisitStmt_(const ProducerConsumer *op) {
  this->Push(op->body);
}

void CodeGenStackVM::VisitStmt_(const For *op) {
  CHECK(is_zero(op->min));
  int vid = this->AllocVarID(op->loop_var.get());
  this->PushOp(StackVM::PUSH_I64, 0);
  int64_t loop_head = this->GetPC();
  this->PushOp(StackVM::STORE_HEAP, vid);
  this->PushOp(StackVM::LOAD_HEAP, vid);
  this->Push(op->extent);
  this->PushOp(StackVM::LT_I64);
  int64_t label_fjump = this->GetPC();
  int64_t foward_jump = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->body);
  this->PushOp(StackVM::LOAD_HEAP, vid);
  this->PushOp(StackVM::PUSH_I64, 1);
  this->PushOp(StackVM::ADD_I64);
  int64_t label_bjump = this->GetPC();
  int64_t backward_jump = this->PushOp(StackVM::RJUMP, 0);
  int64_t loop_end = this->GetPC();
  this->PushOp(StackVM::POP);
  this->SetOperand(foward_jump, loop_end - label_fjump);
  this->SetOperand(backward_jump, loop_head - label_bjump);
}

void CodeGenStackVM::VisitStmt_(const Block *op) {
  this->Push(op->first);
  if (op->rest.defined()) this->Push(op->rest);
}

void CodeGenStackVM::VisitStmt_(const Evaluate *ev) {
  if (is_const(ev->value)) return;
  const Call* op = ev->value.as<Call>();
  if (op && op->is_intrinsic(intrinsic::tvm_struct_set)) {
    CHECK_EQ(op->args.size(), 4U);
    this->Push(op->args[0]);
    this->Push(op->args[3]);
    const IntImm* index = op->args[1].as<IntImm>();
    CHECK(index != nullptr);
    StackVM::Code code;
    code.op_code = StackVM::TVM_STRUCT_SET;
    vm_.code.push_back(code);
    code.v_int = index->value;
    vm_.code.push_back(code);
    code.v_int = op->args[2].as<IntImm>()->value;
    vm_.code.push_back(code);
  } else {
    this->Push(ev->value);
    this->PushOp(StackVM::POP);
  }
}

void CodeGenStackVM::VisitStmt_(const IfThenElse *op) {
  this->Push(op->condition);
  int64_t label_ejump = this->GetPC();
  int64_t else_jump = this->PushOp(StackVM::RJUMP_IF_FALSE, 0);
  this->PushOp(StackVM::POP);
  this->Push(op->then_case);
  if (op->else_case.defined()) {
    int64_t label_then_jump = this->GetPC();
    int64_t then_jump = this->PushOp(StackVM::RJUMP, 0);
    int64_t else_begin = this->GetPC();
    this->SetOperand(else_jump, else_begin - label_ejump);
    this->PushOp(StackVM::POP);
    this->Push(op->else_case);
    int64_t if_end = this->GetPC();
    this->SetOperand(then_jump, if_end - label_then_jump);
  } else {
    int64_t if_end = this->GetPC();
    this->SetOperand(else_jump, if_end - label_ejump);
    this->PushOp(StackVM::POP);
  }
}

void CodeGenStackVM::VisitStmt_(const LetStmt *op) {
  this->Push(op->value);
  int64_t vid = this->AllocVarID(op->var.get());
  this->PushOp(StackVM::STORE_HEAP, static_cast<int>(vid));
  this->Push(op->body);
}

void CodeGenStackVM::VisitExpr_(const Ramp *op) {
  LOG(FATAL) << "Ramp is not supported";
}

void CodeGenStackVM::VisitExpr_(const Broadcast *op) {
  LOG(FATAL) << "Broadcast is not supported";
}

void CodeGenStackVM::VisitExpr_(const Select *op) {
  this->Push(op->true_value);
  this->Push(op->false_value);
  this->Push(op->condition);
  this->PushOp(StackVM::SELECT);
}

void CodeGenStackVM::VisitStmt_(const AssertStmt *op) {
  if (op->message.as<StringImm>()) {
    int sid = this->GetStrID(op->message.as<StringImm>()->value);
    this->Push(op->condition);
    this->PushOp(StackVM::ASSERT, sid);
  }
  this->Push(op->body);
}

void CodeGenStackVM::VisitStmt_(const AttrStmt *op) {
  this->Push(op->body);
}

void CodeGenStackVM::VisitExpr_(const Let *op) {
  this->Push(op->value);
  int64_t vid = this->AllocVarID(op->var.get());
  this->PushOp(StackVM::STORE_HEAP, static_cast<int>(vid));
  this->Push(op->body);
}
}  // namespace codegen
}  // namespace tvm
