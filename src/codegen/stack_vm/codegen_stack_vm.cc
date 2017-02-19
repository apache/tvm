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

CodeGenStackVM::FType& CodeGenStackVM::vtable() {  // NOLINT(*)
  static FType inst; return inst;
}

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
  static const FType& f = vtable();
  f(n, this);
  if (debug_) {
    this->PushOp(StackVM::ASSERT_SP, 0);
  }
}

void CodeGenStackVM::Push(const Expr& n) {
  static const FType& f = vtable();
  f(n, this);
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

void CodeGenStackVM::PushCallPacked(
    int fid, const std::vector<int>& arg_type_codes) {
  StackVM::Code code;
  // CALL_PACKED_FUNC
  code.op_code = StackVM::CALL_PACKED_FUNC;
  vm_.code.push_back(code);
  // num_args
  code.v_int = static_cast<int>(arg_type_codes.size());
  vm_.code.push_back(code);
  // fid
  code.v_int = fid;
  vm_.code.push_back(code);
  // type codes.
  for (int tcode : arg_type_codes) {
    code.v_int = tcode;
    vm_.code.push_back(code);
  }
}

int CodeGenStackVM::GetVarID(const Variable* v) const {
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
      << "Find undefined Variable " << v->name_hint;
  return it->second;
}

void CodeGenStackVM::Push_(const ir::Load* op) {
  this->PushOp(StackVM::LOAD_HEAP, GetVarID(op->buffer_var.get()));
  if (op->type == UInt(32) && op->index.as<IntImm>()) {
    this->PushOp(StackVM::ARRAY_LOAD_UINT32, op->index.as<IntImm>()->value);
  } else {
    this->Push(op->index);
    this->PushOp(StackVM::PUSH_I64, op->type.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
    this->PushOp(StackVM::GetLoad(Type2TVMType(op->type)));
  }
}
void CodeGenStackVM::Push_(const ir::Store* op) {
  this->PushOp(StackVM::LOAD_HEAP, GetVarID(op->buffer_var.get()));
  this->Push(op->index);
  this->PushOp(StackVM::PUSH_I64, op->value.type().element_of().bytes());
  this->PushOp(StackVM::MUL_I64);
  this->PushOp(StackVM::ADDR_ADD);
  this->Push(op->value);
  this->PushOp(StackVM::GetStore(Type2TVMType(op->value.type())));
}

void CodeGenStackVM::Push_(const ir::Allocate* op) {
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

void CodeGenStackVM::Push_(const ir::Call* op) {
  if (op->is_intrinsic(Call::address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    this->PushOp(StackVM::LOAD_HEAP, GetVarID(l->buffer_var.get()));
    this->Push(l->index);
    this->PushOp(StackVM::PUSH_I64, l->type.element_of().bytes());
    this->PushOp(StackVM::MUL_I64);
    this->PushOp(StackVM::ADDR_ADD);
  } else if (op->is_intrinsic(intrinsic::tvm_api_load_arg)) {
    CHECK_EQ(op->args.size(), 3U);
    this->Push(op->args[0]);
    this->Push(op->args[1]);
    this->Push(op->args[2]);
    if (op->type.is_handle()) {
      this->PushOp(StackVM::TVM_LOAD_ARG_HANDLE);
    } else if (op->type.is_float()) {
      this->PushOp(StackVM::TVM_LOAD_ARG_FP64);
    } else if (op->type.is_int() || op->type.is_uint()) {
      this->PushOp(StackVM::TVM_LOAD_ARG_INT64);
    } else {
      LOG(FATAL) << "donot know how to handle type" << op->type;
    }
  } else if (op->is_intrinsic(intrinsic::tvm_array_get_field)) {
    CHECK_EQ(op->args.size(), 2U);
    this->Push(op->args[0]);
    switch (op->args[1].as<IntImm>()->value) {
      case intrinsic::kData: PushOp(StackVM::TVM_ARRAY_GET_DATA); break;
      case intrinsic::kShape: PushOp(StackVM::TVM_ARRAY_GET_SHAPE); break;
      case intrinsic::kStrides: PushOp(StackVM::TVM_ARRAY_GET_STRIDES); break;
      case intrinsic::kNDim: PushOp(StackVM::TVM_ARRAY_GET_NDIM); break;
      case intrinsic::kTypeCode: PushOp(StackVM::TVM_ARRAY_GET_TYPE_CODE); break;
      case intrinsic::kTypeBits: PushOp(StackVM::TVM_ARRAY_GET_TYPE_BITS); break;
      case intrinsic::kTypeLanes: PushOp(StackVM::TVM_ARRAY_GET_TYPE_LANES); break;
      default: LOG(FATAL) << "unknown field code";
    }
  } else if (op->is_intrinsic(intrinsic::tvm_call_packed)) {
    CHECK_GE(op->args.size(), 1U);
    const StringImm* s = op->args[0].as<StringImm>();
    CHECK(s != nullptr) << "tvm_call_global expect first argument as function name";
    for (size_t i = 1; i < op->args.size(); ++i) {
      this->Push(op->args[i]);
    }
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
    // get the argument type code.
    std::vector<int> arg_type_codes;
    for (size_t i = 1; i < op->args.size(); ++i) {
      Type t = op->args[i].type();
      int code = t.code();
      int lanes = t.lanes();
      CHECK_EQ(lanes, 1);
      arg_type_codes.push_back(code);
    }
    this->PushCallPacked(fid, arg_type_codes);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    this->Push(op->args[0]);
    this->PushOp(StackVM::PUSH_I64, 0);
    this->PushOp(StackVM::EQ_I64);
  } else {
    this->HandleUnknownCall(op);
  }
}

void CodeGenStackVM::HandleUnknownCall(const ir::Call* op) {
  LOG(FATAL) << "donot know how to handle call " << op->name;
}

inline void PushBinary(StackVM::OpCode op_int64,
                       const Expr& a,
                       const Expr& b,
                       CodeGenStackVM* p) {
  p->Push(a);
  p->Push(b);
  Type t = a.type();
  if (t.is_int()) {
    p->PushOp(op_int64);
  } else if (t.is_uint()) {
    if (t.bits() <= 32) {
      p->PushOp(op_int64);
    } else {
      LOG(FATAL) << "Cannot handle uint64_t in StackVM";
    }
  } else {
    p->PushOp(StackVM::CodeI64ToF64(op_int64));
  }
}

inline void PushCast(Type dst,
                     Type src,
                     CodeGenStackVM* p) {
  if (dst.is_int()) {
    if (src.is_int()) return;
    if (src.is_uint() && src.bits() <= 32) return;
  } else if (dst.is_uint() && dst.bits() <= 32) {
    if (src.is_int()) return;
    if (src.is_uint() && src.bits() <= 32) return;
  } else if (dst.is_float()) {
    if (src.is_float()) return;
  }
  LOG(FATAL) << "Cannot handle cast " << src << " to " << dst;
}

TVM_STATIC_IR_FUNCTOR(CodeGenStackVM, vtable)
.set_dispatch<StringImm>([](const StringImm *op, CodeGenStackVM *p) {
    int sid = p->GetStrID(op->value);
    p->PushOp(StackVM::PUSH_I64, sid);
  })
.set_dispatch<IntImm>([](const IntImm *op, CodeGenStackVM *p) {
    CHECK(op->value >= std::numeric_limits<int>::min() &&
          op->value <= std::numeric_limits<int>::max())
        << "Int constant exceed bound";
    p->PushOp(StackVM::PUSH_I64, static_cast<int>(op->value));
  })
.set_dispatch<UIntImm>([](const UIntImm *op, CodeGenStackVM *p) {
    CHECK(op->value <= std::numeric_limits<int>::max())
        << "Int constant exceed bound";
    p->PushOp(StackVM::PUSH_I64, static_cast<int>(op->value));
  })
.set_dispatch<FloatImm>([](const FloatImm *op, CodeGenStackVM *p) {
    LOG(FATAL) << "Float Imm is not supported";
  });

TVM_STATIC_IR_FUNCTOR(CodeGenStackVM, vtable)
.set_dispatch<Variable>([](const Variable *op, CodeGenStackVM* p) {
    int vid = p->GetVarID(op);
    p->PushOp(StackVM::LOAD_HEAP, vid);
  })
.set_dispatch<Cast>([](const Cast *op, CodeGenStackVM* p) {
    p->Push(op->value);
    PushCast(op->type, op->value.type(), p);
  })
.set_dispatch<Add>([](const Add *op, CodeGenStackVM* p) {
    PushBinary(StackVM::ADD_I64, op->a, op->b, p);
  })
.set_dispatch<Sub>([](const Sub *op, CodeGenStackVM* p) {
    PushBinary(StackVM::SUB_I64, op->a, op->b, p);
  })
.set_dispatch<Mul>([](const Mul *op, CodeGenStackVM* p) {
    PushBinary(StackVM::MUL_I64, op->a, op->b, p);
  })
.set_dispatch<Div>([](const Div *op, CodeGenStackVM* p) {
    PushBinary(StackVM::DIV_I64, op->a, op->b, p);
  })
.set_dispatch<Mod>([](const Mod *op, CodeGenStackVM* p) {
    PushBinary(StackVM::MOD_I64, op->a, op->b, p);
  })
.set_dispatch<Min>([](const Min *op, CodeGenStackVM* p) {
    p->Push(op->a);
    p->Push(op->b);
    p->PushOp(StackVM::PUSH_VALUE, -1);
    p->PushOp(StackVM::PUSH_VALUE, -1);
    p->PushOp(StackVM::LT_I64);
    p->PushOp(StackVM::SELECT);
  })
.set_dispatch<Max>([](const Max *op, CodeGenStackVM* p) {
    p->Push(op->a);
    p->Push(op->b);
    p->PushOp(StackVM::PUSH_VALUE, 0);
    p->PushOp(StackVM::PUSH_VALUE, -2);
    p->PushOp(StackVM::LT_I64);
    p->PushOp(StackVM::SELECT);
  })
.set_dispatch<EQ>([](const EQ *op, CodeGenStackVM* p) {
    PushBinary(StackVM::EQ_I64, op->a, op->b, p);
  })
.set_dispatch<LE>([](const LE *op, CodeGenStackVM* p) {
    PushBinary(StackVM::LE_I64, op->a, op->b, p);
  })
.set_dispatch<NE>([](const NE *op, CodeGenStackVM* p) {
    PushBinary(StackVM::EQ_I64, op->a, op->b, p);
    p->PushOp(StackVM::NOT);
  })
.set_dispatch<LT>([](const LT *op, CodeGenStackVM* p) {
    PushBinary(StackVM::LT_I64, op->a, op->b, p);
  })
.set_dispatch<GE>([](const GE *op, CodeGenStackVM* p) {
    PushBinary(StackVM::LT_I64, op->a, op->b, p);
    p->PushOp(StackVM::NOT);
  })
.set_dispatch<GT>([](const GT *op, CodeGenStackVM* p) {
    PushBinary(StackVM::LE_I64, op->a, op->b, p);
    p->PushOp(StackVM::NOT);
  })
.set_dispatch<And>([](const And *op, CodeGenStackVM* p) {
    p->Push(op->a);
    int64_t pc_jump = p->GetPC();
    int64_t opr_index = p->PushOp(StackVM::RJUMP_IF_FALSE, 0);
    p->PushOp(StackVM::POP);
    p->Push(op->b);
    int64_t diff = p->GetPC() - pc_jump;
    p->SetOperand(opr_index, diff);
})
.set_dispatch<Or>([](const Or *op, CodeGenStackVM* p) {
    p->Push(op->a);
    int64_t pc_jump = p->GetPC();
    int64_t opr_index = p->PushOp(StackVM::RJUMP_IF_TRUE, 0);
    p->Push(op->b);
    int64_t diff = p->GetPC() - pc_jump;
    p->SetOperand(opr_index, diff);
})
.set_dispatch<Not>([](const Not* op, CodeGenStackVM* p) {
    p->PushOp(StackVM::NOT);
  });


TVM_STATIC_IR_FUNCTOR(CodeGenStackVM, vtable)
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, CodeGenStackVM* p) {
    p->Push(op->body);
  })
.set_dispatch<For>([](const For *op, CodeGenStackVM* p) {
    CHECK(is_zero(op->min));
    int vid = p->AllocVarID(op->loop_var.get());
    p->PushOp(StackVM::PUSH_I64, 0);
    int64_t loop_head = p->GetPC();
    p->PushOp(StackVM::STORE_HEAP, vid);
    p->PushOp(StackVM::LOAD_HEAP, vid);
    p->Push(op->extent);
    p->PushOp(StackVM::LT_I64);
    int64_t label_fjump = p->GetPC();
    int64_t foward_jump = p->PushOp(StackVM::RJUMP_IF_FALSE, 0);
    p->PushOp(StackVM::POP);
    p->Push(op->body);
    p->PushOp(StackVM::LOAD_HEAP, vid);
    p->PushOp(StackVM::PUSH_I64, 1);
    p->PushOp(StackVM::ADD_I64);
    int64_t label_bjump = p->GetPC();
    int64_t backward_jump = p->PushOp(StackVM::RJUMP, 0);
    int64_t loop_end = p->GetPC();
    p->PushOp(StackVM::POP);
    p->SetOperand(foward_jump, loop_end - label_fjump);
    p->SetOperand(backward_jump, loop_head - label_bjump);
  })
.set_dispatch<Block>([](const Block *op, CodeGenStackVM* p) {
    p->Push(op->first);
    if (op->rest.defined()) p->Push(op->rest);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, CodeGenStackVM* p) {
    if (is_const(op->value)) return;
    p->Push(op->value);
    p->PushOp(StackVM::POP);
  })
.set_dispatch<IfThenElse>([](const IfThenElse *op, CodeGenStackVM* p) {
    p->Push(op->condition);
    int64_t label_ejump = p->GetPC();
    int64_t else_jump = p->PushOp(StackVM::RJUMP_IF_FALSE, 0);
    p->PushOp(StackVM::POP);
    p->Push(op->then_case);
    if (op->else_case.defined()) {
      int64_t label_then_jump = p->GetPC();
      int64_t then_jump = p->PushOp(StackVM::RJUMP, 0);
      int64_t else_begin = p->GetPC();
      p->SetOperand(else_jump, else_begin - label_ejump);
      p->PushOp(StackVM::POP);
      p->Push(op->else_case);
      int64_t if_end = p->GetPC();
      p->SetOperand(then_jump, if_end - label_then_jump);
    } else {
      int64_t if_end = p->GetPC();
      p->SetOperand(else_jump, if_end - label_ejump);
      p->PushOp(StackVM::POP);
    }
  })
.set_dispatch<LetStmt>([](const LetStmt *op, CodeGenStackVM* p) {
    p->Push(op->value);
    int64_t vid = p->AllocVarID(op->var.get());
    p->PushOp(StackVM::STORE_HEAP, vid);
    p->Push(op->body);
  })
.set_dispatch<Ramp>([](const Ramp *op, CodeGenStackVM* p) {
    LOG(FATAL) << "Ramp is not supported";
  })
.set_dispatch<Broadcast>([](const Broadcast *op, CodeGenStackVM* p) {
    LOG(FATAL) << "Broadcast is not supported";
  })
.set_dispatch<Select>([](const Select *op, CodeGenStackVM* p) {
    p->Push(op->true_value);
    p->Push(op->false_value);
    p->Push(op->condition);
    p->PushOp(StackVM::SELECT);
  })
.set_dispatch<AssertStmt>([](const AssertStmt *op, CodeGenStackVM* p) {
    if (op->message.as<StringImm>()) {
      int sid = p->GetStrID(op->message.as<StringImm>()->value);
      p->Push(op->condition);
      p->PushOp(StackVM::ASSERT, sid);
    }
  })
.set_dispatch<AttrStmt>([](const AttrStmt *op, CodeGenStackVM* p) {
    p->Push(op->body);
  })
.set_dispatch<Let>([](const Let *op, CodeGenStackVM* p) {
    p->Push(op->value);
    int64_t vid = p->AllocVarID(op->var.get());
    p->PushOp(StackVM::STORE_HEAP, vid);
    p->Push(op->body);
  })
.set_dispatch<Load>([](const Load *op, CodeGenStackVM* p) {
    p->Push_(op);
  })
.set_dispatch<Store>([](const Store *op, CodeGenStackVM* p) {
    p->Push_(op);
  })
.set_dispatch<Allocate>([](const Allocate *op, CodeGenStackVM* p) {
    p->Push_(op);
  })
.set_dispatch<Call>([](const Call *op, CodeGenStackVM* p) {
    p->Push_(op);
  });
}  // namespace codegen
}  // namespace tvm
