/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_c.cc
 */
#include <iomanip>
#include <cctype>
#include "./codegen_c.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

using namespace ir;

std::string CodeGenC::Compile(LoweredFunc f,
                              bool output_ssa) {
  print_ssa_form_ = output_ssa;
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  this->stream << "void " << f->name << "(";
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.type().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }
      stream << ' ';
    }
    if (handle_data_type_.count(v.get())) {
      PrintType(handle_data_type_.at(v.get()), stream);
      stream << "*";
    } else {
      PrintType(v.type(), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n";
  return stream.str();
}

void CodeGenC::PrintStmt(const Stmt& n) {
  static const FPrintStmt& f = vtable_print_stmt();
  f(n, this);
}

std::string CodeGenC::SSAGetID(std::string src, Type t) {
  if (name_alloc_map_.count(src)) return src;
  auto it = ssa_assign_map_.find(src);
  if (it != ssa_assign_map_.end()) {
    if (scope_mark_.at(it->second.scope_id)) {
      return it->second.vid;
    }
  }

  this->PrintIndent();
  SSAEntry e;
  e.vid = GetUniqueName("_");
  e.scope_id = static_cast<int>(scope_mark_.size() - 1);
  ssa_assign_map_[src] = e;
  if (src.length() > 3 &&
      src[0] == '(' && src[src.length() - 1] == ')') {
    src = src.substr(1, src.length() - 2);
  }
  PrintType(t, stream);
  stream << ' ' << e.vid << " = " << src << ";\n";
  return e.vid;
}

void CodeGenC::PrintExpr(const Expr& n, std::ostream& os) {  // NOLINT(*)
  static const FPrintExpr& f = vtable_print_expr();
  if (print_ssa_form_) {
    std::ostringstream temp;
    f(n, temp, this);
    os << SSAGetID(temp.str(), n.type());
  } else {
    f(n, os, this);
  }
}

std::string CodeGenC::GetUniqueName(std::string prefix) {
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      std::string name = os.str();
      if (name_alloc_map_.count(name) == 0) {
        prefix = name;
        break;
      }
    }
  }
  name_alloc_map_[prefix] = 0;
  return prefix;
}

std::string CodeGenC::AllocVarID(const Variable* v) {
  CHECK(!var_idmap_.count(v))
      << "Need input to be in SSA form dup " << v->name_hint;
  std::string key = v->name_hint;
  for (size_t i = 0; i < key.size(); ++i) {
    if (key[i] == '.') key[i] = '_';
  }
  std::string vid = GetUniqueName(key);
  var_idmap_[v] = vid;
  return vid;
}

std::string CodeGenC::GetVarID(const Variable* v) const {
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
      << "Find undefined Variable " << v->name_hint;
  return it->second;
}

bool CodeGenC::HandleTypeMatch(const Variable* buf_var, Type t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const Variable* buf_var, Type t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    CHECK(it->second == t)
        << "conflicting buf var type";
  }
}

void CodeGenC::PrintIndent() {
  for (int i = 0; i < this->indent; ++i) {
    this->stream << ' ';
  }
}

void CodeGenC::MarkConst(std::string vid) {
  if (print_ssa_form_) {
    auto it = ssa_assign_map_.find(vid);
    if (it == ssa_assign_map_.end()) {
      SSAEntry e;
      e.vid = vid;
      e.scope_id = 0;
      ssa_assign_map_[vid] = e;
    } else {
      CHECK_EQ(it->second.vid, vid);
    }
  }
}

void CodeGenC::PrintType(Type t, std::ostream& os) const {  // NOLINT(*)
  CHECK_EQ(t.lanes(), 1)
      << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*"; return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float"; return;
    }
    if (t.bits() == 64) {
      os << "double"; return;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 8: case 16: case 32: case 64: {
        os << "uint" << t.bits() << "_t"; return;
      }
      case 1: os << "int"; return;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8: case 16: case 32: case 64: {
        os << "int" << t.bits() << "_t";  return;
      }
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

CodeGenC::FPrintStmt& CodeGenC::vtable_print_stmt() {  // NOLINT(*)
  static FPrintStmt inst; return inst;
}

CodeGenC::FPrintExpr& CodeGenC::vtable_print_expr() {  // NOLINT(*)
  static FPrintExpr inst; return inst;
}

inline void PrintConst(const IntImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  if (op->type == Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->type, os);
    os << ")" << op->value;
  }
}

inline void PrintConst(const UIntImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  if (op->type == UInt(32)) {
    std::ostringstream temp;
    temp << op->value << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->type, os);
    os << ")" << op->value;
  }
}

inline void PrintConst(const FloatImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  switch (op->type.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->type.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->type, os);
      os << ')' << std::scientific <<op->value << 'f';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
  }
}

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_expr)
.set_dispatch<IntImm>([](const IntImm *op, std::ostream& os, CodeGenC *p) {  // NOLINT(*)
    PrintConst(op, os, p);
  })
.set_dispatch<UIntImm>([](const UIntImm *op, std::ostream& os, CodeGenC *p) {  // NOLINT(*)
    PrintConst(op, os, p);
  })
.set_dispatch<FloatImm>([](const FloatImm *op, std::ostream& os, CodeGenC *p) { // NOLINT(*)
    PrintConst(op, os, p);
  })
.set_dispatch<StringImm>([](const StringImm *op, std::ostream& os, CodeGenC *p) { // NOLINT(*)
    os << "\"" << op->value << "\"";
  });

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char *opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  if (op->type.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->type, op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsitc(const Call* op,
                                  const char *opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenC* p) {
  if (op->type.lanes() == 1) {
    CHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->type, op->args[0], op->args[1], os);
  }
}

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_expr)
.set_dispatch<Cast>([](const Cast *op, std::ostream& os, CodeGenC *p) {  // NOLINT(*)
    p->PrintType(op->type, os);
    os << '(';
    p->PrintExpr(op->value, os);
    os << ')';
  })
.set_dispatch<Variable>([](const Variable *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << p->GetVarID(op);
  })
.set_dispatch<Add>([](const Add *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "+", os, p);
  })
.set_dispatch<Sub>([](const Sub *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "-", os, p);
  })
.set_dispatch<Mul>([](const Mul *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "*", os, p);
  })
.set_dispatch<Div>([](const Div *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "/", os, p);
  })
.set_dispatch<Mod>([](const Mod *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "%", os, p);
})
.set_dispatch<Min>([](const Min *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "min", os, p);
})
.set_dispatch<Max>([](const Max *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "max", os, p);
})
.set_dispatch<EQ>([](const EQ *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "==", os, p);
})
.set_dispatch<NE>([](const NE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "!=", os, p);
})
.set_dispatch<LT>([](const LT *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "<", os, p);
})
.set_dispatch<LE>([](const LE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "<=", os, p);
})
.set_dispatch<GT>([](const GT *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, ">", os, p);
})
.set_dispatch<GE>([](const GE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, ">=", os, p);
})
.set_dispatch<And>([](const And *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "&&", os, p);
})
.set_dispatch<Or>([](const Or *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, "||", os, p);
})
.set_dispatch<Not>([](const Not *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << '!';
    p->PrintExpr(op->a, os);
  });

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_stmt)
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, CodeGenC* p) {
    p->PrintStmt(op->body);
  })
.set_dispatch<Block>([](const Block *op, CodeGenC* p) {
    p->PrintStmt(op->first);
    if (op->rest.defined()) p->PrintStmt(op->rest);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, CodeGenC* p) {
    if (is_const(op->value)) return;
    const Call* call = op->value.as<Call>();

    if (call && call->is_intrinsic(intrinsic::tvm_storage_sync)) {
      p->PrintStorageSync(call->args[0].as<StringImm>()->value);
    } else {
      std::string vid = p->PrintExpr(op->value);
      p->PrintIndent();
      p->stream << "(void)" << vid << ";\n";
    }
  });


#define DISPATCH_EXPR(OP)                            \
  set_dispatch<OP>([](const OP *op, std::ostream&os, CodeGenC* p) { \
      p->PrintExpr(op, os); })

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_expr)
.DISPATCH_EXPR(Load)
.DISPATCH_EXPR(Call)
.DISPATCH_EXPR(Let)
.DISPATCH_EXPR(Ramp)
.DISPATCH_EXPR(Broadcast)
.DISPATCH_EXPR(Select);


void CodeGenC::PrintExpr(const Call *op, std::ostream& os) {  // NOLINT(*)
  CodeGenC* p = this;
  if (op->is_intrinsic(Call::bitwise_and)) {
    PrintBinaryIntrinsitc(op, " & ", os, p);
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    PrintBinaryIntrinsitc(op, " ^ ", os, p);
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    PrintBinaryIntrinsitc(op, " | ", os, p);
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(~";
    p->PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->is_intrinsic(Call::shift_left)) {
    PrintBinaryIntrinsitc(op, " << ", os, p);
  } else if (op->is_intrinsic(Call::shift_right)) {
    PrintBinaryIntrinsitc(op, " >> ", os, p);
  } else if (op->is_intrinsic(Call::address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    p->PrintType(l->type.element_of(), os);
    os << " *)" << p->GetVarID(l->buffer_var.get())
       << " + ";
    p->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_api_load_arg)) {
    CHECK_EQ(op->args.size(), 3U);
    if (!op->type.is_handle()) {
      os << '(';
      p->PrintType(op->type, os);
      os << ')';
    }
    os << "(((TVMArg*)";
    p->PrintExpr(op->args[0], os);
    os << ")[" << op->args[2] << "].";
    if (op->type.is_handle()) {
      os << "v_handle";
    } else if (op->type.is_float()) {
      os << "v_double";
    } else if (op->type.is_int() || op->type.is_uint()) {
      os << "v_long";
    } else {
      LOG(FATAL) << "donot know how to handle type" << op->type;
    }
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_array_get_field)) {
    CHECK_EQ(op->args.size(), 2U);
    os << "(((TVMArray*)";
    p->PrintExpr(op->args[0], os);
    os << ")->";
    switch (op->args[1].as<IntImm>()->value) {
      case intrinsic::kData: os << "data"; break;
      case intrinsic::kShape: os << "shape"; break;
      case intrinsic::kStrides: os << "strides"; break;
      case intrinsic::kNDim: os << "ndim"; break;
      case intrinsic::kTypeCode: os << "dtype.type_code"; break;
      case intrinsic::kTypeBits: os << "dtype.bits"; break;
      case intrinsic::kTypeLanes: os << "dtype.lanes"; break;
      default: LOG(FATAL) << "unknown field code";
    }
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(";
    p->PrintExpr(op->args[0], os);
    os << " == NULL)";
  } else {
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      p->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  }
}

void CodeGenC::PrintVecBinaryOp(
    const std::string&op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os <<"(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

inline bool TryGetRamp1Base(Expr index, int lanes, Expr *base) {
  const Ramp* r = index.as<Ramp>();
  if (!r) return false;
  if (!is_one(r->stride)) return false;
  CHECK_EQ(r->lanes, lanes);
  *base = r->base;
  return true;
}

// Print a reference expression to a buffer.
void CodeGenC::PrintBufferRef(
    const Variable* buffer,
    Type t, Expr index,
    std::ostream& os) {  // NOLINT(*)
  std::string vid = GetVarID(buffer);
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t)) {
      os << "((";
      PrintType(t, os);
      os << "*)" << vid << ')';
    } else {
      os << vid;
    }
    os << '[';
    PrintExpr(index, os);
    os << ']';
  } else {
    // Buffer declared as vector type.
    // optimize for case where it is in register,
    if (HandleTypeMatch(buffer, t)) {
      // optimize for constant access
      int offset;
      if (arith::GetConstInt(index, &offset)) {
        CHECK_EQ(offset % t.lanes(), 0)
            << "Find unaligned vector load to a vector type";
        os << vid << '[' << (offset / t.lanes()) << ']';
        return;
      }
    }
    os << "((";
    PrintType(t, os);
    os << "*)(";
    if (!HandleTypeMatch(buffer, t.element_of())) {
      os << '(';
      PrintType(t.element_of(), os);
      os << "*)";
    }
    os << vid << " + ";
    PrintExpr(index, os);
    os << "))[0]";
  }
}

void CodeGenC::PrintExpr(const Load* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->type.lanes();
  if (op->type.lanes() == 1) {
    this->PrintBufferRef(op->buffer_var.get(), op->type, op->index, os);
  } else {
    Expr base;
    if (TryGetRamp1Base(op->index, op->type.lanes(), &base)) {
      this->PrintVecLoad(op->buffer_var.get(), op->type, base, os);
    } else {
      // Load elements seperately
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string svalue = GetUniqueName("_");
      {
        // delcare type.
        this->PrintIndent();
        this->PrintType(op->type, stream);
        stream << ' ' << svalue << ";\n";
      }
      std::string vid = GetVarID(op->buffer_var.get());
      Type elem_type = op->type.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          PrintType(elem_type, os);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.type(), i, value_temp);
        value_temp << ']';
        PrintVecElemStore(svalue, op->type, i, value_temp.str());
      }
      os << svalue;
    }
  }
}

void CodeGenC::PrintStmt(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    this->PrintIndent();
    std::string value = this->PrintExpr(op->value);
    this->PrintBufferRef(op->buffer_var.get(), t, op->index, stream);
    stream << " = " << value << ";\n";
  } else {
    Expr base;
    if (TryGetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        Type elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, op->index.type(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.type(), i, stream);
        stream << ";\n";
      }
    }
  }
}

void CodeGenC::PrintVecElemLoad(const std::string& vec,
                                Type t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i;
}

void CodeGenC::PrintVecElemStore(const std::string& vec,
                                 Type t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i
         << " = " << value << ";\n";
}

void CodeGenC::PrintVecLoad(const Variable* buffer,
                            Type t, Expr base,
                            std::ostream& os) {
  PrintBufferRef(buffer, t, base, os);
}

void CodeGenC::PrintVecStore(const Variable* buffer,
                             Type t, Expr base,
                             const std::string& value) {
  this->PrintIndent();
  PrintBufferRef(buffer, t, base, stream);
  stream << " = " << value << ";\n";
}

void CodeGenC::PrintExpr(const Let* op, std::ostream& os) {  // NOLINT(*)
  CHECK(print_ssa_form_)
      << "LetExpr is only supported by print SSA form";
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
}

void CodeGenC::PrintExpr(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Ramp: not supported ";
}

void CodeGenC::PrintExpr(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenC::PrintExpr(const Select* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Select: not supported ";
}

// Disoatch back to member functions
TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_stmt)
.set_dispatch<LetStmt>([](const LetStmt *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<Store>([](const Store *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<Allocate>([](const Allocate *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<AttrStmt>([](const AttrStmt *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<AssertStmt>([](const AssertStmt *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<For>([](const For *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<IfThenElse>([](const IfThenElse *op, CodeGenC* p) { p->PrintStmt(op); });

void CodeGenC::PrintThreadIndexExpr(
    std::string thread_tag, std::ostream& os) { // NOLINT(*)
  os << thread_tag;
}

void CodeGenC::PrintStorageSync(const std::string& sync) { // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_EQ(scope, "global");
}

void CodeGenC::PrintStmt(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    CHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.type() == Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* "
             << AllocVarID(op->var.get())
             << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)"  << value << ";\n";
    } else {
      PrintType(op->var.type(), this->stream);
      this->stream << ' '
                   << AllocVarID(op->var.get())
                   << " = " << value << ";\n";
    }
  }
  PrintStmt(op->body);
}

void CodeGenC::PrintStmt(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope = alloc_storage_scope_.at(buffer);
    PrintStorageScope(scope, stream);
    PrintType(op->type, stream);
    stream << ' '<< vid << '['
           << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenC::PrintStmt(const AttrStmt* op) {
  if (op->type_key == ir::attr::thread_extent) {
    IterVar iv(op->node.node_);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        this->PrintIndent();
        PrintType(iv->var.type(), stream);
        stream << ' '
               << AllocVarID(iv->var.get())
               << " = ";
        PrintThreadIndexExpr(iv->thread_tag, stream);
        stream << ";\n";
      }
    }
  } else if (op->type_key == ir::attr::storage_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImm>()->value;
  }
  this->PrintStmt(op->body);
}

void CodeGenC::PrintStmt(const AssertStmt* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (op->message.as<StringImm>()) {
    // GLOG style check
    stream << "CHECK(" << cond << ") << \""
           << op->message.as<StringImm>()->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
}

int CodeGenC::BeginScope() {
  int sid = static_cast<int>(scope_mark_.size());
  scope_mark_.push_back(true);
  indent += 2;
  return sid;
}

void CodeGenC::EndScope(int scope_id) {
  scope_mark_[scope_id] = false;
  indent -= 2;
}

void CodeGenC::PrintStmt(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::PrintStmt(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if (" << cond << ") {\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}


}  // namespace codegen
}  // namespace tvm
