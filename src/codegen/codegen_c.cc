/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_c.cc
 */
#include "./codegen_c.h"

namespace tvm {
namespace codegen {

using namespace ir;

std::string CodeGenC::Compile(
    Stmt stmt, std::string fun_name,
    Array<Var> args, bool output_ssa) {
  print_ssa_form_ = output_ssa;
  // skip the first underscore, so SSA variable starts from _1
  if (print_ssa_form_) GetUniqueName("_");

  this->indent += 2;
  this->stream << "void " << fun_name << "(";
  for (size_t i = 0; i < args.size(); ++i) {
    Var v = args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    PrintType(v.type(), stream);
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PrintStmt(stmt);
  this->indent -= 2;
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
    return it->second;
  } else {
    this->PrintIndent();
    std::string id = GetUniqueName("_");
    ssa_assign_map_[src] = id;
    if (src.length() > 3 &&
        src[0] == '(' && src[src.length() - 1] == ')') {
      src = src.substr(1, src.length() - 2);
    }
    PrintType(t, stream);
    stream << ' ' << id << " = " << src << ";\n";
    return id;
  }
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

bool CodeGenC::BufferTypeMatch(const Variable* buf_var, Type t) const {
  auto it = alloc_buf_type_.find(buf_var);
  if (it == alloc_buf_type_.end()) return false;
  return it->second == t;
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
      ssa_assign_map_[vid] = vid;
    } else {
      CHECK_EQ(it->second, vid);
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
      temp << op->value;
      if (op->type.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->type, os);
      os << ')' << op->value << 'f';
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
  });

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char *opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  os << '(';
  p->PrintExpr(op->a, os);
  os << opstr;
  p->PrintExpr(op->b, os);
  os << ')';
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
    PrintBinaryExpr(op, " + ", os, p);
  })
.set_dispatch<Sub>([](const Sub *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " - ", os, p);
  })
.set_dispatch<Mul>([](const Mul *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " * ", os, p);
  })
.set_dispatch<Div>([](const Div *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " / ", os, p);
  })
.set_dispatch<Mod>([](const Mod *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " % ", os, p);
})
.set_dispatch<Min>([](const Min *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << "min(";
    p->PrintExpr(op->a, os);
    os << ", ";
    p->PrintExpr(op->b, os);
    os << ")";
})
.set_dispatch<Max>([](const Max *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << "max(";
    p->PrintExpr(op->a, os);
    os << ", ";
    p->PrintExpr(op->b, os);
    os << ")";
})
.set_dispatch<EQ>([](const EQ *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " == ", os, p);
})
.set_dispatch<NE>([](const NE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " != ", os, p);
})
.set_dispatch<LT>([](const LT *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " < ", os, p);
})
.set_dispatch<LE>([](const LE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " <= ", os, p);
})
.set_dispatch<GT>([](const GT *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " > ", os, p);
})
.set_dispatch<GE>([](const GE *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " >= ", os, p);
})
.set_dispatch<And>([](const And *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " && ", os, p);
})
.set_dispatch<Or>([](const Or *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    PrintBinaryExpr(op, " || ", os, p);
})
.set_dispatch<Not>([](const Not *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << '!';
    p->PrintExpr(op->a, os);
  })
.set_dispatch<Call>([](const Call *op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      p->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  });

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_stmt)
.set_dispatch<AssertStmt>([](const AssertStmt *op, CodeGenC* p) {
    std::string cond = p->PrintExpr(op->condition);
    p->PrintIndent();
    p->stream << "assert(" << cond << ");\n";
  })
.set_dispatch<ProducerConsumer>([](const ProducerConsumer *op, CodeGenC* p) {
    p->PrintStmt(op->body);
  })
.set_dispatch<For>([](const For *op, CodeGenC* p) {
    std::string extent = p->PrintExpr(op->extent);
    p->PrintIndent();
    std::string vid = p->AllocVarID(op->loop_var.get());
    CHECK(is_zero(op->min));
    p->stream << "for (";
    p->PrintType(op->loop_var.type(), p->stream);
    p->stream << ' ' << vid << " = 0; "
              << vid << " < " << extent
              << "; ++" << vid << ") {\n";
    p->indent += 2;
    p->PrintStmt(op->body);
    p->indent -= 2;
    p->PrintIndent();
    p->stream << "}\n";
  })
.set_dispatch<Block>([](const Block *op, CodeGenC* p) {
    p->PrintStmt(op->first);
    if (op->rest.defined()) p->PrintStmt(op->rest);
  })
.set_dispatch<Evaluate>([](const Evaluate *op, CodeGenC* p) {
    if (is_const(op->value)) return;
    std::string vid = p->PrintExpr(op->value);
    p->PrintIndent();
    p->stream << "(void)" << vid << ";\n";
  })
.set_dispatch<IfThenElse>([](const IfThenElse *op, CodeGenC* p) {
    std::string cond = p->PrintExpr(op->condition);
    p->PrintIndent();
    p->stream << "if (" << cond << ") {\n";
    p->indent += 2;
    p->PrintStmt(op->then_case);
    p->indent -= 2;
    if (op->else_case.defined()) {
      p->PrintIndent();
      p->stream << "} else {\n";
      p->indent += 2;
      p->PrintStmt(op->else_case);
      p->indent -= 2;
    }
    p->PrintIndent();
    p->stream << "}\n";
});


#define DISPATCH_EXPR(OP)                            \
  set_dispatch<OP>([](const OP *op, std::ostream&os, CodeGenC* p) { \
      p->PrintExpr(op, os); })

TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_expr)
.DISPATCH_EXPR(Load)
.DISPATCH_EXPR(Let)
.DISPATCH_EXPR(Ramp)
.DISPATCH_EXPR(Broadcast)
.DISPATCH_EXPR(Select);

void CodeGenC::PrintExpr(const Load* op, std::ostream& os) {  // NOLINT(*)
  std::string vid = GetVarID(op->buffer_var.get());
  if (!BufferTypeMatch(op->buffer_var.get(), op->type)) {
    os << "((const ";
    PrintType(op->type, os);
    os << "*)" << vid << ')';
  } else {
    os << vid;
  }
  os << '[';
  PrintExpr(op->index, os);
  os << ']';
}

void CodeGenC::PrintExpr(const Let* op, std::ostream& os) {  // NOLINT(*)
  CHECK(print_ssa_form_)
      << "LetExpr is only supported by print SSA form";
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
}

void CodeGenC::PrintExpr(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "not supported ";
}

void CodeGenC::PrintExpr(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "not supported ";
}

void CodeGenC::PrintExpr(const Select* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "not supported ";
}

// Disoatch back to member functions
TVM_STATIC_IR_FUNCTOR(CodeGenC, vtable_print_stmt)
.set_dispatch<LetStmt>([](const LetStmt *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<Store>([](const Store *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<Allocate>([](const Allocate *op, CodeGenC* p) { p->PrintStmt(op); })
.set_dispatch<AttrStmt>([](const AttrStmt *op, CodeGenC* p) { p->PrintStmt(op); });


void CodeGenC::PrintStmt(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    CHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    PrintType(op->var.type(), this->stream);
    this->stream << ' '
       << AllocVarID(op->var.get())
       << " = " << value << ";\n";
  }
  PrintStmt(op->body);
}

void CodeGenC::PrintStmt(const Store* op) {
  std::string index = this->PrintExpr(op->index);
  std::string value = this->PrintExpr(op->value);
  this->PrintIndent();
  std::string vid = GetVarID(op->buffer_var.get());
  if (!BufferTypeMatch(op->buffer_var.get(), op->value.type())) {
    this->stream << "((";
    PrintType(op->value.type(), this->stream);
    this->stream << "*)" << vid << ')';
  } else {
    this->stream << vid;
  }
  this->stream << '[' << index
               << "] = " << value
               << ";\n";
}

void CodeGenC::PrintStmt(const Allocate* op) {
  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  std::string vid = AllocVarID(op->buffer_var.get());
  CHECK(!op->new_expr.defined());
  CHECK(!is_zero(op->condition));
  CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
  PrintType(op->type, stream);
  stream << ' '<< vid << '['
         << constant_size << "]\n;";
  this->PrintStmt(op->body);
}

void CodeGenC::PrintStmt(const AttrStmt* op) {
  if (op->type_key == "scope") {
    IterVar iv(op->node.node_);
    if (iv->thread_tag.length() != 0) {
      this->PrintIndent();
      PrintType(iv->var.type(), stream);
      stream << ' '
             << AllocVarID(iv->var.get())
             << " = " << iv->thread_tag << ";\n";
    }
  }
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
