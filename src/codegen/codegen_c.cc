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

void CodeGenC::Init(bool output_ssa) {
  print_ssa_form_ = output_ssa;
}

void CodeGenC::InitFuncState(LoweredFunc f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}
void CodeGenC::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  this->InitFuncState(f);

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
  this->stream << "}\n\n";
}

std::string CodeGenC::Finish() {
  return decl_stream.str() + stream.str();
}

void CodeGenC::PrintExpr(const Expr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.type());
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenC::PrintSSAAssign(
    const std::string& target, const std::string& src, Type t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
  if (src.length() > 3 &&
      src[0] == '(' && src[src.length() - 1] == ')') {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetBufferRef(
    const Variable* buffer,
    Type t, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = volatile_buf_.count(buffer);
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t) || is_vol) {
      os << "((";
      if (is_vol) {
        os << "volatile ";
      }
      if (scope.length() != 0) {
        PrintStorageScope(scope, os);
      }
      os << ' ';
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
    if (HandleTypeMatch(buffer, t) && !is_vol) {
      // optimize for constant access
      int offset;
      if (arith::GetConstInt(index, &offset)) {
        CHECK_EQ(offset % t.lanes(), 0)
            << "Find unaligned vector load to a vector type";
        os << vid << '[' << (offset / t.lanes()) << ']';
        return os.str();
      }
    }
    os << "((";
    if (is_vol) {
      os << "volatile ";
    }
    if (scope.length() != 0) {
      PrintStorageScope(scope, os);
    }
    os << ' ';
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
  return os.str();
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

std::string CodeGenC::GetVecLoad(const Variable* buffer,
                                   Type t, Expr base) {
  return GetBufferRef(buffer, t, base);
}

void CodeGenC::PrintVecStore(const Variable* buffer,
                             Type t, Expr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(buffer, t, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

void CodeGenC::PrintThreadIndexExpr(
    std::string thread_tag, std::ostream& os) { // NOLINT(*)
  os << thread_tag;
}

void CodeGenC::PrintStorageSync(const Call* op) { // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_EQ(scope, "global");
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

void CodeGenC::VisitExpr_(const IntImm *op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const UIntImm *op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImm *op, std::ostream& os) { // NOLINT(*)
  os << "\"" << op->value << "\"";
}

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
void CodeGenC::VisitExpr_(const Cast *op, std::ostream& os) {  // NOLINT(*)
  this->PrintType(op->type, os);
  os << '(';
  this->PrintExpr(op->value, os);
  os << ')';
}
void CodeGenC::VisitExpr_(const Variable *op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenC::VisitExpr_(const Add *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenC::VisitExpr_(const Sub *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenC::VisitExpr_(const Mul *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenC::VisitExpr_(const Div *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenC::VisitExpr_(const Mod *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenC::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenC::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenC::VisitExpr_(const EQ *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenC::VisitExpr_(const NE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenC::VisitExpr_(const LT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenC::VisitExpr_(const LE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenC::VisitExpr_(const GT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenC::VisitExpr_(const GE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenC::VisitExpr_(const And *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenC::VisitExpr_(const Or *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenC::VisitExpr_(const Not *op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenC::VisitExpr_(const Call *op, std::ostream& os) {  // NOLINT(*)
  if (op->call_type == Call::Extern ||
      op->call_type == Call::PureExtern) {
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      this->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    PrintBinaryIntrinsitc(op, " & ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    PrintBinaryIntrinsitc(op, " ^ ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    PrintBinaryIntrinsitc(op, " | ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(~";
    this->PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->is_intrinsic(Call::shift_left)) {
    PrintBinaryIntrinsitc(op, " << ", os, this);
  } else if (op->is_intrinsic(Call::shift_right)) {
    PrintBinaryIntrinsitc(op, " >> ", os, this);
  } else if (op->is_intrinsic(Call::address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    this->PrintType(l->type.element_of(), os);
    os << " *)" << this->GetVarID(l->buffer_var.get())
       << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_api_load_arg)) {
    CHECK_EQ(op->args.size(), 3U);
    if (!op->type.is_handle()) {
      os << '(';
      this->PrintType(op->type, os);
      os << ')';
    }
    os << "(((TVMArg*)";
    this->PrintExpr(op->args[0], os);
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
    this->PrintExpr(op->args[0], os);
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
    this->PrintExpr(op->args[0], os);
    os << " == NULL)";
  } else {
    if (op->call_type == Call::Intrinsic ||
        op->call_type == Call::PureIntrinsic) {
      LOG(FATAL) << "Unresolved intrinsic " << op->name
                 << " with return type " << op->type;
    } else {
      LOG(FATAL) << "Unresolved call type " << op->call_type;
    }
  }
}

void CodeGenC::PrintVecBinaryOp(
    const std::string& op, Type t,
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

void CodeGenC::VisitExpr_(const Load* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    std::string ref = GetBufferRef(op->buffer_var.get(), op->type, op->index);
    os << ref;
  } else {
    Expr base;
    if (TryGetRamp1Base(op->index, op->type.lanes(), &base)) {
      std::string ref = GetVecLoad(op->buffer_var.get(), op->type, base);
      os << ref;
    } else {
      // load seperately.
      std::string svalue = GetUniqueName("_");
      this->PrintIndent();
      this->PrintType(op->type, stream);
      stream << ' ' << svalue << ";\n";
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string vid = GetVarID(op->buffer_var.get());
      Type elem_type = op->type.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          PrintType(elem_type, value_temp);
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

void CodeGenC::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(op->buffer_var.get(), t, op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
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

void CodeGenC::VisitExpr_(const Let* op, std::ostream& os) {  // NOLINT(*)
  CHECK(print_ssa_form_)
      << "LetExpr is only supported by print SSA form";
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
}

void CodeGenC::VisitExpr_(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Ramp: not supported ";
}

void CodeGenC::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenC::VisitExpr_(const Select* op, std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeGenC::VisitStmt_(const LetStmt* op) {
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

void CodeGenC::VisitStmt_(const Allocate* op) {
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

void CodeGenC::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::thread_extent) {
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
  } else if (op->attr_key == ir::attr::storage_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImm>()->value;
  } else if (op->attr_key == ir::attr::volatile_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    volatile_buf_.insert(v);
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AssertStmt* op) {
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

void CodeGenC::VisitStmt_(const For* op) {
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

void CodeGenC::VisitStmt_(const IfThenElse* op) {
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

void CodeGenC::VisitStmt_(const Block *op) {
  PrintStmt(op->first);
  if (op->rest.defined()) PrintStmt(op->rest);
}

void CodeGenC::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();

  if (call && call->is_intrinsic(intrinsic::tvm_storage_sync)) {
    this->PrintStorageSync(call);
  } else {
    std::string vid = this->PrintExpr(op->value);
    this->PrintIndent();
    this->stream << "(void)" << vid << ";\n";
  }
}

void CodeGenC::VisitStmt_(const ProducerConsumer *op) {
  PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
