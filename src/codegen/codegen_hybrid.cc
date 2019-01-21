/*!  Copyright (c) 2019 by Contributors
 * \file codegen_hybrid.cc
 */
#include <iomanip>
#include <cctype>
#include "codegen_hybrid.h"
#include "../pass/ir_util.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

using namespace ir;

void CodeGenHybrid::Init(bool simple_mode) {
  simple_mode_ = simple_mode;
}

void CodeGenHybrid::InitFuncState(LoweredFunc f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}

std::string CodeGenHybrid::GetVarID(const Variable* v) {
  auto it = var_idmap_.find(v);
  if (!simple_mode_) {
    CHECK(it != var_idmap_.end()) << "Find undefined Variable " << v->name_hint;
  } else {
    if (it == var_idmap_.end())
      return AllocVarID(v);
  }
  return it->second;
}

void CodeGenHybrid::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  GetUniqueName("def");
  GetUniqueName("for");
  GetUniqueName("in");
  GetUniqueName("range");
  GetUniqueName("unroll");
  GetUniqueName("vectorize");
  GetUniqueName("parallel");
  GetUniqueName("if");
  GetUniqueName("else");
  GetUniqueName("and");
  GetUniqueName("or");
  GetUniqueName("not");
}

void CodeGenHybrid::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  stream << "def " << f->name << "(";
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    stream << ' ' << v->name_hint;
  }
  stream << "):\n";
  int func_scope = BeginScope();
  PrintStmt(f->body);
  EndScope(func_scope);
}

std::string CodeGenHybrid::Finish() {
  return decl_stream.str() + stream.str();
}

void CodeGenHybrid::PrintExpr(const Expr& n, std::ostream& os) {  // NOLINT(*)
  VisitExpr(n, os);
}

void CodeGenHybrid::PrintSSAAssign(const std::string& target, const std::string& src, Type t) {
  LOG(FATAL) << "Python backend does not support SSA format.";
}

// Print a reference expression to a buffer.
std::string CodeGenHybrid::GetBufferRef(
    Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  os << vid << "[";
  PrintExpr(index, os);
  os << "]";
  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeGenHybrid::GetStructRef(
    Type t, const Expr& buffer, const Expr& index, int kind) {
  if (kind < intrinsic::kArrKindBound_) {
    std::ostringstream os;
    os << "(((TVMArray*)";
    this->PrintExpr(buffer, os);
    os << ")";
    if (kind == intrinsic::kArrAddr) {
      os << " + ";
      this->PrintExpr(index, os);
      os << ")";
      return os.str();
    }
    os << '[';
    this->PrintExpr(index, os);
    os << "].";
    // other case: get fields.
    switch (kind) {
      case intrinsic::kArrData: os << "data"; break;
      case intrinsic::kArrShape: os << "shape"; break;
      case intrinsic::kArrStrides: os << "strides"; break;
      case intrinsic::kArrNDim: os << "ndim"; break;
      case intrinsic::kArrTypeCode: os << "dtype.code"; break;
      case intrinsic::kArrTypeBits: os << "dtype.bits"; break;
      case intrinsic::kArrByteOffset: os << "byte_offset"; break;
      case intrinsic::kArrTypeLanes: os << "dtype.lanes"; break;
      case intrinsic::kArrDeviceId: os << "ctx.device_id"; break;
      case intrinsic::kArrDeviceType: os << "ctx.device_type"; break;
      default: LOG(FATAL) << "unknown field code";
    }
    os << ')';
    return os.str();
  } else {
    CHECK_LT(kind, intrinsic::kTVMValueKindBound_);
    std::ostringstream os;
    os << "(((TVMValue*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].";
    if (t.is_handle()) {
      os << "v_handle";
    } else if (t.is_float()) {
      os << "v_float64";
    } else if (t.is_int()) {
      os << "v_int64";
    } else {
      LOG(FATAL) << "Do not know how to handle type" << t;
    }
    os << ")";
    return os.str();
  }
}


bool CodeGenHybrid::HandleTypeMatch(const Variable* buf_var, Type t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenHybrid::RegisterHandleType(const Variable* buf_var, Type t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    CHECK(it->second == t) << "conflicting buf var type";
  }
}

std::string CodeGenHybrid::CastFromTo(std::string value, Type from, Type target) {
  if (from == target) return value;
  std::ostringstream os;
  this->PrintType(target, os);
  os << "(" << value << ")";
  return os.str();
}

void CodeGenHybrid::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "to be implemented";
}

void CodeGenHybrid::PrintStorageSync(const Call* op) { // NOLINT(*)
  LOG(FATAL) << "to be implemented";
}

void CodeGenHybrid::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_EQ(scope, "global");
}

void CodeGenHybrid::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  CHECK_EQ(t.lanes(), 1) << "do not yet support vector types";
  CHECK(!t.is_handle()) << "Buffer type cannot be a handle!";
  if (t.is_float()) {
    CHECK(t.bits() == 32 || t.bits() == 64);
    os << "float" << t.bits();
  } else if (t.is_uint() || t.is_int()) {
    switch (t.bits()) {
      case 8: case 16: case 32: case 64: {
        os << "int" << t.bits(); return;
      }
      case 1: os << "int"; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Python type";
}

void CodeGenHybrid::VisitExpr_(const IntImm *op, std::ostream& os) {  // NOLINT(*)
  os << op->value;
}
void CodeGenHybrid::VisitExpr_(const UIntImm *op, std::ostream& os) {  // NOLINT(*)
  os << op->value;
}
void CodeGenHybrid::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  os << std::scientific << op->value;
}
void CodeGenHybrid::VisitExpr_(const StringImm *op, std::ostream& os) { // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char *opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenHybrid* p) {
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
    LOG(FATAL) << "vec bin op to be implemented";
  }
}

inline void PrintBinaryIntrinsitc(const Call* op,
                                  const char *opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenHybrid* p) {
  if (op->type.lanes() == 1) {
    CHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    LOG(FATAL) << "vec bin intrin to be implemented";
  }
}
void CodeGenHybrid::VisitExpr_(const Cast *op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.type(), op->type);
}
void CodeGenHybrid::VisitExpr_(const Variable *op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenHybrid::VisitExpr_(const Add *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenHybrid::VisitExpr_(const Sub *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenHybrid::VisitExpr_(const Mul *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenHybrid::VisitExpr_(const Div *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenHybrid::VisitExpr_(const Mod *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenHybrid::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenHybrid::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenHybrid::VisitExpr_(const EQ *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenHybrid::VisitExpr_(const NE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenHybrid::VisitExpr_(const LT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenHybrid::VisitExpr_(const LE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenHybrid::VisitExpr_(const GT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenHybrid::VisitExpr_(const GE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenHybrid::VisitExpr_(const And *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenHybrid::VisitExpr_(const Or *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenHybrid::VisitExpr_(const Not *op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenHybrid::VisitExpr_(const Call *op, std::ostream& os) {  // NOLINT(*)
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
  } /*else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    os << "(";
    PrintExpr(op->args[0], os);
    os << " ? ";
    PrintExpr(op->args[1], os);
    os << " : ";
    PrintExpr(op->args[2], os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    this->PrintType(l->type.element_of(), os);
    os << " *)" << this->GetVarID(l->buffer_var.get())
       << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    os << GetStructRef(
        op->type, op->args[0], op->args[1],
        op->args[2].as<IntImm>()->value);
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
  }*/
}

void CodeGenHybrid::VisitExpr_(const Load* op, std::ostream& os) {  // NOLINT(*)
  // int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    std::string ref = GetBufferRef(op->type, op->buffer_var.get(), op->index);
    os << ref;
  } else {
    LOG(FATAL) << "vec load to be supported";
  }
}

void CodeGenHybrid::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << "\n";
  } else {
    LOG(FATAL) << "Vectorized store is not supported yet...";
  }
}

void CodeGenHybrid::VisitExpr_(const Let* op, std::ostream& os) {  // NOLINT(*)
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

void CodeGenHybrid::VisitExpr_(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  // TODO(@were): Support vectorization access in both frontend and backend
  LOG(FATAL) << "ramp to be supported yet";
}

void CodeGenHybrid::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenHybrid::VisitExpr_(const Select* op, std::ostream& os) {  // NOLINT(*)
  PrintExpr(op->true_value, os);
  os << " if ";
  PrintExpr(op->condition, os);
  os << " else ";
  PrintExpr(op->false_value, os);
  os << "\n";
}

void CodeGenHybrid::VisitStmt_(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  stream << AllocVarID(op->var.get())
         << " = " << value << ";\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  PrintIndent();
  stream << vid << " = allocate((";
  for (size_t i = 0; i < op->extents.size(); ++i) {
    if (!i) stream << ", ";
    stream << PrintExpr(op->extents[i]);
  }
  stream << "), \"" << op-> type << "\")\n";
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::thread_extent) {
    LOG(FATAL) << "Thread binding support yet!\n";
  } else if (op->attr_key == ir::attr::storage_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImm>()->value;
  } else if (op->attr_key == ir::attr::volatile_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    volatile_buf_.insert(v);
  }
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AssertStmt* op) {
  //TODO(@were): Support AssertStmt in both hybrid parser and here
  LOG(FATAL) << "assert to be supported yet!\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for " << vid << " in " << "range(" << extent << "):\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  EndScope(for_scope);
  PrintIndent();
}

void CodeGenHybrid::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if " << cond << ":\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    EndScope(else_scope);
  }
}

void CodeGenHybrid::VisitStmt_(const Block *op) {
  PrintStmt(op->first);
  if (op->rest.defined()) PrintStmt(op->rest);
}

void CodeGenHybrid::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  std::string str = PrintExpr(op->value);
  if (!str.empty())
    stream << str << "\n";
}

void CodeGenHybrid::VisitStmt_(const ProducerConsumer *op) {
  PrintStmt(op->body);
}

TVM_REGISTER_API("hybrid._HybridDump")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Stmt stmt;
    if (args[0].IsNodeType<Stmt>()) {
      stmt = args[0];
    } else if (args[0].IsNodeType<Expr>()) {
      stmt = Evaluate::make(args[0]);
    }
    CodeGenHybrid generator;
    generator.PrintStmt(stmt);
    *ret = generator.Finish();
  });
}  // namespace codegen
}  // namespace tvm
