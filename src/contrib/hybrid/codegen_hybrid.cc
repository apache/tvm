/*!  Copyright (c) 2019 by Contributors
 * \file codegen_hybrid.cc
 */
#include <iomanip>
#include <cctype>
#include "codegen_hybrid.h"

namespace tvm {
namespace contrib {

using namespace ir;

std::string CodeGenHybrid::GetUniqueName(std::string prefix) {
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] == '.') prefix[i] = '_';
  }
  auto it = ids_allocated_.find(prefix);
  if (it != ids_allocated_.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      std::string name = os.str();
      if (ids_allocated_.count(name) == 0) {
        prefix = name;
        break;
      }
    }
  }
  ids_allocated_[prefix] = 0;
  return prefix;
}

std::string CodeGenHybrid::Finish() {
  return stream.str();
}

void CodeGenHybrid::PrintType(Type t, std::ostream &os) {
  if (t.is_float()) {
    os << "float";
    CHECK(t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  } else if (t.is_int()) {
    os << "int";
    CHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  } else {
    CHECK(t.is_uint()) << "Unsupported type " << t;
    os << "uint";
    CHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  }
  os << t.bits();
}

void CodeGenHybrid::VisitExpr_(const IntImm *op, std::ostream& os) {  // NOLINT(*)
  os << op->value;
}
void CodeGenHybrid::VisitExpr_(const UIntImm *op, std::ostream& os) {  // NOLINT(*)
  PrintType(op->type, os);
  os << "(" << op->value << ")";
}
void CodeGenHybrid::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  os << std::setprecision(20) << op->value;
}
void CodeGenHybrid::VisitExpr_(const StringImm *op, std::ostream& os) { // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char *opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenHybrid* p) {
  CHECK(op->type.lanes() == 1)  << "vec bin op not implemented";
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
}

inline void PrintBinaryIntrinsitc(const Call* op,
                                  const char *opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenHybrid* p) {
  CHECK(op->type.lanes() == 1)  << "vec bin intrin not implemented";
  CHECK_EQ(op->args.size(), 2U);
  os << '(';
  p->PrintExpr(op->args[0], os);
  os << opstr;
  p->PrintExpr(op->args[1], os);
  os << ')';
}

void CodeGenHybrid::VisitExpr_(const Cast *op, std::ostream& os) {  // NOLINT(*)
  if (op->type == op->value.type()) {
    PrintExpr(op->value, stream);
  } else {
    PrintType(op->type, os);
    os << "(";
    PrintExpr(op->value, os);
    os << ")";
  }
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
  if (op->call_type == Call::Halide) {
    os << GetTensorID(op->func, op->value_index);
    os << "[";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i) os << ", ";
      std::stringstream idx;
      PrintExpr(op->args[i], idx);
      os << idx.str();
    }
    os << "]";
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    PrintBinaryIntrinsitc(op, "&", os, this);
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    PrintBinaryIntrinsitc(op, "^", os, this);
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    PrintBinaryIntrinsitc(op, "|", os, this);
  } else if (op->is_intrinsic(Call::shift_left)) {
    PrintBinaryIntrinsitc(op, "<<", os, this);
  } else if (op->is_intrinsic(Call::shift_right)) {
    PrintBinaryIntrinsitc(op, ">>", os, this);
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(~";
    PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    PrintExpr(op->args[1], os);
    os << " if ";
    PrintExpr(op->args[0], os);
    os << " else ";
    PrintExpr(op->args[2], os);
  } else {
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  }
}

void CodeGenHybrid::VisitExpr_(const Load* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Phase 0 has no Load(s)!";
}

void CodeGenHybrid::VisitStmt_(const Store* op) {
  LOG(FATAL) << "Phase 0 has no Store(s)!";
}

void CodeGenHybrid::VisitExpr_(const Let* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Phase 0 has no Let(s)!";
}

void CodeGenHybrid::VisitStmt_(const Allocate* op) {
  LOG(FATAL) << "Phase 0 has no Allocate(s)!";
}

void CodeGenHybrid::VisitExpr_(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Ramp to be supported yet";
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
  stream << GetVarID(op->var.get()) << " = " << value << ";\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AttrStmt* op) {
  // TODO(@were): Support thread and buffer binding
  if (op->attr_key == ir::attr::thread_extent) {
    LOG(FATAL) << "Thread binding support yet!\n";
  } else if (op->attr_key == ir::attr::realize_scope) {
    auto v = FunctionRef(op->node.node_);
    alloc_storage_scope_[v] = op->value.as<StringImm>()->value;
  }
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const Realize *op) {
  CHECK(alloc_storage_scope_.count(op->func));
  if (!alloc_storage_scope_[op->func].empty()) {
    PrintIndent();
    stream << GetTensorID(op->func, op->value_index) << " = allocate((";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (i) stream << ", ";
      stream << PrintExpr(op->bounds[i]->extent);
    }
    if (op->bounds.size() == 1) stream << ", ";
    stream << "), \"";
    PrintType(op->type, stream); 
    stream << "\", '";
    stream << alloc_storage_scope_[op->func] << "')\n";
  }
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AssertStmt* op) {
  PrintIndent();
  stream << "assert ";
  PrintExpr(op->condition, stream);
  stream << ", ";
  PrintExpr(op->message, stream);
  stream << "\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const Provide* op) {
  PrintIndent();
  stream << GetTensorID(op->func, op->value_index);
  stream << "[";
  for (size_t i = 0; i < op->args.size(); ++i) {
    if (i) stream << ", ";
    PrintExpr(op->args[i], stream);
  }
  stream << "] = ";
  PrintExpr(op->value, stream);
  stream << "\n";
}

void CodeGenHybrid::VisitStmt_(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = GetVarID(op->loop_var.get());
  stream << "for " << vid << " in " << "range(" << extent << "):\n";
  indent_ += tab_;
  PrintStmt(op->body);
  indent_ -= tab_;
}

void CodeGenHybrid::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if " << cond << ":\n";
  indent_ += tab_;
  PrintStmt(op->then_case);
  indent_ -= tab_;

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "else:\n";
    indent_ += tab_;
    PrintStmt(op->else_case);
    indent_ -= tab_;
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
  PrintIndent();
  stream << "# producing " << op->func->func_name() << "\n";
  PrintStmt(op->body);
  PrintIndent();
  stream << "# produced " << op->func->func_name() << "\n";
}

void CodeGenHybrid::PrintIndent() {
  stream << std::string(indent_, ' ');
}

std::string CodeGenHybrid::GetVarID(const Variable *v) {
  auto node = v->GetNodePtr().get();
  if (id_map_.count(node)) {
    return id_map_[node];
  }
  return id_map_[node] = GetUniqueName(v->name_hint);
}

std::string CodeGenHybrid::GetTensorID(const FunctionRef &func, int value_index) {
  auto node = func.get();
  if (id_map_.count(node)) {
    return id_map_[node];
  }
  std::string name_hint = func->func_name();
  if (func->num_outputs() > 1) {
    name_hint += "_v" + std::to_string(value_index);
  }
  return id_map_[node] = GetUniqueName(name_hint);
}

void CodeGenHybrid::DumpStmt(const Stmt &stmt,
                             const Array<NodeRef> &inputs,
                             const Array<Tensor> &outputs,
                             const std::string &name) {
  stream << "@tvm.hybrid.script\n"
         << "def " << name << "(";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i) stream << ", ";
    if (auto tensor = inputs[i].as<TensorNode>()) {
      stream << GetTensorID(tensor->op, tensor->value_index);
    } else {
      auto var = inputs[i].as<Variable>();
      CHECK(var) << "Input should either be a tensor or a variable!";
      stream << GetVarID(var);
    }
  }
  stream << "):\n";
  indent_ += tab_;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PrintIndent();
    stream << GetTensorID(outputs[i]->op, outputs[i]->value_index)
           << " = output_tensor((";
    for (size_t j = 0; j < outputs[i]->shape.size(); ++j) {
      if (j) stream << ", ";
      PrintExpr(outputs[i]->shape[j], stream);
    }
    if (outputs[i]->shape.size() == 1)
      stream << ", ";
    stream << "), '" << outputs[i]->dtype << "')\n";
  }
  PrintStmt(stmt);
  PrintIndent();
  stream << "return ";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i) stream << ", ";
    stream << GetTensorID(outputs[i]->op, outputs[i]->value_index);
  }
  stream << "\n";
}

TVM_REGISTER_GLOBAL("hybrid.dump")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    CodeGenHybrid codegen;
    if (args.size() == 4)
      codegen.DumpStmt(args[0], args[1], args[2], args[3]);
    else
      codegen.DumpStmt(args[0], args[1], args[2]);
    *rv = codegen.Finish();
  });
}  // namespace contrib
}  // namespace tvm
