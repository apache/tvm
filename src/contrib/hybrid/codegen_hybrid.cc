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
 * \file codegen_hybrid.cc
 */
#include <tvm/runtime/registry.h>
#include <iomanip>
#include <cctype>
#include "codegen_hybrid.h"

namespace tvm {
namespace contrib {

using runtime::TVMArgs;
using runtime::TVMRetValue;

using namespace tir;

std::string dot_to_underscore(std::string s) {
  for (auto &ch : s)
    if (ch == '.') ch = '_';
  return s;
}

std::string CodeGenHybrid::GetUniqueName(std::string prefix) {
  prefix = dot_to_underscore(prefix);
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

void CodeGenHybrid::PrintType(DataType t, std::ostream &os) {
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

void CodeGenHybrid::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << op->value;
}

void CodeGenHybrid::VisitExpr_(const FloatImmNode* op, std::ostream& os) { // NOLINT(*)
  PrintType(op->dtype, os);
  os << "(" << std::setprecision(20) << op->value << ")";
}
void CodeGenHybrid::VisitExpr_(const StringImmNode* op, std::ostream& os) { // NOLINT(*)
  os << "'" << op->value << "'";
}

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenHybrid* p) {
  CHECK(op->dtype.lanes() == 1)  << "vec bin op not implemented";
  if (isalpha(opstr[0])) {
    os << opstr << '(';
    p->PrintExpr(op->a, os);
    os << ", ";
    p->PrintExpr(op->b, os);
    os << ')';
  } else {
    os << '(';
    p->PrintExpr(op->a, os);
    if (!strcmp(opstr, "&&")) opstr = "and";
    if (!strcmp(opstr, "||")) opstr = "or";
    os << ' ' << opstr << ' ';
    p->PrintExpr(op->b, os);
    os << ')';
  }
}

inline void PrintBinaryIntrinsitc(const CallNode* op,
                                  const char* opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenHybrid* p) {
  CHECK(op->dtype.lanes() == 1)  << "vec bin intrin not implemented";
  CHECK_EQ(op->args.size(), 2U);
  os << '(';
  p->PrintExpr(op->args[0], os);
  os << opstr;
  p->PrintExpr(op->args[1], os);
  os << ')';
}

void CodeGenHybrid::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->dtype == op->value.dtype()) {
    PrintExpr(op->value, stream);
  } else {
    PrintType(op->dtype, os);
    os << "(";
    PrintExpr(op->value, os);
    os << ")";
  }
}

void CodeGenHybrid::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenHybrid::VisitExpr_(const AddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenHybrid::VisitExpr_(const SubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenHybrid::VisitExpr_(const MulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}

void CodeGenHybrid::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->dtype.is_int())
    PrintBinaryExpr(op, "//", os, this);
  else
    PrintBinaryExpr(op, "/", os, this);
}

void CodeGenHybrid::VisitExpr_(const FloorDivNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->dtype.is_int())
    PrintBinaryExpr(op, "//", os, this);
  else
    PrintBinaryExpr(op, "/", os, this);
}

void CodeGenHybrid::VisitExpr_(const ModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}

void CodeGenHybrid::VisitExpr_(const FloorModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenHybrid::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenHybrid::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenHybrid::VisitExpr_(const EQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenHybrid::VisitExpr_(const NENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenHybrid::VisitExpr_(const LTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenHybrid::VisitExpr_(const LENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenHybrid::VisitExpr_(const GTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenHybrid::VisitExpr_(const GENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenHybrid::VisitExpr_(const AndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenHybrid::VisitExpr_(const OrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenHybrid::VisitExpr_(const NotNode* op, std::ostream& os) {  // NOLINT(*)
  os << "not ";
  PrintExpr(op->a, os);
}

void CodeGenHybrid::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->call_type == CallNode::Halide) {
    os << GetTensorID(op->func, op->value_index);
    os << "[";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i) os << ", ";
      std::stringstream idx;
      PrintExpr(op->args[i], idx);
      os << idx.str();
    }
    os << "]";
  } else if (op->is_intrinsic(CallNode::bitwise_and)) {
    PrintBinaryIntrinsitc(op, "&", os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_xor)) {
    PrintBinaryIntrinsitc(op, "^", os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_or)) {
    PrintBinaryIntrinsitc(op, "|", os, this);
  } else if (op->is_intrinsic(CallNode::shift_left)) {
    PrintBinaryIntrinsitc(op, "<<", os, this);
  } else if (op->is_intrinsic(CallNode::shift_right)) {
    PrintBinaryIntrinsitc(op, ">>", os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_not)) {
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

void CodeGenHybrid::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Phase 0 has no Load(s)!";
}

void CodeGenHybrid::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Phase 0 has no Store(s)!";
}

void CodeGenHybrid::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Phase 0 has no Let(s)!";
}

void CodeGenHybrid::VisitStmt_(const AllocateNode* op) {
  LOG(FATAL) << "Phase 0 has no Allocate(s)!";
}

void CodeGenHybrid::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Ramp to be supported yet";
}

void CodeGenHybrid::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenHybrid::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  PrintExpr(op->true_value, os);
  os << " if ";
  PrintExpr(op->condition, os);
  os << " else ";
  PrintExpr(op->false_value, os);
  os << "\n";
}

void CodeGenHybrid::VisitStmt_(const LetStmtNode* op) {
  std::string value = PrintExpr(op->value);
  stream << GetVarID(op->var.get()) << " = " << value << ";\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    auto iter_var = op->node.as<IterVarNode>();
    CHECK(iter_var);
    binds_[iter_var->var.get()] = dot_to_underscore(iter_var->var->name_hint);
    PrintIndent();
    stream << "for " << binds_[iter_var->var.get()] << " in bind('"
           << iter_var->var->name_hint << "', ";
    PrintExpr(op->value, stream);
    stream << "):\n";
    indent_ += tab_;
    PrintStmt(op->body);
    indent_ -= tab_;
  } else if (op->attr_key == tir::attr::realize_scope) {
    auto v = Downcast<FunctionRef>(op->node);
    alloc_storage_scope_[v] = op->value.as<StringImmNode>()->value;
    PrintStmt(op->body);
  } else {
    // For now we ignore the unsupported AttrStmt
    PrintStmt(op->body);
  }
}

void CodeGenHybrid::VisitStmt_(const RealizeNode* op) {
  CHECK(alloc_storage_scope_.count(op->func));
  if (!alloc_storage_scope_[op->func].empty()) {
    PrintIndent();
    stream << GetTensorID(op->func, op->value_index) << " = allocate((";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (i) stream << ", ";
      stream << PrintExpr(op->bounds[i]->extent);
    }
    if (op->bounds.size() == 1) stream << ", ";
    stream << "), '";
    PrintType(op->dtype, stream);
    stream << "', '";
    stream << alloc_storage_scope_[op->func] << "')\n";
  }
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const AssertStmtNode* op) {
  PrintIndent();
  stream << "assert ";
  PrintExpr(op->condition, stream);
  stream << ", ";
  PrintExpr(op->message, stream);
  stream << "\n";
  PrintStmt(op->body);
}

void CodeGenHybrid::VisitStmt_(const ProvideNode* op) {
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

void CodeGenHybrid::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = GetVarID(op->loop_var.get());
  stream << "for " << vid << " in " << "range(" << extent << "):\n";
  indent_ += tab_;
  PrintStmt(op->body);
  indent_ -= tab_;
}

bool is_noop(const Stmt &stmt) {
  if (!stmt.defined())
    return true;
  if (auto eval = stmt.as<EvaluateNode>())
    return is_const(eval->value);
  return false;
}

void CodeGenHybrid::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if " << cond << ":\n";
  indent_ += tab_;
  PrintStmt(op->then_case);
  indent_ -= tab_;

  if (!is_noop(op->else_case)) {
    PrintIndent();
    stream << "else:\n";
    indent_ += tab_;
    PrintStmt(op->else_case);
    indent_ -= tab_;
  }
}

void CodeGenHybrid::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt);
  }
}

void CodeGenHybrid::VisitStmt_(const EvaluateNode* op) {
  if (is_const(op->value)) return;
  std::string str = PrintExpr(op->value);
  if (!str.empty())
    stream << str << "\n";
}

void CodeGenHybrid::PrintIndent() {
  stream << std::string(indent_, ' ');
}

std::string CodeGenHybrid::GetVarID(const VarNode *v) {
  if (binds_.count(v))
    return binds_[v];
  auto key = std::make_pair(static_cast<const Object*>(v), 0);
  if (id_map_.count(key)) {
    return id_map_[key];
  }
  return id_map_[key] = GetUniqueName(v->name_hint);
}

std::string CodeGenHybrid::GetTensorID(const FunctionRef &func, int value_index) {
  auto key = std::make_pair(func.get(), value_index);
  if (id_map_.count(key)) {
    return id_map_[key];
  }
  std::string name_hint = func->func_name();
  if (func->num_outputs() > 1) {
    name_hint += "_v" + std::to_string(value_index);
  }
  return id_map_[key] = GetUniqueName(name_hint);
}

void CodeGenHybrid::ReserveKeywords() {
  GetUniqueName("def");
  GetUniqueName("for");
  GetUniqueName("in");
  GetUniqueName("range");
  GetUniqueName("True");
  GetUniqueName("False");
  GetUniqueName("unroll");
  GetUniqueName("const_range");
  GetUniqueName("parallel");
  GetUniqueName("vectorize");
  GetUniqueName("bind");
  GetUniqueName("threadIdx.x");
  GetUniqueName("threadIdx.y");
  GetUniqueName("threadIdx.z");
  GetUniqueName("blockIdx.x");
  GetUniqueName("blockIdx.y");
  GetUniqueName("blockIdx.z");
  GetUniqueName("vthread");
  GetUniqueName("allocate");
  GetUniqueName("output_tensor");
  GetUniqueName("sqrt");
  GetUniqueName("log");
  GetUniqueName("tanh");
  GetUniqueName("power");
  GetUniqueName("exp");
  GetUniqueName("sigmoid");
  GetUniqueName("popcount");
  GetUniqueName("likely");
  GetUniqueName("int8");
  GetUniqueName("int16");
  GetUniqueName("int32");
  GetUniqueName("int64");
  GetUniqueName("uint8");
  GetUniqueName("uint16");
  GetUniqueName("uint32");
  GetUniqueName("uint64");
  GetUniqueName("float16");
  GetUniqueName("float32");
  GetUniqueName("float64");
  GetUniqueName("ceil_div");
  GetUniqueName("max_num_threads");
}

void CodeGenHybrid::DumpStmt(const Stmt &stmt,
                             const Array<ObjectRef> &inputs,
                             const Array<Tensor> &outputs,
                             const std::string &name) {
  ReserveKeywords();
  GetUniqueName(name);

  stream << "def " << name << "(";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i) stream << ", ";
    if (auto tensor = inputs[i].as<TensorNode>()) {
      stream << GetTensorID(tensor->op, tensor->value_index);
    } else {
      auto var = inputs[i].as<VarNode>();
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

TVM_REGISTER_GLOBAL("hybrid._Dump")
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
