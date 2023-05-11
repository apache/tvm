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
#include "codegen_hybrid.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>

#include <cctype>
#include <iomanip>

namespace tvm {
namespace contrib {

using runtime::TVMArgs;
using runtime::TVMRetValue;

using namespace tir;

std::string dot_to_underscore(std::string s) {
  for (auto& ch : s)
    if (ch == '.') ch = '_';
  return s;
}

std::string CodeGenHybrid::Finish() { return stream.str(); }

void CodeGenHybrid::PrintType(DataType t, std::ostream& os) {
  if (t.is_float()) {
    os << "float";
    ICHECK(t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  } else if (t.is_int()) {
    os << "int";
    ICHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  } else if (t.is_bfloat16()) {
    os << "bfloat";
    ICHECK(t.bits() == 16);
  } else {
    ICHECK(t.is_uint()) << "Unsupported type " << t;
    os << "uint";
    ICHECK(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
  }
  os << t.bits();
}

void CodeGenHybrid::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << op->value;
}

void CodeGenHybrid::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintType(op->dtype, os);
  os << "(" << std::setprecision(20) << op->value << ")";
}
void CodeGenHybrid::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "'" << op->value << "'";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenHybrid* p) {
  ICHECK(op->dtype.lanes() == 1) << "vec bin op not implemented";
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

inline void PrintBinaryIntrinsitc(const CallNode* op, const char* opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenHybrid* p) {
  ICHECK(op->dtype.lanes() == 1) << "vec bin intrin not implemented";
  ICHECK_EQ(op->args.size(), 2U);
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

void CodeGenHybrid::VisitExpr_(const ProducerLoadNode* op, std::ostream& os) {  // NOLINT(*)
  auto tensor = Downcast<Tensor>(op->producer);

  os << GetTensorID(tensor);
  os << "[";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i) os << ", ";
    std::stringstream idx;
    PrintExpr(op->indices[i], idx);
    os << idx.str();
  }
  os << "]";
}
void CodeGenHybrid::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::bitwise_and())) {
    PrintBinaryIntrinsitc(op, "&", os, this);
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    PrintBinaryIntrinsitc(op, "^", os, this);
  } else if (op->op.same_as(builtin::bitwise_or())) {
    PrintBinaryIntrinsitc(op, "|", os, this);
  } else if (op->op.same_as(builtin::shift_left())) {
    PrintBinaryIntrinsitc(op, "<<", os, this);
  } else if (op->op.same_as(builtin::shift_right())) {
    PrintBinaryIntrinsitc(op, ">>", os, this);
  } else if (op->op.same_as(builtin::bitwise_not())) {
    ICHECK_EQ(op->args.size(), 1U);
    os << "(~";
    PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->op.same_as(builtin::if_then_else())) {
    PrintExpr(op->args[1], os);
    os << " if ";
    PrintExpr(op->args[0], os);
    os << " else ";
    PrintExpr(op->args[2], os);
  } else if (op->op.same_as(builtin::call_pure_extern()) ||
             op->op.same_as(builtin::call_extern())) {
    StringImm fname = Downcast<StringImm>(op->args[0]);
    os << fname << "(";
    for (size_t i = 1; i < op->args.size(); i++) {
      PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  } else {
    auto* ptr_op = op->op.as<OpNode>();
    ICHECK(ptr_op != nullptr);
    std::string name = ptr_op->name;
    ICHECK_EQ(name.compare(0, 4, "tir."), 0);
    os << name.substr(4) << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  }
}

void CodeGenHybrid::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Phase 0 has no BufferLoad(s)!";
}

void CodeGenHybrid::VisitStmt_(const BufferStoreNode* op) {
  LOG(FATAL) << "Phase 0 has no BufferStore(s)!";
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

void CodeGenHybrid::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
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
    ICHECK(iter_var);
    binds_[iter_var->var.get()] = dot_to_underscore(iter_var->var->name_hint);
    PrintIndent();
    stream << "for " << binds_[iter_var->var.get()] << " in bind('" << iter_var->var->name_hint
           << "', ";
    PrintExpr(op->value, stream);
    stream << "):\n";
    indent_ += tab_;
    PrintStmt(op->body);
    indent_ -= tab_;
  } else {
    // For now we ignore the unsupported AttrStmt
    PrintStmt(op->body);
  }
}

void CodeGenHybrid::VisitStmt_(const ProducerRealizeNode* op) {
  auto tensor = Downcast<Tensor>(op->producer);
  if (!op->storage_scope.empty()) {
    PrintIndent();
    stream << GetTensorID(tensor) << " = allocate((";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (i) stream << ", ";
      stream << PrintExpr(op->bounds[i]->extent);
    }
    if (op->bounds.size() == 1) stream << ", ";
    stream << "), '";
    PrintType(tensor->dtype, stream);
    stream << "', '";
    stream << op->storage_scope << "')\n";
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

void CodeGenHybrid::VisitStmt_(const ProducerStoreNode* op) {
  auto tensor = Downcast<Tensor>(op->producer);
  PrintIndent();
  stream << GetTensorID(tensor);
  stream << "[";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i) stream << ", ";
    PrintExpr(op->indices[i], stream);
  }
  stream << "] = ";
  PrintExpr(op->value, stream);
  stream << "\n";
}

void CodeGenHybrid::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = GetVarID(op->loop_var.get());
  stream << "for " << vid << " in "
         << "range(" << extent << "):\n";
  indent_ += tab_;
  PrintStmt(op->body);
  indent_ -= tab_;
}

bool is_noop(const Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (auto eval = stmt.as<EvaluateNode>()) return is_const_int(eval->value);
  return false;
}

void CodeGenHybrid::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if " << cond << ":\n";
  indent_ += tab_;
  PrintStmt(op->then_case);
  indent_ -= tab_;

  if (op->else_case && !is_noop(op->else_case.value())) {
    PrintIndent();
    stream << "else:\n";
    indent_ += tab_;
    PrintStmt(op->else_case.value());
    indent_ -= tab_;
  }
}

void CodeGenHybrid::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt);
  }
}

void CodeGenHybrid::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  std::string str = PrintExpr(op->value);
  if (!str.empty()) stream << str << "\n";
}

void CodeGenHybrid::PrintIndent() { stream << std::string(indent_, ' '); }

std::string CodeGenHybrid::GetVarID(const VarNode* v) {
  if (binds_.count(v)) return binds_[v];
  auto key = std::make_pair(static_cast<const Object*>(v), 0);
  if (id_map_.count(key)) {
    return id_map_[key];
  }
  return id_map_[key] = ids_allocated->FreshName(v->name_hint);
}

std::string CodeGenHybrid::GetTensorID(const Tensor& tensor) {
  auto key = std::make_pair(tensor->op.get(), tensor->value_index);
  if (id_map_.count(key)) {
    return id_map_[key];
  }
  std::string name_hint = tensor->op->name;
  if (tensor->op->num_outputs() > 1) {
    name_hint += "_v" + std::to_string(tensor->value_index);
  }
  return id_map_[key] = ids_allocated->FreshName(name_hint);
}

void CodeGenHybrid::ReserveKeywords() {
  ids_allocated->ReserveName("def");
  ids_allocated->ReserveName("for");
  ids_allocated->ReserveName("in");
  ids_allocated->ReserveName("range");
  ids_allocated->ReserveName("True");
  ids_allocated->ReserveName("False");
  ids_allocated->ReserveName("unroll");
  ids_allocated->ReserveName("const_range");
  ids_allocated->ReserveName("parallel");
  ids_allocated->ReserveName("vectorize");
  ids_allocated->ReserveName("bind");
  ids_allocated->ReserveName("threadIdx.x");
  ids_allocated->ReserveName("threadIdx.y");
  ids_allocated->ReserveName("threadIdx.z");
  ids_allocated->ReserveName("blockIdx.x");
  ids_allocated->ReserveName("blockIdx.y");
  ids_allocated->ReserveName("blockIdx.z");
  ids_allocated->ReserveName("vthread");
  ids_allocated->ReserveName("allocate");
  ids_allocated->ReserveName("output_tensor");
  ids_allocated->ReserveName("sqrt");
  ids_allocated->ReserveName("log");
  ids_allocated->ReserveName("tanh");
  ids_allocated->ReserveName("power");
  ids_allocated->ReserveName("exp");
  ids_allocated->ReserveName("sigmoid");
  ids_allocated->ReserveName("popcount");
  ids_allocated->ReserveName("likely");
  ids_allocated->ReserveName("int8");
  ids_allocated->ReserveName("int16");
  ids_allocated->ReserveName("int32");
  ids_allocated->ReserveName("int64");
  ids_allocated->ReserveName("uint8");
  ids_allocated->ReserveName("uint16");
  ids_allocated->ReserveName("uint32");
  ids_allocated->ReserveName("uint64");
  ids_allocated->ReserveName("float16");
  ids_allocated->ReserveName("float32");
  ids_allocated->ReserveName("float64");
  ids_allocated->ReserveName("ceil_div");
  ids_allocated->ReserveName("max_num_threads");
}

void CodeGenHybrid::DumpStmt(const Stmt& stmt, const Array<ObjectRef>& inputs,
                             const Array<Tensor>& outputs, const std::string& name) {
  ReserveKeywords();
  ids_allocated->ReserveName(name);

  stream << "def " << name << "(";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i) stream << ", ";
    if (auto tensor = inputs[i].as<Tensor>()) {
      stream << GetTensorID(tensor.value());
    } else {
      auto var = inputs[i].as<VarNode>();
      ICHECK(var) << "Input should either be a tensor or a variable!";
      stream << GetVarID(var);
    }
  }
  stream << "):\n";
  indent_ += tab_;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PrintIndent();
    stream << GetTensorID(outputs[i]) << " = output_tensor((";
    for (size_t j = 0; j < outputs[i]->shape.size(); ++j) {
      if (j) stream << ", ";
      PrintExpr(outputs[i]->shape[j], stream);
    }
    if (outputs[i]->shape.size() == 1) stream << ", ";
    stream << "), '" << outputs[i]->dtype << "')\n";
  }
  PrintStmt(stmt);
  PrintIndent();
  stream << "return ";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i) stream << ", ";
    stream << GetTensorID(outputs[i]);
  }
  stream << "\n";
}

TVM_REGISTER_GLOBAL("hybrid._Dump").set_body([](TVMArgs args, TVMRetValue* rv) {
  CodeGenHybrid codegen;
  if (args.size() == 4)
    codegen.DumpStmt(args[0], args[1], args[2], args[3]);
  else
    codegen.DumpStmt(args[0], args[1], args[2]);
  *rv = codegen.Finish();
});
}  // namespace contrib
}  // namespace tvm
