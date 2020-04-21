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
 * \file codegen_c.cc
 */
#include <iomanip>
#include <cctype>
#include "codegen_c.h"
#include "../../arith/pattern_match.h"
#include "../../arith/compute_expr.h"

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenC::Init(bool output_ssa) {
  print_ssa_form_ = output_ssa;
}

void CodeGenC::InitFuncState(const PrimFunc& f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}

void CodeGenC::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  GetUniqueName("extern");
  GetUniqueName("void");
  GetUniqueName("int");
  GetUniqueName("float");
  GetUniqueName("double");
  GetUniqueName("char");
  GetUniqueName("unsigned");
  GetUniqueName("short");
  GetUniqueName("long");
  GetUniqueName("if");
  GetUniqueName("else");
  GetUniqueName("switch");
  GetUniqueName("case");
  GetUniqueName("default");
  GetUniqueName("for");
  GetUniqueName("do");
  GetUniqueName("while");
  GetUniqueName("goto");
  GetUniqueName("register");
  GetUniqueName("continue");
  GetUniqueName("break");
  GetUniqueName("typedef");
  GetUniqueName("struct");
  GetUniqueName("enum");
  GetUniqueName("union");
  GetUniqueName("return");
}

void CodeGenC::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix();
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
        stream << ' ';
      }

      PrintType(GetType(v), stream);
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && restrict_keyword_.length() != 0) {
        stream << ' ' << restrict_keyword_;
      }
    } else {
      PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenC::PrintFuncPrefix() {
  stream << "void";
}

void CodeGenC::PrintFinalReturn() {
}

std::string CodeGenC::Finish() {
  return decl_stream.str() + stream.str();
}

void CodeGenC::PrintExpr(const PrimExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.dtype());
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenC::PrintSSAAssign(
    const std::string& target, const std::string& src, DataType t) {
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
    DataType t, const VarNode* buffer, PrimExpr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = IsVolatile(buffer);
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t) || is_vol) {
      os << "((";
      if (is_vol) {
        os << "volatile ";
      }
      // Scope may not be part of type.
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      os << ' ';
      PrintType(t, os);
      os << "*)" << vid << ')';
    } else {
      os << vid;
    }
    os << "[(";
    PrintExpr(index, os);
    os << ")";
    if (t.bits() == 4 ||
        (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << ']';
  } else {
    // Buffer declared as vector type.
    // optimize for case where it is in register,
    if (HandleTypeMatch(buffer, t) && !is_vol) {
      // optimize for constant access
      if (auto* ptr = index.as<tir::IntImmNode>()) {
        int64_t offset = ptr->value;
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
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, os);
    }
    os << ' ';
    PrintType(t, os);
    os << "*)(";
    if (!HandleTypeMatch(buffer, t.element_of())) {
      os << '(';
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      os << ' ';
      PrintType(t.element_of(), os);
      os << "*)";
    }
    os << vid << " + (";
    PrintExpr(index, os);
    os << ")";
    if (t.bits() == 4 ||
        (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << "))[0]";
  }
  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetStructRef(
    DataType t, const PrimExpr& buffer, const PrimExpr& index, int kind) {
  if (kind < intrinsic::kArrKindBound_) {
    std::ostringstream os;
    os << "(((DLTensor*)";
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

bool CodeGenC::HandleTypeMatch(const VarNode* buf_var, DataType t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const VarNode* buf_var, DataType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    CHECK(it->second == t)
        << "conflicting buf var type";
  }
}

void CodeGenC::PrintVecElemLoad(const std::string& vec,
                                DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeGenC::PrintVecElemStore(const std::string& vec,
                                 DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i
         << " = " << value << ";\n" << std::dec;
}

std::string CodeGenC::GetVecLoad(
    DataType t, const VarNode* buffer, PrimExpr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeGenC::PrintVecStore(const VarNode* buffer,
                             DataType t, PrimExpr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

std::string CodeGenC::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeGenC::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "not implemented";
}

void CodeGenC::PrintStorageSync(const CallNode* op) { // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_EQ(scope, "global");
}

void CodeGenC::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
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


void CodeGenC::PrintType(const Type& type, std::ostream& os) { // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}


inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  if (op->dtype == DataType::Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->dtype, os);
    os << ")" << op->value;
  }
}


inline void PrintUIntConst(DataType dtype, uint64_t val, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  if (dtype == DataType::UInt(32)) {
    std::ostringstream temp;
    temp << val << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(dtype, os);
    os << ")" << val;
  }
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->dtype, os);
      os << ')' << std::scientific <<op->value << 'f';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenC::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenC::VisitExpr_(const FloatImmNode* op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImmNode* op, std::ostream& os) { // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
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
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsic(const CallNode* op,
                                  const char* opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    CHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->args[0], op->args[1], os);
  }
}
void CodeGenC::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
}
void CodeGenC::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenC::VisitExpr_(const AddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenC::VisitExpr_(const SubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenC::VisitExpr_(const MulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenC::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenC::VisitExpr_(const ModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenC::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenC::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenC::VisitExpr_(const EQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenC::VisitExpr_(const NENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenC::VisitExpr_(const LTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenC::VisitExpr_(const LENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenC::VisitExpr_(const GTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenC::VisitExpr_(const GENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenC::VisitExpr_(const AndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenC::VisitExpr_(const OrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenC::VisitExpr_(const NotNode* op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->call_type == CallNode::Extern ||
      op->call_type == CallNode::PureExtern) {
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      this->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  } else if (op->is_intrinsic(CallNode::bitwise_and)) {
    PrintBinaryIntrinsic(op, " & ", os, this);
  } else if (op->is_intrinsic(intrinsic::tvm_large_uint_imm)) {
    CHECK_EQ(op->args.size(), 2U);
    uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
    uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
    uint64_t val = (high << 32U) | low;
    PrintUIntConst(op->dtype, val, os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_xor)) {
    PrintBinaryIntrinsic(op, " ^ ", os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_or)) {
    PrintBinaryIntrinsic(op, " | ", os, this);
  } else if (op->is_intrinsic(CallNode::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(~";
    this->PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->is_intrinsic(CallNode::shift_left)) {
    PrintBinaryIntrinsic(op, " << ", os, this);
  } else if (op->is_intrinsic(CallNode::shift_right)) {
    PrintBinaryIntrinsic(op, " >> ", os, this);
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    os << "(";
    PrintExpr(op->args[0], os);
    os << " ? ";
    PrintExpr(op->args[1], os);
    os << " : ";
    PrintExpr(op->args[2], os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const LoadNode *l = op->args[0].as<LoadNode>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    this->PrintType(l->dtype.element_of(), os);
    os << " *)" << this->GetVarID(l->buffer_var.get())
       << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    os << GetStructRef(
        op->dtype, op->args[0], op->args[1],
        op->args[2].as<IntImmNode>()->value);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(";
    this->PrintExpr(op->args[0], os);
    os << " == NULL)";
  } else if (op->is_intrinsic(CallNode::reinterpret)) {
    // generate (*( TYPE *)(&(ARG)))
    os << "(*(";
    this->PrintType(op->dtype, os);
    os << " *)(&(";
    this->PrintExpr(op->args[0], os);
    os << ")))";
  } else if (op->is_intrinsic(CallNode::isnan)) {
    os << "(";
    this->PrintExpr(op->args[0], os);
    os << " != ";
    this->PrintExpr(op->args[0], os);
    os << ")";
  } else {
    if (op->call_type == CallNode::Intrinsic ||
        op->call_type == CallNode::PureIntrinsic) {
      LOG(FATAL) << "Unresolved intrinsic " << op->name
                 << " with return type " << op->dtype;
    } else {
      LOG(FATAL) << "Unresolved call type " << op->call_type;
    }
  }
}

void CodeGenC::PrintVecBinaryOp(
    const std::string& op, DataType t,
    PrimExpr lhs, PrimExpr rhs, std::ostream& os) {  // NOLINT(*)
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

void CodeGenC::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  // delcare type.
  if (op->dtype.lanes() == 1) {
    std::string ref = GetBufferRef(op->dtype, op->buffer_var.get(), op->index);
    HandleVolatileLoads(ref, op, os);
  } else {
    CHECK(is_one(op->predicate))
        << "predicated load is not supported";

    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(op->index)) {
      std::string ref = GetVecLoad(op->dtype, op->buffer_var.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
              value_temp << ' ';
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenC::VisitStmt_(const StoreNode* op) {
  DataType t = op->value.dtype();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    CHECK(is_one(op->predicate))
        << "Predicated store is not supported";
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, t.lanes()).Match(op->index)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        DataType elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
              stream << ' ';
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, op->index.dtype(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.dtype(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenC::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

void CodeGenC::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  CHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenC::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  LOG(FATAL) << "Shuffle: not supported ";
}

void CodeGenC::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenC::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeGenC::VisitStmt_(const LetStmtNode* op) {
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    CHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* "
             << AllocVarID(op->var.get())
             << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)"  << value << ";\n";
    } else {
      PrintType(op->var.dtype(), this->stream);
      this->stream << ' '
                   << AllocVarID(op->var.get())
                   << " = " << value << ";\n";
    }
  }
  PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AllocateNode* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const VarNode* buffer = op->buffer_var.as<VarNode>();
    std::string scope = alloc_storage_scope_.at(buffer);
    PrintStorageScope(scope, stream);
    stream << ' ';
    PrintType(op->dtype, stream);
    stream << ' '<< vid << '['
           << constant_size << "];\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  } else if (op->attr_key == tir::attr::storage_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImmNode>()->value;
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    CHECK(v);
    volatile_buf_.insert(v);
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AssertStmtNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    stream << "CHECK(" << cond << ") << \"" << str->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
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

void CodeGenC::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt);
  }
}

void CodeGenC::VisitStmt_(const EvaluateNode* op) {
  if (is_const(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->is_intrinsic(intrinsic::tvm_storage_sync)) {
      this->PrintStorageSync(call); return;
    } else if (call->is_intrinsic(intrinsic::tvm_struct_set)) {
      CHECK_EQ(call->args.size(), 4);
      std::string value = PrintExpr(call->args[3]);
      std::string ref = GetStructRef(
          call->args[3].dtype(),
          call->args[0],
          call->args[1],
          call->args[2].as<IntImmNode>()->value);
      this->PrintIndent();
      this->stream << ref << " = " << value << ";\n";
      return;
    }
  }
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << "(void)" << vid << ";\n";
  }
}

void CodeGenC::PrintVecElemLoadExpr(
    DataType t, int i, const std::string& value, std::ostream& os) {
  CHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }

  if (i == 0) {
    os << "((";
    PrintType(t, os);
    os << t.lanes() << ")(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "))";
  }
  return;
}

}  // namespace codegen
}  // namespace tvm
