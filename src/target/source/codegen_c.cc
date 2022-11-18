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
#include "codegen_c.h"

#include <tvm/arith/analyzer.h>

#include <cctype>
#include <iomanip>

#include "../../arith/pattern_match.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenC::Init(bool output_ssa) { print_ssa_form_ = output_ssa; }

void CodeGenC::InitFuncState(const PrimFunc& f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}

void CodeGenC::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->ReserveName("_");
  name_supply_->ReserveName("extern");
  name_supply_->ReserveName("void");
  name_supply_->ReserveName("int");
  name_supply_->ReserveName("float");
  name_supply_->ReserveName("double");
  name_supply_->ReserveName("char");
  name_supply_->ReserveName("unsigned");
  name_supply_->ReserveName("short");
  name_supply_->ReserveName("long");
  name_supply_->ReserveName("if");
  name_supply_->ReserveName("else");
  name_supply_->ReserveName("switch");
  name_supply_->ReserveName("case");
  name_supply_->ReserveName("default");
  name_supply_->ReserveName("for");
  name_supply_->ReserveName("do");
  name_supply_->ReserveName("while");
  name_supply_->ReserveName("goto");
  name_supply_->ReserveName("register");
  name_supply_->ReserveName("continue");
  name_supply_->ReserveName("break");
  name_supply_->ReserveName("typedef");
  name_supply_->ReserveName("struct");
  name_supply_->ReserveName("enum");
  name_supply_->ReserveName("union");
  name_supply_->ReserveName("return");
}

void CodeGenC::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix();
  this->PrintExtraAttrs(f);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
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

      if (no_alias) {
        PrintRestrict(v, stream);
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

void CodeGenC::PrintFuncPrefix() { stream << "void"; }

void CodeGenC::PrintExtraAttrs(const PrimFunc& f) {}

void CodeGenC::PrintFinalReturn() {}

std::string CodeGenC::Finish() { return decl_stream.str() + stream.str(); }

void CodeGenC::PrintExpr(const PrimExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.dtype());
  } else {
    VisitExpr(n, os);
  }
}

static bool CheckOutermostBracketMatch(const std::string& s);

void CodeGenC::PrintSSAAssign(const std::string& target, const std::string& src, DataType t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
  if (CheckOutermostBracketMatch(src)) {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetBufferRef(DataType t, const BufferNode* buffer, PrimExpr index) {
  const VarNode* buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  bool is_vol = IsVolatile(buffer_var);

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = PrintExpr(index);
  if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetStructRef(DataType t, const PrimExpr& buffer, const PrimExpr& index,
                                   int kind) {
  if (kind < builtin::kArrKindBound_) {
    std::ostringstream os;
    os << "(((DLTensor*)";
    this->PrintExpr(buffer, os);
    os << ")";
    if (kind == builtin::kArrAddr) {
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
      case builtin::kArrData:
        os << "data";
        break;
      case builtin::kArrShape:
        os << "shape";
        break;
      case builtin::kArrStrides:
        os << "strides";
        break;
      case builtin::kArrNDim:
        os << "ndim";
        break;
      case builtin::kArrTypeCode:
        os << "dtype.code";
        break;
      case builtin::kArrTypeBits:
        os << "dtype.bits";
        break;
      case builtin::kArrByteOffset:
        os << "byte_offset";
        break;
      case builtin::kArrTypeLanes:
        os << "dtype.lanes";
        break;
      case builtin::kArrDeviceId:
        os << "device.device_id";
        break;
      case builtin::kArrDeviceType:
        os << "device.device_type";
        break;
      default:
        LOG(FATAL) << "unknown field code";
    }
    os << ')';
    return os.str();
  } else {
    ICHECK_LT(kind, builtin::kTVMValueKindBound_);
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
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

void CodeGenC::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeGenC::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i << " = " << value << ";\n" << std::dec;
}

std::string CodeGenC::GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeGenC::PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
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

void CodeGenC::BindThreadIndex(const IterVar& iv) { LOG(FATAL) << "not implemented"; }

void CodeGenC::PrintStorageSync(const CallNode* op) {  // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(scope, "global");
}

inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
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

inline void PrintUIntConst(DataType dtype, uint64_t val, std::ostream& os,
                           CodeGenC* p) {  // NOLINT(*)
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

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
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
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenC::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenC::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
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

inline void PrintBinaryIntrinsic(const CallNode* op, const char* opstr,
                                 std::ostream& os,  // NOLINT(*)
                                 CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    ICHECK_EQ(op->args.size(), 2U);
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
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinaryExpr(op, "%", os, this);
  } else {
    ICHECK(op->dtype.is_float()) << "Expected floating point or integer dtype in Mod, but got "
                                 << op->dtype;
    if (op->dtype.bits() == 32) {
      PrintBinaryExpr(op, "fmodf", os, this);
    } else if (op->dtype.bits() == 64) {
      PrintBinaryExpr(op, "fmod", os, this);
    } else {
      ICHECK(false)
          << "Non single or double precision floating point in Mod, expected 32 or 64 bits but got "
          << op->dtype.bits() << " bits.";
    }
  }
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

void CodeGenC::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                               bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  os << global_symbol << "(";
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

void CodeGenC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);

    if (op->op.same_as(builtin::tvm_check_return())) {
      const CallNode* call = op->args[2].as<CallNode>();
      os << "if (";
      VisitExpr_(call, os);
      os << " != ";
      PrintExpr(op->args[0], os);
      os << " ) return ";
      PrintExpr(op->args[1], os);
    } else if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      ICHECK_GE(op->args.size(), 1U);
      auto func = Downcast<StringImm>(op->args[0]);
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), op_attr_global_symbol_[call_op],
                            op->args, false, os);
    } else if (op->op.same_as(builtin::bitwise_and())) {
      PrintBinaryIntrinsic(op, " & ", os, this);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      ICHECK_EQ(op->args.size(), 2U);
      uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
      uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
      uint64_t val = (high << 32U) | low;
      PrintUIntConst(op->dtype, val, os, this);
    } else if (op->op.same_as(builtin::bitwise_xor())) {
      PrintBinaryIntrinsic(op, " ^ ", os, this);
    } else if (op->op.same_as(builtin::bitwise_or())) {
      PrintBinaryIntrinsic(op, " | ", os, this);
    } else if (op->op.same_as(builtin::bitwise_not())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(~";
      this->PrintExpr(op->args[0], os);
      os << ')';
    } else if (op->op.same_as(builtin::shift_left())) {
      PrintBinaryIntrinsic(op, " << ", os, this);
    } else if (op->op.same_as(builtin::shift_right())) {
      PrintBinaryIntrinsic(op, " >> ", os, this);
    } else if (op->op.same_as(builtin::if_then_else())) {
      os << "(";
      PrintExpr(op->args[0], os);
      os << " ? ";
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    } else if (op->op.same_as(builtin::address_of())) {
      const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
      ICHECK(op->args.size() == 1 && load);
      ICHECK_EQ(load->indices.size(), 1) << "CodeGenC only supports flat memory allocations.";
      os << "(&(" << GetBufferRef(load->dtype, load->buffer.get(), load->indices[0]) << "))";
    } else if (op->op.same_as(builtin::tvm_struct_get())) {
      ICHECK_EQ(op->args.size(), 3U);
      os << GetStructRef(op->dtype, op->args[0], op->args[1], op->args[2].as<IntImmNode>()->value);
    } else if (op->op.same_as(builtin::isnullptr())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " == NULL)";
    } else if (op->op.same_as(builtin::reinterpret())) {
      int ssa_scope = BeginScope();
      std::string rhs = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
      os << "(*(";
      this->PrintType(op->dtype, os);
      os << " *)(&(" << rhs << ")))";
      EndScope(ssa_scope);
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " != ";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else if (op->op.same_as(builtin::lookup_param())) {
      ICHECK_EQ(op->args.size(), 1);
      const StringImmNode* str = op->args[0].as<StringImmNode>();
      ICHECK(str != nullptr);
      os << "__tvm_param__" << str->value;
    } else {
      LOG(FATAL) << "Unresolved call " << op->op;
    }
  } else {
    ICHECK(op->op.as<GlobalVarNode>());
    LOG(FATAL) << "Do not yet support cross function call";
  }
}

void CodeGenC::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os << "(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

void CodeGenC::VisitStmt_(const AllocateConstNode* op) {
  std::string symbol_name = op->buffer_var->name_hint;
  int64_t num_elements = 1;
  const auto& data = op->data.value();

  for (int64_t dim : data.Shape()) {
    num_elements *= dim;
  }

  decl_stream << "\n"
              << "#ifdef __cplusplus\n"
              << "extern \"C\" {\n"
              << "#endif\n"
              << "static const ";

  PrintType(data.DataType(), decl_stream);

  // Allocate the global static variable
  decl_stream << " __attribute__((section(\".rodata.tvm\"), "
              << "aligned(" << constants_byte_alignment_->value << "))) " << symbol_name << "["
              << num_elements << "] = {\n";
  NDArrayDataToC(data, 4, decl_stream);

  decl_stream << "};\n"
              << "#ifdef __cplusplus\n"
              << "}  // extern \"C\"\n"
              << "#endif\n";
  var_idmap_[op->buffer_var.operator->()] = symbol_name;
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const DeclBufferNode* op) { this->PrintStmt(op->body); }

void CodeGenC::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Unexpected deprecated LoadNode.  Use BufferLoadNode instead.";
}

void CodeGenC::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  int lanes = op->dtype.lanes();
  // delcare type.
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
    HandleVolatileLoads(ref, op, os);
  } else {
    bool can_vector_load = false;
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(index)) {
      const RampNode* ramp = index.as<RampNode>();
      ICHECK(ramp);
      arith::ModularSet me = arith::Analyzer().modular_set(ramp->base);
      // The condition: {k * coeff + base} divisible by the alignment for any k
      if (me->coeff % op->dtype.lanes() == 0 && me->base % op->dtype.lanes() == 0) {
        can_vector_load = true;
      }
    }

    if (can_vector_load) {
      std::string ref = GetVecLoad(op->dtype, op->buffer.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(index), index.dtype());
      std::string vid = GetVarID(buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenC::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected deprecated StoreNode.  Use BufferStoreNode instead.";
}

void CodeGenC::VisitStmt_(const BufferStoreNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->buffer->data;

  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(value_dtype, op->buffer.get(), index_expr);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    arith::PVar<PrimExpr> base;

    if (arith::ramp(base, 1, value_dtype.lanes()).Match(index_expr)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer.get(), value_dtype, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements separately
      std::string index = SSAGetID(PrintExpr(index_expr), index_expr.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      std::string vid = GetVarID(buffer_var.get());
      for (int i = 0; i < value_dtype.lanes(); ++i) {
        this->PrintIndent();
        DataType elem_type = value_dtype.element_of();
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          stream << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, index_expr.dtype(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.dtype(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenC::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  std::string value = PrintExpr(op->value);
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

void CodeGenC::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

void CodeGenC::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  LOG(FATAL) << "Shuffle: not supported ";
}

void CodeGenC::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
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
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {
      PrintType(op->var.dtype(), this->stream);
      this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
  }
  PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  PrintStorageScope(scope, stream);

  PrintType(op->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

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
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    volatile_buf_.insert(v);
  } else if (op->attr_key == tir::attr::pragma_import_c) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    ICHECK(value != nullptr);
    decl_stream << value->value;
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AssertStmtNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    stream << "ICHECK(" << cond << ") << \"" << str->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  stream << "while (" << PrintExpr(op->condition) << ") {\n";
  int while_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(while_scope);
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

  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
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
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->op.same_as(builtin::tvm_storage_sync())) {
      this->PrintStorageSync(call);
      return;
    } else if (call->op.same_as(builtin::tvm_struct_set())) {
      ICHECK_EQ(call->args.size(), 4);
      int kind = call->args[2].as<IntImmNode>()->value;
      std::string ref = GetStructRef(call->args[3].dtype(), call->args[0], call->args[1], kind);
      std::string value = PrintExpr(call->args[3]);
      std::string cast;
      if (kind == builtin::kArrStrides) {
        // cast void* to int64_t*
        cast = call->args[3]->dtype.is_handle() ? "(int64_t*)" : "";
      } else if (kind == builtin::kArrDeviceType) {
        // cast int to enum
        cast = "(DLDeviceType)";
      }
      this->PrintIndent();
      this->stream << ref << " = " << cast << value << ";\n";
      return;
    }
  }
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << vid << ";\n";
  }
}

void CodeGenC::PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
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
    os << ")(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "))";
  }
  return;
}

void CodeGenC::PrintRestrict(const Var& v, std::ostream& os) {
  if (restrict_keyword_.length() != 0) {
    os << ' ' << restrict_keyword_;
  }
}

static bool CheckOutermostBracketMatch(const std::string& s) {
  if (!s.empty() && s.front() == '(' && s.back() == ')') {
    size_t len = s.size();
    int n_unmatched = 0;
    for (size_t i = 0; i < len; ++i) {
      if (s[i] == '(') {
        n_unmatched++;
      } else if (s[i] == ')') {
        n_unmatched--;
      }
      if (n_unmatched == 0) {
        return i == len - 1;
      }
    }
  }
  return false;
}

}  // namespace codegen
}  // namespace tvm
