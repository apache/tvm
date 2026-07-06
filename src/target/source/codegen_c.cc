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
#include <tvm/ffi/cast.h>
#include <tvm/ir/unique_name_supply.h>

#include <cctype>
#include <iomanip>

#include "../../arith/pattern_match.h"
#include "../../tirx/ir/buffer_common.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

using namespace tirx;

void CodeGenC::Init(bool output_ssa) { print_ssa_form_ = output_ssa; }

void CodeGenC::InitFuncState(const PrimFunc& f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  pointer_offset_vars_.clear();
  CodeGenSourceBase::ClearFuncState();
  ReserveKeywordsAsUnique();
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

void CodeGenC::PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                                      std::ostream& os) {
  PrintFuncPrefix(os);
  PrintType(func->ret_type, os);
  PrintExtraAttrs(func, os);
  os << " " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    tirx::Var v = func->params[i];

    if (i > 0) {
      os << ", ";
    }

    if (auto it = alloc_storage_scope_.find(v.get()); it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }

    auto is_tensormap_ptr = [&]() -> bool {
      if (auto* ptr = v->ty.as<PointerTypeNode>()) {
        return ptr->element_type.as<TensorMapTypeNode>();
      }
      return false;
    };
    if (is_tensormap_ptr()) {
      os << "const __grid_constant__ CUtensorMap";
    } else {
      PrintType(v->ty, os);
    }

    bool no_alias = func->HasNonzeroAttr(tirx::attr::kNoAlias);
    bool is_handle = v->ty.as<PointerTypeNode>();
    auto* ptr = v->ty.as<PointerTypeNode>();
    if (ptr && ptr->element_type.as<TensorMapTypeNode>()) {
      is_handle = false;
    }
    if (no_alias && is_handle) {
      PrintRestrict(v, os);
    }

    os << " " << AllocVarID(v.get());
  }
  os << ")";

  // Register handle data type
  // TODO(tvm-team): consider simply keep type info in the
  // type annotation(via a normalizing rewriting).
  for (const auto& param : func->params) {
    if (auto* ptr = param->ty.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(param.get(), ffi::GetRef<PrimType>(prim));
      }
    }
  }
}

void CodeGenC::DeclareFunction(const GlobalVar& gvar, const PrimFunc& func) {
  if (internal_functions_.count(gvar)) {
    return;
  }

  auto function_name = [&]() -> ffi::String {
    if (auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      auto name = global_symbol.value();
      TVM_FFI_ICHECK(!func_name_supply_->ContainsName(name))
          << "Function " << gvar << " must use global symbol " << name
          << ", but this name has already been used.";
      func_name_supply_->ReserveName(name);
      return name;
    } else {
      func_name_supply_->ReserveName(gvar->name_hint);
      return gvar->name_hint;
    }
  }();
  if (function_name == ffi::symbol::tvm_ffi_main) {
    has_tvm_ffi_main_func_ = true;
  }
  internal_functions_.insert({gvar, function_name});

  InitFuncState(func);
  PrintFunctionSignature(function_name, func, fwd_decl_stream);
  fwd_decl_stream << ";\n";
}

ffi::String CodeGenC::GetFunctionName(const GlobalVar& gvar) {
  auto it = internal_functions_.find(gvar);
  TVM_FFI_ICHECK(it != internal_functions_.end())
      << "Attempted to find name of " << gvar
      << ", but no function with this GlobalVar has been declared";
  return it->second;
}

void CodeGenC::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  DeclareFunction(gvar, f);
  auto function_name = GetFunctionName(gvar);

  // clear previous generated state.
  InitFuncState(f);

  PrintFunctionSignature(function_name, f, stream);
  stream << " {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenC::PrintFuncPrefix(std::ostream& os) {}

void CodeGenC::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {}

std::string CodeGenC::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << fwd_decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenC::PrintExpr(const PrimExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.ty());
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenC::PrintExpr(const Expr& n, std::ostream& os) {  // NOLINT(*)
  if (auto prim = n.as<PrimExpr>()) {
    PrintExpr(prim.value(), os);
  } else if (auto* var = n.as<VarNode>()) {
    VisitExpr_(var, os);
  } else if (auto* call = n.as<CallNode>()) {
    VisitExpr_(call, os);
  } else {
    TVM_FFI_THROW(TypeError) << "CodeGenC cannot print non-primitive expression " << n.GetTypeKey();
  }
}

static bool CheckOutermostBracketMatch(const std::string& s);

void CodeGenC::PrintSSAAssign(const std::string& target, const std::string& src, const Type& t) {
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
std::string CodeGenC::GetBufferRef(const PrimType& t, const BufferNode* buffer, PrimExpr index) {
  const VarNode* buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  bool is_vol = IsVolatile(buffer_var);

  auto ptr_cast = [this, is_vol, scope](const PrimType& pointed_to) {
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

  const PrimType& buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = PrintExpr(index);
  if ((t.bits() == 4 && !t.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) ||
      (t.bits() == 1 && t.MatchesCode(DLDataTypeCode::kDLInt))) {
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
  } else if (t.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) && t.lanes() == 1) {
    // float4_e2m1fn: sizeof(__nv_fp4_e2m1) = 1 byte, but data is packed
    // 2 elements per byte.  Divide element index by 2 to get byte offset.
    // This returns an lvalue so it works for address_of() and stores.
    // Nibble extraction (for loads) is handled in VisitExpr_(BufferLoadNode*).
    os << "*(" << ptr_cast(t) << "(" << vid << " + " << index_str << " / 2))";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetStructRef(const Type& t, const Expr& buffer, const PrimExpr& index,
                                   int kind) {
  if (kind < builtin::kDLTensorKindBound_) {
    std::ostringstream os;
    os << "(((DLTensor*)";
    this->PrintExpr(buffer, os);
    os << ")";
    if (kind == builtin::kDLTensorAddr) {
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
      case builtin::kDLTensorData:
        os << "data";
        break;
      case builtin::kDLTensorShape:
        os << "shape";
        break;
      case builtin::kDLTensorStrides:
        os << "strides";
        break;
      case builtin::kDLTensorNDim:
        os << "ndim";
        break;
      case builtin::kDLTensorTypeCode:
        os << "dtype.code";
        break;
      case builtin::kDLTensorTypeBits:
        os << "dtype.bits";
        break;
      case builtin::kDLTensorByteOffset:
        os << "byte_offset";
        break;
      case builtin::kDLTensorTypeLanes:
        os << "dtype.lanes";
        break;
      case builtin::kDLTensorDeviceId:
        os << "device.device_id";
        break;
      case builtin::kDLTensorDeviceType:
        os << "device.device_type";
        break;
      default:
        TVM_FFI_THROW(InternalError) << "unknown field code";
    }
    os << ')';
    return os.str();
  } else if (kind == builtin::kTVMFFIAnyTypeIndex) {
    std::ostringstream os;
    os << "(((TVMFFIAny*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].type_index)";
    return os.str();
  } else if (kind == builtin::kTVMFFIAnyZeroPadding) {
    std::ostringstream os;
    os << "(((TVMFFIAny*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].zero_padding)";
    return os.str();
  } else if (kind == builtin::kTVMFFIAnyUnionValue) {
    std::ostringstream os;
    os << "(((TVMFFIAny*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].";
    if (t.as<PointerTypeNode>()) {
      os << "v_ptr";
    } else if (PrimType prim_type = t.as_or_throw<PrimType>();
               prim_type.MatchesCode(DLDataTypeCode::kDLFloat)) {
      os << "v_float64";
    } else if (prim_type.MatchesCode(DLDataTypeCode::kDLInt)) {
      os << "v_int64";
    } else {
      TVM_FFI_THROW(InternalError) << "Do not know how to handle type" << t;
    }
    os << ")";
    return os.str();
  } else if (kind == builtin::kInt64ArrayElem) {
    std::ostringstream os;
    os << "(((int64_t*)";
    this->PrintExpr(buffer, os);
    os << ")[";
    this->PrintExpr(index, os);
    os << "])";
    return os.str();
  } else {
    TVM_FFI_THROW(RuntimeError) << "Unsupported type index: " << kind;
    TVM_FFI_UNREACHABLE();
  }
}

bool CodeGenC::HandleTypeMatch(const VarNode* buf_var, const PrimType& t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const VarNode* buf_var, const PrimType& t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_.emplace(buf_var, t);
  } else {
    TVM_FFI_ICHECK(it->second == t) << "conflicting buf var type";
  }
}

void CodeGenC::RegisterHandleTypeFromPointer(const tirx::Var& var, const Expr* value) {
  if (value == nullptr) return;
  auto* call = value->as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::ptr_byte_offset())) return;
  std::optional<PrimType> value_dtype = [&]() {
    if (auto prim_value = value->as<PrimExpr>()) {
      return tirx::GetPointerType(GetType(prim_value.value()));
    }
    return tirx::GetPointerType((*value)->ty);
  }();
  if (!value_dtype.has_value()) return;
  RegisterHandleType(var.get(), value_dtype.value());
  pointer_offset_vars_.insert(var.get());
}

void CodeGenC::PrintVecElemLoad(const std::string& vec, const PrimType& t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeGenC::PrintVecElemStore(const std::string& vec, const PrimType& t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i << " = " << value << ";\n" << std::dec;
}

std::string CodeGenC::GetVecLoad(const PrimType& t, const BufferNode* buffer, PrimExpr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeGenC::PrintVecStore(const BufferNode* buffer, const PrimType& t, PrimExpr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

void CodeGenC::PrintVecConstructor(const PrimType& t, std::ostream& os) {  // NOLINT(*)
  PrintType(t, os);
}

std::string CodeGenC::CastFromTo(std::string value, const PrimType& from, const PrimType& target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeGenC::BindThreadIndex(const IterVar& iv) {
  TVM_FFI_THROW(InternalError) << "not implemented";
}

void CodeGenC::PrintStorageSync(const CallNode* op) {  // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  TVM_FFI_ICHECK_EQ(scope, "global");
}

inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  if (op->ty.as_or_throw<PrimType>() == PrimType::Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->ty.as_or_throw<PrimType>(), os);
    os << ")" << op->value;
  }
}

inline void PrintUIntConst(const PrimType& dtype, uint64_t val, std::ostream& os,
                           CodeGenC* p) {  // NOLINT(*)
  if (dtype == PrimType::UInt(32)) {
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
  switch (op->ty.as_or_throw<PrimType>().bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->ty.as_or_throw<PrimType>().bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->ty.as_or_throw<PrimType>(), os);
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      TVM_FFI_THROW(InternalError)
          << "Bad bit-width for float: " << op->ty.as_or_throw<PrimType>()->dtype << "\n";
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
  PrimType op_ty = op->ty.template as_or_throw<PrimType>();
  if (op_ty.lanes() == 1) {
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
    p->PrintVecBinaryOp(opstr, op->ty.template as_or_throw<PrimType>(), op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsic(const CallNode* op, const char* opstr,
                                 std::ostream& os,  // NOLINT(*)
                                 CodeGenC* p) {
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  if (op_ty.lanes() == 1) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op_ty, op->args[0].as_or_throw<PrimExpr>(),
                        op->args[1].as_or_throw<PrimExpr>(), os);
  }
}
void CodeGenC::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.ty(), op->ty.as_or_throw<PrimType>());
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
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  if (op_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    PrintBinaryExpr(op, "%", os, this);
  } else {
    TVM_FFI_ICHECK(op_ty.MatchesCode(DLDataTypeCode::kDLFloat))
        << "Expected floating point or integer dtype in Mod, but got "
        << op->ty.as_or_throw<PrimType>()->dtype;
    if (op_ty.bits() == 32) {
      PrintBinaryExpr(op, "fmodf", os, this);
    } else if (op_ty.bits() == 64) {
      PrintBinaryExpr(op, "fmod", os, this);
    } else {
      TVM_FFI_ICHECK(false)
          << "Non single or double precision floating point in Mod, expected 32 or 64 bits but got "
          << op_ty.bits() << " bits.";
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

void CodeGenC::PrintCallExtern(Type ret_type, ffi::String global_symbol,
                               const ffi::Array<Expr>& args, bool skip_first_arg,
                               std::ostream& os) {  // NOLINT(*)
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
  if (auto opt_call_op = op->op.as<Op>()) {
    auto call_op = opt_call_op.value();

    if (op->op.same_as(builtin::ret())) {
      os << "return ";
      PrintExpr(op->args[0], os);
    } else if (op->op.same_as(builtin::continue_loop())) {
      os << "continue;";
    } else if (op->op.same_as(builtin::break_loop())) {
      os << "break;";
    } else if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      TVM_FFI_ICHECK_GE(op->args.size(), 1U);
      auto func = op->args[0].as_or_throw<StringImm>();
      ffi::Array<Expr> args = op->args;
      this->PrintCallExtern(op->ty, func->value, args, true, os);

      // If the call_extern refers to an function within the IRModule, then
      // the forward declaration is already provided from DeclareFunction.
      if (!func_name_supply_->ContainsName(func->value)) {
        ffi::Array<Type> arg_types;
        for (size_t i = 1; i < op->args.size(); i++) {
          if (auto prim = op->args[i].as<PrimExpr>()) {
            arg_types.push_back(GetType(prim.value()));
          } else if (auto var = op->args[i].as<Var>()) {
            arg_types.push_back(var.value()->ty);
          } else {
            arg_types.push_back(op->args[i]->ty);
          }
        }
        Type ret_type = op->ty;
        this->GenerateForwardFunctionDeclarations(func->value, arg_types, ret_type);
      }
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      ffi::Array<Expr> args = op->args;
      this->PrintCallExtern(op->ty, op_attr_global_symbol_[call_op], args, false, os);
    } else if (op->op.same_as(builtin::bitwise_and())) {
      PrintBinaryIntrinsic(op, " & ", os, this);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 2U);
      uint64_t low = static_cast<uint64_t>(op->args[0].as_or_throw<IntImm>()->value);
      uint64_t high = static_cast<uint64_t>(op->args[1].as_or_throw<IntImm>()->value);
      uint64_t val = (high << 32U) | low;
      PrintUIntConst(op->ty.as_or_throw<PrimType>(), val, os, this);
    } else if (op->op.same_as(builtin::bitwise_xor())) {
      PrintBinaryIntrinsic(op, " ^ ", os, this);
    } else if (op->op.same_as(builtin::bitwise_or())) {
      PrintBinaryIntrinsic(op, " | ", os, this);
    } else if (op->op.same_as(builtin::bitwise_not())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1U);
      os << "(~";
      this->PrintExpr(op->args[0], os);
      os << ')';
    } else if (op->op.same_as(builtin::shift_left())) {
      PrintBinaryIntrinsic(op, " << ", os, this);
    } else if (op->op.same_as(builtin::shift_right())) {
      PrintBinaryIntrinsic(op, " >> ", os, this);
    } else if (op->op.same_as(builtin::if_then_else())) {
      // conditional that skips eval if cond evals to false
      std::string result = name_supply_->FreshName("condval");
      std::string cond = PrintExpr(op->args[0]);
      this->PrintIndent();
      PrintType(op->ty, this->stream);
      this->stream << " " << result << ";\n";
      this->PrintIndent();
      this->stream << "if (" << cond << ") {\n";
      {
        int then_scope = this->BeginScope();
        std::string true_val = PrintExpr(op->args[1]);
        this->PrintIndent();
        this->stream << result << " = " << true_val << ";\n";
        this->EndScope(then_scope);
        this->PrintIndent();
        this->stream << "} else {\n";
      }
      {
        int else_scope = this->BeginScope();
        std::string false_val = PrintExpr(op->args[2]);
        this->PrintIndent();
        this->stream << result << " = " << false_val << ";\n";
        this->EndScope(else_scope);
        this->PrintIndent();
        this->stream << "}\n";
      }
      os << result;
    } else if (op->op.same_as(builtin::address_of())) {
      const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
      TVM_FFI_ICHECK(op->args.size() == 1);
      if (load) {
        TVM_FFI_ICHECK_EQ(load->indices.size(), 1)
            << "CodeGenC only supports flat memory allocations.";
        const VarNode* data = load->buffer->data.get();
        if (pointer_offset_vars_.count(data) && HandleTypeMatch(data, load->buffer->dtype) &&
            !IsVolatile(data)) {
          os << "(" << GetVarID(data) << " + ";
          this->PrintExpr(load->indices[0], os);
          os << ")";
        } else {
          os << "(&("
             << GetBufferRef(load->ty.as_or_throw<PrimType>(), load->buffer.get(), load->indices[0])
             << "))";
        }
      } else {
        auto* var = op->args[0].as<tirx::VarNode>();
        TVM_FFI_ICHECK(var)
            << "Builtin address_of() expects the argument to be a BufferLoad or Var, but "
            << "received argument " << op->args[0];
        if (auto* ptr = var->ty.as<PointerTypeNode>()) {
          if (ptr->element_type.as<TensorMapTypeNode>()) {
            os << "((unsigned long long)(&(";
            this->PrintExpr(op->args[0], os);
            os << ")))";
          } else {
            os << "(&(";
            this->PrintExpr(op->args[0], os);
            os << "))";
          }
        } else {
          os << "(&(";
          this->PrintExpr(op->args[0], os);
          os << "))";
        }
      }
    } else if (op->op.same_as(builtin::tvm_struct_get())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 3U);
      os << GetStructRef(op->ty, op->args[0], op->args[1].as_or_throw<PrimExpr>(),
                         op->args[2].as<IntImmNode>()->value);
    } else if (op->op.same_as(builtin::isnullptr())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1U);
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " == NULL)";
    } else if (op->op.same_as(builtin::ptr_byte_offset())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 3U);
      os << "((";
      PrintType(op->args[2].as_or_throw<PrimExpr>().ty(), os);
      os << "*)(((char*)";
      this->PrintExpr(op->args[0], os);
      os << ") + ";
      this->PrintExpr(op->args[1], os);
      os << "))";
    } else if (op->op.same_as(builtin::handle_add_byte_offset())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 2U);
      os << "((void*)((char*)";
      this->PrintExpr(op->args[0], os);
      os << " + ";
      this->PrintExpr(op->args[1], os);
      os << "))";
    } else if (op->op.same_as(builtin::reinterpret())) {
      if (const auto* pointer_type = op->ty.as<PointerTypeNode>()) {
        os << "((";
        if (const auto* element_type = pointer_type->element_type.as<PrimTypeNode>()) {
          this->PrintType(ffi::GetRef<PrimType>(element_type), os);
        } else {
          os << "void";
        }
        os << "*)";
        this->PrintExpr(op->args[0], os);
        os << ")";
        return;
      }
      PrimType target_dtype = op->ty.as_or_throw<PrimType>();
      if (op->args[0]->ty.as<PointerTypeNode>()) {
        TVM_FFI_ICHECK(target_dtype.IsScalar() && target_dtype.bits() == 64 &&
                       target_dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
            << "Pointer reinterpret requires a scalar 64-bit integer target, but got "
            << target_dtype;
        os << "((";
        this->PrintType(target_dtype, os);
        os << ")(uintptr_t)(";
        this->PrintExpr(op->args[0], os);
        os << "))";
        return;
      }
      PrimType source_dtype = op->args[0].as_or_throw<PrimExpr>().ty();
      TVM_FFI_ICHECK_EQ(target_dtype.lanes() * target_dtype.bits(),
                        source_dtype.lanes() * source_dtype.bits())
          << "reinterpret expects source and target to have the same number of bits";
      int ssa_scope = BeginScope();
      std::string rhs = SSAGetID(PrintExpr(op->args[0]), source_dtype);
      os << "(*(";
      this->PrintType(target_dtype, os);
      os << " *)(&(" << rhs << ")))";
      EndScope(ssa_scope);
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " != ";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else if (op->op.same_as(builtin::lookup_param())) {
      TVM_FFI_ICHECK_EQ(op->args.size(), 1);
      const StringImmNode* str = op->args[0].as<StringImmNode>();
      TVM_FFI_ICHECK(str != nullptr);
      os << "__tvm_param__" << str->value;
    } else if (op->op.same_as(builtin::tvm_thread_invariant())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      TVM_FFI_THROW(InternalError) << "Unresolved call " << op->op;
    }
  } else if (auto opt = op->op.as<GlobalVar>()) {
    auto gvar = opt.value();
    auto callee_name = GetFunctionName(gvar);
    ffi::Array<Expr> args = op->args;
    PrintCallExtern(op->ty, callee_name, args, false, os);
  } else {
    TVM_FFI_THROW(InternalError) << "CodeGenC: Unknown operation " << op->op
                                 << " is neither a recognized built-in, "
                                 << "nor a GlobalVar reference to another function in the IRModule";
  }
}

void CodeGenC::PrintVecBinaryOp(const std::string& op, const PrimType& t, PrimExpr lhs,
                                PrimExpr rhs, std::ostream& os) {  // NOLINT(*)
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

void CodeGenC::VisitStmt_(const DeclBufferNode* op) {
  // DeclBuffer is a flat statement with no body — nothing to emit.
}

void CodeGenC::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";
  TVM_FFI_ICHECK(!op->predicate.defined()) << "Predicated buffer load is not supported.";

  PrimType value_ty = op->ty.as_or_throw<PrimType>();
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  const PrimType& element_ty = op->buffer->dtype;

  int lanes = value_ty.lanes();
  // delcare type.
  if (value_ty.lanes() == element_ty.lanes()) {
    std::string ref = GetBufferRef(op->ty.as_or_throw<PrimType>(), op->buffer.get(), index);
    if (value_ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) && value_ty.lanes() == 1) {
      // GetBufferRef returns an lvalue: *(ptr + index/2), which reads the
      // full byte.  Extract the correct nibble (low for even, high for odd).
      std::string index_str = PrintExpr(index);
      std::ostringstream nibble;
      nibble << "([](__nv_fp4_storage_t v) { __nv_fp4_e2m1 t; t.__x = v; return t; })"
             << "(((" << ref << ").__x >> ((" << index_str << " % 2) * 4)) & 0xF)";
      HandleVolatileLoads(nibble.str(), op, os);
    } else {
      HandleVolatileLoads(ref, op, os);
    }
  } else {
    bool can_vector_load = false;
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, value_ty.lanes()).Match(index)) {
      const RampNode* ramp = index.as<RampNode>();
      TVM_FFI_ICHECK(ramp);
      arith::ModularSet me = arith::Analyzer()->modular_set(ramp->base);
      // The condition: {k * coeff + base} divisible by the alignment for any k
      if (me->coeff % value_ty.lanes() == 0 && me->base % value_ty.lanes() == 0) {
        can_vector_load = true;
      }
    }

    if (value_ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) && lanes != 1) {
      // A float4_e2m1fn element has 4 bits, which is an incomplete byte.
      // So we cannot vector load it.
      can_vector_load = false;
    }
    if (can_vector_load) {
      std::string ref = GetVecLoad(op->ty.as_or_throw<PrimType>(), op->buffer.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(index), index.ty());
      std::string vid = GetVarID(buffer_var.get());
      PrimType elem_type = value_ty.WithLanes(1);
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (buffer_var->ty.as<PointerTypeNode>()) {
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
        PrintVecElemLoad(sindex, index.ty(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->ty.as_or_throw<PrimType>(), i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenC::VisitStmt_(const BufferStoreNode* op) {
  TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";
  TVM_FFI_ICHECK(!op->predicate.defined()) << "Predicated buffer store is not supported.";

  PrimType value_ty = op->value.ty();
  const PrimType& element_ty = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->buffer->data;

  if (value_ty.lanes() == element_ty.lanes()) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(value_ty, op->buffer.get(), index_expr);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    arith::PVar<PrimExpr> base;

    if (arith::ramp(base, 1, value_ty.lanes()).Match(index_expr) &&
        value_ty.code() != DLDataTypeCode::kDLFloat4_e2m1fn) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer.get(), value_ty, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements separately
      std::string index = SSAGetID(PrintExpr(index_expr), index_expr.ty());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.ty());
      std::string vid = GetVarID(buffer_var.get());
      for (int i = 0; i < value_ty.lanes(); ++i) {
        this->PrintIndent();
        PrimType elem_type = value_ty.WithLanes(1);
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          stream << "((";
          if (buffer_var->ty.as<PointerTypeNode>()) {
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
        PrintVecElemLoad(index, index_expr.ty(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.ty(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenC::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    TVM_FFI_ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  RegisterHandleTypeFromPointer(op->var, &op->value);
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    bool is_pointer = op->var->ty.as<PointerTypeNode>();
    if (is_pointer && handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), this->stream);
      this->stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), this->stream);
      this->stream << "*)" << value << ";\n";
    } else {
      PrintType(op->var->ty, this->stream);
      this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
  }
  os << PrintExpr(op->body);
  // Pop the defined var from var_idmap when exiting its scope.
  // We do this because it is hard to completely avoid a same LetNode appearing
  // at different places.
  bool removed = var_idmap_.erase(op->var.get());
  TVM_FFI_ICHECK(removed);
}

void CodeGenC::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // NOTE: C have comma expression so cannot use (int2)(v0, v1)
  // instead should use int2(v0, v1)
  PrintType(op->ty.as_or_throw<PrimType>(), os);
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1) os << ", ";
  }
  os << ")";
}

void CodeGenC::VisitExpr_(const ShuffleNode* op, std::ostream& os) {  // NOLINT(*)
  // Shuffle support
  // vec = concat(vectors)
  // result = (vec[indices[0]], vec[indices[1]], ...)
  //
  // print shuffle as:
  // target_dtype(e0, e1, e2, .. en)

  // construct the concat
  std::vector<std::string> concat_vec;
  // NOTE: important to print expr first
  // in case each expr have their own nested expressions
  // print each elements
  if (op->vectors.size() > 1) {
    for (const PrimExpr& vec : op->vectors) {
      std::string vec_value = this->PrintExpr(vec);
      if (vec.ty().lanes() == 1) {
        concat_vec.push_back(vec_value);
      } else {
        // print out each element
        for (int i = 0; i < vec.ty().lanes(); ++i) {
          // access i-th element of each vector
          std::ostringstream vec_elem_strm;
          vec_elem_strm << vec_value << "[" << i << "]";
          concat_vec.push_back(vec_elem_strm.str());
        }
      }
    }
  } else {
    // Extract elements from a single vector-type value.
    std::string vec_value = "(" + this->PrintExpr(op->vectors[0]) + ")";
    if (op->vectors[0].ty().lanes() == 1) {
      concat_vec.push_back(vec_value);
    } else {
      // print out each element
      for (int i = 0; i < op->vectors[0].ty().lanes(); ++i) {
        // access i-th element of each vector
        std::ostringstream vec_elem_strm;
        PrintVecElemLoad(vec_value, op->vectors[0].ty(), i, vec_elem_strm);
        concat_vec.push_back(vec_elem_strm.str());
      }
    }
  }
  if (op->indices.size() == 1) {
    // This is an extract element
    TVM_FFI_ICHECK(op->indices[0]->IsInstance<IntImmNode>())
        << "The ShuffleNode indices are expected to be constants at codegen time. However, "
        << "a non-constant index is " << op->indices[0]
        << ". Please avoid using ShuffleNode or eliminate the ShuffleNode with loop unroll or "
        << "vectorize.";
    int64_t idx = op->indices[0].as_or_throw<IntImm>()->value;
    TVM_FFI_ICHECK_LT(idx, concat_vec.size());
    os << concat_vec[idx];
  } else {
    // Print the shuffle as vector constructor
    // vec(e0, e1, e2, .. en)
    PrintVecConstructor(op->ty.as_or_throw<PrimType>(), os);
    os << '(';
    for (size_t i = 0; i < op->indices.size(); ++i) {
      if (i != 0) os << ", ";
      TVM_FFI_ICHECK(op->indices[i]->IsInstance<IntImmNode>())
          << "The ShuffleNode indices are expected to be constants at codegen time. However, "
          << "a non-constant index is " << op->indices[i]
          << ". Please avoid using ShuffleNode or eliminate the ShuffleNode with loop unroll or "
          << "vectorize.";
      os << concat_vec[op->indices[i].as_or_throw<IntImm>()->value];
    }
    os << ')';
  }
}

void CodeGenC::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  TVM_FFI_THROW(InternalError) << "Broadcast: not supported ";
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

void CodeGenC::VisitStmt_(const BindNode* op) {
  RegisterHandleTypeFromPointer(op->var, &op->value);
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    bool is_pointer = op->var->ty.as<PointerTypeNode>();
    if (is_pointer && handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {
      PrintType(op->var->ty, this->stream);
      this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
  }
}

void CodeGenC::VisitStmt_(const AllocBufferNode* op) {
  TVM_FFI_ICHECK(op->buffer.defined());
  std::string vid = AllocVarID(op->buffer->data.get());

  this->PrintIndent();
  const auto& shape = op->buffer->shape;
  size_t constant_size = 1;
  for (const auto& dim : shape) {
    const IntImmNode* dim_imm = dim.as<IntImmNode>();
    TVM_FFI_ICHECK(dim_imm) << "Can only handle constant size stack allocation for now";
    constant_size *= dim_imm->value;
  }
  TVM_FFI_ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer->data);
  alloc_storage_scope_[op->buffer->data.get()] = scope;
  PrintStorageScope(scope, stream);

  PrintType(op->buffer->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
  if (op->annotations.count(tirx::attr::kVolatile)) {
    MarkVolatile(op->buffer->data.get());
  }
}

void CodeGenC::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = op->node.as_or_throw<IterVar>();
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  } else if (op->attr_key == tirx::attr::pragma_import_c) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    TVM_FFI_ICHECK(value != nullptr);
    decl_stream << value->value;
  }
  this->PrintStmt(op->body);
}

void CodeGenC::PrintEscapedCString(const std::string& str, std::ostream& os) {
  os << "\"";
  for (unsigned char c : str) {
    switch (c) {
      case '"':
        os << "\\\"";
        break;
      case '\\':
        os << "\\\\";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\t':
        os << "\\t";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\a':
        os << "\\a";
        break;
      case '\b':
        os << "\\b";
        break;
      case '\f':
        os << "\\f";
        break;
      case '\v':
        os << "\\v";
        break;
      case '\0':
        os << "\\0";
        break;
      default:
        if (c < 0x20 || c >= 0x7f) {
          // Non-printable: emit as hex escape
          os << "\\x" << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(c)
             << std::dec;
        } else {
          os << static_cast<char>(c);
        }
        break;
    }
  }
  os << "\"";
}

void CodeGenC::VisitStmt_(const AssertStmtNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  int num_parts = static_cast<int>(op->message_parts.size());
  if (num_parts > 0) {
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
    stream << "const char* __tvm_assert_parts[" << num_parts << "] = {";
    for (int i = 0; i < num_parts; ++i) {
      if (i > 0) stream << ", ";
      PrintEscapedCString(op->message_parts[i]->value, stream);
    }
    stream << "};\n";
    PrintIndent();
    stream << "TVMFFIErrorSetRaisedFromCStrParts(";
    PrintEscapedCString(op->error_kind->value, stream);
    stream << ", __tvm_assert_parts, " << num_parts << ");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
}

void CodeGenC::VisitStmt_(const ForNode* op) {
  std::string begin_str = PrintExpr(op->min);
  PrimExpr end = is_zero(op->min) ? op->extent : arith::Analyzer()->Simplify(op->min + op->extent);
  std::string end_str = PrintExpr(end);
  std::string step_str = op->step.has_value() ? PrintExpr(*op->step) : "";
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for (";
  PrintType(op->loop_var.ty(), stream);
  stream << ' ' << vid << " = " << begin_str << "; " << vid << " < " << end_str << "; ";
  if (step_str.empty()) {
    stream << "++" << vid;
  } else {
    stream << vid << " += " << step_str;
  }
  stream << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  stream << "#pragma unroll 1\n";
  PrintIndent();
  stream << "while (1) {\n";
  int while_scope = BeginScope();
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if (!(" << cond << ")) { break; }\n";
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const BreakNode* op) {
  PrintIndent();
  stream << "break;\n";
}

void CodeGenC::VisitStmt_(const ContinueNode* op) {
  PrintIndent();
  stream << "continue;\n";
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
  if (auto value = op->value.as<PrimExpr>(); value && is_const_int(value.value())) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->op.same_as(builtin::tvm_storage_sync())) {
      this->PrintStorageSync(call);
      return;
    } else if (call->op.same_as(builtin::tvm_struct_set())) {
      TVM_FFI_ICHECK_EQ(call->args.size(), 4);
      int kind = call->args[2].as<IntImmNode>()->value;
      Type store_ty = call->args[3]->ty;
      std::string ref =
          GetStructRef(store_ty, call->args[0], call->args[1].as_or_throw<PrimExpr>(), kind);
      std::string value = PrintExpr(call->args[3]);
      std::string cast;

      auto store_prim_type = store_ty.as<PrimType>();
      bool clears_union = store_ty.as<PointerTypeNode>() ||
                          (store_prim_type && store_prim_type.value().bits() < 64);
      if (kind == builtin::kTVMFFIAnyUnionValue && clears_union) {
        this->PrintIndent();
        // when we set any union value, we need to be careful to
        // clear off the union value to zero if the set size is less than 64 bits
        this->stream << GetStructRef(PrimType::Int(64), call->args[0],
                                     call->args[1].as_or_throw<PrimExpr>(), kind)
                     << " = 0;\n";
      }

      if (kind == builtin::kDLTensorStrides) {
        // cast void* to int64_t*
        cast = store_ty.as<PointerTypeNode>() ? "(int64_t*)" : "";
      } else if (kind == builtin::kDLTensorDeviceType) {
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

void CodeGenC::PrintVecElemLoadExpr(const PrimType& t, int i, const std::string& value,
                                    std::ostream& os) {
  int lanes = t.lanes();
  TVM_FFI_ICHECK_GT(lanes, 1);
  if (t.bits() == 8 && t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }

  if (i == 0) {
    // NOTE: C have comma expression so cannot use (float2)(v0, v1)
    // instead should use float2(v0, v1)
    os << "(";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != lanes - 1) {
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
