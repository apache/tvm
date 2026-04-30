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
 * \file tvm_ffi_binder.cc
 * \brief Helper utility to match and bind packed function arguments.
 */
#include "tvm_ffi_binder.h"

#include <tvm/ffi/cast.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>

#include "ir_utils.h"

namespace tvm {
namespace tirx {

using ffi::reflection::AccessPath;
using ffi::reflection::AccessStep;

// ============================================================
// Constructor
// ============================================================

TVMFFIABIBuilder::TVMFFIABIBuilder(const ffi::String& func_name, const ffi::Array<Var>& params,
                                   const ffi::Map<Var, Buffer>& buffer_map,
                                   const Var& v_packed_args, const Var& v_num_packed_args,
                                   const PrimExpr& device_type, const PrimExpr& device_id)
    : func_name_(func_name),
      params_(params),
      buffer_map_(buffer_map),
      v_packed_args_(v_packed_args),
      device_type_(device_type),
      device_id_(device_id) {
  // Build function signature string
  std::ostringstream os;
  os << func_name << "(";
  for (size_t i = 0; i < params.size(); ++i) {
    if (i > 0) os << ", ";
    Var param = params[i];
    if (buffer_map.count(param)) {
      Buffer buf = buffer_map[param];
      std::string buf_name = buf->name;
      os << buf_name << ": Tensor([";
      for (size_t j = 0; j < buf->shape.size(); ++j) {
        if (j > 0) os << ", ";
        std::ostringstream shape_os;
        shape_os << buf->shape[j];
        os << shape_os.str();
      }
      os << "], " << buf->dtype << ")";
      param_names_[static_cast<int>(i)] = buf_name;
    } else {
      os << param->name_hint << ": " << param.dtype();
      param_names_[static_cast<int>(i)] = param->name_hint;
    }
  }
  os << ")";
  func_signature_ = os.str();
  sig_imm_ = StringImm(func_signature_);

  // Emit argument count check (early check — must execute before any loads)
  int num_args = static_cast<int>(params.size());
  EmitAssert(v_num_packed_args == num_args, "TypeError",  //
             "Expected ", std::to_string(num_args), " arguments", when_calling_imm_, sig_imm_, "`");

  // Emit null-pointer check for packed args (early check)
  if (num_args > 0) {
    EmitAssert(!Call(DataType::Bool(), builtin::isnullptr(), {v_packed_args}),
               "TypeError",  //
               "args pointer is NULL", when_calling_imm_, sig_imm_, "`");
  }
}

// ============================================================
// EmitTypeIndexCheck
// ============================================================

void TVMFFIABIBuilder::EmitTypeIndexCheck(int param_index, const PrimExpr& cond,
                                          const std::string& expected_type) {
  EmitAssert(cond, "TypeError",  //
             "Mismatched type on argument #", std::to_string(param_index), when_calling_imm_,
             sig_imm_, "`,\n  expected ", expected_type);
}

// ============================================================
// RenderAccessPath
// ============================================================

ffi::String TVMFFIABIBuilder::RenderAccessPath(const AccessPath& path) const {
  ffi::Array<AccessStep> steps = path->ToSteps();
  std::ostringstream os;
  bool first_printed = false;
  bool consumed_root_array_item = false;
  for (size_t i = 0; i < steps.size(); ++i) {
    const AccessStep& step = steps[i];
    if (step->kind == ffi::reflection::AccessKind::kAttr) {
      std::string field_name = step->key.cast<ffi::String>();
      if (!first_printed) {
        os << field_name;
        first_printed = true;
      } else {
        os << "." << field_name;
      }
    } else if (step->kind == ffi::reflection::AccessKind::kArrayItem) {
      int64_t index = step->key.cast<int64_t>();
      if (!first_printed && !consumed_root_array_item) {
        consumed_root_array_item = true;
        if (i + 1 < steps.size() && steps[i + 1]->kind == ffi::reflection::AccessKind::kAttr) {
          continue;
        }
        auto it = param_names_.find(static_cast<int>(index));
        if (it != param_names_.end()) {
          os << it->second;
          first_printed = true;
        }
      } else {
        os << "[" << index << "]";
      }
    } else if (step->kind == ffi::reflection::AccessKind::kMapItem) {
      os << "[" << step->key << "]";
    }
  }
  return os.str();
}

// ============================================================
// GetParamIndex
// ============================================================

int TVMFFIABIBuilder::GetParamIndex(const AccessPath& path) const {
  ffi::Array<AccessStep> steps = path->ToSteps();
  if (steps.size() >= 1 && steps[0]->kind == ffi::reflection::AccessKind::kArrayItem) {
    return static_cast<int>(steps[0]->key.cast<int64_t>());
  }
  return -1;
}

// ============================================================
// BindScalar (scalar bind with AccessPath)
// ============================================================

bool TVMFFIABIBuilder::BindScalar(const PrimExpr& arg, const PrimExpr& value,
                                  const AccessPath& path, bool with_lets) {
  TVM_FFI_ICHECK_EQ(arg.dtype(), value.dtype());
  if (arg.as<VarNode>()) {
    Var v_arg = Downcast<Var>(arg);
    auto it = var_defs_.find(v_arg.get());
    if (it == var_defs_.end()) {
      // First bind: define the variable
      if (with_lets) {
        var_defs_.emplace(v_arg.get(), VarDefInfo{arg, path});
        init_nest_.emplace_back(Bind(v_arg, value));
      } else {
        var_defs_.emplace(v_arg.get(), VarDefInfo{value, path});
      }
      return true;
    } else {
      // Duplicate bind: create rich assertion with both paths
      PrimExpr prev_value = it->second.value;
      PrimExpr scond = analyzer_.Simplify(prev_value == value);
      if (is_zero(scond)) {
        TVM_FFI_THROW(InternalError) << "Bind have an unmet assertion: " << prev_value
                                     << " == " << value << " at " << RenderAccessPath(path);
      }
      if (!is_one(scond)) {
        ffi::String current_path_str = RenderAccessPath(path);
        ffi::String first_path_str = RenderAccessPath(it->second.first_def_path);
        int param_index = GetParamIndex(path);
        ffi::Array<StringImm> parts;
        parts.push_back(StringImm("Mismatched "));
        parts.push_back(StringImm(current_path_str));
        if (param_index >= 0) {
          parts.push_back(StringImm(" on argument #"));
          parts.push_back(StringImm(std::to_string(param_index)));
        }
        parts.push_back(when_calling_imm_);
        parts.push_back(sig_imm_);
        if (!first_path_str.empty() && first_path_str != current_path_str) {
          parts.push_back(StringImm("`,\n  expected to match "));
          parts.push_back(StringImm(first_path_str));
        } else {
          parts.push_back(StringImm("`,\n  expected matching value"));
        }
        asserts_.emplace_back(AssertStmt(scond, StringImm("ValueError"), parts));
      }
    }
  } else {
    // Non-Var expression (e.g. batch_size + 1): defer assertion to Finalize()
    // so display-var substitution can render human-readable names.
    PrimExpr scond = analyzer_.Simplify(arg == value);
    if (is_zero(scond)) {
      TVM_FFI_THROW(InternalError) << "Bind have an unmet assertion: " << arg << " == " << value
                                   << " at " << RenderAccessPath(path);
    }
    if (!is_one(scond)) {
      pending_const_asserts_.push_back({scond, path, arg});
    }
  }
  return false;
}

// ============================================================
// ExprPathRenderer (generic expr-to-string via ExprFunctor)
// ============================================================

/*!
 * \brief Render PrimExpr to string with variable names replaced by AccessPath names.
 *
 * Uses ExprFunctor for generic dispatch over all expression types.
 * The default TIR printer sanitizes Var name_hints (e.g. "B.shape[0]" -> "B_shape_0_")
 * and adds type annotations (e.g. T.int64(1)). This functor preserves original path
 * names and uses plain integer formatting for human-readable error messages.
 */
class ExprPathRenderer : public ExprFunctor<std::string(const PrimExpr&)> {
 public:
  using FVarName = std::function<std::string(const VarNode*)>;
  explicit ExprPathRenderer(FVarName f_var_name) : f_var_name_(std::move(f_var_name)) {}

 protected:
  std::string VisitExpr_(const VarNode* op) final { return f_var_name_(op); }
  std::string VisitExpr_(const IntImmNode* op) final { return std::to_string(op->value); }
  std::string VisitExpr_(const FloatImmNode* op) final {
    std::ostringstream os;
    os << op->value;
    return os.str();
  }
  std::string VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }
  std::string VisitExpr_(const AddNode* op) final { return BinOp(op->a, " + ", op->b); }
  std::string VisitExpr_(const SubNode* op) final { return BinOp(op->a, " - ", op->b); }
  std::string VisitExpr_(const MulNode* op) final { return BinOp(op->a, " * ", op->b); }
  std::string VisitExpr_(const DivNode* op) final { return BinOp(op->a, " / ", op->b); }
  std::string VisitExpr_(const ModNode* op) final { return BinOp(op->a, " % ", op->b); }
  std::string VisitExpr_(const FloorDivNode* op) final { return FuncOp("floordiv", op->a, op->b); }
  std::string VisitExpr_(const FloorModNode* op) final { return FuncOp("floormod", op->a, op->b); }
  std::string VisitExpr_(const MinNode* op) final { return FuncOp("min", op->a, op->b); }
  std::string VisitExpr_(const MaxNode* op) final { return FuncOp("max", op->a, op->b); }
  // Fallback: use operator<< for unhandled expression types.
  std::string VisitExprDefault_(const ffi::Object* op) final {
    std::ostringstream os;
    os << ffi::GetRef<PrimExpr>(static_cast<const PrimExprNode*>(op));
    return os.str();
  }

 private:
  std::string BinOp(const PrimExpr& a, const char* op, const PrimExpr& b) {
    return VisitExpr(a) + op + VisitExpr(b);
  }
  std::string FuncOp(const char* name, const PrimExpr& a, const PrimExpr& b) {
    return std::string(name) + "(" + VisitExpr(a) + ", " + VisitExpr(b) + ")";
  }
  FVarName f_var_name_;
};

// ============================================================
// RenderPendingAsserts
// ============================================================

void TVMFFIABIBuilder::RenderPendingAsserts() {
  if (pending_const_asserts_.empty()) return;

  ExprPathRenderer renderer([this](const VarNode* v) -> std::string {
    auto it = var_defs_.find(v);
    if (it != var_defs_.end()) {
      ffi::String path = RenderAccessPath(it->second.first_def_path);
      if (!path.empty()) return std::string(path);
    }
    return std::string(v->name_hint);
  });

  for (auto& pending : pending_const_asserts_) {
    std::string display_str = renderer.VisitExpr(pending.expected_expr);

    ffi::String path_str = RenderAccessPath(pending.path);
    int param_index = GetParamIndex(pending.path);
    ffi::Array<StringImm> parts;
    parts.push_back(StringImm("Invalid "));
    parts.push_back(StringImm(path_str));
    if (param_index >= 0) {
      parts.push_back(StringImm(" on argument #"));
      parts.push_back(StringImm(std::to_string(param_index)));
    }
    parts.push_back(when_calling_imm_);
    parts.push_back(sig_imm_);
    parts.push_back(StringImm("`,\n  expected "));
    parts.push_back(StringImm(display_str));
    asserts_.emplace_back(AssertStmt(pending.condition, StringImm("ValueError"), parts));
  }
  pending_const_asserts_.clear();
}

// ============================================================
// Finalize
// ============================================================

TVMFFIABIBuilder::Result TVMFFIABIBuilder::Finalize() {
  RenderPendingAsserts();
  Result result;
  result.var_defs = std::move(var_defs_);
  result.init_nest = std::move(init_nest_);
  result.asserts = std::move(asserts_);
  result.decl_buffers = std::move(decl_buffers_);
  return result;
}

// ============================================================
// BindArray (array bind with AccessPath)
// ============================================================

void TVMFFIABIBuilder::BindArray(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                                 const AccessPath& base_path) {
  TVM_FFI_ICHECK_EQ(arg.size(), value.size())
      << "Array size mismatch at " << RenderAccessPath(base_path);
  for (size_t i = 0; i < arg.size(); ++i) {
    AccessPath elem_path = base_path->ArrayItem(i);
    BindScalar(arg[i], value[i], elem_path, false);
  }
}

// ============================================================
// BindBuffer (buffer-to-buffer bind with AccessPath)
// ============================================================

void TVMFFIABIBuilder::BindBuffer(const Buffer& arg, const Buffer& value, AccessPath base_path,
                                  bool fuzzy_match) {
  TVM_FFI_ICHECK_EQ(arg.scope(), value.scope())
      << "Argument " << arg->name << " Buffer bind scope mismatch";
  TVM_FFI_ICHECK_EQ(arg->dtype, value->dtype)
      << "Argument " << arg->name << " Buffer bind data type mismatch";
  if (value->data_alignment % arg->data_alignment != 0) {
    LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                 << " required alignment=" << arg->data_alignment
                 << ", provided alignment=" << value->data_alignment;
  }

  if (value->elem_offset.defined()) {
    if (is_zero(arg->elem_offset)) {
      TVM_FFI_ICHECK(is_zero(value->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << arg->elem_offset
          << ", provided elem_offset=" << value->elem_offset;
    }
    AccessPath data_path = base_path->Attr(ffi::String("data"));
    BindScalar(arg->data, value->data, data_path, false);
    AccessPath offset_path = base_path->Attr(ffi::String("elem_offset"));
    if (BindScalar(arg->elem_offset, value->elem_offset, offset_path, false)) {
      if (arg->offset_factor > 1) {
        PrimExpr offset = value->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), arg->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        PrimExpr acond = analyzer_.Simplify(truncmod(offset, factor) == zero);
        if (is_zero(acond)) {
          TVM_FFI_THROW(InternalError)
              << "Bind have an unmet assertion at " << RenderAccessPath(offset_path);
        }
        if (!is_one(acond)) {
          int param_index = GetParamIndex(base_path);
          int data_bytes = GetVectorBytes(arg->dtype);
          EmitAssert(acond, "ValueError",  //
                     "Misaligned buffer data on argument #", std::to_string(param_index),
                     when_calling_imm_, sig_imm_, "`,\n  expected data alignment=",
                     std::to_string(arg->offset_factor * data_bytes), " bytes");
        }
      }
    }
  }

  AccessPath shape_path = base_path->Attr(ffi::String("shape"));
  AccessPath strides_path = base_path->Attr(ffi::String("strides"));

  if (arg->shape.size() < value->shape.size()) {
    TVM_FFI_ICHECK(fuzzy_match) << "Buffer size mismatch at " << RenderAccessPath(base_path);
    size_t diff = value->shape.size() - arg->shape.size();
    for (size_t i = 0; i < diff; ++i) {
      TVM_FFI_ICHECK(is_one(analyzer_.Simplify(value->shape[i])))
          << "Buffer shape mismatch at " << RenderAccessPath(base_path) << ": " << arg->shape
          << " vs " << value->shape;
    }
    for (size_t i = 0; i < arg->shape.size(); ++i) {
      AccessPath shape_k_path = shape_path->ArrayItem(i);
      BindScalar(arg->shape[i], value->shape[i + diff], shape_k_path, false);
    }
    if (value->strides.size() != 0) {
      TVM_FFI_ICHECK_EQ(arg->strides.size(), arg->shape.size());
      TVM_FFI_ICHECK_EQ(value->strides.size(), value->shape.size());
      for (size_t i = 0; i < arg->strides.size(); ++i) {
        AccessPath strides_k_path = strides_path->ArrayItem(i);
        BindScalar(arg->strides[i], value->strides[i + diff], strides_k_path, false);
      }
    }
  } else {
    BindArray(arg->shape, value->shape, shape_path);
    BindArray(arg->strides, value->strides, strides_path);
  }
}

// ============================================================
// Per-dtype type-check + value-load methods
// ============================================================

/*! \brief Load the i-th packed argument as the given type. */
PrimExpr TVMFFIABIBuilder::LoadTVMFFIAnyUnionValue(const Var& v_packed_args, int param_index,
                                                   DataType arg_type) {
  ffi::Array<PrimExpr> call_args{v_packed_args, IntImm(DataType::Int(32), param_index),
                                 IntImm(DataType::Int(32), builtin::kTVMFFIAnyUnionValue)};
  DataType api_type = APIType(arg_type);
  PrimExpr res = Call(api_type, builtin::tvm_struct_get(), call_args);
  if (api_type != arg_type) {
    res = Cast(arg_type, res);
  }
  return res;
}

PrimExpr TVMFFIABIBuilder::DecodeParamOpaqueHandle(int param_index, const Var& type_index) {
  // ── Type check: accept handle-like types ───────────────────
  std::string expected_type = buffer_map_.count(params_[param_index]) ? "Tensor" : "pointer";
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFINone ||
                         type_index == ffi::TypeIndex::kTVMFFIOpaquePtr ||
                         type_index == ffi::TypeIndex::kTVMFFIDLTensorPtr ||
                         type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin,
                     expected_type);

  // ── Load value and apply tensor offset ─────────────────────
  const int64_t object_cell_offset = sizeof(TVMFFIObject);
  static_assert(sizeof(TVMFFIObject) == 24);
  PrimExpr arg_value =
      LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, params_[param_index].dtype());
  PrimExpr handle_from_tensor = Call(DataType::Handle(), tirx::builtin::handle_add_byte_offset(),
                                     {arg_value, IntImm(DataType::Int(32), object_cell_offset)});
  return Select(type_index == ffi::TypeIndex::kTVMFFITensor, handle_from_tensor, arg_value);
}

PrimExpr TVMFFIABIBuilder::DecodeParamBool(int param_index, const Var& type_index) {
  // ── Type check: accept bool or int ─────────────────────────
  EmitTypeIndexCheck(
      param_index,
      type_index == ffi::TypeIndex::kTVMFFIBool || type_index == ffi::TypeIndex::kTVMFFIInt,
      "boolean");
  return Cast(DataType::Bool(),
              LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, DataType::Int(64)));
}

PrimExpr TVMFFIABIBuilder::DecodeParamInt(int param_index, const Var& type_index, DataType dtype) {
  // ── Type check: accept int or bool ─────────────────────────
  EmitTypeIndexCheck(
      param_index,
      type_index == ffi::TypeIndex::kTVMFFIInt || type_index == ffi::TypeIndex::kTVMFFIBool, "int");
  return LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, dtype);
}

PrimExpr TVMFFIABIBuilder::DecodeParamFloat(int param_index, const Var& type_index,
                                            DataType dtype) {
  // ── Type check: accept float, int, or bool ─────────────────
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFIFloat ||
                         type_index == ffi::TypeIndex::kTVMFFIInt ||
                         type_index == ffi::TypeIndex::kTVMFFIBool,
                     "float");
  return tirx::Select(
      type_index == ffi::TypeIndex::kTVMFFIFloat,
      /* true_value = */ LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, dtype),
      /* false_value = */
      Cast(dtype, LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, DataType::Int(64))));
}

// ============================================================
// DecodeParam (primary public method)
// ============================================================

void TVMFFIABIBuilder::DecodeParam(int param_index) {
  Var param = params_[param_index];
  DataType dtype = param.dtype();

  // Extract type_index from packed_args
  Var type_index(param->name_hint + ".type_index", DataType::Int(32));
  init_nest_.push_back(
      Bind(type_index, tirx::Call(DataType::Int(32), builtin::tvm_struct_get(),
                                  {v_packed_args_, IntImm(DataType::Int(32), param_index),
                                   IntImm(DataType::Int(32), builtin::kTVMFFIAnyTypeIndex)})));

  // Type-check and load value via per-dtype dispatch
  PrimExpr arg_value;
  if (dtype.is_handle()) {
    arg_value = DecodeParamOpaqueHandle(param_index, type_index);
  } else if (dtype.is_bool()) {
    arg_value = DecodeParamBool(param_index, type_index);
  } else if (dtype.is_int() || dtype.is_uint()) {
    arg_value = DecodeParamInt(param_index, type_index, dtype);
  } else {
    TVM_FFI_ICHECK(dtype.is_float());
    arg_value = DecodeParamFloat(param_index, type_index, dtype);
  }

  // Bind scalar param to loaded value (defines vars before buffer binds reference them)
  AccessPath param_path = AccessPath::Root()->Extend(AccessStep::ArrayItem(param_index));
  BindScalar(param, arg_value, param_path, true);
}

// ============================================================
// DecodeAllParams (primary public method)
// ============================================================

void TVMFFIABIBuilder::DecodeAllParams() {
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(params_.size());

  // Phase 1: Decode each packed argument (type-check, value load, scalar bind)
  for (int i = 0; i < num_args; ++i) {
    DecodeParam(i);
  }

  // Phase 2: Bind DLTensor buffers (shape, strides, dtype, device checks)
  for (int i = 0; i < num_args; ++i) {
    Var param = params_[i];
    if (buffer_map_.count(param)) {
      Buffer buffer = buffer_map_[param];
      AccessPath param_path =
          AccessPath::Root()->Extend(AccessStep::ArrayItem(i))->Attr(ffi::String(buffer->name));
      DecodeParamDLTensor(buffer, device_type_, device_id_, param,
                          func_name_ + "." + param->name_hint, param_path);
      decl_buffers_.push_back(DeclBuffer(buffer));
    }
  }
}

// ============================================================
// DLTensorGetFieldPtr (helper for shape/strides pointer extraction)
// ============================================================

Var TVMFFIABIBuilder::DLTensorGetFieldPtr(const Var& handle, int field_kind,
                                          const std::string& var_name) {
  Var ptr(var_name, DataType::Handle());
  init_nest_.emplace_back(
      Bind(ptr, TVMStructGet(DataType::Handle(), handle, 0,
                             static_cast<builtin::TVMStructFieldKind>(field_kind))));
  return ptr;
}

// ============================================================
// LoadInt64ArrayElem (helper for shape/strides element access)
// ============================================================

PrimExpr TVMFFIABIBuilder::LoadInt64ArrayElem(const Var& ptr, int index) {
  return TVMStructGet(DataType::ShapeIndex(), ptr, index, builtin::kInt64ArrayElem);
}

// ============================================================
// Strides validation subfunctions
// ============================================================

void TVMFFIABIBuilder::BindCompactStrides(const Buffer& buffer, const Var& strides_ptr,
                                          const PrimExpr& v_strides_is_null,
                                          const AccessPath& param_path) {
  DataType stype = buffer->DefaultIndexType();
  PrimExpr expect_stride = make_const(stype, 1);
  ffi::Array<PrimExpr> conds;
  for (size_t i = buffer->shape.size(); i != 0; --i) {
    size_t k = i - 1;
    PrimExpr svalue = cast(stype, LoadInt64ArrayElem(strides_ptr, k));
    conds.push_back(buffer->shape[k] == 1 || expect_stride == svalue);
    expect_stride = expect_stride * buffer->shape[k];
  }
  if (conds.size() != 0) {
    int param_index = GetParamIndex(param_path);
    Stmt check = AssertStmt(
        foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
              const_true(1), conds),
        StringImm("ValueError"),
        ffi::Array<StringImm>({StringImm("Mismatched "), StringImm(buffer->name),
                               StringImm(".strides on argument #"),
                               StringImm(std::to_string(param_index)), when_calling_imm_, sig_imm_,
                               StringImm("`,\n  expected to be compact array")}));
    check = IfThenElse(Not(v_strides_is_null), check);
    asserts_.emplace_back(SeqStmt({check, Evaluate(0)}));
  }
}

void TVMFFIABIBuilder::BindAutoBroadcastStrides(const Buffer& buffer, const Var& strides_ptr,
                                                const PrimExpr& v_strides_is_null,
                                                const AccessPath& param_path) {
  DataType stype = buffer->DefaultIndexType();
  PrimExpr stride = make_const(stype, 1);
  for (size_t i = buffer->shape.size(); i != 0; --i) {
    size_t k = i - 1;
    PrimExpr value = cast(buffer->shape[k].dtype(), LoadInt64ArrayElem(strides_ptr, k));
    value = tvm::if_then_else(v_strides_is_null, stride, value);
    value = tvm::if_then_else(buffer->shape[k] == 1, 0, value);
    AccessPath strides_k_path = param_path->Attr(ffi::String("strides"))->ArrayItem(k);
    BindScalar(buffer->strides[k], value, strides_k_path, true);
    stride = analyzer_.Simplify(stride * buffer->shape[k]);
  }
}

void TVMFFIABIBuilder::BindRegularStrides(const Buffer& buffer, const Var& strides_ptr,
                                          const Var& shape_ptr, const PrimExpr& v_strides_is_null,
                                          const AccessPath& param_path) {
  PrimExpr stride_from_shape = 1;
  for (int k = buffer->strides.size() - 1; k >= 0; k--) {
    PrimExpr explicit_stride = cast(buffer->shape[k].dtype(), LoadInt64ArrayElem(strides_ptr, k));
    AccessPath strides_k_path = param_path->Attr(ffi::String("strides"))->ArrayItem(k);
    BindScalar(buffer->strides[k],
               tvm::if_then_else(v_strides_is_null, stride_from_shape, explicit_stride),
               strides_k_path, true);
    stride_from_shape *= cast(buffer->shape[k].dtype(), LoadInt64ArrayElem(shape_ptr, k));
  }
}

// ============================================================
// DecodeParamDLTensor (private)
// ============================================================

void TVMFFIABIBuilder::DecodeParamDLTensor(const Buffer& buffer, const PrimExpr& device_type,
                                           const PrimExpr& device_id, const Var& handle,
                                           const std::string& arg_name, AccessPath base_path) {
  const DataType tvm_ndim_type = DataType::Int(32);

  std::string buf_name = buffer->name;
  AccessPath param_path = base_path;
  int param_index = GetParamIndex(base_path);

  // ── Section: Null pointer check ──────────────────────────────
  EmitTypeIndexCheck(param_index, !Call(DataType::Bool(), builtin::isnullptr(), {handle}),
                     "Tensor");

  // ── Section: ndim ────────────────────────────────────────────
  PrimExpr v_ndim = TVMStructGet(tvm_ndim_type, handle, 0, builtin::kDLTensorNDim);
  PrimExpr a_ndim = make_const(tvm_ndim_type, static_cast<int64_t>(buffer->shape.size()));
  EmitAssert(a_ndim == v_ndim, "ValueError",  //
             "Mismatched ", buf_name, ".ndim on argument #", std::to_string(param_index),
             when_calling_imm_, sig_imm_, "`,\n  expected ", std::to_string(buffer->shape.size()));

  // ── Section: dtype ───────────────────────────────────────────
  {
    PrimExpr cond = (TVMStructGet(DataType::UInt(8), handle, 0, builtin::kDLTensorTypeCode) ==
                         IntImm(DataType::UInt(8), buffer->dtype.code()) &&
                     TVMStructGet(DataType::UInt(8), handle, 0, builtin::kDLTensorTypeBits) ==
                         IntImm(DataType::UInt(8), buffer->dtype.bits()) &&
                     TVMStructGet(DataType::UInt(16), handle, 0, builtin::kDLTensorTypeLanes) ==
                         IntImm(DataType::UInt(16), buffer->dtype.lanes()));
    if (!(buffer->dtype == DataType::Int(1) || buffer->dtype == DataType::Int(4) ||
          buffer->dtype == DataType::UInt(4))) {
      std::ostringstream dtype_os;
      dtype_os << buffer->dtype;
      EmitAssert(cond, "TypeError",  //
                 "Mismatched ", buf_name, ".dtype on argument #", std::to_string(param_index),
                 when_calling_imm_, sig_imm_, "`,\n  expected ", dtype_os.str());
    }
  }

  // ── Section: shape ───────────────────────────────────────────
  Var shape_ptr = DLTensorGetFieldPtr(handle, builtin::kDLTensorShape, arg_name + "_shape");
  for (size_t k = 0; k < buffer->shape.size(); ++k) {
    if (buffer->dtype == DataType::Int(4) || buffer->dtype == DataType::UInt(4) ||
        buffer->dtype == DataType::Int(1)) {
      break;
    }
    AccessPath shape_k_path = param_path->Attr(ffi::String("shape"))->ArrayItem(k);
    BindScalar(buffer->shape[k], cast(buffer->shape[k].dtype(), LoadInt64ArrayElem(shape_ptr, k)),
               shape_k_path, true);
  }

  // ── Section: strides ─────────────────────────────────────────
  Var strides_ptr = DLTensorGetFieldPtr(handle, builtin::kDLTensorStrides, arg_name + "_strides");
  PrimExpr v_strides_is_null = Call(DataType::Bool(), builtin::isnullptr(), {strides_ptr});
  if (buffer->strides.size() == 0) {
    BindCompactStrides(buffer, strides_ptr, v_strides_is_null, param_path);
  } else if (buffer->buffer_type == kAutoBroadcast) {
    BindAutoBroadcastStrides(buffer, strides_ptr, v_strides_is_null, param_path);
  } else {
    BindRegularStrides(buffer, strides_ptr, shape_ptr, v_strides_is_null, param_path);
  }

  // ── Section: byte_offset ─────────────────────────────────────
  int data_bytes = GetVectorBytes(buffer->dtype);
  AccessPath byte_offset_path = param_path->Attr(ffi::String("byte_offset"));
  if (const auto* const_offset = buffer->elem_offset.as<IntImmNode>()) {
    BindScalar(make_const(DataType::UInt(64), const_offset->value * data_bytes),
               TVMStructGet(DataType::UInt(64), handle, 0, builtin::kDLTensorByteOffset),
               byte_offset_path, true);
  } else {
    if (BindScalar(buffer->elem_offset,
                   cast(buffer->elem_offset.dtype(),
                        (TVMStructGet(DataType::UInt(64), handle, 0, builtin::kDLTensorByteOffset) /
                         make_const(DataType::UInt(64), data_bytes))),
                   byte_offset_path, true)) {
      if (buffer->offset_factor > 1) {
        PrimExpr offset = buffer->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        PrimExpr acond = analyzer_.Simplify(truncmod(offset, factor) == zero);
        if (is_zero(acond)) {
          TVM_FFI_THROW(InternalError)
              << "Bind have an unmet assertion at " << RenderAccessPath(byte_offset_path);
        }
        if (!is_one(acond)) {
          EmitAssert(acond, "ValueError",  //
                     "Misaligned Tensor data on argument #", std::to_string(param_index),
                     when_calling_imm_, sig_imm_, "`,\n  expected data alignment=",
                     std::to_string(buffer->offset_factor * data_bytes), " bytes");
        }
      }
    }
  }

  // ── Section: device ──────────────────────────────────────────
  {
    PrimExpr actual_device_type =
        TVMStructGet(DataType::Int(32), handle, 0, builtin::kDLTensorDeviceType);
    // Use custom assertion for device_type to show human-readable device name
    if (const auto* const_dt = device_type_.as<IntImmNode>()) {
      PrimExpr cond =
          analyzer_.Simplify(make_const(DataType::Int(32), const_dt->value) == actual_device_type);
      if (!is_one(cond)) {
        std::string device_name = runtime::DLDeviceType2Str(static_cast<int>(const_dt->value));
        EmitAssert(cond, "ValueError",  //
                   "Mismatched ", buf_name, ".device_type on argument #",
                   std::to_string(param_index), when_calling_imm_, sig_imm_, "`,\n  expected ",
                   device_name);
      }
    } else {
      AccessPath device_type_path = param_path->Attr(ffi::String("device_type"));
      BindScalar(device_type_, actual_device_type, device_type_path, true);
    }
    AccessPath device_id_path = param_path->Attr(ffi::String("device_id"));
    BindScalar(device_id_, TVMStructGet(DataType::Int(32), handle, 0, builtin::kDLTensorDeviceId),
               device_id_path, true);
  }

  // ── Section: data pointer ────────────────────────────────────
  {
    AccessPath data_path = param_path->Attr(ffi::String("data"));
    if (BindScalar(buffer->data,
                   TVMStructGet(DataType::Handle(), handle, 0, builtin::kDLTensorData), data_path,
                   true)) {
      Var vptr(buffer->data);

      auto alloc_size = [&]() -> PrimExpr {
        PrimExpr product = IntImm(buffer->DefaultIndexType(), 1);
        for (const auto& dim : buffer->shape) {
          product *= dim;
        }
        return product;
      }();
      // Data pointer null and alignment checks go to asserts_ because alloc_size
      // references buffer->shape which may contain forward-referenced symbolic vars.
      asserts_.emplace_back(AssertStmt(
          alloc_size == 0 || !Call(DataType::Bool(), builtin::isnullptr(), {vptr}),
          StringImm("ValueError"),
          ffi::Array<StringImm>({StringImm(buf_name),
                                 StringImm(" data pointer is NULL on argument #"),
                                 StringImm(std::to_string(param_index)), when_calling_imm_,
                                 sig_imm_, StringImm("`,\n  expected non-NULL data pointer")})));

      if (check_alignment_) {
        // Check data pointer alignment
        if (buffer->data_alignment > 1) {
          PrimExpr ptr_as_int =
              Call(DataType::UInt(64), builtin::reinterpret(), {cast(DataType::Handle(), vptr)});
          PrimExpr align_cond =
              truncmod(ptr_as_int, make_const(DataType::UInt(64), buffer->data_alignment)) ==
              make_const(DataType::UInt(64), 0);
          asserts_.emplace_back(AssertStmt(
              alloc_size == 0 || align_cond, StringImm("ValueError"),
              ffi::Array<StringImm>({StringImm("Misaligned Tensor data on argument #"),
                                     StringImm(std::to_string(param_index)), when_calling_imm_,
                                     sig_imm_, StringImm("`,\n  expected data alignment="),
                                     StringImm(std::to_string(buffer->data_alignment)),
                                     StringImm(" bytes")})));
        }
        // mark alignment of external bufs — must be after the alignment assertion
        // so the compiler does not emit aligned loads before the check fires.
        asserts_.emplace_back(AttrStmt(vptr, tirx::attr::storage_alignment,
                                       IntImm(DataType::Int(32), buffer->data_alignment),
                                       Evaluate(0)));
      } else {
        // Even without alignment check, mark alignment for the compiler.
        init_nest_.emplace_back(AttrStmt(vptr, tirx::attr::storage_alignment,
                                         IntImm(DataType::Int(32), buffer->data_alignment),
                                         Evaluate(0)));
      }
    }
  }
}

}  // namespace tirx
}  // namespace tvm
