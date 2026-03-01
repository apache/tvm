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
 * \brief Helper utility to match and bind arguments.
 */
#include "tvm_ffi_binder.h"

#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

using ffi::reflection::AccessPath;
using ffi::reflection::AccessStep;

void BinderAddAssert(arith::Analyzer* ana, PrimExpr cond, const std::string& arg_name,
                     std::vector<Stmt>* asserts) {
  PrimExpr scond = ana->Simplify(cond);
  if (is_zero(scond)) {
    TVM_FFI_THROW(InternalError) << "Bind have an unmet assertion: " << cond << ", "
                                 << " on argument " << arg_name;
  }
  if (!is_one(scond)) {
    std::ostringstream os;
    os << "Argument " << arg_name << " has an unsatisfied constraint: " << cond;
    asserts->emplace_back(
        AssertStmt(tvm::tir::StringImm("RuntimeError"), scond, {tvm::tir::StringImm(os.str())}));
  }
}

ArgBinder::ArgBinder(std::unordered_map<const VarNode*, PrimExpr>* def_map,
                     const std::string& func_name, const ffi::Array<Var>& params,
                     const ffi::Map<Var, Buffer>& buffer_map)
    : def_map_(def_map) {
  func_name_ = func_name;
  // Build function signature string
  std::ostringstream os;
  os << func_name << "(";
  for (size_t i = 0; i < params.size(); ++i) {
    if (i > 0) os << ", ";
    Var param = params[i];
    if (buffer_map.count(param)) {
      Buffer buf = buffer_map[param];
      // Use the buffer name (e.g. "a") not the handle name (e.g. "a_handle")
      std::string buf_name = buf->name;
      os << buf_name << ": Tensor([";
      for (size_t j = 0; j < buf->shape.size(); ++j) {
        if (j > 0) os << ", ";
        std::ostringstream shape_os;
        shape_os << buf->shape[j];
        os << shape_os.str();
      }
      os << "], " << buf->dtype << ")";
      // Record param_index -> param_name mapping
      param_names_[static_cast<int>(i)] = buf_name;
    } else {
      os << param->name_hint << ": " << param.dtype();
      // Record param_index -> param_name mapping
      param_names_[static_cast<int>(i)] = param->name_hint;
    }
  }
  os << ")";
  func_signature_ = os.str();
}

std::string ArgBinder::RenderAccessPath(const AccessPath& path) const {
  ffi::Array<AccessStep> steps = path->ToSteps();
  std::ostringstream os;
  bool first_printed = false;
  // Track whether we've consumed the root ArrayItem(param_index)
  // so we know the next Attr is the param name (which we skip in favor of
  // looking up param_names_).
  bool consumed_root_array_item = false;
  for (size_t i = 0; i < steps.size(); ++i) {
    const AccessStep& step = steps[i];
    if (step->kind == ffi::reflection::AccessKind::kAttr) {
      std::string field_name = step->key.cast<ffi::String>();
      if (!first_printed) {
        // If this Attr immediately follows a root ArrayItem, use the param name
        // from the mapping (which should be the same as field_name for BindDLTensor).
        if (consumed_root_array_item) {
          // Use the param name from our mapping (already resolved by root ArrayItem)
          os << field_name;
        } else {
          os << field_name;
        }
        first_printed = true;
      } else {
        os << "." << field_name;
      }
    } else if (step->kind == ffi::reflection::AccessKind::kArrayItem) {
      int64_t index = step->key.cast<int64_t>();
      if (!first_printed && !consumed_root_array_item) {
        // This is the root ArrayItem (param_index).
        // Mark it as consumed; the next Attr step will provide the display name.
        consumed_root_array_item = true;
        // Check if there's a following Attr step that will provide the name.
        // If there's no Attr step after (unusual), fall back to param_names_ lookup.
        if (i + 1 < steps.size() &&
            steps[i + 1]->kind == ffi::reflection::AccessKind::kAttr) {
          // The next Attr will handle the display name; skip this ArrayItem.
          continue;
        }
        // No following Attr: use param_names_ directly
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

bool ArgBinder::Bind_(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                      bool with_lets, AccessPath path) {
  TVM_FFI_ICHECK_EQ(arg.dtype(), value.dtype());
  bool has_path = path->depth > 0;
  if (const VarNode* v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      Var v_arg = Downcast<Var>(arg);
      defs_.emplace_back(v_arg);
      // Record first bind path for this variable
      if (first_bind_path_.find(v) == first_bind_path_.end() && has_path) {
        first_bind_path_.emplace(v, path);
      }
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmt(v_arg, value, Evaluate(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      // Duplicate bind: create rich assertion with both paths
      if (!func_signature_.empty()) {
        PrimExpr scond = analyzer_.Simplify(it->second == value);
        if (is_zero(scond)) {
          TVM_FFI_THROW(InternalError)
              << "Bind have an unmet assertion: " << it->second << " == " << value << ", "
              << " on argument " << arg_name;
        }
        if (!is_one(scond)) {
          // Find the first bind path for the variable
          auto path_it = first_bind_path_.find(v);
          // Render the current path description
          std::string current_path_str;
          if (has_path) {
            current_path_str = RenderAccessPath(path);
          } else {
            current_path_str = arg_name;
          }
          std::string first_path_str;
          if (path_it != first_bind_path_.end()) {
            first_path_str = RenderAccessPath(path_it->second);
          }
          // Determine param_index from the path root step
          int param_index = -1;
          if (has_path) {
            ffi::Array<AccessStep> steps = path->ToSteps();
            if (steps.size() >= 1 && steps[0]->kind == ffi::reflection::AccessKind::kArrayItem) {
              param_index = static_cast<int>(steps[0]->key.cast<int64_t>());
            }
          }
          std::ostringstream detail_os;
          detail_os << "Mismatched " << current_path_str;
          if (param_index >= 0) {
            detail_os << " on argument #" << param_index;
          }
          std::string detail_str = detail_os.str();
          std::string expectation;
          if (!first_path_str.empty() && first_path_str != current_path_str) {
            expectation = "to match " + first_path_str;
          } else {
            expectation = "matching value";
          }
          // Build inline rich assert parts
          ffi::Array<StringImm> parts;
          parts.push_back(tvm::tir::StringImm(detail_str + " when calling:\n  `"));
          parts.push_back(tvm::tir::StringImm(func_signature_));
          parts.push_back(tvm::tir::StringImm("`,\n  expected " + expectation));
          asserts_.emplace_back(
              AssertStmt(tvm::tir::StringImm("ValueError"), scond, parts));
        }
      } else {
        BinderAddAssert(&analyzer_, it->second == value, arg_name, &asserts_);
      }
    }
  } else {
    if (!func_signature_.empty()) {
      PrimExpr scond = analyzer_.Simplify(arg == value);
      if (is_zero(scond)) {
        TVM_FFI_THROW(InternalError) << "Bind have an unmet assertion: " << arg << " == " << value
                                     << ", on argument " << arg_name;
      }
      if (!is_one(scond)) {
        // Render path if available
        std::string path_str;
        if (has_path) {
          path_str = RenderAccessPath(path);
        } else {
          path_str = arg_name;
        }
        // Determine param_index from path
        int param_index = -1;
        if (has_path) {
          ffi::Array<AccessStep> steps = path->ToSteps();
          if (steps.size() >= 1 && steps[0]->kind == ffi::reflection::AccessKind::kArrayItem) {
            param_index = static_cast<int>(steps[0]->key.cast<int64_t>());
          }
        }
        std::ostringstream detail_os;
        detail_os << "Invalid " << path_str;
        if (param_index >= 0) {
          detail_os << " on argument #" << param_index;
        }
        std::string detail_str = detail_os.str();
        std::ostringstream expect_os;
        expect_os << arg;
        std::string expectation = expect_os.str();
        // Build inline rich assert parts
        ffi::Array<StringImm> parts;
        parts.push_back(tvm::tir::StringImm(detail_str + " when calling:\n  `"));
        parts.push_back(tvm::tir::StringImm(func_signature_));
        parts.push_back(tvm::tir::StringImm("`,\n  expected " + expectation));
        asserts_.emplace_back(
            AssertStmt(tvm::tir::StringImm("ValueError"), scond, parts));
      }
    } else {
      BinderAddAssert(&analyzer_, arg == value, arg_name, &asserts_);
    }
  }
  return false;
}

void ArgBinder::Bind(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                     bool with_let) {
  Bind_(arg, value, arg_name, with_let, AccessPath::Root());
}

void ArgBinder::BindArray(const ffi::Array<PrimExpr>& arg, const ffi::Array<PrimExpr>& value,
                          const std::string& arg_name) {
  TVM_FFI_ICHECK_EQ(arg.size(), value.size()) << "Argument " << arg_name << " array size mismatch";
  for (size_t i = 0; i < arg.size(); ++i) {
    std::ostringstream os;
    os << arg_name << "[" << i << "]";
    this->Bind(arg[i], value[i], os.str());
  }
}

void ArgBinder::BindBuffer(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                           bool fuzzy_match) {
  TVM_FFI_ICHECK_EQ(arg.scope(), value.scope())
      << "Argument " << arg_name << " Buffer bind scope mismatch";
  TVM_FFI_ICHECK_EQ(arg->dtype, value->dtype)
      << "Argument " << arg_name << " Buffer bind data type mismatch";
  if (value->data_alignment % arg->data_alignment != 0) {
    LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                 << " required alignment=" << arg->data_alignment
                 << ", provided alignment=" << value->data_alignment;
  }

  if (value->elem_offset.defined()) {
    // bind pointer and offset.
    if (is_zero(arg->elem_offset)) {
      TVM_FFI_ICHECK(is_zero(value->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << arg->elem_offset
          << ", provided elem_offset=" << value->elem_offset;
    }

    this->Bind(arg->data, value->data, arg_name + ".data");
    if (Bind_(arg->elem_offset, value->elem_offset, arg_name + ".elem_offset", false,
              AccessPath::Root())) {
      if (arg->offset_factor > 1) {
        PrimExpr offset = value->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), arg->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        BinderAddAssert(&analyzer_, truncmod(offset, factor) == zero, arg_name + ".elem_offset",
                        &asserts_);
      }
    }
  }

  if (arg->shape.size() < value->shape.size()) {
    TVM_FFI_ICHECK(fuzzy_match) << "Argument " << arg_name << " size mismatch";
    size_t diff = value->shape.size() - arg->shape.size();
    for (size_t i = 0; i < diff; ++i) {
      TVM_FFI_ICHECK(is_one(analyzer_.Simplify(value->shape[i])))
          << "Argument " << arg_name << " shape mismatch" << arg->shape << " vs " << value->shape;
    }
    for (size_t i = 0; i < arg->shape.size(); ++i) {
      std::ostringstream os;
      os << arg_name << ".shape[" << i << "]";
      this->Bind(arg->shape[i], value->shape[i + diff], os.str());
    }
    if (value->strides.size() != 0) {
      TVM_FFI_ICHECK_EQ(arg->strides.size(), arg->shape.size());
      TVM_FFI_ICHECK_EQ(value->strides.size(), value->shape.size());
      for (size_t i = 0; i < arg->strides.size(); ++i) {
        std::ostringstream os;
        os << arg_name << ".strides[" << i << "]";
        this->Bind(arg->strides[i], value->strides[i + diff], os.str());
      }
    }
  } else {
    this->BindArray(arg->shape, value->shape, arg_name + ".shape");
    this->BindArray(arg->strides, value->strides, arg_name + ".strides");
  }
}

inline PrimExpr TVMArrayGet(DataType t, Var arr, builtin::TVMStructFieldKind kind) {
  return TVMStructGet(t, arr, 0, kind);
}

void ArgBinder::BindDLTensor(const Buffer& buffer, const PrimExpr& device_type,
                             const PrimExpr& device_id, const Var& handle,
                             const std::string& arg_name, AccessPath base_path) {
  const DataType tvm_shape_type = DataType::ShapeIndex();
  const DataType tvm_ndim_type = DataType::Int(32);
  const Stmt nop = Evaluate(0);

  // Use the buffer's own name for display purposes (e.g. "a" not "a_handle").
  std::string buf_name = buffer->name;

  // base_path represents the parameter (e.g. Root()->ArrayItem(i)->Attr(buf_name))
  // Extend it for sub-fields like shape, strides, etc.
  AccessPath param_path = base_path;

  // Determine param_index from the base_path root step
  int param_index = -1;
  {
    ffi::Array<AccessStep> steps = base_path->ToSteps();
    if (steps.size() >= 1 && steps[0]->kind == ffi::reflection::AccessKind::kArrayItem) {
      param_index = static_cast<int>(steps[0]->key.cast<int64_t>());
    }
  }

  // Helper functions for shape/stride name formatting
  auto shape_handle_name = [&]() { return arg_name + ".shape"; };
  auto stride_handle_name = [&]() { return arg_name + ".strides"; };
  auto array_element_name = [&](const std::string& arr_name, size_t k) {
    std::stringstream ss;
    ss << arr_name << '[' << k << ']';
    return ss.str();
  };
  auto shape_element_name = [&](size_t k) { return array_element_name(shape_handle_name(), k); };
  auto stride_element_name = [&](size_t k) { return array_element_name(stride_handle_name(), k); };

  // Helper to build inline rich assert parts
  auto make_rich_assert = [&](const std::string& kind, PrimExpr cond,
                              const std::string& detail_str, const std::string& expectation,
                              std::vector<Stmt>* asserts) {
    if (func_signature_.empty()) {
      // Fallback: no signature info, use simple message
      std::ostringstream os;
      os << detail_str << ", expected " << expectation;
      asserts->emplace_back(
          AssertStmt(tvm::tir::StringImm(kind), cond, {tvm::tir::StringImm(os.str())}));
    } else {
      ffi::Array<StringImm> parts;
      parts.push_back(tvm::tir::StringImm(detail_str + " when calling:\n  `"));
      parts.push_back(tvm::tir::StringImm(func_signature_));
      parts.push_back(tvm::tir::StringImm("`,\n  expected " + expectation));
      asserts->emplace_back(AssertStmt(tvm::tir::StringImm(kind), cond, parts));
    }
  };

  // null pointer check: TypeError for wrong type (not a tensor)
  {
    std::ostringstream detail;
    detail << "Mismatched type on argument #" << param_index;
    make_rich_assert("TypeError", !Call(DataType::Bool(), builtin::isnullptr(), {handle}),
                     detail.str(), "Tensor", &init_nest_);
  }

  // dimension checks
  PrimExpr v_ndim = TVMArrayGet(tvm_ndim_type, handle, builtin::kArrNDim);
  PrimExpr a_ndim = make_const(tvm_ndim_type, static_cast<int64_t>(buffer->shape.size()));
  {
    std::ostringstream detail;
    detail << "Mismatched " << buf_name << ".ndim on argument #" << param_index;
    std::ostringstream expect;
    expect << buffer->shape.size();
    make_rich_assert("ValueError", a_ndim == v_ndim, detail.str(), expect.str(), &init_nest_);
  }

  // type checks
  PrimExpr cond = (TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeCode) ==
                       IntImm(DataType::UInt(8), buffer->dtype.code()) &&
                   TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeBits) ==
                       IntImm(DataType::UInt(8), buffer->dtype.bits()) &&
                   TVMArrayGet(DataType::UInt(16), handle, builtin::kArrTypeLanes) ==
                       IntImm(DataType::UInt(16), buffer->dtype.lanes()));
  if (!(buffer->dtype == DataType::Int(1) || buffer->dtype == DataType::Int(4) ||
        buffer->dtype == DataType::UInt(4))) {
    std::ostringstream detail;
    detail << "Mismatched " << buf_name << ".dtype on argument #" << param_index;
    std::ostringstream expect;
    expect << buffer->dtype;
    make_rich_assert("TypeError", cond, detail.str(), expect.str(), &asserts_);
  }

  // shape field
  Buffer buf_shape = decl_buffer({IntImm(DataType::Int(32), buffer->shape.size())}, tvm_shape_type,
                                 shape_handle_name());
  Var v_shape(shape_handle_name(), DataType::Handle());
  def_handle_dtype_.Set(v_shape, make_const(tvm_shape_type, 0));
  init_nest_.emplace_back(
      LetStmt(buf_shape->data, TVMArrayGet(DataType::Handle(), handle, builtin::kArrShape), nop));
  init_nest_.emplace_back(DeclBuffer(buf_shape, nop));
  for (size_t k = 0; k < buffer->shape.size(); ++k) {
    if (buffer->dtype == DataType::Int(4) || buffer->dtype == DataType::UInt(4) ||
        buffer->dtype == DataType::Int(1)) {
      break;
    }
    AccessPath shape_k_path = param_path->Attr(ffi::String("shape"))->ArrayItem(k);
    Bind_(buffer->shape[k],
          cast(buffer->shape[k].dtype(), BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)})),
          shape_element_name(k), true, shape_k_path);
  }
  // strides field
  Buffer buf_strides = decl_buffer({IntImm(DataType::Int(32), buffer->strides.size())},
                                   tvm_shape_type, arg_name + ".strides");
  def_handle_dtype_.Set(buf_strides->data, tir::TypeAnnotation(tvm_shape_type));
  init_nest_.emplace_back(LetStmt(
      buf_strides->data, TVMArrayGet(DataType::Handle(), handle, builtin::kArrStrides), nop));
  init_nest_.emplace_back(DeclBuffer(buf_strides, nop));
  PrimExpr v_strides_is_null = Call(DataType::Bool(), builtin::isnullptr(), {buf_strides->data});
  if (buffer->strides.size() == 0) {
    // Assert the buffer is compact
    DataType stype = buffer->DefaultIndexType();
    PrimExpr expect_stride = make_const(stype, 1);
    ffi::Array<PrimExpr> conds;
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      PrimExpr svalue = cast(stype, BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      conds.push_back(buffer->shape[k] == 1 || expect_stride == svalue);
      expect_stride = expect_stride * buffer->shape[k];
    }
    if (conds.size() != 0) {
      std::ostringstream detail;
      detail << "Mismatched " << buf_name << ".strides on argument #" << param_index;
      std::string expectation = "to be compact array";
      ffi::Array<StringImm> parts;
      if (!func_signature_.empty()) {
        parts.push_back(tvm::tir::StringImm(detail.str() + " when calling:\n  `"));
        parts.push_back(tvm::tir::StringImm(func_signature_));
        parts.push_back(tvm::tir::StringImm("`,\n  expected " + expectation));
      } else {
        parts.push_back(tvm::tir::StringImm(detail.str() + ", expected " + expectation));
      }
      Stmt check = AssertStmt(
          tvm::tir::StringImm("ValueError"),
          foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
                const_true(1), conds),
          parts);
      check = IfThenElse(Not(v_strides_is_null), check);
      asserts_.emplace_back(SeqStmt({check, Evaluate(0)}));
    }
  } else if (buffer->buffer_type == kAutoBroadcast) {
    DataType stype = buffer->DefaultIndexType();
    PrimExpr stride = make_const(stype, 1);
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      PrimExpr value =
          cast(buffer->shape[k].dtype(), BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      value = tvm::if_then_else(v_strides_is_null, stride, value);
      value = tvm::if_then_else(buffer->shape[k] == 1, 0, value);
      AccessPath strides_k_path = param_path->Attr(ffi::String("strides"))->ArrayItem(k);
      Bind_(buffer->strides[k], value, stride_element_name(k), true, strides_k_path);
      stride = analyzer_.Simplify(stride * buffer->shape[k]);
    }
  } else {
    PrimExpr stride_from_shape = 1;

    for (int k = buffer->strides.size() - 1; k >= 0; k--) {
      PrimExpr explicit_stride =
          cast(buffer->shape[k].dtype(), BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      AccessPath strides_k_path = param_path->Attr(ffi::String("strides"))->ArrayItem(k);
      Bind_(buffer->strides[k],
            tvm::if_then_else(v_strides_is_null, stride_from_shape, explicit_stride),
            stride_element_name(k), true, strides_k_path);

      stride_from_shape *=
          cast(buffer->shape[k].dtype(), BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)}));
    }
  }
  // Byte_offset field.
  int data_bytes = GetVectorBytes(buffer->dtype);
  AccessPath byte_offset_path = param_path->Attr(ffi::String("byte_offset"));

  if (const auto* const_offset = buffer->elem_offset.as<IntImmNode>()) {
    Bind_(make_const(DataType::UInt(64), const_offset->value * data_bytes),
          TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset),
          arg_name + ".byte_offset", true, byte_offset_path);
  } else {
    if (Bind_(buffer->elem_offset,
              cast(buffer->elem_offset.dtype(),
                   (TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset) /
                    make_const(DataType::UInt(64), data_bytes))),
              arg_name + ".elem_offset", true, byte_offset_path)) {
      if (buffer->offset_factor > 1) {
        PrimExpr offset = buffer->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        // Generate rich alignment assertion
        if (!func_signature_.empty() && param_index >= 0) {
          PrimExpr acond = analyzer_.Simplify(truncmod(offset, factor) == zero);
          if (is_zero(acond)) {
            TVM_FFI_THROW(InternalError)
                << "Bind have an unmet assertion on " << arg_name << ".elem_offset";
          }
          if (!is_one(acond)) {
            std::ostringstream detail;
            detail << "Misaligned Tensor data on argument #" << param_index;
            std::ostringstream expect;
            expect << "data alignment=" << (buffer->offset_factor * data_bytes) << " bytes";
            make_rich_assert("ValueError", acond, detail.str(), expect.str(), &asserts_);
          }
        } else {
          BinderAddAssert(&analyzer_, truncmod(offset, factor) == zero, arg_name + ".elem_offset",
                          &asserts_);
        }
      }
    }
  }
  // device info.
  AccessPath device_type_path = param_path->Attr(ffi::String("device_type"));
  AccessPath device_id_path = param_path->Attr(ffi::String("device_id"));
  Bind_(device_type, TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceType),
        arg_name + ".device_type", true, device_type_path);
  Bind_(device_id, TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceId),
        arg_name + ".device_id", true, device_id_path);

  // Data field.  Because the validation of the data field may depend
  // on a dynamic size defined by the other DLTensor* parameters, this
  // field must be generated last.
  AccessPath data_path = param_path->Attr(ffi::String("data"));
  if (Bind_(buffer->data, TVMArrayGet(DataType::Handle(), handle, builtin::kArrData),
            arg_name + ".data", true, data_path)) {
    Var vptr(buffer->data);

    // Check if the data pointer is NULL.  This check is skipped for
    // size-0 arrays, since CUDA provides a NULL pointer for size-zero
    // allocations.
    auto alloc_size = [&]() -> PrimExpr {
      PrimExpr product = IntImm(buffer->DefaultIndexType(), 1);
      for (const auto& dim : buffer->shape) {
        product *= dim;
      }
      return product;
    }();
    {
      std::ostringstream detail;
      detail << buf_name << " data pointer is NULL on argument #" << param_index;
      make_rich_assert("ValueError",
                       alloc_size == 0 || !Call(DataType::Bool(), builtin::isnullptr(), {vptr}),
                       detail.str(), "non-NULL data pointer", &asserts_);
    }

    def_handle_dtype_.Set(vptr, tir::TypeAnnotation(buffer->dtype));
    // mark alignment of external bufs
    init_nest_.emplace_back(AttrStmt(vptr, tir::attr::storage_alignment,
                                     IntImm(DataType::Int(32), buffer->data_alignment), nop));
  }
}

}  // namespace tir
}  // namespace tvm
