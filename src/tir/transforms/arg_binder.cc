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
 * \file arg_binder.cc
 * \brief Helper utility to match and bind arguments.
 */
#include "arg_binder.h"

#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

void BinderAddAssert(arith::Analyzer* ana, PrimExpr cond, const std::string& arg_name,
                     std::vector<Stmt>* asserts) {
  PrimExpr scond = ana->Simplify(cond);
  if (is_zero(scond)) {
    LOG(FATAL) << "Bind have an unmet assertion: " << cond << ", "
               << " on argument " << arg_name;
  }
  if (!is_one(scond)) {
    std::ostringstream os;
    os << "Argument " << arg_name << " has an unsatisfied constraint: " << cond;
    asserts->emplace_back(AssertStmt(scond, tvm::tir::StringImm(os.str()), Evaluate(0)));
  }
}

bool ArgBinder::Bind_(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                      bool with_lets) {
  ICHECK_EQ(arg.dtype(), value.dtype());
  if (const VarNode* v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      Var v_arg = Downcast<Var>(arg);
      defs_.emplace_back(v_arg);
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmt(v_arg, value, Evaluate(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      BinderAddAssert(&analyzer_, it->second == value, arg_name, &asserts_);
    }
  } else {
    BinderAddAssert(&analyzer_, arg == value, arg_name, &asserts_);
  }
  return false;
}

void ArgBinder::Bind(const PrimExpr& arg, const PrimExpr& value, const std::string& arg_name,
                     bool with_let) {
  Bind_(arg, value, arg_name, with_let);
}

void ArgBinder::BindArray(const Array<PrimExpr>& arg, const Array<PrimExpr>& value,
                          const std::string& arg_name) {
  ICHECK_EQ(arg.size(), value.size()) << "Argument " << arg_name << " array size mismatch";
  for (size_t i = 0; i < arg.size(); ++i) {
    std::ostringstream os;
    os << arg_name << "[" << i << "]";
    this->Bind(arg[i], value[i], os.str());
  }
}

void ArgBinder::BindBuffer(const Buffer& arg, const Buffer& value, const std::string& arg_name,
                           bool fuzzy_match) {
  ICHECK_EQ(arg.scope(), value.scope()) << "Argument " << arg_name << " Buffer bind scope mismatch";
  ICHECK_EQ(arg->dtype, value->dtype)
      << "Argument " << arg_name << " Buffer bind data type mismatch";
  if (value->data_alignment % arg->data_alignment != 0) {
    LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                 << " required_alignment=" << arg->data_alignment
                 << ", provided_alignment=" << value->data_alignment;
  }

  if (value->elem_offset.defined()) {
    // bind pointer and offset.
    if (is_zero(arg->elem_offset)) {
      ICHECK(is_zero(value->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << arg->elem_offset
          << ", provided elem_offset=" << value->elem_offset;
    }

    this->Bind(arg->data, value->data, arg_name + ".data");
    if (Bind_(arg->elem_offset, value->elem_offset, arg_name + ".elem_offset", false)) {
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
    ICHECK(fuzzy_match) << "Argument " << arg_name << " size mismatch";
    size_t diff = value->shape.size() - arg->shape.size();
    for (size_t i = 0; i < diff; ++i) {
      ICHECK(is_one(analyzer_.Simplify(value->shape[i])))
          << "Argument " << arg_name << " shape mismatch" << arg->shape << " vs " << value->shape;
    }
    for (size_t i = 0; i < arg->shape.size(); ++i) {
      std::ostringstream os;
      os << arg_name << ".shape[" << i << "]";
      this->Bind(arg->shape[i], value->shape[i + diff], os.str());
    }
    if (value->strides.size() != 0) {
      ICHECK_EQ(arg->strides.size(), arg->shape.size());
      ICHECK_EQ(value->strides.size(), value->shape.size());
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
                             const std::string& arg_name) {
  const DataType tvm_shape_type = DataType::ShapeIndex();
  const DataType tvm_ndim_type = DataType::Int(32);
  const Stmt nop = Evaluate(0);

  init_nest_.emplace_back(AssertStmt(
      !Call(DataType::Bool(), builtin::isnullptr(), {handle}),
      tvm::tir::StringImm(arg_name + " is expected to have non-NULL DLTensor* pointer"), nop));

  // dimension checks
  PrimExpr v_ndim = TVMArrayGet(tvm_ndim_type, handle, builtin::kArrNDim);

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

  PrimExpr a_ndim = make_const(tvm_ndim_type, static_cast<int64_t>(buffer->shape.size()));
  std::ostringstream ndim_err_msg;
  ndim_err_msg << arg_name << ".ndim is expected to equal " << buffer->shape.size();
  auto msg = tvm::tir::StringImm(ndim_err_msg.str());
  init_nest_.emplace_back(AssertStmt(a_ndim == v_ndim, msg, nop));
  // type checks
  std::ostringstream type_err_msg;
  type_err_msg << arg_name << ".dtype is expected to be " << buffer->dtype;
  PrimExpr cond = (TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeCode) ==
                       IntImm(DataType::UInt(8), buffer->dtype.code()) &&
                   TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeBits) ==
                       IntImm(DataType::UInt(8), buffer->dtype.bits()) &&
                   TVMArrayGet(DataType::UInt(16), handle, builtin::kArrTypeLanes) ==
                       IntImm(DataType::UInt(16), buffer->dtype.lanes()));
  if (!(buffer->dtype == DataType::Int(1) || buffer->dtype == DataType::Int(4) ||
        buffer->dtype == DataType::UInt(4))) {
    auto type_msg = tvm::tir::StringImm(type_err_msg.str());
    asserts_.emplace_back(AssertStmt(cond, type_msg, nop));
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
    Bind_(buffer->shape[k],
          cast(buffer->shape[k].dtype(), BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)})),
          shape_element_name(k), true);
  }
  // strides field
  Buffer buf_strides = decl_buffer({IntImm(DataType::Int(32), buffer->strides.size())},
                                   tvm_shape_type, arg_name + ".strides");
  def_handle_dtype_.Set(buf_strides->data, tir::TypeAnnotation(tvm_shape_type));
  init_nest_.emplace_back(LetStmt(
      buf_strides->data, TVMArrayGet(DataType::Handle(), handle, builtin::kArrStrides), nop));
  init_nest_.emplace_back(DeclBuffer(buf_strides, nop));
  PrimExpr v_strides_is_null = Call(DataType::Bool(1), builtin::isnullptr(), {buf_strides->data});
  if (buffer->strides.size() == 0) {
    // Assert the buffer is compact
    DataType stype = buffer->DefaultIndexType();
    PrimExpr expect_stride = make_const(stype, 1);
    Array<PrimExpr> conds;
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      PrimExpr svalue = cast(stype, BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      conds.push_back(buffer->shape[k] == 1 || expect_stride == svalue);
      expect_stride = expect_stride * buffer->shape[k];
    }
    std::ostringstream stride_err_msg;
    stride_err_msg << stride_handle_name() << ": expected to be compact array";
    if (conds.size() != 0) {
      auto stride_msg = tvm::tir::StringImm(stride_err_msg.str());
      Stmt check = AssertStmt(
          foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
                const_true(1), conds),
          stride_msg, Evaluate(0));
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
      Bind_(buffer->strides[k], value, stride_element_name(k), true);
      stride = analyzer_.Simplify(stride * buffer->shape[k]);
    }
  } else {
    PrimExpr stride_from_shape = 1;

    for (int k = buffer->strides.size() - 1; k >= 0; k--) {
      PrimExpr explicit_stride =
          cast(buffer->shape[k].dtype(), BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));

      Bind_(buffer->strides[k],
            tvm::if_then_else(v_strides_is_null, stride_from_shape, explicit_stride),
            stride_element_name(k), true);

      stride_from_shape *=
          cast(buffer->shape[k].dtype(), BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)}));
    }
  }
  // Byte_offset field.
  int data_bytes = GetVectorBytes(buffer->dtype);

  if (const auto* const_offset = buffer->elem_offset.as<IntImmNode>()) {
    Bind_(make_const(DataType::UInt(64), const_offset->value * data_bytes),
          TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset),
          arg_name + ".byte_offset", true);
  } else {
    if (Bind_(buffer->elem_offset,
              cast(buffer->elem_offset.dtype(),
                   (TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset) /
                    make_const(DataType::UInt(64), data_bytes))),
              arg_name + ".elem_offset", true)) {
      if (buffer->offset_factor > 1) {
        PrimExpr offset = buffer->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        BinderAddAssert(&analyzer_, truncmod(offset, factor) == zero, arg_name + ".elem_offset",
                        &asserts_);
      }
    }
  }
  // device info.
  Bind_(device_type, TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceType),
        arg_name + ".device_type", true);
  Bind_(device_id, TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceId),
        arg_name + ".device_id", true);

  // Data field.  Because the validation of the data field may depend
  // on a dynamic size defined by the other DLTensor* parameters, this
  // field must be generated last.
  if (Bind_(buffer->data, TVMArrayGet(DataType::Handle(), handle, builtin::kArrData),
            arg_name + ".data", true)) {
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
    asserts_.emplace_back(AssertStmt(
        alloc_size == 0 || !Call(DataType::Bool(), builtin::isnullptr(), {vptr}),
        tvm::tir::StringImm(arg_name + " is expected to have non-NULL data pointer"), nop));

    def_handle_dtype_.Set(vptr, tir::TypeAnnotation(buffer->dtype));
    // mark alignment of external bufs
    init_nest_.emplace_back(AttrStmt(vptr, tir::attr::storage_alignment,
                                     IntImm(DataType::Int(32), buffer->data_alignment), nop));
  }
}

}  // namespace tir
}  // namespace tvm
