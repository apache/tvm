/*!
 *  Copyright (c) 2017 by Contributors
 * \file arg_binder.cc
 * \brief Helper utility to match and bind arguments.
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/runtime/device_api.h>
#include "./ir_util.h"
#include "./arg_binder.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

void BinderAddAssert(Expr cond,
                     const std::string& arg_name,
                     std::vector<Stmt>* asserts) {
  cond = Simplify(cond);
  if (is_zero(cond)) {
    LOG(FATAL) << "Bind have unmet assertion: " << cond << ", " << " on argument " << arg_name;
  }
  if (!is_one(cond)) {
    std::ostringstream os;
    os << "Argument " << arg_name << " has a unsatisfied constraint";
    asserts->emplace_back(AssertStmt::make(cond, os.str()));
  }
}

bool ArgBinder::Bind_(const Expr& arg,
                      const Expr& value,
                      const std::string& arg_name,
                      bool with_lets) {
  CHECK_EQ(arg.type(), value.type());
  if (const Variable* v = arg.as<Variable>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      Var v_arg(arg.node_);
      defs_.emplace_back(v_arg);
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmt::make(v_arg, value, Evaluate::make(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      BinderAddAssert(it->second == value, arg_name, &asserts_);
    }
  } else {
    BinderAddAssert(arg == value, arg_name, &asserts_);
  }
  return false;
}

void ArgBinder::Bind(const Expr& arg,
                     const Expr& value,
                     const std::string& arg_name,
                     bool with_let) {
  Bind_(arg, value, arg_name, with_let);
}

void ArgBinder::BindArray(const Array<Expr>& arg,
                          const Array<Expr>& value,
                          const std::string& arg_name) {
  CHECK_EQ(arg.size(), value.size())
      << "Argument " << arg_name << " array size mismatch";
  for (size_t i = 0; i < arg.size(); ++i) {
    std::ostringstream os;
    os << arg_name << "[" << i << "]";
    this->Bind(arg[i], value[i], os.str());
  }
}

void ArgBinder::BindBuffer(const Buffer& arg,
                           const Buffer& value,
                           const std::string& arg_name) {
  CHECK_EQ(arg->scope, value->scope)
      << "Argument " << arg_name
      << " Buffer bind scope mismatch";
  this->Bind(arg->data, value->data, arg_name + ".data");
  this->BindArray(arg->shape, value->shape, arg_name + ".shape");
  this->BindArray(arg->strides, value->strides, arg_name + ".strides");
  this->Bind(arg->elem_offset, value->elem_offset, arg_name + ".elem_offset");
}

inline Expr TVMArrayGet(Type t, Var arr, intrinsic::TVMStructFieldKind kind) {
  return TVMStructGet(t, arr, 0, kind);
}

inline Stmt AssertNull(Var handle, std::string msg) {
  return AssertStmt::make(Call::make(
      Bool(1), intrinsic::tvm_handle_is_null,
      {handle}, Call::PureIntrinsic), msg);
}

void ArgBinder::BindDLTensor(const Buffer& buffer,
                             const Expr& device_type,
                             const Expr& device_id,
                             const Var& handle,
                             const std::string& arg_name) {
  const Type tvm_shape_type = TVMShapeIndexType();
  const Type tvm_ndim_type = Int(32);
  const Stmt nop = Evaluate::make(0);
  // dimension checks
  Expr v_ndim = TVMArrayGet(tvm_ndim_type, handle, intrinsic::kArrNDim);
  Expr a_ndim = make_const(tvm_ndim_type,
                           static_cast<int64_t>(buffer->shape.size()));
  std::ostringstream ndim_err_msg;
  ndim_err_msg << arg_name
               << ".ndim is expected to equal "
               << buffer->shape.size();
  asserts_.emplace_back(AssertStmt::make(a_ndim == v_ndim, ndim_err_msg.str()));
  // type checks
  Type dtype = buffer->dtype;
  std::ostringstream type_err_msg;
  type_err_msg << arg_name << ".dtype is expected to be " << dtype;
  Expr cond = (TVMArrayGet(UInt(8), handle, intrinsic::kArrTypeCode) ==
               UIntImm::make(UInt(8), dtype.code()) &&
               TVMArrayGet(UInt(8), handle, intrinsic::kArrTypeBits) ==
               UIntImm::make(UInt(8), dtype.bits()) &&
               TVMArrayGet(UInt(16), handle, intrinsic::kArrTypeLanes) ==
               UIntImm::make(UInt(16), dtype.lanes()));
  asserts_.emplace_back(AssertStmt::make(cond, type_err_msg.str()));
  // data field
  if (Bind_(buffer->data, TVMArrayGet(Handle(), handle, intrinsic::kArrData),
            arg_name + ".data", true)) {
    Var vptr(buffer->data);
    def_handle_dtype_.Set(vptr, make_const(buffer->dtype, 0));
    // mark alignment of external bufs
    init_nest_.emplace_back(AttrStmt::make(
        vptr, ir::attr::storage_alignment,
        IntImm::make(Int(32), runtime::kAllocAlignment), nop));
  }

  Var v_shape(arg_name + ".shape", Handle());
  def_handle_dtype_.Set(v_shape, make_const(tvm_shape_type, 0));
  init_nest_.emplace_back(LetStmt::make(
      v_shape, TVMArrayGet(Handle(), handle, intrinsic::kArrShape), nop));
  for (size_t k = 0; k < buffer->shape.size(); ++k) {
    std::ostringstream field_name;
    field_name << v_shape->name_hint << '[' << k << ']';
    Bind_(buffer->shape[k],
          cast(buffer->shape[k].type(),
               Load::make(tvm_shape_type, v_shape,
                          IntImm::make(Int(32), k), const_true(1))),
          field_name.str(), true);
  }
  // strides field
  Var v_strides(arg_name + ".strides", Handle());
  def_handle_dtype_.Set(v_strides, make_const(tvm_shape_type, 0));
  init_nest_.emplace_back(LetStmt::make(
      v_strides, TVMArrayGet(Handle(), handle, intrinsic::kArrStrides),
      nop));
  if (buffer->strides.size() == 0) {
    std::ostringstream stride_err_msg;
    stride_err_msg << arg_name << ".strides:"
                   << " expected to be nullptr for contiguous array";
    init_nest_.emplace_back(AssertNull(v_strides, stride_err_msg.str()));
  } else {
    for (size_t k = 0; k < buffer->strides.size(); ++k) {
      std::ostringstream field_name;
      field_name << v_strides->name_hint << '[' << k << ']';
      Bind_(buffer->strides[k],
            cast(buffer->shape[k].type(),
                 Load::make(tvm_shape_type, v_strides,
                            IntImm::make(Int(32), k), const_true(1))),
            field_name.str(), true);
    }
  }
  // Byte_offset field.
  int data_bytes = GetVectorBytes(buffer->dtype);
  int64_t const_offset;
  if (arith::GetConst(buffer->elem_offset, &const_offset)) {
    Bind_(make_const(UInt(64), const_offset * data_bytes),
               TVMArrayGet(UInt(64), handle, intrinsic::kArrByteOffset),
          arg_name + ".byte_offset", true);
  } else {
    Bind_(buffer->elem_offset,
           cast(buffer->elem_offset.type(),
                (TVMArrayGet(UInt(64), handle, intrinsic::kArrByteOffset) /
                 make_const(UInt(64), data_bytes))),
          arg_name + ".elem_offset", true);
  }
  // device info.
  Bind_(device_type,
        TVMArrayGet(Int(32), handle, intrinsic::kArrDeviceType),
        arg_name + ".device_type", true);
  Bind_(device_id,
        TVMArrayGet(Int(32), handle, intrinsic::kArrDeviceId),
        arg_name + ".device_id", true);
}

}  // namespace ir
}  // namespace tvm
