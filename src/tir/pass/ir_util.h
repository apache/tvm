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
 * \file ir_util.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_PASS_IR_UTIL_H_
#define TVM_TIR_PASS_IR_UTIL_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/runtime/device_api.h>
#include <vector>

namespace tvm {
namespace tir {
/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body);

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<std::vector<Stmt> >& nest, Stmt body);

/*!
 * \brief update array with an unary function
 * \param arr array
 * \param fupdate an unary function
 * \tparam T type of array element
 * \tparam F type of the unary function
 * \return if update happens, return the new array, else return the
 *  original array
 */
template<typename T, typename F>
inline Array<T> UpdateArray(Array<T> arr, F fupdate) {
  std::vector<T> new_arr(arr.size());
  bool changed = false;
  for (size_t i = 0; i < arr.size(); ++i) {
    T old_elem = arr[i];
    T new_elem = fupdate(old_elem);
    if (!new_elem.same_as(old_elem)) changed = true;
    new_arr[i] = new_elem;
  }
  if (!changed) {
    return arr;
  } else {
    return Array<T>(new_arr);
  }
}

/*!
 * \brief Get construct from struct
 * \param dtype The data type.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \return the get expression.
 */
inline PrimExpr TVMStructGet(
    DataType dtype, Var handle, int index,
    intrinsic::TVMStructFieldKind kind) {
  Array<PrimExpr> args ={
    handle,
    make_const(DataType::Int(32), index),
    make_const(DataType::Int(32), static_cast<int>(kind))};
  return CallNode::make(dtype, intrinsic::tvm_struct_get, args, CallNode::PureIntrinsic);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline PrimExpr AddressOffset(Var handle, DataType dtype, int offset) {
  return CallNode::make(
      DataType::Handle(), intrinsic::tvm_address_of,
      {LoadNode::make(dtype, handle, make_const(DataType::Int(32), offset * dtype.lanes()),
                  const_true(dtype.lanes()))},
      CallNode::PureIntrinsic);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline PrimExpr AddressOffset(Var handle, DataType dtype, PrimExpr offset) {
  if (dtype.lanes() != 1) {
    offset = offset * make_const(offset.dtype(), dtype.lanes());
    offset = RampNode::make(offset, make_const(offset.dtype(), 1), dtype.lanes());
  }
  return CallNode::make(
      DataType::Handle(), intrinsic::tvm_address_of,
      {LoadNode::make(dtype, handle, offset,
                  const_true(dtype.lanes()))},
      CallNode::PureIntrinsic);
}

/*!
 * \brief Set value into struct.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \param value The value to be set.
 * \return the set stmt.
 */
inline Stmt TVMStructSet(
    Var handle, int index,
    intrinsic::TVMStructFieldKind kind, PrimExpr value) {
  Array<PrimExpr> args ={
    handle,
    make_const(DataType::Int(32), index),
    make_const(DataType::Int(32), static_cast<int>(kind)),
    value};
  return EvaluateNode::make(
      CallNode::make(DataType::Int(32), intrinsic::tvm_struct_set, args, CallNode::Intrinsic));
}

/*!
 * \brief Get the type that is passed around TVM PackedFunc API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline DataType APIType(DataType t) {
  if (t.is_handle()) return t;
  CHECK_EQ(t.lanes(), 1)
      << "Cannot pass vector type through packed API.";
  if (t.is_uint() || t.is_int()) return DataType::Int(64);
  CHECK(t.is_float());
  return DataType::Float(64);
}

/*!
 * \brief Rule to get allocation alignment requirement for a given const array.
 * \param type The type of allocation.
 * \param const_size The constant size of the array.
 * \return the alignment
 */
inline int GetTempAllocaAlignment(DataType type, int32_t const_size) {
  int align = runtime::kTempAllocaAlignment;
  if (const_size > 0) {
    int64_t const_s = static_cast<int64_t>(const_size) * type.bits() * type.lanes() / 8;
    while (align > const_s) {
      align = align / 2;
    }
  }
  return align;
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_PASS_IR_UTIL_H_
