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
 * \file ir_utils.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_TRANSFORMS_IR_UTILS_H_
#define TVM_TIR_TRANSFORMS_IR_UTILS_H_

#include <tvm/arith/int_set.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/with.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <limits>
#include <string>
#include <unordered_map>
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
template <typename T, typename F>
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
inline PrimExpr TVMStructGet(DataType dtype, Var handle, int index,
                             builtin::TVMStructFieldKind kind) {
  Array<PrimExpr> args = {handle, make_const(DataType::Int(32), index),
                          make_const(DataType::Int(32), static_cast<int>(kind))};
  return Call(dtype, builtin::tvm_struct_get(), args);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline PrimExpr AddressOffset(Var handle, DataType dtype, int offset) {
  return Call(DataType::Handle(), builtin::address_of(),
              {Load(dtype, handle, make_const(DataType::Int(32), offset * dtype.lanes()),
                    const_true(dtype.lanes()))});
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
    offset = Ramp(offset, make_const(offset.dtype(), 1), dtype.lanes());
  }
  return Call(DataType::Handle(), builtin::address_of(),
              {Load(dtype, handle, offset, const_true(dtype.lanes()))});
}

/*!
 * \brief Set value into struct.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \param value The value to be set.
 * \return the set stmt.
 */
inline Stmt TVMStructSet(Var handle, int index, builtin::TVMStructFieldKind kind, PrimExpr value) {
  Array<PrimExpr> args = {handle, make_const(DataType::Int(32), index),
                          make_const(DataType::Int(32), static_cast<int>(kind)), value};
  return Evaluate(Call(DataType::Int(32), builtin::tvm_struct_set(), args));
}

/*!
 * \brief Get the type that is passed around TVM PackedFunc API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline DataType APIType(DataType t) {
  if (t.is_handle()) return t;
  ICHECK_EQ(t.lanes(), 1) << "Cannot pass vector type through packed API.";
  if (t.is_uint() || t.is_int()) return DataType::Int(64);
  ICHECK(t.is_float());
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

/*!
 * \brief Create an int32 constant
 * \param index the value of the constant
 * \return the PrimExpr that represents the constant
 */
inline PrimExpr ConstInt32(size_t index) {
  ICHECK_LE(index, std::numeric_limits<int>::max());
  return make_const(DataType::Int(32), static_cast<int>(index));
}

/*!
 * \brief Allocate TVMValues on the stack
 * \param type type of allocation
 * \param num number of TVMValues to allocate
 * \return PrimExpr representing the TVMValue
 */
inline PrimExpr StackAlloca(std::string type, size_t num) {
  Array<PrimExpr> args = {StringImm(type), ConstInt32(num)};
  return Call(DataType::Handle(), builtin::tvm_stack_alloca(), args);
}

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
Stmt ConvertSSA(Stmt stmt);

/*!
 * \brief Return the storage scope associated with a buffer variable.
 * \param buffer_var The input buffer variable.
 * \return A string representing the storage scope of this buffer variable.
 */
String GetPtrStorageScope(Var buffer_var);

/*!
 * \brief Convert match buffer target buffer access indices to original one.
 * \param indices The indices of the target buffer
 * \return The indices of source buffer.
 */
Array<PrimExpr> ConvertIndices(const MatchBufferRegion& match_buffer,
                               const Array<PrimExpr>& indices);

/*!
 * \brief Convert match buffer target buffer region to original one.
 * \param region The sub-region of the target buffer
 * \return The region of source buffer.
 */
Region ConvertRegion(const MatchBufferRegion& match_buffer, const Region& region);

/*!
 * \brief Check if a given PrimFunc originated from a TE schedule.
 *
 * Internally this checks for the `from_legacy_te_schedule` attr of the PrimFunc.
 *
 * \param f PrimFunc to check
 * \return Whether or not the PrimFunc was created from a te schedule
 */
Bool IsFromLegacyTESchedule(PrimFunc f);

/*!
 * \brief Helper to solve related variable's bound within conditional scope.
 * \param condition The condition expression to solve.
 * \param dom_map Global domain map of related vars.
 */
Map<Var, Range> GetVarBoundsFromCondition(
    const PrimExpr& condition, const std::unordered_map<const VarNode*, arith::IntSet>& dom_map);

/*!
 *\brief Context helper to update domain map within conditional scope.
 *
 * Assume the condition is `0 <= i && i < 9` and global domain of i is [0, 20], thus `bounds[i]` is
 *[0, 8]. Then `With<ConditionalBoundsContext> ctx(&dom_map, bounds, true)` step into scope where
 *dom_map[i] is [0, 8] and `With<ConditionalBoundsContext> ctx(&dom_map, bounds, false)` step into
 *scope where dom_map[i] is [9, 20]
 */
class ConditionalBoundsContext {
 private:
  friend class With<ConditionalBoundsContext>;
  /*!
   * \brief Construct a condition bounds context.
   * \param dom_map The global domain map to be updated.
   * \param true_branch_bounds The condition bounds map.
   * \param is_true_branch Whether step into the branch where condition bounds holds.
   */
  ConditionalBoundsContext(std::unordered_map<const VarNode*, arith::IntSet>* dom_map,
                           const Map<Var, Range>& true_branch_bounds, bool is_true_branch);
  void EnterWithScope();
  void ExitWithScope();

  /*! \brief global domain map to updated */
  std::unordered_map<const VarNode*, arith::IntSet>* dom_map_;
  /*! \brief var bounds on true branch */
  const Map<Var, Range>& true_branch_bounds_;
  /*! \brief whether is on true branch */
  bool is_true_branch_;
  /*! \brief used to record and restore original var bounds */
  std::unordered_map<const VarNode*, arith::IntSet> origin_map_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_IR_UTILS_H_
