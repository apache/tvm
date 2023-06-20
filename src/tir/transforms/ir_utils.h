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
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/with.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body);

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
  PrimExpr offset_expr = make_const(DataType::Int(32), offset * dtype.lanes());
  Buffer dummy_buf(handle, dtype, {offset_expr + 1}, {}, 0, handle->name_hint, 0, 0, kDefault);
  BufferLoad buf_load(dummy_buf, {offset_expr});

  return Call(DataType::Handle(), builtin::address_of(), {buf_load});
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

  Buffer dummy_buf(handle, dtype.element_of(), {offset + 1}, {}, 0, handle->name_hint, 0, 0,
                   kDefault);
  BufferLoad buf_load(dummy_buf, {offset});

  return Call(DataType::Handle(), builtin::address_of(), {buf_load});
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
  ICHECK(!t.is_void()) << "Cannot pass void type through packed API.";
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
 * \brief Get stride aware buffer allocation shape from buffer.
 * \param buffer The buffer object.
 * \return shape The shape considering buffer strides.
 */
Array<PrimExpr> GetBufferAllocationShape(const Buffer& buffer);

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
 * \brief Context helper to update domain map within conditional scope.
 * Assume the condition is `0 <= i && i < 9` and domain of i is [0, 20], Then
 * `With<ConditionalBoundsContext> ctx(condition, &relax_map, &hint_map, &constraints)`
 * step into scope where dom_map[i] is [0, 8]; and
 * `With<ConditionalBoundsContext> ctx(!condition, &relax_map, &hint_map, &constraints)`
 * step into scope where dom_map[i] is [9, 20]
 */
class ConditionalBoundsContext {
 private:
  friend class With<ConditionalBoundsContext>;
  /*!
   * \brief Construct a condition bounds context.
   * \param condition The condition holds on true branch.
   * \param relax_map The domain map for relaxed vars to update.
   * \param hint_map The domain map for free vars to update.
   * \param pending_conditions The stack of unresolved constraints.
   */
  ConditionalBoundsContext(const PrimExpr& condition,
                           std::unordered_map<const VarNode*, arith::IntSet>* relax_map,
                           std::unordered_map<const VarNode*, arith::IntSet>* hint_map,
                           std::vector<PrimExpr>* pending_constraints);
  void EnterWithScope();
  void ExitWithScope();

  /*! \brief Helper to solve related variable's bound within conditional scope.*/
  Optional<arith::IntConstraints> TrySolveCondition();

  /*! \brief the condition holds on true branch. */
  const PrimExpr& condition_;
  /*! \brief domain map for relaxed vars to update */
  std::unordered_map<const VarNode*, arith::IntSet>* relax_map_;
  /*! \brief domain map for free vars to update */
  std::unordered_map<const VarNode*, arith::IntSet>* hint_map_;
  /*! \brief unresolved condition stack */
  std::vector<PrimExpr>* pending_conditions_;
  /*! \brief used to record and restore original var bounds */
  std::unordered_map<const VarNode*, arith::IntSet> origin_map_;
  /*! \brief used to record unresolved conditions num. */
  size_t origin_pending_conditions_num_;
};

// Information of tensor core fragment.
struct FragmentInfo {
  // fragment shape
  int m, n, k;
  // fragment layout (row-major or column-major)
  std::string layout;
  // scope of the fragment (wmma.matrix_a, wmma.matrix_b, or wmma.accumulator)
  std::string scope;
  FragmentInfo() = default;
  FragmentInfo(int _m, int _n, int _k, const std::string& _layout, const std::string& _scope)
      : m(_m), n(_n), k(_k), layout(_layout), scope(_scope) {}

  int GetSize() const {
    if (scope == "wmma.matrix_a") {
      return m * k;
    } else if (scope == "wmma.matrix_b") {
      return n * k;
    } else if (scope == "wmma.accumulator") {
      return m * n;
    } else {
      ICHECK(0);
      throw;
    }
  }
};

/*!
 * \brief Extract information of tensor core fragment from the IR.
 * \param stmt The stmt to visit.
 * \return Map from buffer variables to the fragment info.
 */
std::unordered_map<const VarNode*, FragmentInfo> GetTensorCoreFragmentInfo(const Stmt& stmt);

// Return the queue id and the in-flight count associated with the given
// attr::async_wait_queue_scope annotation.
std::pair<PrimExpr, PrimExpr> GetAsyncWaitAttributes(const AttrStmtNode* op);

/*!
 * \brief Bind a subset of parameter tensors to constants, replacing them by AllocateConst nodes.
 * \param f The function to bind constants to.
 * \param constants Raw constant data. If the size of this array is N, the last N parameter tensors
 * will be removed from the signature and instead AllocateConst nodes will be introduced in the
 * function body.
 * \return The updated function.
 */
PrimFunc BindParams(PrimFunc f, const Array<runtime::NDArray>& constants);

/*! \brief The quad used by StorageAlign for (buffer_idx, axis, factor, offset) */
using StorageAlignTuple = Array<Integer>;
/*! \brief A list of StorageAlignTuple, used by StorageAlign */
using StorageAlignAnnotation = Array<StorageAlignTuple>;
/*!
 * \brief Collect storage alignment annotations for all buffer vars within body.
 * \param body The stmt to collect.
 * \return The result dict from buffer var to storage align annotations.
 */
std::unordered_map<Var, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual>
CollectStorageAlignAnnotation(const Stmt& body);
/*!
 * \brief Split string separated by "," to get wmma fragment dimension size.
 * \param  shape_str The string to split.
 * \param  scope The scope to match.
 * \return The result pair of fragment dimension size.
 */
std::pair<int32_t, int32_t> GetWmmaFragmentDimSize(const std::string& shape_str,
                                                   const std::string& scope);

/*! \brief Check if a PrimFunc is a host function
 *
 * \param func The function to be inspected
 *
 * \return True if the function is known to run on the host, false if
 * the function is known to run on the device.  If it cannot be
 * determined (e.g. a function without a tvm::attr::kTarget
 * attribute), returns std::nullopt.
 */
std::optional<bool> IsHostFunc(const PrimFunc& func);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_IR_UTILS_H_
