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
 * \file tvm/tirx/op/gpu.h
 * \brief Gpu operations for TIRx.
 */
#ifndef TVM_TIRX_OP_GPU_H_
#define TVM_TIRX_OP_GPU_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm::tirx {

/*!
 * \brief Return from a GPU thread without returning a function value.
 */
TVM_DLL const Op& thread_return_op();

/*!
 * \brief Thread-set filter predicate. Used as the condition of an IfThenElse
 * to narrow the active thread set A for the then-branch. Two forms:
 *   filter(var, lo, hi)   -- range form, true iff var in [lo, hi)
 *   filter(var, cond)     -- predicate form (e.g. var == k); true iff cond
 * `var` must be a ScopeIdDef-declared Var at parse time (Verifier Rule 2).
 */
TVM_DLL const Op& filter_op();

/*!
 * \brief Analysis-only active-thread selector.
 *
 * ``selector(var, pred)`` denotes the unique value of ``var`` in the current
 * active domain for which ``pred`` is true. It is used only inside
 * ExecContext/DispatchContext metadata, for predicates such as
 * ``ptx.elect_sync()`` whose selected lane cannot be inferred structurally.
 */
TVM_DLL const Op& selector_op();

/*!
 * \brief Mark a condition to be thread invariant.
 *
 * Arguments, in order:
 * - args[0]: cond, The condition.
 */
TVM_DLL const Op& tvm_thread_invariant_op();

/*!
 * \brief Synchronize accesses in a storage scope.
 *
 * Arguments, in order:
 * - args[0]: storage_scope, The storage scope.
 */
TVM_DLL const Op& tvm_storage_sync_op();

/*!
 * \brief Read the value from a selected lane in the warp.
 *
 * Arguments, in order:
 * - args[0]: mask, The mask.
 * - args[1]: value, The value to use.
 * - args[2]: warp_id, The warp identifier.
 * - args[3]: width, The width.
 * - args[4]: warp_size, The number of threads per warp.
 */
TVM_DLL const Op& tvm_warp_shuffle_op();

/*!
 * \brief Read the value from a lower lane in the warp.
 *
 * Arguments, in order:
 * - args[0]: mask, The mask.
 * - args[1]: value, The value to use.
 * - args[2]: offset, The offset.
 * - args[3]: width, The width.
 * - args[4]: warp_size, The number of threads per warp.
 */
TVM_DLL const Op& tvm_warp_shuffle_up_op();

/*!
 * \brief Read the value from a higher lane in the warp.
 *
 * Arguments, in order:
 * - args[0]: mask, The mask.
 * - args[1]: value, The value to use.
 * - args[2]: offset, The offset.
 * - args[3]: width, The width.
 * - args[4]: warp_size, The number of threads per warp.
 */
TVM_DLL const Op& tvm_warp_shuffle_down_op();

/*!
 * \brief Read the value from the lane selected by an XOR mask.
 *
 * Arguments, in order:
 * - args[0]: mask, The mask.
 * - args[1]: value, The value to use.
 * - args[2]: lane_mask, The lane mask.
 * - args[3]: width, The width.
 * - args[4]: warp_size, The number of threads per warp.
 */
TVM_DLL const Op& tvm_warp_shuffle_xor_op();

/*!
 * \brief Return the bit mask of active lanes in the warp.
 */
TVM_DLL const Op& tvm_warp_activemask_op();

/*!
 * \brief Cross-thread reduction with an explicit typed combiner and identities.
 *
 * void tvm_thread_allreduce(LambdaExpr combine, Expr identity, Expr values,
 *                           PrimExpr predicate, Expr destinations, Expr thread_axes);
 *
 * For N values, combine binds lhs[0:N] followed by rhs[0:N] and returns an
 * N-element Tuple, or a scalar when N is one. Identity, values, destinations
 * and thread_axes may each be a scalar or an explicit Tuple of fields.
 * Each value, identity, pair of parameters and result have
 * the same primitive type. Inactive inputs are replaced by their identities.
 * Destinations are N tensor loads at index zero of one-element result temporaries
 * (optionally cast for boolean storage). Each result temporary must be accessed
 * only at index zero. Thread axes are reduction thread variables or zero for
 * simplified unit axes.
 * Other thread indices remain fixed. The operation writes the reduced values
 * to the destination tensors and returns void.
 */
TVM_DLL const Op& tvm_thread_allreduce_op();

/*!
 * \brief Dot product of two int8x4 vectors and add an optional accumulator.
 *
 * Arguments, in order:
 * - args[0]: vec1, The first input vector.
 * - args[1]: vec2, The second input vector.
 * - args[2]: acc, The accumulator.
 */
TVM_DLL const Op& dp4a_op();

inline ffi::Array<Expr> GetAllreduceFields(const Expr& value) {
  if (const auto* tuple = value.as<tvm::TupleNode>()) return tuple->fields;
  return {value};
}

/*!
 * \brief Return from a GPU thread without returning a function value.
 *
 * \param span The location of this operation in the source.
 * \return The thread return expression.
 */
TVM_DLL PrimExpr thread_return(Span span = Span());

}  // namespace tvm::tirx

#endif  // TVM_TIRX_OP_GPU_H_
