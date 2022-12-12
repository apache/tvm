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
 * \file control_flow_graph.h
 * \brief Utility for extracting and interacting with buffer touch points
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/container/array.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TVM_TIR_ANALYSIS_CONTROL_FLOW_GRAPH_H_
#define TVM_TIR_ANALYSIS_CONTROL_FLOW_GRAPH_H_

namespace tvm {
namespace tir {

/*! \brief Represents an interaction with a buffer */
struct BufferTouch {
  enum class AccessType {
    /*! \brief Buffer access occurs in BufferLoad */
    Read,

    /*! \brief Buffer access occurs in BufferStore */
    Write,

    /*! \brief Buffer access occurs in tir::builtin::assume() */
    Assume,
  };

  BufferTouch(Buffer buffer, PrimExpr predicate, PrimExpr value)
      : buffer(buffer),
        predicate(predicate),
        value(value),
        loop_var_expressions({}),
        touch_type(AccessType::Assume) {}

  BufferTouch(Buffer buffer, PrimExpr predicate, PrimExpr value,
              std::vector<std::pair<Var, PrimExpr>> loop_var_expressions, AccessType touch_type)
      : buffer(buffer),
        predicate(predicate),
        value(value),
        loop_var_expressions(loop_var_expressions),
        touch_type(touch_type) {}

  /*! \brief The buffer being touched */
  Buffer buffer;

  /*! \brief A predicate that is true when this touch applies
   *
   * May be in terms of axis variables to indicate touches that impact
   * only a portion of a buffer.
   */
  PrimExpr predicate;

  /*! \brief The value in this buffer after the touch
   *
   * May be in terms of axis variables to indicate a known
   * non-constant value.  May be in terms of a BufferLoad to indicate
   * an unknown value.
   */
  PrimExpr value;

  /*! \brief Active loops during the buffer touch
   *
   * The vector contains one entry for each loop that contains the
   * buffer touch.  The `Var` item in each entry is the loop variable
   * itself.  The `PrimExpr` item is an expression for the loop
   * variable in terms of the buffer axis variables in
   * `ControlFlowGraph::axis_var_lookup_`.
   *
   * Used to construct boolean expressions indicating whether the loop
   * iteration that performs this touch has been reached.
   */
  std::vector<std::pair<Var, PrimExpr>> loop_var_expressions;

  /*! \brief How the buffer was interacted with
   *
   * When used as a constraint (e.g. in BufferState), should use
   * Assume.
   */
  AccessType touch_type{AccessType::Assume};

  /*! \brief Generate a boolean expression that is true for indices
   *  accessed by this touch during this iteration or a previous
   *  loop iteration.
   *
   * Used during forward propagation, to track known values that were
   * written in the current loop iteration, or in a preceding loop
   * iteration.
   */
  PrimExpr BeforeLoopIteration() const;

  /*! \brief Generate a boolean expression that is true for indices
   *  accessed by this touch during this loop iteration.
   *
   * Used during speculative no-op insertion checks, to specify which
   * indices must be later overwritten for a store to have no impact
   * on final results.
   */
  PrimExpr AtLoopIteration() const;

  /*! \brief Generate a boolean expression that is true for indices
   *  accessed by this touch during this loop iteration or a
   *  subsequent loop iteration.
   *
   * Used during backward propagation, to track indices that that are
   * overwritten in the current loop iteration or in a later loop
   * iteration.
   */
  PrimExpr AfterLoopIteration() const;

  /* \brief Checks if this touch affects a subset of indices of another
   *
   * Returns true if the indices accessed by this touch are a subset
   * of predicate is true can be proven to be a subset of the other
   * subset.  Returns false if it cannot be proven to be a subset of
   * ther other subset.
   */
  bool IsSubsetOf(const BufferTouch& other, arith::Analyzer* analyzer) const;

  /* \brief Checks if this touch affects distinct indices from another
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsDistinctFrom(const BufferTouch& other, arith::Analyzer* analyzer) const;

  /* \brief Checks if this touch affects distinct indices from another
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsEquivalentTo(const BufferTouch& other, arith::Analyzer* analyzer) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);
};

/*! \brief Represents the known state of buffers at a specific point */
class BufferState {
 public:
  /*! Default constructor
   *
   * Initialize the buffer state with no known information.
   */
  BufferState() {}

  /*! \brief Replace BufferLoad instances with known values
   *
   * \param expr The expression to be updated.
   *
   * \param axis_var_lookup A map from buffer to the variables
   * representing positions along the buffer's axes.
   *
   * \param analyzer The analyzer to use when validating a
   * constraint's predicate.
   *
   * \returns The modified expression.  If no substitutions are made,
   * the original expression is returned.
   */
  PrimExpr SubstituteKnownBufferValues(PrimExpr expr,
                                       const Map<Buffer, Array<Var>>& axis_var_lookup,
                                       arith::Analyzer* analyzer) const;

  /*! \brief Apply a condition to all known constraints
   *
   * For example, when propagating pre-loop constraints into the body
   * of a loop, add a condition that the loop iterator is zero.
   *
   * \param condition The condition to apply
   */
  void AddCondition(const PrimExpr& condition);

  /*! \brief Perform a variable substitution for all constraints
   *
   * For example, when propagating constraints from the end of a loop
   * to the beginning, replace `i` with `i-1`.
   *
   * \param var_remap The variable remapping to apply.
   */
  void Substitute(const Map<Var, PrimExpr>& var_remap, arith::Analyzer* analyzer);

  /*! \brief Simplify the predicate of all constraints
   *
   * \param analyzer The analyzer with which to simplify
   */
  void Simplify(arith::Analyzer* analyzer);

  /*! \brief Update the known buffer values based on buffer touches
   *
   * For any Write or Assume touches, update the known values.  For
   * any Read touches, ignore.  Used to determine known values at the
   * end of a control flow block, given the known values at the start.
   *
   * \param axis_var_lookup A map from buffer to the variables
   * representing positions along the buffer's axes.
   *
   * \param touch_points The buffer touch points to apply
   *
   * \param analyzer The analyzer to use for simplifications
   */
  void ApplyTouches(const Map<Buffer, Array<Var>>& axis_var_lookup,
                    const std::vector<BufferTouch>& touch_points, arith::Analyzer* analyzer);

  /*! \brief Update unused buffer locations based on buffer touches
   *
   * For any Write, mark the written-to indices as unused.  (That is,
   * immediately prior to assigning `buf[i] = expr`, the value stored
   * at `buf[i]` is irrelevant.)  For any Read, mark the read-from
   * indices as used.  This method is used to determine unused buffer
   * indices at the start of a control flow block, given the unused
   * buffer indices values at the end.
   *
   * \param axis_var_lookup A map from buffer to the variables
   * representing positions along the buffer's axes.
   *
   * \param touch_points The buffer touch points to apply
   *
   * \param analyzer The analyzer to use for simplifications
   */
  void BackpropUnusedIndices(const Map<Buffer, Array<Var>>& axis_var_lookup,
                             const std::vector<BufferTouch>& touch_points,
                             arith::Analyzer* analyzer);

  /*! \brief Remove free parameters from the constraints
   *
   * \param free_predicate_parameters
   *
   * \param analyzer The analyzer with which to simplify after removal
   */
  void RemoveFreeParameters(const Map<Var, Range>& free_predicate_parameters,
                            arith::Analyzer* analyzer);

  /*! \brief Check if two buffer states are equivalent
   *
   * \param other
   *
   * \param analyzer The analyzer used to check equality of PrimExpr
   *
   * \return True if the two states are provably equivalent, false otherwise.
   */
  bool IsEquivalentTo(const BufferState& other, arith::Analyzer* analyzer) const;

  /* \brief Add known values provided by another state
   *
   * \param other The state with which to merge constraints
   *
   * \param analyzer The analyzer with which to simplify the result
   */
  void Union(const BufferState& other, arith::Analyzer* analyzer);

  /* \brief Remove all known values not consistent with another state
   *
   * \param other The state with which to merge constraints
   *
   * \param analyzer The analyzer with which to simplify the result
   */
  void Intersection(const BufferState& other, arith::Analyzer* analyzer);

  friend std::ostream& operator<<(std::ostream& os, const BufferState&);

 private:
  friend class ControlFlowGraph;
  /*! \brief The known constraints */
  std::vector<BufferTouch> constraints_;
};

/*! \brief Represents the flow of control through a `tir::Stmt`
 *
 * This class contains an internal representation of the possible
 * control flow that may occur during execution of a `tir::Stmt`.  It
 * consists of a collection of ControlFlowBlock objects, each of which
 * represents a subset of operations performed during execution, along
 * with edges that represent allowed transitions between
 * `ControlFlowBlock`.
 *
 * In addition, the following restrictions are used.
 *
 * 1. Each block may have at most two predecessors, and at most two
 *    successors.
 *
 * 2. Within each block, values stored in a buffer do not change.
 *    That is, encountering a `BufferStore` node requires creating a
 *    new block.
 *
 * For example, consider the following PrimFunc
 *
 * ```python
 * @T.prim_func
 * def func(T.Buffer[16, "float32"]):
 *     for i in T.serial(16):
 *         if i < 8:
 *              B[i] = i
 *         else:
 *              B[i] = i-8
 * ```
 *
 * The control flow graph would have eight control blocks.
 *
 * 1. function_entry, from the start of the function through the
 *    evaluation of the loop's extent.
 *
 *    Predecessors: n/a
 *    Successors: loop_start
 *
 * 2. loop_start, after entering the body of the loop, through the
 *    evaluation of the conditional `i < 8`
 *
 *    Predecessors: function_entry, after_conditional
 *    Successors: then_clause_start, else_clause_start
 *
 * 3. then_clause_start, after entering the then_clause of `i < 8`,
 *    through evaluation of the value `i`.
 *
 *    Predecessors: loop_start
 *    Successors: then_clause_end
 *
 * 4. then_clause_end, after storing to `B[i]` prior to exiting the
 *    then_clause.
 *
 *    Predecessors: then_clause_start
 *    Successors: after_conditional
 *
 * 5. else_clause_start, after entering the else_clause of `i < 8`,
 *    through evaluation of the value `i-8`.
 *
 *    Predecessors: loop_start
 *    Successors: else_clause_end
 *
 * 6. else_clause_end, after storing to `B[i]` prior to exiting the
 *    else_clause.
 *
 *    Predecessors: else_clause_start
 *    Successors: after_conditional
 *
 * 7. after_conditional, after the end of the if/then/else, before the
 *    end of the loop body
 *
 *    Predecessors: then_clause_end, else_clause_end
 *    Successors: loop_start, after_loop
 *
 * 8. after_loop, after the loop
 *
 *    Predecessors: after_conditional
 *    Successors: n/a
 *
 *
 * By identifying `BufferStore` nodes whose value does not depend on
 * values stored in input buffers (e.g. initializing `buf[i] = 0.0`),
 * or whose values are provided using `builtin::assume()`
 * (e.g. `T.assume(buf[i] == 0.0)`), the value stored in a buffer at
 * those indices may be known for a given control block.  These known
 * values can then be propagated forward to successor blocks, to be
 * used in context-dependent simplifications.
 *
 * In addition to the allowed transitions between control-flow
 * blocks, each block also tracks the buffer touch points; which
 * indices are read from a buffer, which values are written to which
 * indices of a buffer, and assumptions are provided using
 * `builtin::assume()`; that occur during the control-flow block.
 *
 * Note: The current implementation only tracks the values of
 * buffers that are constrained to a specific value, and does not
 * track inequalities that may partially constrain buffer values.
 * That is, entering a scoped context with a data-dependent equality
 * condition (e.g. `if buf[i] == value`) is tracked, but entering a
 * scoped context with a data-dependent inequality condition
 * (e.g. `if buf[i] > value`) is not tracked.
 */
class ControlFlowGraph {
 public:
  /* \brief Extract the touch pattern from a TIR statement
   */
  explicit ControlFlowGraph(const Stmt& stmt, size_t max_revisits = 5);

  /* \brief Check if a write is overwritten without impacting final results
   *
   * \param store The store to be examined
   *
   * \param context The context in which the buffer store occurs, used
   * to identify the control-flow block in which the store occurs.  In
   * most cases, this will be the same object as the `store` itself.
   *
   * \param analyzer The analyzer to be used for simplifications
   *
   * \return True if the specified store can be proven to be
   * overwritten without contributing to any later statements.
   * Returns false otherwise.
   */
  bool IsOverwrittenWithoutEffect(const BufferStore& store, const Stmt& context) const;

  /* \brief Simplify the expression, assuming it occurs within the given context
   *
   * \param expr The expression to be simplified.  Does not need to
   * have occurred within the statement used to construct this
   * BufferTouchPattern.
   *
   * \param context The statement where this expression occurred, or
   * is to be inserted.  Must occur within the statement used to
   * construct this BufferTouchPattern.
   *
   * \param analyzer The analyzer to be used for simplifications
   *
   * \returns The simplified statement
   */
  PrimExpr SimplifyInContext(PrimExpr expr, const Stmt& context, arith::Analyzer* analyzer) const;

  /*! \brief Remove the specified BufferStore from the control-flow
   *  graph
   *
   * Removing the specified store, which may reflow known values.
   * This is necessary when simplifying sequential stores of the same
   * value.  Otherwise, the first could be removed as a no-op because
   * it is overwritten by the second, and the second could be removed
   * as a no-op because it is the same value as the first.
   *
   * \param store The store to remove
   */
  void RemoveStore(const tir::BufferStore& store);

  friend std::ostream& operator<<(std::ostream& os, const ControlFlowGraph& pattern);

 private:
  /*! \brief Return index variables representing locations within a
   *   buffer.
   *
   * For a given buffer, will always return the same set of variables.
   *
   * \param buf The buffer being accessed
   *
   * \param indices The indices at which the buffer is being accessed.
   * These are used to set the dtype of the buffer axis variables.
   *
   * \returns Variables representing a position along the buffer's axis.
   */
  Array<Var> GetIndexVariables(const Buffer& buf, const Array<PrimExpr>& indices);

  /*! \brief Return index variables representing locations within a
   *   buffer, if they have been generated before.
   *
   * For a given buffer, will always return the same set of variables.
   *
   * \param buf The buffer being accessed
   *
   * \returns Variables representing a position along the buffer's axis.
   */
  Optional<Array<Var>> GetIndexVariables(const Buffer& buf) const;

  /*! \brief Propagate known values from known BufferStore/assume
   *  subsequent control flow blocks
   *
   * \param flow_from If specified, re-flow only from that block.
   */
  void ForwardPropagateKnownValues(std::optional<size_t> flow_from = std::nullopt);

  /*! \brief Propagate overwritten/unused indices to preceding control
   *  flow blocks
   *
   * \param flow_from If specified, re-flow only from that block.
   */
  void BackwardPropagateUnusedValues(std::optional<size_t> flow_from = std::nullopt);

  struct ControlFlowEdge {
    /* \brief The source block of the control flow edge
     *
     * Lookup index into `control_flow_`
     */
    size_t index;

    /*! \brief Variable remaps
     *
     * e.g. Replacing loop iterator `i` with `i-1` when following an
     * edge from the end of a loop to the beginning of the loop.
     */
    Map<Var, PrimExpr> var_remap;

    /*! \brief Condition that must to true after following this edge
     *
     * This is applied after variable remapping.  For example, `i >
     * loop_min` when following the an edge from the end of a loop to
     * the beginning of the loop.
     */
    Optional<PrimExpr> post_condition;
  };
  friend std::ostream& operator<<(std::ostream& os, const ControlFlowEdge& edge);

  struct ControlFlowBlock {
    struct LoopEntry {
      Var loop_var;
      PrimExpr loop_min;
      PrimExpr loop_max;
      Range loop_range;
    };

    /*! \brief Loop iterators that are active during this block */
    std::vector<LoopEntry> active_loop_iterators;

    /*! \brief Loop-dependent Let bindings that may appear within the block */
    Map<Var, PrimExpr> let_bindings_using_loop;

    /*! \brief Predicate that must be true to have reached this block */
    PrimExpr scope_predicate{Bool(true)};

    /*! \brief All known values prior to executing the block */
    BufferState known_at_block_start;

    /*! \brief All known values after executing the block */
    BufferState known_at_block_end;

    /*! \brief Indices whose value at the start of the block is known to be unused */
    BufferState unused_at_block_start;

    /*! \brief Indices whose value at the end of the block is known to be unused */
    BufferState unused_at_block_end;

    /* \brief Buffer touches that occur within the block
     *
     * All buffer touches within a block can be treated as occurring
     * simultaneously.
     */
    std::vector<BufferTouch> touch_points;

    /* \brief The blocks that occur after this block
     *
     * Lookup index into `control_flow_`
     */
    std::vector<ControlFlowEdge> successors;

    /* \brief The blocks that occur before this block */
    std::vector<ControlFlowEdge> predecessors;

    /* \brief Construct a BufferTouch instance within this
     * ControlFlowBlock
     *
     * \param graph The mutable ControlFlowGraph that owns the buffer
     * touch.  Any free parameters used in the BufferTouch's predicate
     * will be tracked by the ControlFlowGraph.
     *
     * \param buf The Buffer being accessed
     *
     * \param indices The indices at which the buffer is accessed, in
     * terms of the loop variables.
     *
     * \param touch_type The type of touch being generated
     *
     * \param known_expr_value The value being written to the buffer
     *
     * \returns The newly generated BufferTouch
     */
    BufferTouch MakeBufferTouch(ControlFlowGraph* graph, const Buffer& buf,
                                const Array<PrimExpr>& indices, BufferTouch::AccessType touch_type,
                                PrimExpr known_value_expr) const;

    /* \brief Construct a BufferTouch instance as if it occurred in
     * this ControlFlowBlock
     *
     * Used when speculative checking if a BufferStore could be
     * inserted.
     *
     * \param buf The Buffer being accessed
     *
     * \param index_variables The variables representing location
     * within a buffer, with one variable for each axis of the buffer.
     *
     * \param indices The indices at which the buffer is accessed, in
     * terms of the loop variables.
     *
     * \param touch_type The type of touch being generated
     *
     * \param known_expr_value The value being written to the buffer
     *
     * \returns The newly generated BufferTouch, and a map specifying
     * all free parameters that may occur in the BufferTouch's
     * predicate.
     */
    std::pair<BufferTouch, Map<Var, Range>> MakeBufferTouch(const Buffer& buf,
                                                            Array<Var> index_variables,
                                                            Array<PrimExpr> indices,
                                                            BufferTouch::AccessType touch_type,
                                                            PrimExpr known_value_expr) const;
  };
  friend std::ostream& operator<<(std::ostream& os, const ControlFlowBlock& pattern);

  /* \brief The control flow that occurs within the analyzed statement */
  std::vector<ControlFlowBlock> control_flow_;

  /* \brief A lookup into control_flow_
   *
   * A map to look up the control flow block that contains the
   * statement.
   */
  std::unordered_map<const StmtNode*, size_t> control_flow_lookup_;

  /*! \brief A map from free parameters to their range
   *
   * A BufferStore/BufferLoad has indices in terms of loop iterators,
   * while the internal BufferTouch must have predicate in terms of
   * the buffer's axes.  While converting to the internal BufferTouch,
   * reduction axes show up as free parameters.  Tracking the range of
   * the free parameters allows them to be removed later, by requiring
   * a predicate to be true for all values of the free parameters.
   */
  Map<Var, Range> free_predicate_parameters_;

  /*! \brief Ranges of iterators found in the analyzed statement */
  Map<Var, Range> iterator_ranges_;

  /* \brief A map from buffer to the variables representing positions
   * along the buffer's axes.
   *
   * This is stored here, rather than as part of the BufferState or
   * BufferTouch, to ensure that all access of a buffer use the same
   * variables to represent the buffer's axes, reducing the amount of
   * variable substitution required.
   */
  Map<Buffer, Array<Var>> axis_var_lookup_;

  /* \brief Assumptions that do not depend on buffer values
   *
   * These may be collected as part of the handling of `builtin::assume()`, and do not depend on any
   * buffer.  Since TIR only allows mutable values as part of buffers, these assumptions may be used
   * anywhere the
   */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class ControlFlowGraphBuilder;

  /*! \brief The maximum number of revisits while flowing constraints */
  size_t max_revisits_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_ANALYSIS_CONTROL_FLOW_GRAPH_H_
