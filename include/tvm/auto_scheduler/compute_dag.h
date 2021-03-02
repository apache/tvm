/*r
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
 * \file tvm/auto_scheduler/compute_dag.h
 * \brief The auto-scheduler's computational graph and related program analyses.
 *
 * We convert a compute declaration described by `tvm.compute` (could be a single operator or a
 * subgraph) to a ComputeDAG. It keeps the input/output tensors, all operations in the DAG, and
 * some static analysis results for the DAG (e.g. the total float operation count, consumer/producer
 * relations of operations, whether an operation stage should be tiled/compute inlined ...).
 * These analyses can help the search policy to make decisions during the search.
 * ComputeDAG is also responsible for the interaction between auto-scheduler's `LoopState` and
 * TVM schedule (e.g. applying the `LoopState` transform steps to a TVM schedule, providing
 * `LoopState` with extra information got from TVM schedule ...).
 */

#ifndef TVM_AUTO_SCHEDULER_COMPUTE_DAG_H_
#define TVM_AUTO_SCHEDULER_COMPUTE_DAG_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/te/schedule.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace auto_scheduler {

/*! \brief Static analyzer for a ComputeDAG */
class AccessAnalyzerNode : public Object {
 public:
  template <class T>
  using OperationMap = std::unordered_map<te::Operation, T, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Map an operation to all operations it reads from.
   * For each operation pair, use a two-dimensional array for multiple multi-dimensional accesses
   * The inner vector represents the indices of multi-dimensional access.*/
  OperationMap<OperationMap<std::vector<std::vector<PrimExpr>>>> read_from;
  /*! \brief Map an operation to all operations it is read by.
   * For each operation pair, use a two-dimensional array for multiple multi-dimensional accesses
   * The inner vector represents the indices of multi-dimensional access.*/
  OperationMap<OperationMap<std::vector<std::vector<PrimExpr>>>> read_by;
  /*! \brief Store the number of common outer iterators for operation pairs that have
   * read-write relations. */
  OperationMap<OperationMap<int>> num_common_outer_iterators;
  /*! \brief Store whether the operation is an op with only simple access.
   *  (e.g., injective, broadcast and elementwise ops without reduction) */
  OperationMap<bool> is_simple_access;
  /*! \brief Store whether the operation is strictly inlineable
   * (e.g., injective, broadcast and elementwise without reduction, branch or expensive operations)
   */
  OperationMap<bool> is_strictly_inlineable;
  /*! \brief Store whether the operation needs multi-level tiling
   * (e.g., computation-intensive ops with data reuse opportunity like matmul, conv2d) */
  OperationMap<bool> needs_multi_level_tiling;
  /*! \brief Store whether the operation is an output operation */
  OperationMap<bool> is_output;
  /*! \brief Store the topological order of operations */
  Array<te::Operation> ops_topo_order;

  static constexpr const char* _type_key = "auto_scheduler.AccessAnalyzer";
  TVM_DECLARE_FINAL_OBJECT_INFO(AccessAnalyzerNode, Object);
};

/*!
 * \brief Managed reference to AccessAnalyzerNode.
 * \sa AccessAnalyzerNode
 */
class AccessAnalyzer : public ObjectRef {
 public:
  explicit AccessAnalyzer(const Array<te::Tensor>& tensors);

  /*!
   * \brief Return whether this operation is an op with simple access
   * (e.g., injective, broadcast and elementwise ops without reduction)
   * \param op The operation
   */
  TVM_DLL bool IsSimpleAccess(const te::Operation& op) const;

  /*!
   * \brief Return whether this operation is strictly inlineable
   * (e.g., injective, broadcast and elementwise without reduction, branch or expensive operations)
   * \param op The operation
   */
  TVM_DLL bool IsStrictlyInlineable(const te::Operation& op) const;

  /*!
   * \brief Return whether this operation needs multi-level tiling
   * (e.g., computation-intensive ops with data reuse opportunity like matmul, conv2d)
   * \param op The operation
   */
  TVM_DLL bool NeedsMultiLevelTiling(const te::Operation& op) const;

  /*!
   * \brief Return whether this operation is an output operation
   * \param op The operation
   */
  TVM_DLL bool IsOutput(const te::Operation& op) const;

  /*!
   * \brief Get all consumers of an operation
   * \param state The current loop state
   * \param op The operation
   * \return The set of consumers
   * \note This function propagates the relation for inlined ops
   */
  TVM_DLL std::unordered_set<te::Operation, ObjectHash, ObjectEqual> GetConsumers(
      const State& state, const te::Operation& op) const;

  /*!
   * \brief Get all producers of an operation
   * \param state The current loop state
   * \param op The operation
   * \return The set of producers
   * \note This function propagates the relation for inlined ops
   */
  TVM_DLL std::unordered_set<te::Operation, ObjectHash, ObjectEqual> GetProducers(
      const State& state, const te::Operation& op) const;

  /*!
   * \brief Get all direct producers of an operation
   * \param op The operation
   * \return The set of direct producers
   * \note This function DOES NOT propagate the relation for inlined ops
   */
  TVM_DLL std::unordered_set<te::Operation, ObjectHash, ObjectEqual> GetDirectProducers(
      const te::Operation& op) const;

  /*!
   * \brief Get the number of common outer iterators.
   * \param op The operation
   * \param target_op The target operation
   * \note This function propagates the relation for chains with multiple ops.
   */
  TVM_DLL int GetNumCommonOuterIterator(const te::Operation& op,
                                        const te::Operation& target_op) const;

  /*!
   * \brief Return whether two operations are elementwise-matched
   *  (e.g. conv2d and relu are elementwise-matched)
   * \note This function propagates the relation for chains with multiple ops.
   */
  TVM_DLL bool ElementWiseMatch(const te::Operation& op, const te::Operation& target_op) const;

  TVM_DEFINE_OBJECT_REF_METHODS(AccessAnalyzer, ObjectRef, AccessAnalyzerNode);
};

/*! \brief The auto-scheduler's computational graph and related program analyses. */
class ComputeDAGNode : public Object {
 public:
  /*!
   * \brief Input and output tensors.
   * This is used as the input of `tvm.lower` or `tvm.build`.
   */
  Array<te::Tensor> tensors;
  /*! \brief All used operations in topo order. */
  Array<te::Operation> ops;
  /*! \brief The number of float operations in this ComputeDAG. */
  double flop_ct;
  /*! \brief The initial state without any transform steps. */
  State init_state;
  /*! \brief The static read-write access analyzer. */
  AccessAnalyzer access_analyzer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensors", &tensors);
    v->Visit("ops", &ops);
    v->Visit("flop_ct", &flop_ct);
    v->Visit("init_state", &init_state);
    v->Visit("access_analyzer", &access_analyzer);
  }

  static constexpr const char* _type_key = "auto_scheduler.ComputeDAG";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeDAGNode, Object);
};

/*!
 * \brief Options for applying layout rewrite.
 * This is an optimization to rewrite the layout of input tensors according to the schedule we get.
 */
enum class LayoutRewriteOption : int {
  /*! \brief Do not perform layout rewrite. */
  NoRewrite = 0,
  /*! \brief Insert layout transformation stages for input placeholders in the compute DAG */
  InsertTransformStage = 1,
  /*!
   * \brief Do not insert layout transformation stages and assume the input placeholders
   * are pre-transformed.
   * \note The lowered function with this option does not accept the origial input shapes,
   * so this option must be used along with `AutoSchedulerLayoutRewrite` pass in Relay.
   */
  RewriteForPreTransformed = 2,
};

/*!
 * \brief Managed reference to ComputeDAGNode.
 * \sa ComputeDAGNode
 */
class ComputeDAG : public ObjectRef {
 public:
  /*! \brief Construct a DAG from a list of output tensors.
   * \param tensors `te::Tensor`s for a compute declaration.
   */
  TVM_DLL explicit ComputeDAG(Array<te::Tensor> tensors);

  /*! \brief Construct a DAG based on a schedule.
   * \param sch `te::Schedule`s for a compute declaration.
   */
  TVM_DLL explicit ComputeDAG(const te::Schedule& sch);

  /*!
   * \brief Rewrite the layout of placeholder specified by attr `layout_free_placeholders`
   * according to the loop nest derived with `transform_steps`.
   * \param transform_steps Transform steps of a state.
   * \param layout_rewrite Different options in layout rewrite.
   * \return The updated ComputeDAG after layout rewrite.
   */
  ComputeDAG RewriteLayout(Array<Step>* transform_steps, LayoutRewriteOption layout_rewrite) const;

  /*!
   * \brief Apply the history transform steps to get a TVM schedule.
   * \param transform_steps Transform steps of a state.
   * \param stages The list of stages after applying the steps.
   * Pass a valid pointer if this information needs to be used outside this function.
   * \param stage_to_axes The map that stores all axes for one stage.
   * Pass a valid pointer if this information needs to be used outside this function.
   * \param layout_rewrite Rewrite the layout of placeholders specified by
   * attr `layout_free_placeholders`.
   * \return A `te.schedule` and the an Array of `te.Tensor` to be used in `tvm.lower`
   * or `tvm.build`.
   */
  std::pair<te::Schedule, Array<te::Tensor>> ApplySteps(
      const Array<Step>& transform_steps, Array<te::Stage>* stages = nullptr,
      StageToAxesMap* stage_to_axes = nullptr,
      LayoutRewriteOption layout_rewrite = LayoutRewriteOption::NoRewrite) const;

  /*!
   * \brief Print transform steps as equivalent python schedule API.
   * This can be used for debugging.
   * \param transform_steps Transform steps of a state.
   * \return The Python schedule code.
   */
  String PrintStepsAsPython(const Array<Step>& transform_steps) const;

  /*!
   * \brief Print the compute DAG to a string. This is also used to generate the ComputeDAG hash.
   * \param simple_mode Simple mode will only include the op names and brief compute.
   * \return The ComputeDAG in a string.
   */
  String PrintDAG(bool simple_mode = false) const;

  /*!
   * \brief Fill the correct bound information for a given state by calling ir_pass::InferBound.
   * The states can lose complete bound information after some transform steps (e.g., compute_at).
   * We can call this function to infer and fill all the bound information.
   * This function calls TVM InferBound pass internally to get the bound.
   * The returned state of this function is guaranteed to have complete bound information.
   * \param state The input state.
   * \return The State with complete bound information
   */
  State InferBound(const State& state) const;

  /*!
   * \brief Fill the correct bound information for the given states by calling ir_pass::InferBound.
   * The states can lose complete bound information after some transform steps (e.g., compute_at).
   * We can call this function to infer and fill all the bound information.
   * This function calls TVM InferBound pass internally to get the bound.
   * The returned state of this function is guaranteed to have complete bound information.
   * \param states The input states.
   * \return The States with complete bound information.
   * \note The returned array will contains empty State, if there're infer bound failure on some
   * states.
   */
  Array<State> InferBound(const Array<State>& states) const;

  /*!
   * \brief Since some steps may change the ComputeDAG (e.g. CacheRead/CacheWrite), the initial
   * ComputeDAG may not be up-to-date. This function replays the given transform steps from the
   * initial state and returns an up-to-date ComputeDAG.
   * \param steps The steps to be replayed. Usually we'll filter out the unused steps to speed up
   * the replay process, since we only intend to get a ComputeDAG with the up-to-date op stage
   * structure.
   * \return The up-to-date ComputeDAG.
   */
  ComputeDAG ReplayAndGetDAG(const Array<Step>& steps) const;

  static constexpr const char* layout_free_placeholders_key = "layout_free_placeholders";

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);
};

/*!
 *  \brief Get the orginal shape from a rewritten layout string.
 *  \param rewritten_layout The layout after auto-scheduler's layout rewrite.
 *  \param axis_names Specifiy the names of axes.
 *  \return shape The original shape.
 */
Array<PrimExpr> GetShapeFromRewrittenLayout(String rewritten_layout, Array<String> axis_names);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_COMPUTE_DAG_H_
