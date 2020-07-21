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
 * \file auto_scheduler/loop_state.h
 * \brief The definition of the "state" in search.
 *
 * Each LoopState corresponds to a schedule for its ComputeDAG.
 * A LoopState consists of: 1. a current loop structure; 2. a list of transformation steps used to
 * construct the loop structure.
 * The loop structure keeps a preview of how the schedule will finally look like after lowering the
 * current state (e.g. number of iterators, the extent of each iterator, the compute_at locations
 * ...).
 * During the schedule search process, the loop structure can provide search policy with necessary
 * information on how to manipulate the current state.
 * The transform history is a sequence of `TransformStep` which will finally be mapped to TVM
 * schedule primitives. The steps can also be used for the serialization of a state.
 *
 * The LoopState can be seen as a lightweight loop structure IR specifically for schedule search.
 * We don't use the existing TVM IR but to extend a new structure on it is because:
 * 1. We want fast incremental change to the loop structures. The search policy needs to get the
 * immediate loop structures update rather than after TVM lowering;
 * 2. We want serializable transform history for replay, backtracking, and mutation;
 * 3. We may create some macro schedule primitives that represent the combination of several
 * TVM schedule primitives.
 *
 * When the search is complete, we will lower the state to TVM IR with TVM's schedule primitives.
 * Since we share a lot of common objects during search, the transformation is implemented in
 * copy on write style. All objects are immutable, which is similar to TVM IR.
 */

#ifndef TVM_AUTO_SCHEDULER_LOOP_STATE_H_
#define TVM_AUTO_SCHEDULER_LOOP_STATE_H_

#include <tvm/runtime/container.h>

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transform_step.h"

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;

class ComputeDAG;

/*! \brief The type of a stage. */
enum class StageKind : int {
  /*! \brief A placeholder stage. */
  kPlaceholder = 0,
  /*! \brief A compute stage. */
  kCompute = 1
};

/*! \brief The type of compute location. */
enum class ComputeAtKind : int {
  /*! \brief Compute at root. */
  kRoot = 0,
  /*! \brief Compute inlined. */
  kInlined = 1,
  /*! \brief Compute at some iterator. */
  kIter = 2,
};

/*! \brief Stage-level attributes. */
struct StageAttributes {
  /*! \brief The maximum steps for the pragma `auto_unroll_max_step`. */
  int auto_unroll_max_step;
  /*! \brief The storage offset for the schedule primitive `storage_align`. */
  int storage_offset;
};

/*!
 * \brief A op stage in the compute declaration.
 * Similar to te::Stage in `include/tvm/te/schedule.h`.
 */
class StageNode : public Object {
 public:
  /*! \brief The operator of this stage */
  te::Operation op;
  /*! \brief The iterators in this stage. */
  Array<Iterator> iters;
  /*! \brief The type of this stage. */
  StageKind op_type;
  /*! \brief The compute location of this stage. */
  ComputeAtKind compute_at;
  /*! \brief Other stage-level attributes. */
  StageAttributes attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("iters", &iters);
    v->Visit("op_type", &op_type);
    v->Visit("compute_at", &compute_at);
  }

  static constexpr const char* _type_key = "auto_scheduler.Stage";
  TVM_DECLARE_FINAL_OBJECT_INFO(StageNode, Object);
};

/*!
 * \brief Managed reference to StageNode.
 * \sa StageNode
 */
class Stage : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param op A `te::Operation`.
   */
  explicit Stage(te::Operation op);
  /*!
   * \brief The constructor.
   * \param op A `te::Operation`.
   * \param op_type The stage type of this op.
   * \param iters The iterators of this op.
   * \param compute_at The compute at type of this op.
   * \param attrs Other stage-level attributes.
   */
  Stage(te::Operation op, StageKind op_type, const Array<Iterator>& iters, ComputeAtKind compute_at,
        StageAttributes attrs);

  TVM_DEFINE_OBJECT_REF_METHODS(Stage, ObjectRef, StageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StageNode);
};

/*! \brief Use stage_id to represent a stage. */
using StageKey = int;
/*! \brief Use stage_id and iter_id to represent a iterator. */
using IterKey = std::pair<int, int>;

/*!
 * \brief stores the compute_at relation between stages
 * This stores a bi-directional mapping from stages and iter:
 * 1. Stage to its attached iterator
 * 2. Iterator to the stage attached to it
 * You can use AttachMapNode::stage_to_attach_iter and AttachMapNode::iter_to_attached_stages
 * to query the relations
 */
class AttachMapNode : public Object {
 public:
  /*! \brief A Map to store the mapping of stage to its attached iterator. */
  std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
  /*! \brief A Map to store the mapping of iterator to the stage attached to it. */
  std::unordered_map<IterKey, std::vector<StageKey>> iter_to_attached_stages;

  static constexpr const char* _type_key = "auto_scheduler.AttachMap";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachMapNode, Object);
};

/*!
 * \brief Managed reference to AttachMapNode.
 * \sa AttachMapNode
 */
class AttachMap : public ObjectRef {
 public:
  /*!
   * \brief Process the stage/iterator mapping after compute at.
   * \param stage_id The index of the stage to be compute at.
   * \param target_stage_id The index of stage that this step will compute at to.
   * \param target_iter_id The index of iterator in target stage that this step will compute at to.
   */
  void SetComputeAtIter(int stage_id, int target_stage_id, int target_iter_id);
  /*!
   * \brief This is a public wrapper of `DeleteStageEntry`. To delete the entry of a specific stage.
   * \param stage_id The index of the stage to be compute at.
   */
  void DeleteStage(int stage_id);
  /*!
   * \brief Find the relations of original iterators in AttachMap, and update them with the new
   * iterators. Both `stage_to_attach_iter` and `iter_to_attached_stages` will be updated.
   * \param original_iters The original IterKey.
   * \param new_iters The new IterKey to update.
   */
  void UpdateIters(const std::vector<IterKey>& original_iters,
                   const std::vector<IterKey>& new_iters);

  TVM_DEFINE_OBJECT_REF_METHODS(AttachMap, ObjectRef, AttachMapNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachMapNode);

 private:
  /*!
   * \brief To delete the entry of a specific stage. This will remove the items related to this
   * stage in both `stage_to_attach_iter` and `iter_to_attached_stages` map.
   * \param pnode A mutable pointer to AttachMapNode.
   * \param stage_id The index of stage that will be removed from the map.
   */
  static void DeleteStageEntry(AttachMapNode* pnode, int stage_id);
};

/*!
 * \brief A state in the search process.
 * It consists of the current loop structure and a list of transformation steps used to construct
 * it.
 * Each State corresponds to a specific schedule for its ComputeDAG.
 */
class StateNode : public Object {
 public:
  /*! \brief Current stages and loop structures. */
  Array<Stage> stages;
  /*! \brief History transformation steps. */
  Array<Step> transform_steps;
  /*!
   * \brief The attach relations of stages and iterators. This is used to track the compute at
   * operation.
   */
  AttachMap attach_map;
  /*!
   * \brief Indicate whether this state has unfilled tile sizes. A concrete state means that all
   * tile sizes of the state is filled. Only concrete state can be apply to TVM schedule.
   */
  bool concrete;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("stages", &stages);
    v->Visit("transform_steps", &transform_steps);
    v->Visit("concrete", &concrete);
  }

  static constexpr const char* _type_key = "auto_scheduler.State";
  TVM_DECLARE_FINAL_OBJECT_INFO(StateNode, Object);

 private:
  /*!
   * \brief The up-to-date ComputeDAG of this state, used for some steps that may change the
   * stage structure of the ComputeDAG (e.g. CacheReadStep/CacheWriteStep which Will be added
   * later).
   * The default value is an empty ObjectRef. (means no modification to the original DAG)
   */
  ObjectRef current_compute_dag;
};

/*!
 * \brief Managed reference to StateNode.
 * \sa StateNode
 */
class State : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param ops `te::Operation`s for a compute declaration.
   */
  explicit State(const Array<te::Operation>& ops);

  /*!
   * \brief Print the state to a human readable string.
   * \param delete_trivial_loop True for skipping the trivial loops.
   * (undefined or extent == 1, default set to True)
   * \return The human readable state structure.
   */
  String ToStr(bool delete_trivial_loop = true) const;

  /*!
   * \brief General call step functions with a runtime dynamic dispatcher. This will re-apply all
   * the transform steps from the initial state.
   * \param dag The original ComputeDAG of this state.
   * \note The input `dag` is different from the class member `current_compute_dag`.
   * This function takes the initial ComputeDAG as input to replay all the history. While the
   * `current_compute_dag` is used to track the current stage status, for some transform step may
   * change the op stage structure.
   */
  void ApplySteps(const ComputeDAG& dag);

  /********** Step APIs working on single stage **********/

  /*!
   * \brief Schedule primitive corresponds to te.bind.
   * \param stage_id The index of the stage to be binded.
   * \param it The iterator to be binded.
   * \param thread_type The thread type to be binded. We dirctly use the IteratorAnnotation as
   * this input.
   * \return The iterator result after binded.
   */
  Iterator bind(int stage_id, const Iterator& it, IteratorAnnotation thread_type);
  /*!
   * \brief Schedule primitive corresponds to te.parallel.
   * \param stage_id The index of the stage to be paralleled.
   * \param it The iterator to be paralleled.
   * \return The iterator result after parallel.
   */
  Iterator parallel(int stage_id, const Iterator& it);
  /*!
   * \brief Schedule primitive corresponds to te.unroll.
   * \param stage_id The index of the stage to be unrolled.
   * \param it The iterator to be unrolled.
   * \param max_unroll The max unroll limit. Iterator with extent larger than this limit will be
   * skipped.
   * \return The iterator result after unrolled.
   */
  Iterator unroll(int stage_id, const Iterator& it, int max_unroll = -1);
  /*!
   * \brief Schedule primitive corresponds to te.vectorize.
   * \param stage_id The index of the stage to be vectorized.
   * \param it The iterator to be vectorized.
   * \return The iterator result after vectorize.
   */
  Iterator vectorize(int stage_id, const Iterator& it);
  /*!
   * \brief Schedule primitive corresponds to te.fuse.
   * \param stage_id The index of the stage to be fused.
   * \param iters The iterators to be fused.
   * \return The iterator result after fuse.
   * \note If the iterators to be fused have stages attached at them(by compute_at), the fused
   * result will become the new attach point.
   */
  Iterator fuse(int stage_id, const Array<Iterator>& iters);
  /*!
   * \brief Schedule primitive corresponds to te.reorder.
   * \param stage_id The index of the stage to be reordered.
   * \param order The expected iterator order.
   */
  void reorder(int stage_id, const Array<Iterator>& order);
  /*!
   * \brief Schedule primitive corresponds to te.split.
   * \param stage_id The index of the stage to be split.
   * \param it The iterator to be split.
   * \param lengths The multiple split factors. Can be None to be filled by search policy.
   * \param inner_to_outer Whether the factor go from inner to outer, or from outer to inner.
   * \return The iterator results after split.
   * \note If we do split on an iterator which has stages attached at it(by compute_at), the inner
   * most iterator of split results will become the new attach point.
   */
  Array<Iterator> split(int stage_id, const Iterator& it, const Array<Optional<Integer>>& lengths,
                        bool inner_to_outer = true);

  /********** Step APIs working on multiple stages **********/

  /*!
   * \brief Schedule primitive corresponds to te.compute_at.
   * \param stage_id The index of the stage to be reordered.
   * \param target_stage_id The index of stage that this step will compute at to.
   * \param target_iter The iterator in target stage that this step will compute at to.
   * \note After compute_at, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state to get the complete bound information.
   */
  void compute_at(int stage_id, int target_stage_id, const Iterator& target_iter);
  /*!
   * \brief Schedule primitive corresponds to te.compute_inline.
   * \param stage_id The index of the stage to be reordered.
   */
  void compute_inline(int stage_id);
  /*!
   * \brief Schedule primitive corresponds to te.compute_root.
   * \param stage_id The index of the stage to be reordered.
   * \note After compute_root, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state to get the complete bound information.
   */
  void compute_root(int stage_id);

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

// Hash and equal function for State
namespace std {

/*! \brief The hash function for auto_scheduler::State. */
template <>
struct hash<::tvm::auto_scheduler::State> {
  std::size_t operator()(const ::tvm::auto_scheduler::State& state) const {
    return tvm::runtime::ObjectHash()(state.ToStr());
  }
};

/*!
 * \brief The equal_to function for auto_scheduler::State.
 * We use the schedule result(its string format) of a state to check if two states are `euqal`.
 * Equal States: 1. the transform steps are totally the same; 2. even with different steps, two
 * states may still result in a same schedule. e.g. To split a axis with extent 512 to 3 parts
 * [8, 16, 4]. We can split from inner to outter by factors [16, 4], while we can get a same result
 * to split from outter to inner by factors [8, 16])
 */
template <>
struct equal_to<::tvm::auto_scheduler::State> {
  bool operator()(const ::tvm::auto_scheduler::State& lhs,
                  const ::tvm::auto_scheduler::State& rhs) const {
    return lhs.ToStr() == rhs.ToStr();
  }
};

}  // namespace std

#endif  // TVM_AUTO_SCHEDULER_LOOP_STATE_H_
