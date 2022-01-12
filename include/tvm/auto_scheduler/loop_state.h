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
 * \brief The definition of the "state" in the search.
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
 * schedule primitives. The steps are also used for the serialization of a state.
 *
 * The LoopState can be seen as a lightweight loop structure IR specifically for schedule search.
 * We don't use the existing TVM IR but to extend a new structure on it is because:
 * 1. We want fast incremental change to the loop structures. The search policy needs to get the
 * immediate loop structures update rather than after TVM lowering;
 * 2. We want serializable transform history for replay, backtracking, and mutation;
 * 3. We may create some macro schedule primitives that represent the combination of several
 * TVM schedule primitives.
 *
 * When the search is finished, we will lower the state to TVM IR with TVM's schedule primitives.
 * Since we share a lot of common objects during search, the transformation is implemented in
 * copy on write style. All objects are immutable, which is similar to TVM IR.
 */

#ifndef TVM_AUTO_SCHEDULER_LOOP_STATE_H_
#define TVM_AUTO_SCHEDULER_LOOP_STATE_H_

#include <dmlc/common.h>
#include <tvm/auto_scheduler/transform_step.h>

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

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
   * \param op The source operation
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
  struct IterKeyHash {
    std::size_t operator()(const IterKey& k) const {
      return ::dmlc::HashCombine(std::hash<int>()(k.first), std::hash<int>()(k.second));
    }
  };

  /*! \brief A Map to store the mapping of stage to its attached iterator. */
  std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
  /*! \brief A Map to store the mapping of iterator to the stages attached to it. */
  std::unordered_map<IterKey, std::vector<StageKey>, IterKeyHash> iter_to_attached_stages;

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
   * \param stage_id The index of the source stage of computed at.
   * \param target_stage_id The index of stage that this step will compute at to.
   * \param target_iter_id The index of target iterator in the target stage.
   */
  void SetComputeAtIter(int stage_id, int target_stage_id, int target_iter_id);

  /*!
   * \brief Delete the entry of a specific stage. This is a public wrapper of `DeleteStageEntry`.
   * \param stage_id The index of the stage to be deleted.
   */
  void DeleteStage(int stage_id);

  /*!
   * \brief Find the relations of original iterators in AttachMap, and update them with the new
   * iterators. Both `stage_to_attach_iter` and `iter_to_attached_stages` will be updated.
   * \param original_iters The original IterKey.
   * \param new_iters The new IterKey for replacing the old ones.
   */
  void UpdateIters(const std::vector<IterKey>& original_iters,
                   const std::vector<IterKey>& new_iters);

  /*!
   * \brief Traverse through `stage_to_attach_iter` and `iter_to_attached_stages` map, add offset
   * to stage indexes that are larger than the start_id. Used for steps that insert new stages to
   * ComputeDAG (e.g., CacheRead/CacheWrite step).
   * \param start_id The index threshold. This function only adds offset for stages
   * with indices larger then this threshold.
   * \param offset The index offset to be added to the stage index.
   * \return The updated AttachMap after applying stage index offset.
   */
  AttachMap ApplyStageIdOffset(int start_id, int offset = 1) const;

  TVM_DEFINE_OBJECT_REF_METHODS(AttachMap, ObjectRef, AttachMapNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachMapNode);

 private:
  /*!
   * \brief Delete the entry of a specific stage. This will remove the items related to this
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
  /*! \brief The up-to-date ComputeDAG of this state. The default value is an empty NullOpt,
   * meaning the dag of this state is the same as the original ComputeDAG in the SearchTask.
   * Otherwise, the stored value is the up-to-date ComputeDAG for this state, meaning some steps
   * (e.g., CacheReadStep/CacheWriteStep) have modified the ComputeDAG.
   */
  Optional<ObjectRef> current_compute_dag;
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
   * \brief Pretty-print the state to a human readable string.
   * \param delete_trivial_loop True for skipping the trivial loops.
   * (undefined or extent == 1, default set to True)
   * \return The human readable string.
   */
  String ToStr(bool delete_trivial_loop = true) const;

  /********** Step APIs working on a single stage **********/
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::bind`.
   * \param stage_id The index of the stage to be binded.
   * \param it The iterator to be binded.
   * \param thread_type The thread type.
   * \return The new iterator after binding.
   */
  TVM_DLL Iterator bind(int stage_id, const Iterator& it, IteratorAnnotation thread_type);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::parallel`.
   * \param stage_id The index of the stage to be paralleled.
   * \param it The iterator to be paralleled.
   * \return The new iterator after parallel.
   */
  TVM_DLL Iterator parallel(int stage_id, const Iterator& it);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::unroll`.
   * \param stage_id The index of the stage to be unrolled.
   * \param it The iterator to be unrolled.
   * \param max_unroll The max unroll limit. Iterator with extent larger than this limit will be
   * skipped.
   * \return The new iterator after unroll.
   */
  TVM_DLL Iterator unroll(int stage_id, const Iterator& it, int max_unroll = -1);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::vectorize`.
   * \param stage_id The index of the stage to be vectorized.
   * \param it The iterator to be vectorized.
   * \return The new iterator after vectorization.
   */
  TVM_DLL Iterator vectorize(int stage_id, const Iterator& it);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::fuse`.
   * \param stage_id The index of the stage to be fused.
   * \param iters The iterators to be fused.
   * \return The iterator result after fuse.
   * \note If the iterators to be fused have stages attached at them(by compute_at), the fused
   * result will become the new attach point.
   */
  TVM_DLL Iterator fuse(int stage_id, const Array<Iterator>& iters);
  /*!
   * \brief The schedule primitive corresponding to `te.Stage.pragma`.
   * \param stage_id The index of the stage to add pragma.
   * \param it The iterator to add pragma.
   * \param pragma_type The pragma string.
   */
  TVM_DLL void pragma(int stage_id, const Iterator& it, const String& pragma_type);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::reorder`.
   * \param stage_id The index of the stage to be reordered.
   * \param order The expected iterator order.
   */
  TVM_DLL void reorder(int stage_id, const Array<Iterator>& order);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::split`.
   * \param stage_id The index of the stage to be split.
   * \param it The iterator to be split.
   * \param lengths The multiple split factors. Can be None to be filled by search policy.
   * \param inner_to_outer Whether the factors go from inner to outer, or from outer to inner.
   * \return The new iterator after splitting.
   * \note If we do split on an iterator which has stages attached at it(by compute_at), the inner
   * most iterator of split results will become the new attach point.
   */
  TVM_DLL Array<Iterator> split(int stage_id, const Iterator& it,
                                const Array<Optional<Integer>>& lengths,
                                bool inner_to_outer = true);
  /*!
   * \brief The schedule primitive similar to split, but uses split factors from previous steps.
   * \param stage_id The index of the stage to be split.
   * \param it The iterator to be split.
   * \param src_step_id The index of the split step to be followed in the history.
   * \param n_split The number of split level.
   * \return The split new Iterators.
   */
  TVM_DLL Array<Iterator> follow_split(int stage_id, const Iterator& it, int src_step_id,
                                       int n_split);
  /*!
   * \brief The schedule primitive similar to split, but uses split factors from
   * fused previous steps.
   * \param stage_id The index of the stage to be split.
   * \param it The iterator to be split.
   * \param src_step_ids The indices of the split steps to be followed in the history.
   * \param level Use the length in this split level.
   * \param factor_or_nparts True to use `factor` for split from inner to outer,
      False to use `nparts` for split from outer to inner.
   * \return The split new Iterators.
   */
  TVM_DLL Array<Iterator> follow_fused_split(int stage_id, const Iterator& it,
                                             const Array<Integer>& src_step_ids, int level,
                                             bool factor_or_nparts);
  /*!
   * \brief The schedule primitive corresponding to `te.Stage.storage_align`.
   * \param stage_id The index of the stage to be aligned.
   * \param it The iterator to be aligned.
   * \param factor The factor in alignment specification.
   * \param offset The offset in the alignment specification.
   */
  TVM_DLL void storage_align(int stage_id, const Iterator& it, int factor, int offset);

  /********** Step APIs working on multiple stages **********/
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::compute_at`.
   * \param stage_id The index of the source stage of computed at.
   * \param target_stage_id The index of stage that this step will compute at to.
   * \param target_iter The indiex of the target iterator in the target stage.
   * \note After compute_at, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state if you need the complete bound information.
   */
  TVM_DLL void compute_at(int stage_id, int target_stage_id, const Iterator& target_iter);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::compute_inline`.
   * \param stage_id The index of the stage to be marked compute inlined.
   */
  TVM_DLL void compute_inline(int stage_id);
  /*!
   * \brief The schedule primitive corresponding to `te::Stage::compute_root`.
   * \param stage_id The index of the stage to be marked compute at root.
   * \note After compute_root, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state if you need the complete bound information.
   */
  TVM_DLL void compute_root(int stage_id);

  /********** Step APIs adding new stages **********/
  /*!
   * \brief The schedule primitive corresponding to `te::Schedule::cache_read`.
   * \param stage_id The index of the stage to be cache_read.
   * \param scope_name The scope name of the newly added stage.
   * \param reader_stage_ids The indices of reader stages.
   * \param dag The original ComputeDAG of this state.
   * \note Cache read step will add an extra stage to the original ComputeDAG (at the back of the
   * target stage), an up-to-date ComputeDAG is stored in State's `current_compute_dag`.
   */
  TVM_DLL int cache_read(int stage_id, const String& scope_name,
                         const Array<Integer>& reader_stage_ids, const ComputeDAG& dag);
  /*!
   * \brief The schedule primitive corresponding to `te::Schedule::cache_write`.
   * \param stage_id The index of the stage to be cache_write.
   * \param scope_name The scope name of the newly added stage.
   * \param dag The original ComputeDAG of this state.
   * \note Cache write step will add an extra stage to the original ComputeDAG (in the front of the
   * target stage), an up-to-date ComputeDAG is stored in State's `current_compute_dag`.
   * This step will cache write all output tensors of the target stage.
   */
  TVM_DLL int cache_write(int stage_id, const String& scope_name, const ComputeDAG& dag);
  /*!
   * \brief The schedule primitive corresponding to `te::Schedule::rfactor`.
   * \param stage_id The index of the iterator to be factored.
   * \param it The iterator to be factored.
   * \param factor_iter_id The position where the new iterator is placed.
   * \param dag The original ComputeDAG of this state.
   * \note Rfactor step will add an extra stage to the original ComputeDAG (in the front of the
   * target stage), an up-to-date ComputeDAG is stored in State's `current_compute_dag`.
   */
  TVM_DLL int rfactor(int stage_id, const Iterator& it, int factor_iter_id, const ComputeDAG& dag);

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

// Hash and equal function for State
namespace std {

/*!
 * \brief The equal_to function for auto_scheduler::State.
 * This function checks the equality by looking at the lowered string format of states.
 * If two states with different transform history have the same lowered string format,
 * they will be considered being equal.
 */
template <>
struct equal_to<::tvm::auto_scheduler::State> {
  bool operator()(const ::tvm::auto_scheduler::State& lhs,
                  const ::tvm::auto_scheduler::State& rhs) const {
    return lhs.ToStr() == rhs.ToStr();
  }
};

/*! \brief The hash function for auto_scheduler::State. */
template <>
struct hash<::tvm::auto_scheduler::State> {
  std::size_t operator()(const ::tvm::auto_scheduler::State& state) const {
    return tvm::runtime::ObjectHash()(state.ToStr());
  }
};

}  // namespace std

#endif  // TVM_AUTO_SCHEDULER_LOOP_STATE_H_
