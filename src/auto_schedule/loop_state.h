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
 * \file auto_schedule/loop_state.h
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

#ifndef TVM_AUTO_SCHEDULE_LOOP_STATE_H_
#define TVM_AUTO_SCHEDULE_LOOP_STATE_H_

#include <tvm/runtime/container.h>

#include <functional>

#include "transform_step.h"

namespace tvm {
namespace auto_schedule {

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

/*! \brief The type of an iterator. */
enum class IteratorKind : int {
  /*! \brief Spatial iterator. */
  kSpatial = 0,
  /*! \brief Reduction iterator. */
  kReduction = 1,
  /*! \brief Fused spatial and reduction iterator. */
  kMixed = 2,
  /*! \brief Special iterator. (e.g. virtual root iterator) */
  kSpecial = 3
};

/*! \brief The type of an iterator's annotation. */
enum class IteratorAnnotation : int {
  /*! \brief This iterator has no annotation. */
  kNone = 0,
  /*! \brief This iterator has been unrolled. */
  kUnroll = 1,
  /*! \brief This iterator has been vectorized. */
  kVectorize = 2,
  /*! \brief This iterator has been paralleld. */
  kParallel = 3,
  /*! \brief This iterator has been bind to vthread. */
  kVThread = 4,
  /*! \brief This iterator has been bind to blockIdx.x. */
  kBlockX = 5,
  /*! \brief This iterator has been bind to threadIdx.x. */
  kThreadX = 6,
  /*! \brief This iterator has been bind to blockIdx.y. */
  kBlockY = 7,
  /*! \brief This iterator has been bind to threadIdx.y. */
  kThreadY = 8,
  /*! \brief This iterator has been mapped with a tensorize intrinsic. */
  kTensorized = 9
};

/*!
 * \brief A for loop iterator
 * Similar to tvm::IterVar in `include/tvm/tir/expr.h`
 */
class IteratorNode : public Object {
 public:
  /*! \brief The name of this iterator. */
  String name;
  /*! \brief The range of this iterator. */
  Range range;
  /*! \brief The iterator type of this iterator. */
  IteratorKind iter_kind;
  /*! \brief The annotation type of this iterator. */
  IteratorAnnotation annotation;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("range", &range);
  }

  static constexpr const char* _type_key = "auto_schedule.Iterator";
  TVM_DECLARE_FINAL_OBJECT_INFO(IteratorNode, Object);
};

/*!
 * \brief Managed reference to IteratorNode.
 * \sa IteratorNode
 */
class Iterator : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of this iterator.
   * \param range The range of this iterator.
   * \param iter_kind The iterator type of this iterator.
   * \param annotation The annotation type of this iterator.
   */
  Iterator(String name, Range range, IteratorKind iter_kind, IteratorAnnotation annotation);

  TVM_DEFINE_OBJECT_REF_METHODS(Iterator, ObjectRef, IteratorNode);
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
 * Similar to te::Stage in `include/schedule.h`.
 */
class StageNode : public Object {
 public:
  /*! \brief The operator of this stage */
  te::Operation op;
  /*! \brief The type of this stage. */
  StageKind op_type;
  /*! \brief The iterators in this stage. */
  Array<Iterator> iters;
  /*! \brief The compute location of this stage. */
  ComputeAtKind compute_at;
  /*! \brief Other stage-level attributes. */
  StageAttributes attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("iters", &iters);
  }

  static constexpr const char* _type_key = "auto_schedule.Stage";
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
   * \brief Indicate whether this state has unfilled tile sizes. A concrete state means that all
   * tile sizes of the state is filled. Only concrete state can be apply to TVM schedule.
   */
  bool concrete;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("stages", &stages);
    v->Visit("transform_steps", &transform_steps);
    v->Visit("concrete", &concrete);
  }

  static constexpr const char* _type_key = "auto_schedule.State";
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
   * \brief General do step functions with a runtime dynamic dispatcher. This will re-apply all the
   * transform steps with the initial state.
   * \param dag The original ComputeDAG of this state.
   * \note This is different from the class member `current_compute_dag`, for some transform step
   * may change the op stage structure of the ComputeDAG.
   */
  void DoSteps(const ComputeDAG& dag);

  /* Step APIs for State. */

  /*!
   * \brief Schedule primitive corresponds to te.reorder.
   * \param stage_id The index of the stage to be reordered.
   * \param order The expected iterator order.
   */
  void reorder(int stage_id, const Array<Iterator>& order);
  /*!
   * \brief Schedule primitive corresponds to te.split.
   * \param stage_id The index of the stage to be split.
   * \param it The iterator the be split.
   * \param lengths The multiple split factors. Can be None to be filled by search policy.
   * \param inner_to_outer Whether the factor go from inner to outer, or from outer to inner.
   * \return The iterator results after split.
   */
  Array<Iterator> split(int stage_id, const Iterator& it, const Array<Optional<Integer>>& lengths,
                        bool inner_to_outer = true);
  /*!
   * \brief Schedule primitive corresponds to te.fuse.
   * \param stage_id The index of the stage to be fused.
   * \param iters The iterators to be fused.
   * \return The iterator result after fuse.
   */
  Iterator fuse(int stage_id, const Array<Iterator>& iters);

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);

 private:
  /* Do transform steps
   * Note: The following functions only change loop state but do not change transform_history.
   * We separate these functions out, so you can call them for replay easily given history steps */

  /*!
   * \brief Apply reorder step to current state.
   * \param step A ReorderStep.
   */
  void DoReorderStep(const ReorderStep& step);
  /*!
   * \brief Apply split step to current state.
   * \param step A SplitStep.
   * \return The iterator results after split.
   */
  Array<Iterator> DoSplitStep(const SplitStep& step);
  /*!
   * \brief Apply fuse step to current state.
   * \param step A FuseStep.
   * \return The iterator result after fuse.
   */
  Iterator DoFuseStep(const FuseStep& step);

  /*!
   * \brief Common function for DoSplitStep and DoFollowSplitStep(Will be added later).
   * \param stage_id The index of the stage to be split.
   * \param iter_id The index of the iterator to be split.
   * \param lengths The multiple split factors.
   * \param inner_to_outer The split direction.
   * \return The iterator results after split.
   */
  Array<Iterator> DoSplitStepCommon(int stage_id, int iter_id,
                                    const Array<Optional<Integer>>& lengths, bool inner_to_outer);
};

}  // namespace auto_schedule
}  // namespace tvm

// Hash and equal function for State
namespace std {

/*! \brief The hash function for auto_schedule::State. */
template <>
struct hash<::tvm::auto_schedule::State> {
  std::size_t operator()(const ::tvm::auto_schedule::State& state) const {
    return tvm::runtime::ObjectHash()(state.ToStr());
  }
};

/*!
 * \brief The equal_to function for auto_schedule::State.
 * We use the schedule result(its string format) of a state to check if two states are `euqal`.
 * Equal States: 1. the transform steps are totally the same; 2. even with different steps, two
 * states may still result in a same schedule. e.g. To split a axis with extent 512 to 3 parts
 * [8, 16, 4]. We can split from inner to outter by factors [16, 4], while we can get a same result
 * to split from outter to inner by factors [8, 16])
 */
template <>
struct equal_to<::tvm::auto_schedule::State> {
  bool operator()(const ::tvm::auto_schedule::State& lhs,
                  const ::tvm::auto_schedule::State& rhs) const {
    return lhs.ToStr() == rhs.ToStr();
  }
};

}  // namespace std

#endif  // TVM_AUTO_SCHEDULE_LOOP_STATE_H_
