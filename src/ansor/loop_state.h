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
 * \file ansor/loop_state.h
 * \brief The definition of the "state" in search. A state consists the current loop structure
 * and the transform history to reach its current loop structure.
 * To enable flexible manipulation of the loop structures, we implemented a lightweight loop
 * structure IR (Intermediate Representation) based on the original TVM IR but specifically
 * for schedule search.
 *
 * We don't use the existing TVM IR but to extend a new Sketch IR on it is because:
 * 1. We want fast incremental change to the loop structures;
 * 2. We want serializable transform history for replay, backtracking, and mutation;
 * 3. We may create some macro schedule primitives that represent the combination of several
 * TVM schedule primitives.
 *
 * After the search is done, we will lower this IR to TVM IR with TVM's schedule primitives.
 * Because we share a lot common objects during search, the transformation is implemented in
 * copy on write style. All objects are immutable, which is similar to TVM IR.
 */

#ifndef TVM_ANSOR_LOOP_STATE_H_
#define TVM_ANSOR_LOOP_STATE_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "compute_dag.h"
#include "transform_step.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

/*! \brief The type of a stage. */
enum StageType {
  /*! \brief A placeholder stage. */
  kPlaceholder = 0,
  /*! \brief A compute stage. */
  kCompute = 1
};

/*! \brief The type of compute location. */
enum ComputeAtType {
  /*! \brief Compute at root. */
  kRoot = 0,
  /*! \brief Compute inlined. */
  kInlined = 1,
  /*! \brief Compute at some iterator. */
  kIter = 2,
};

/*! \brief The type of an iterator. */
enum IteratorType {
  /*! \brief Spatial iterator. */
  kSpace = 0,
  /*! \brief Reduction iterator. */
  kReduce = 1,
  /*! \brief Fused spatial and reduction iterator. */
  kMixed = 2,
  /*! \brief Special iterator. (e.g. virtual root iterator) */
  kSpecial = 3
};

/*! \brief The type of an iterator's annotation. */
enum IteratorAnnotation {
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

// forward declaration
class Iterator;

/*!
 * \brief A for loop iterator
 * Similar to tvm::IterVar in `include/tvm/tir/expr.h`
 */
class IteratorNode : public Object {
 public:
  /*! \brief The name of this iterator. */
  std::string name;
  /*! \brief The target range of this iterator. */
  Range range;
  /*! \brief The iterator type of this iterator. */
  IteratorType iter_type;
  /*! \brief The annotation type of this iterator. */
  IteratorAnnotation annotation;
  /*! \brief The original iterators before fusion. */
  std::vector<Iterator> ori_iters;
  /*! \brief The extra attributes of this iterator. */
  std::string attr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("range", &range);
    v->Visit("attr", &attr);
  }

  static constexpr const char* _type_key = "ansor.Iterator";
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
   * \param range The target range of this iterator.
   * \param iter_type The iterator type of this iterator.
   * \param annotation The annotation type of this iterator.
   * \param ori_iters The original iterators before fusion.
   * \param attr The extra attribute of this iterator.
   */
  Iterator(std::string name, Range range, IteratorType iter_type, IteratorAnnotation annotation,
           const std::vector<Iterator>* ori_iters = nullptr, std::string attr = "");

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
  StageType op_type;
  /*! \brief The iterators in this stage. */
  std::vector<Iterator> iters;
  /*! \brief The compute location of this stage. */
  ComputeAtType compute_at;
  /*! \brief Other stage-level attributes. */
  StageAttributes attrs;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("op", &op); }

  static constexpr const char* _type_key = "ansor.Stage";
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
   * \param iters The iterators of this op. (copy)
   * \param compute_at The compute at type of this op.
   * \param attrs Other stage-level attributes.
   */
  Stage(te::Operation op, StageType op_type, const std::vector<Iterator>& iters,
        ComputeAtType compute_at, StageAttributes attrs);
  /*!
   * \brief The constructor.
   * \param op A `te::Operation`.
   * \param op_type The stage type of this op.
   * \param iters The iterators of this op. (move)
   * \param compute_at The compute at type of this op.
   * \param attrs Other stage-level attributes.
   */
  Stage(te::Operation op, StageType op_type, std::vector<Iterator>&& iters,
        ComputeAtType compute_at, StageAttributes attrs);

  TVM_DEFINE_OBJECT_REF_METHODS(Stage, ObjectRef, StageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StageNode);
};

/*!
 * \brief A State in the search process.
 * It consists of the current loop structure and the history steps to reach this State.
 * Each State corresponds to a specific schedule for the target ComputeDAG.
 */
class StateNode : public Object {
 public:
  /*! \brief Current stages and loop structures. */
  std::vector<Stage> stages;
  /*! \brief History transformation steps. */
  std::vector<Step> transform_steps;
  /*! \brief Indicate whether this state has unfilled tile sizes. */
  bool complete;
  /*!
   * \brief The up-to-date ComputeDAG of this state, used for some steps that may change the
   * stage structure of the ComputeDAG, for exp. CacheReadStep/CacheWriteStep(Will be added later).
   * The default value is an empty NodeRef. (means no modification to the original DAG)
   */
  ComputeDAG task_dag;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("complete", &complete);
    v->Visit("task_dag", &task_dag);
  }

  static constexpr const char* _type_key = "ansor.State";
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
   * \brief The constructor.
   * \param stages Stages of the target state.
   * \param transform_steps Transform steps of the target state.
   * \param complete Indicate whether this state has unfilled tile sizes.
   */
  State(const std::vector<Stage>& stages, const std::vector<Step>& transform_steps, bool complete);

  /*!
   * \brief Schedule primitive corresponds to te.reorder.
   * \param stage_id The index of the target stage.
   * \param order The target iterator order.
   */
  void reorder(int stage_id, const std::vector<Iterator>& order);
  /*!
   * \brief Schedule primitive corresponds to te.split.
   * \param stage_id The index of the target stage.
   * \param it The target iterator.
   * \param lengths The target split factors. Can be None to be filled by search policy.
   * \param inner_to_outer True for split from inner to outer & False for outer to inner.
   * \return The iterator results after split.
   */
  std::vector<Iterator> split(int stage_id, const Iterator& it,
                              const std::vector<PrimExpr>& lengths, bool inner_to_outer = true);
  /*!
   * \brief Schedule primitive corresponds to te.fuse.
   * \param stage_id The index of the target stage.
   * \param iters The target iterators to be fused.
   * \return The iterator result after fuse.
   */
  Iterator fuse(int stage_id, const std::vector<Iterator>& iters);

  /*!
   * \brief General do step functions with a runtime dynamic dispatcher.
   * \param steps The target transform steps.
   * \param dag The target ComputeDAG.
   */
  void DoSteps(const std::vector<Step>& steps, const ComputeDAG& dag);

  /*!
   * \brief Print the state to a string.
   * \param delete_trivial_loop True for skipping the trivial loops.
   * (undefined or extent == 1, default set to True)
   * \return The human readable state structure.
   */
  std::string ToStr(bool delete_trivial_loop = true) const;

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
  std::vector<Iterator> DoSplitStep(const SplitStep& step);
  /*!
   * \brief Apply fuse step to current state.
   * \param step A FuseStep.
   * \return The iterator result after fuse.
   */
  Iterator DoFuseStep(const FuseStep& step);

  /*!
   * \brief Common function for DoSplitStep and DoFollowSplitStep(Will be added later).
   * \param stage_id The index of the target stage.
   * \param iter_id The index of the target iterator.
   * \param lengths The target split factors.
   * \param inner_to_outer The split direction.
   * \return The iterator results after split.
   */
  std::vector<Iterator> DoSplitStepCommon(int stage_id, int iter_id,
                                          const std::vector<PrimExpr>& lengths,
                                          bool inner_to_outer);
};

}  // namespace ansor
}  // namespace tvm

// Hash and equal function for State
namespace std {

/*! \brief The hash function for ansor::State. */
template <>
struct hash<::tvm::ansor::State> {
  std::size_t operator()(const ::tvm::ansor::State& state) const {
    return std::hash<std::string>()(state.ToStr());
  }
};

/*! \brief The equal_to function for ansor::State. */
template <>
struct equal_to<::tvm::ansor::State> {
  bool operator()(const ::tvm::ansor::State& lhs, const ::tvm::ansor::State& rhs) const {
    return lhs.ToStr() == rhs.ToStr();
  }
};

}  // namespace std

#endif  // TVM_ANSOR_LOOP_STATE_H_
