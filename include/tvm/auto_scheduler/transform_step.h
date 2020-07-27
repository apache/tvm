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
 * \file auto_scheduler/transform_step.h
 * \brief Transformation steps. These steps are used to manipulate the LoopState.
 *        They are similar to the schedule primitives in te::Stage.
 *
 * \note How to add a new transform step:
 * Take fuse step for example:
 * 1. Define class `FuseStepNode`, `FuseStep` in `transform_steps.h`, and implement its first
 *    construction function `FuseStep::FuseStep()` in `transform_steps.cc`.
 * 2. Implement `FuseStepNode::ApplyToSchedule()` and `FuseStepNode::PrintAsPythonAPI()`.
 *    - In these two functions you need to lower this step with tvm's te schedule API
 * 3. Implement `FuseStepNode::ApplyToState` and the state API `State::fuse`.
 *    - In these two functions you need to incrementally update all data structures in State with
 *      CopyOnWrite style.
 * 4. Add your step implementation to `StepApplyToState`, `StepApplyToSchedule` and
 *    `StepPrintAsPythonAPI`, make sure it works.
 * 5. Log record serialization support:
 *    - Add `FuseStepNode::WriteToRecord` which takes a mutable JSONWriter pointer as input and
 *      output the record to it.
 *    - Add another construction function that takes a mutable JSONReader as input, this will get a
 *      step record from the reader and create the step.
 *    - Add the step implementation to `StepReadFromRecord`.
 * 6. Add its corresponding Python API to `loop_state.py` and necessary unit test, the test should
 *    at lease consists of two parts: the functional test and the record serialization test.
 */

#ifndef TVM_AUTO_SCHEDULER_TRANSFORM_STEP_H_
#define TVM_AUTO_SCHEDULER_TRANSFORM_STEP_H_

#include <dmlc/common.h>
#include <dmlc/json.h>
#include <tvm/node/node.h>
#include <tvm/te/schedule.h>

namespace tvm {
namespace auto_scheduler {

typedef Map<tvm::te::Stage, Array<tir::IterVar>, ObjectHash, ObjectEqual> StageToAxesMap;

/*!
 * \brief Update the current stage IterVar information to StageToAxesMap.
 * \param stage A te::Stage Object.
 * \param stage_to_axes A mutable pointer to StageToAxesMap, this map will be updated.
 */
void UpdateStageToAxesMap(const te::Stage& stage, StageToAxesMap* stage_to_axes);

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
  /*! \brief This iterator has been bind to blockIdx.y. */
  kBlockZ = 9,
  /*! \brief This iterator has been bind to threadIdx.y. */
  kThreadZ = 10,
  /*! \brief This iterator has been mapped with a tensorize intrinsic. */
  kTensorize = 11
};

extern const char* IteratorAnnotationString[];

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
    v->Visit("iter_kind", &iter_kind);
    v->Visit("annotation", &annotation);
  }

  static constexpr const char* _type_key = "auto_scheduler.Iterator";
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

/*!
 * \brief The base class of transformation steps. Each step has its corresponding tvm.te
 * schedule primitives.
 */
class StepNode : public Object {
 public:
  /*! \brief The index of the stage. */
  int stage_id;

  /*!
   * \brief Serialize the current step record to JSONWriter.
   * \param writer The output JSONWriter.
   */
  virtual void WriteToRecord(dmlc::JSONWriter* writer) const = 0;

  static constexpr const char* _type_key = "auto_scheduler.Step";
  TVM_DECLARE_BASE_OBJECT_INFO(StepNode, Object);
};

/*!
 * \brief Managed reference to StepNode.
 * \sa StepNode
 */
class Step : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Step, ObjectRef, StepNode);
};

// Forward declaration
class State;
class ComputeDAG;

/*!
 * \brief Read a step record from JSONReader and create the corresponding step.
 * \param reader The input JSONReader.
 */
Step StepReadFromRecord(dmlc::JSONReader* reader);

/*!
 * \brief Apply the step to State.
 * \param step The step to be applied to State.
 * \param state A mutable pointer to state, which will be updated.
 * \param dag The original ComputeDAG of this state.
 */
void StepApplyToState(const Step& step, State* state, const ComputeDAG& dag);

/*!
 * \brief Apply the step to tvm.schedule.
 * \param step The step to be applied to tvm.schedule.
 * \param stages The `te::Stage`s used in TVM scheduler applying.
 * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
 * \param schedule A mutable pointer to a `te::Schedule`. This is required by some steps which need
 * `te::Schedule` API. (e.g. CacheRead/CacheWrite step)
 */
void StepApplyToSchedule(const Step& step, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                         te::Schedule* schedule);

/*!
 * \brief Print the step as equivalent python schedule API.
 * \param step The step to be applied to python API.
 * \param stages The `te::Stage`s used in TVM scheduler applying.
 * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
 * \param schedule A mutable pointer to a te::Schedule. This is required by some steps. (e.g.
 * CacheRead/CacheWrite step)
 * \return Python schedule code.
 */
String StepPrintAsPythonAPI(const Step& step, Array<te::Stage>* stages,
                            StageToAxesMap* stage_to_axes, te::Schedule* schedule);

/********** Steps working on single stage **********/

/*!
 * \brief Annotation step that corresponds to vectorize, parallel, unroll and thread binding.
 * (i.e. te::Stage::vectorize, te::Stage::parallel, te::Stage::vectorize, te::Stage::bind)
 */
class AnnotationStepNode : public StepNode {
 public:
  /*! \brief The index of the iterator to add annotation. */
  int iter_id;
  /*! \brief The annotation type of this step. */
  IteratorAnnotation annotation;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \return The iterator result after annotate.
   */
  Iterator ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   */
  void ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "AN";

  static constexpr const char* _type_key = "auto_scheduler.AnnotationStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(AnnotationStepNode, Object);
};

/*!
 * \brief Managed reference to AnnotationStepNode.
 * \sa AnnotationStepNode
 */
class AnnotationStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to add annotation.
   * \param iter_id The index of the iterator to add annotation.
   * \param ann The annotation type of this step.
   */
  AnnotationStep(int stage_id, int iter_id, IteratorAnnotation ann);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit AnnotationStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(AnnotationStep, Step, AnnotationStepNode);
};

/*! \brief Fuse step that corresponds to te::Stage::fuse */
class FuseStepNode : public StepNode {
 public:
  /*! \brief The ids of iterators to fuse. */
  Array<Integer> fused_ids;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \return The iterator result after fuse.
   * \note If the iterators to be fused have stages attached at them(by compute_at), the fused
   * result will become the new attach point.
   */
  Iterator ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return The iterator result after fuse.
   */
  tir::IterVar ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "FU";

  static constexpr const char* _type_key = "auto_scheduler.FuseStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuseStepNode, Object);
};

/*!
 * \brief Managed reference to FuseStepNode.
 * \sa FuseStepNode
 */
class FuseStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be fused.
   * \param fused_ids The index of the iterators to be fused.
   */
  FuseStep(int stage_id, const Array<Integer>& fused_ids);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit FuseStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(FuseStep, Step, FuseStepNode);
};

/*! \brief Reorder step that corresponds to te::Stage::reorder */
class ReorderStepNode : public StepNode {
 public:
  /*!
   * \brief The iterator ids after reorder.
   * This array should specify the order of all iterators.
   */
  Array<Integer> after_ids;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   */
  void ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   */
  void ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "RE";

  static constexpr const char* _type_key = "auto_scheduler.ReorderStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderStepNode, Object);
};

/*!
 * \brief Managed reference to ReorderStepNode.
 * \sa ReorderStepNode
 */
class ReorderStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be reordered.
   * \param after_ids The expected indexes of the iterators after reorder.
   */
  ReorderStep(int stage_id, const Array<Integer>& after_ids);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit ReorderStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(ReorderStep, Step, ReorderStepNode);
};

/*!
 * \brief Split step that corresponds to te::Stage::split with additional
 *  support of multiple-level of factors
 */
class SplitStepNode : public StepNode {
 public:
  /*! \brief The id of the iter to split. */
  int iter_id;
  /*! \brief The extent length of the axis to split. */
  Optional<Integer> extent;
  /*! \brief The split factors. */
  Array<Optional<Integer>> lengths;
  /*!
   * \brief If true, the `lengths` denote the lengths of iterators
   * from inner level to outer level
   */
  bool inner_to_outer;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \return The iterator results after split.
   * \note If we do split on an iterator which has stages attached at it(by compute_at), the inner
   * most iterator of split results will become the new attach point.
   */
  Array<Iterator> ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return The iterator results after split.
   */
  Array<tir::IterVar> ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "SP";

  static constexpr const char* _type_key = "auto_scheduler.SplitStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitStepNode, Object);
};

/*!
 * \brief Managed reference to SplitStepNode.
 * \sa SplitStepNode
 */
class SplitStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be split.
   * \param iter_id The index of the iterator to be split.
   * \param extent The extent length of the axis to split.
   * \param lengths The multiple split factors. Can be None to be filled by search policy.
   * \param inner_to_outer The split direction.
   */
  SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
            const Array<Optional<Integer>>& lengths, bool inner_to_outer);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit SplitStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(SplitStep, Step, SplitStepNode);
};

/********** Steps working on multiple stages **********/

/*! \brief Compute at step that corresponds to te::Stage::compute_at */
class ComputeAtStepNode : public StepNode {
 public:
  /*! \brief The index of stage that this step will compute at to. */
  int target_stage_id;
  /*! \brief The index of iterator in target stage that this step will compute at to. */
  int target_iter_id;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \note After compute_at, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state to get the complete bound information.
   */
  void ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   */
  void ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "CA";

  static constexpr const char* _type_key = "auto_scheduler.ComputeAtStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeAtStepNode, Object);
};

/*!
 * \brief Managed reference to ComputeAtStepNode.
 * \sa ComputeAtStepNode
 */
class ComputeAtStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be computed at.
   * \param target_stage_id The index of stage that this step will compute at to.
   * \param target_iter_id The index of iterator in target stage that this step will compute at to.
   */
  ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit ComputeAtStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeAtStep, Step, ComputeAtStepNode);
};

/*! \brief Compute inline step that corresponds to te::Stage::compute_inline */
class ComputeInlineStepNode : public StepNode {
 public:
  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   */
  void ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return The iterator result after fuse.
   */
  void ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "CI";

  static constexpr const char* _type_key = "auto_scheduler.ComputeInlineStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeInlineStepNode, Object);
};

/*!
 * \brief Managed reference to ComputeInlineStepNode.
 * \sa ComputeInlineStepNode
 */
class ComputeInlineStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be marked compute inlined.
   */
  explicit ComputeInlineStep(int stage_id);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit ComputeInlineStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeInlineStep, Step, ComputeInlineStepNode);
};

/*! \brief Compute root step that corresponds to te::Stage::compute_root */
class ComputeRootStepNode : public StepNode {
 public:
  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \note After compute_root, we need careful dependency analysis to compute the accurate bound
   * information. However, it is relatively expensive and complicated, so we just fill "None" as
   * bound for the newly created iterators.
   * Call ComputeDAG::InferBound on the updated state to get the complete bound information.
   */
  void ApplyToState(State* state) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return The iterator result after fuse.
   */
  void ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  static constexpr const char* record_prefix_str = "CR";

  static constexpr const char* _type_key = "auto_scheduler.ComputeRootStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeRootStepNode, Object);
};

/*!
 * \brief Managed reference to ComputeRootStepNode.
 * \sa ComputeRootStepNode
 */
class ComputeRootStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be marked compute at root.
   */
  explicit ComputeRootStep(int stage_id);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit ComputeRootStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeRootStep, Step, ComputeRootStepNode);
};

/********** Primitives adding new stages **********/

/*!
 * \brief Cache read step that corresponds to te::Schedule::cache_read.
 * \note Cache read step will add an extra stage to the original ComputeDAG, a up-to-date ComputeDAG
 * is stored in State's `current_compute_dag`.
 */
class CacheReadStepNode : public StepNode {
 public:
  /*! \brief The scope name of the newly added read stage. (e.g. local, shared, global) */
  String scope_name;
  /*! \brief The indices of read stages. */
  Array<Integer> reader_stage_ids;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \param dag The original ComputeDAG of this state.
   * \return The index of the new added stage.
   */
  int ApplyToState(State* state, const ComputeDAG& dag) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \param schedule A mutable pointer to a te::Schedule.
   * \return The output Tensor of the new added stage.
   */
  te::Tensor ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                             te::Schedule* schedule) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \param schedule A mutable pointer to a te::Schedule.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                          te::Schedule* schedule) const;

  static constexpr const char* record_prefix_str = "CHR";

  static constexpr const char* _type_key = "auto_scheduler.CacheReadStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadStepNode, Object);
};

/*!
 * \brief Managed reference to CacheReadStepNode.
 * \sa CacheReadStepNode
 */
class CacheReadStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be cache read.
   * \param scope_name The scope name of the newly added read stage.
   * \param reader_stage_ids The indices of read stages.
   */
  CacheReadStep(int stage_id, String scope_name, const Array<Integer>& reader_stage_ids);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit CacheReadStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(CacheReadStep, Step, CacheReadStepNode);
};

/*!
 * \brief Cache write step that corresponds to te::Schedule::cache_write.
 * \note Cache write step will add an extra stage to the original ComputeDAG, a up-to-date
 * ComputeDAG is stored in State's `current_compute_dag`.
 * This step will cache write all output tensors of the target stage.
 */
class CacheWriteStepNode : public StepNode {
 public:
  /*! \brief The scope name of the newly added compute stage. (e.g. local, shared, global) */
  String scope_name;

  void WriteToRecord(dmlc::JSONWriter* writer) const final;

  /*!
   * \brief Apply the current step to State.
   * \param state A mutable pointer to state, which will be updated.
   * \param dag The original ComputeDAG of this state.
   * \return The index of the new added stage.
   */
  int ApplyToState(State* state, const ComputeDAG& dag) const;

  /*!
   * \brief Apply the current step to tvm.schedule.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \param schedule A mutable pointer to a te::Schedule.
   * \return The output Tensors of the new added stage.
   */
  Array<te::Tensor> ApplyToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                    te::Schedule* schedule) const;

  /*!
   * \brief Print the current step as equivalent python schedule API.
   * \param stages The `te::Stage`s used in TVM scheduler applying.
   * \param stage_to_axes The `te::Stage` and `tir::IterVar` map.
   * \param schedule A mutable pointer to a te::Schedule.
   * \return Python schedule code.
   */
  String PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                          te::Schedule* schedule) const;

  static constexpr const char* record_prefix_str = "CHW";

  static constexpr const char* _type_key = "auto_scheduler.CacheWriteStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteStepNode, Object);
};

/*!
 * \brief Managed reference to CacheWriteStepNode.
 * \sa CacheWriteStepNode
 */
class CacheWriteStep : public Step {
 public:
  /*!
   * \brief The constructor.
   * \param stage_id The index of the stage to be cache write.
   * \param scope_name The scope name of the newly added compute stage.
   */
  CacheWriteStep(int stage_id, String scope_name);

  /*!
   * \brief The constructor used to read a step record from JSONReader and create the
   * corresponding step.
   * \param reader The input JSONReader.
   */
  explicit CacheWriteStep(dmlc::JSONReader* reader);

  TVM_DEFINE_OBJECT_REF_METHODS(CacheWriteStep, Step, CacheWriteStepNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_TRANSFORM_STEP_H_
