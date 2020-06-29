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
 * \file ansor/transform_step.h
 * \brief Transformation steps. For each schedule primitive, there is a corresponding transform
 * step. The implementation of each step consists of 2 parts:
 * - transform_step.cc: How each step interact with TVM system
 * - loop_state.cc:     How each step reflect on LoopState
 *
 * \note Adding a new transform step.
 * Take fuse step for example:
 * 1. Define class `FuseStepNode`, `FuseStep` in `transform_steps.h`, and implement its construction
 *    function `FuseStep::FuseStep(...)` in `transform_steps.cc`
 * 2. Implement `FuseStepNode::ApplyToSchedule` and `FuseStepNode::PrintAsPythonAPI`.
 *    - In these two functions you need to lower this step with tvm's te schedule API
 * 3. Implement `State::fuse` and `State::DoFuseStep`.
 *    - In these two functions you need to incrementally update all data structures in State with
 *      CopyOnWrite style
 * 4. Add you step to `ComputeDAG::ReplaySteps` and make sure it works.
 * 5. Add serialization support in `struct Handler<std::vector<::tvm::ansor::Step> >`
 *    in `serialization.cc`.
 * 6. Add hash support in `struct hash<::tvm::ansor::Step>`. (search for this function in this file)
 * 7. Add its corresponding Python API to `loop_state.py` and necessary unit test.
 */

#ifndef TVM_ANSOR_TRANSFORM_STEP_H_
#define TVM_ANSOR_TRANSFORM_STEP_H_

#include <dmlc/common.h>
#include <tvm/node/node.h>
#include <tvm/te/schedule.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tvm {
namespace ansor {

typedef std::unordered_map<tvm::te::Stage, std::vector<tir::IterVar>, ObjectHash, ObjectEqual>
    StageToAxesMap;

class Step;

/*! \brief The base class for a transformation step */
class StepNode : public Object {
 public:
  /*! \brief The index of the target stage. */
  int stage_id;

  /*!
   * \brief Print step as equivalent python schedule API.
   * \param stages A pointer to `te::Stage` vector.
   * \param stage_to_axes A pointer to StageToAxesMap.
   * \param schedule A pointer to `te::Schedule`.
   * \param transform_steps Transform steps of the target state.
   * \return Python schedule code.
   */
  virtual std::string PrintAsPythonAPI(std::vector<te::Stage>* stages,
                                       StageToAxesMap* stage_to_axes, te::Schedule* schedule,
                                       const std::vector<Step>& transform_steps) const = 0;

  static constexpr const char* _type_key = "ansor.Step";
  TVM_DECLARE_BASE_OBJECT_INFO(StepNode, Object);
};

class Step : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Step, ObjectRef, StepNode);
};

/*! \brief Reorder step that corresponds to te::Stage::reorder */
class ReorderStepNode : public StepNode {
 public:
  /*!
   * \brief The iterator ids after reorder.
   * This array should specify the order of all iterators.
   */
  std::vector<int> after_ids;

  /*!
   * \brief Apply the current state to tvm.schedule
   * \param stages A pointer to `te::Stage` vector.
   * \param stage_to_axes A pointer to StageToAxesMap.
   */
  void ApplyToSchedule(std::vector<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                               te::Schedule* schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.ReorderStep";
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
   * \param stage_id The index of the target stage.
   * \param after_ids The index of the iterators after reorder.
   */
  ReorderStep(int stage_id, const std::vector<int>& after_ids);

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
  PrimExpr extent;
  /*! \brief The split factors. */
  std::vector<PrimExpr> lengths;
  /*!
   * \brief If true, the `lengths` denote the lengths of iterators
   * from inner level to outer level
   */
  bool inner_to_outer;

  /*!
   * \brief Apply the current state to tvm.schedule
   * \param stages A pointer to `te::Stage` vector.
   * \param stage_to_axes A pointer to StageToAxesMap.
   * \return The iterator results after split.
   */
  std::vector<tir::IterVar> ApplyToSchedule(std::vector<te::Stage>* stages,
                                            StageToAxesMap* stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                               te::Schedule* schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.SplitStep";
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
   * \param stage_id The index of the target stage.
   * \param extent The index of the target iterator.
   * \param lengths The extent length of the axis to split.
   * \param inner_to_outer The split direction.
   */
  SplitStep(int stage_id, int iter_id, PrimExpr extent, const std::vector<PrimExpr>& lengths,
            bool inner_to_outer);

  TVM_DEFINE_OBJECT_REF_METHODS(SplitStep, Step, SplitStepNode);
};

/*! \brief Fuse step that corresponds to te::Stage::fuse */
class FuseStepNode : public StepNode {
 public:
  /*! \brief The ids of iterators to fuse. */
  std::vector<int> fused_ids;

  /*!
   * \brief Apply the current state to tvm.schedule
   * \param stages A pointer to `te::Stage` vector.
   * \param stage_to_axes A pointer to StageToAxesMap.
   * \return The iterator result after fuse.
   */
  tir::IterVar ApplyToSchedule(std::vector<te::Stage>* stages, StageToAxesMap* stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                               te::Schedule* schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.FuseStep";
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
   * \param stage_id The index of the target stage.
   * \param fused_ids The index of the target iterators to be fused.
   */
  FuseStep(int stage_id, const std::vector<int>& fused_ids);

  TVM_DEFINE_OBJECT_REF_METHODS(FuseStep, Step, FuseStepNode);
};

}  // namespace ansor
}  // namespace tvm

// Hash and equal function for Step
namespace std {

/*! \brief The hash function of each transform step. */
template <>
struct hash<::tvm::ansor::Step> {
  std::size_t operator()(const ::tvm::ansor::Step& step) const {
    // clang-format off
    if (auto ps = step.as<::tvm::ansor::ReorderStepNode>()) {
      return ::dmlc::HashCombine(1,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id), ps->after_ids));
    } else if (auto ps = step.as<::tvm::ansor::SplitStepNode>()) {
      size_t ret = ::dmlc::HashCombine(2,
                   ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                   ::dmlc::HashCombine(std::hash<int>()(ps->iter_id), ps->inner_to_outer)));
      for (const auto& len : ps->lengths) {
        if (len.defined()) {
          auto pint = len.as<::tvm::tir::IntImmNode>();
          CHECK(pint != nullptr);
          ret = ::dmlc::HashCombine(ret, pint->value);
        } else {
          ret = ::dmlc::HashCombine(ret, 0x5D);  // a magic number
        }
      }
      return ret;
    } else if (auto ps = step.as<::tvm::ansor::FuseStepNode>()) {
      return ::dmlc::HashCombine(3,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id), ps->fused_ids));
    } else {
      LOG(FATAL) << "Invalid step";
    }
    return 0;
    // clang-format on
  }
};
}  // namespace std

#endif  // TVM_ANSOR_TRANSFORM_STEP_H_
