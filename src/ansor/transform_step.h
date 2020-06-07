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
 * \brief  Transformation steps. For each schedule primitive, there is a corresponding transform step.
 *
 * \Note How to add a new transform step.
 * Take fuse for example:
 * 1. Define class FuseStepNode, FuseStep in transform_steps.h, and implement its make function
 *    in  FuseStepNode::make(...)  transform_steps.cc
 * 2. Implement FuseStepNode::ApplyToSchedule and FuseStepNode::PrintAsPythonAPI.
 *    - In these two functions you need to lower this step with tvm's schedule API
 * 3. Implement State::fuse and State::DoFuseStep.
 *    - In these two functions you need to incrementally update all data structures in State with
 *       CopyOnWrite style
 * 4. Add you step to ComputeDAG::ReplaySteps and make sure it works.
 * 5. Add serialization support in `struct Handler<std::vector<::tvm::ansor::Step> >`
 *    (in serialization.cc)
 * 6. Add hash support in `struct hash<::tvm::ansor::Step>` (search for this function in this file)
 */

#ifndef TVM_ANSOR_TRANSFORM_STEP_H_
#define TVM_ANSOR_TRANSFORM_STEP_H_

#include <dmlc/common.h>
#include <string>
#include <vector>
#include "loop_state.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

/*! \brief Reorder step that corresponds to te::Stage::reorder */
class ReorderStepNode: public StepNode {
 public:
  std::vector<int> after_ids;  // The iterator ids after reorder.
  // This array should specify the order of all iterators.

  static ReorderStep make(int stage_id, const std::vector<int>& after_ids);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.ReorderStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(ReorderStep, Step, ReorderStepNode);

/*! \brief Split step that corresponds to te::Stage::split with additional
 *  support of multiple-level of factors */
class SplitStepNode: public StepNode {
 public:
  int iter_id;                    // The id of the iter to split
  PrimExpr extent;                // the extent length of the axis to split
  std::vector<PrimExpr> lengths;  // The split factors
  bool inner_to_outer;            // If true, the `lengths` denote the lengths of
                                  // iterators from inner level to outer level

  static SplitStep make(int stage_id, int iter_id, PrimExpr extent,
                        const std::vector<PrimExpr>& lengths,
                        bool inner_to_outer);

  std::vector<IterVar> ApplyToSchedule(std::vector<te::Stage> *stages,
                                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.SplitStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(SplitStep, Step, SplitStepNode);

/*! \brief Similar to SplitStepNode, but use split factor from another step
 * (i.e. Follow another split step) */
class FollowSplitStepNode: public StepNode {
 public:
  int iter_id;      // The id of the iter to split
  int src_step_id;  // The index of the split step to follow in the history
  int n_split;      // The number of split level

  static FollowSplitStep make(int stage_id, int iter_id,
                              int src_step_id, int n_split);

  void ExtractSplitLengths(const std::vector<Step>& transform_steps,
                           std::vector<PrimExpr>* lengths) const;

  std::vector<IterVar> ApplyToSchedule(std::vector<te::Stage> *stages,
                                       StageToAxesMap *stage_to_axes,
                                       const std::vector<Step>& transform_steps) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.FollowSplitStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(FollowSplitStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(FollowSplitStep, Step, FollowSplitStepNode);

/*! \brief Similar to FollowSplitStep, but use split factors from multiple steps.
 *  \Note This can be used for the split in cooperative fetching
 */
class FollowFusedSplitStepNode: public StepNode {
 public:
  int iter_id;                    // The id of the iter to split
  std::vector<int> src_step_ids;  // The indices of the split steps to follow in the history
  int level;                      // Use the length in this split level
  bool factor_or_nparts;          // If this is true, use factor. Otherwise, use nparts

  static FollowFusedSplitStep make(int stage_id, int iter_id,
                                   const std::vector<int>& src_step_ids,
                                   int level, bool factor_or_nparts);

  PrimExpr ExtractSplitLength(const std::vector<Step>& transform_steps) const;

  std::vector<IterVar> ApplyToSchedule(std::vector<te::Stage> *stages,
                                       StageToAxesMap *stage_to_axes,
                                       const std::vector<Step>& transform_steps) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.FollowFusedSplitStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(FollowFusedSplitStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(FollowFusedSplitStep, Step, FollowFusedSplitStepNode);

/*! \brief Fuse step that corresponds to te::Stage::fuse */
class FuseStepNode: public StepNode {
 public:
  std::vector<int> fused_ids;  // The ids of iterators to fuse

  static FuseStep make(int stage_id, const std::vector<int>& fused_ids);

  IterVar ApplyToSchedule(std::vector<te::Stage> *stages,
                          StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.FuseStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuseStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(FuseStep, Step, FuseStepNode);

/*! \brief Annotation step that corresponds to vectorize, parallel, unroll and thread binding.
 * (i.e. te::Stage::vectorize, te::Stage::parallel, te::Stage::vectorize, te::Stage::bind)
 */
class AnnotationStepNode: public StepNode {
 public:
  int iter_id;
  IteratorAnnotation annotation;

  static AnnotationStep make(int stage_id, int iter_id, IteratorAnnotation ann);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.AnnotationStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(AnnotationStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(AnnotationStep, Step, AnnotationStepNode);

/*! \brief Fuse step that corresponds to te::Stage::compute_at */
class ComputeAtStepNode: public StepNode {
 public:
  int target_stage_id;
  int target_iter_id;

  static ComputeAtStep make(int stage_id, int target_stage_id,
                            int target_iter_id);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.ComputeAtStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeAtStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(ComputeAtStep, Step, ComputeAtStepNode);

/*! \brief Fuse step that corresponds to te::Stage::compute_root */
class ComputeRootStepNode: public StepNode {
 public:
  static ComputeRootStep make(int stage_id);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.ComputeRootStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeRootStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(ComputeRootStep, Step, ComputeRootStepNode);

/*! \brief Fuse step that corresponds to te::Stage::compute_inline */
class ComputeInlineStepNode: public StepNode {
 public:
  static ComputeInlineStep make(int stage_id);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.ComputeInlineStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeInlineStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(ComputeInlineStep, Step, ComputeInlineStepNode);

/*! \brief Cache read step that corresponds to te::Schedule::cache_read */
class CacheReadStepNode: public StepNode {
 public:
  std::string scope_name;
  std::vector<int> reader_stage_ids;

  static CacheReadStep make(int stage_id, std::string scope_name,
      const std::vector<int>& reader_stage_id);

  te::Tensor ApplyToSchedule(std::vector<te::Stage> *stages,
      StageToAxesMap *stage_to_axes, te::Schedule *schedule) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.CacheReadStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(CacheReadStep, Step, CacheReadStepNode);

/*! \brief Cache read step that corresponds to te::Schedule::cache_write
 *  \Note This step will cache_write all output tensors of target stage */
class CacheWriteStepNode: public StepNode {
 public:
  std::string scope_name;

  static CacheWriteStep make(int stage_id, std::string scope_name);

  Array<te::Tensor> ApplyToSchedule(std::vector<te::Stage> *stages,
      StageToAxesMap *stage_to_axes, te::Schedule *schedule) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.CacheWriteStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(CacheWriteStep, Step, CacheWriteStepNode);

/*! \brief Cache read step that corresponds to te::Schedule::pragma */
class PragmaStepNode: public StepNode {
 public:
  int iter_id;
  std::string pragma_type;

  static PragmaStep make(int stage_id, int iter_id, std::string pragma_type);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.PragmaStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(PragmaStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(PragmaStep, Step, PragmaStepNode);

/*! \brief Reduction factor step that corresponds to te::Schedule::rfactor */
class RfactorStepNode: public StepNode {
 public:
  int iter_id;
  int factor_iter_id;

  static RfactorStep make(int stage_id, int iter_id, int factor_iter_id);

  Array<te::Tensor> ApplyToSchedule(std::vector<te::Stage> *stages,
                                StageToAxesMap *stage_to_axes,
                                te::Schedule *schedule) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.RfactorStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(RfactorStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(RfactorStep, Step, RfactorStepNode);

/*! \brief Storage align step that corresponds to te::Schedule::storage_align */
class StorageAlignStepNode: public StepNode {
 public:
  int iter_id;
  int factor;
  int offset;

  static StorageAlignStep make(int stage_id, int iter_id, int factor,
                               int offset);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
                       StageToAxesMap *stage_to_axes) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.StorageAlignStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(StorageAlignStepNode, Object);
};
TVM_DEFINE_COW_OBJECT_REF(StorageAlignStep, Step, StorageAlignStepNode);

}  // namespace ansor
}  // namespace tvm

// Hash and equal function for Step
namespace std {

template <>
struct hash<::tvm::ansor::Step> {
  std::size_t operator()(const ::tvm::ansor::Step& step) const {
    if (auto ps = step.as<::tvm::ansor::ReorderStepNode>()) {
      return ::dmlc::HashCombine(1,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                                 ps->after_ids));
    } else if (auto ps = step.as<::tvm::ansor::SplitStepNode>()) {
      size_t ret =  ::dmlc::HashCombine(2,
                    ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                    ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                        ps->inner_to_outer)));
      for (const auto& len : ps->lengths) {
        if (len.defined()) {
          auto pint = len.as<::tvm::tir::IntImmNode>();
          CHECK(pint != nullptr);
          ret = ::dmlc::HashCombine(ret, pint->value);
        } else {
          ret = ::dmlc::HashCombine(ret, 0x5D);  // a magic number
        }
        return ret;
      }
    } else if (auto ps = step.as<::tvm::ansor::FollowSplitStepNode>()) {
      return ::dmlc::HashCombine(3,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->src_step_id),
                                 ps->n_split))));
    } else if (auto ps = step.as<::tvm::ansor::FollowFusedSplitStepNode>()) {
      return ::dmlc::HashCombine(4,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
             ::dmlc::HashCombine(std::hash<vector<int>>()(ps->src_step_ids),
             ::dmlc::HashCombine(std::hash<int>()(ps->level),
                                 ps->factor_or_nparts)))));
    } else if (auto ps = step.as<::tvm::ansor::FuseStepNode>()) {
      return ::dmlc::HashCombine(5,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                                 ps->fused_ids));
    } else if (auto ps = step.as<::tvm::ansor::AnnotationStepNode>()) {
      return ::dmlc::HashCombine(6,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 static_cast<int>(ps->annotation))));
    } else if (auto ps = step.as<::tvm::ansor::ComputeAtStepNode>()) {
      return ::dmlc::HashCombine(7,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->target_stage_id),
                                 ps->target_iter_id)));
    } else if (auto ps = step.as<::tvm::ansor::ComputeRootStepNode>()) {
      return ::dmlc::HashCombine(8,
                                 ps->stage_id);
    } else if (auto ps = step.as<::tvm::ansor::ComputeInlineStepNode>()) {
      return ::dmlc::HashCombine(9,
                                 ps->stage_id);
    } else if (auto ps = step.as<::tvm::ansor::CacheReadStepNode>()) {
      return ::dmlc::HashCombine(10,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<std::string>()(ps->scope_name),
                                 ps->reader_stage_ids)));
    } else if (auto ps = step.as<::tvm::ansor::CacheWriteStepNode>()) {
      return ::dmlc::HashCombine(11,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                                 ps->scope_name));
    } else if (auto ps = step.as<::tvm::ansor::PragmaStepNode>()) {
      return ::dmlc::HashCombine(12,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 ps->pragma_type)));
    } else if (auto ps = step.as<::tvm::ansor::RfactorStepNode>()) {
      return ::dmlc::HashCombine(13,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 ps->factor_iter_id)));
    } else if (auto ps = step.as<::tvm::ansor::StorageAlignStepNode>()) {
      return ::dmlc::HashCombine(14,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->factor),
                                 ps->offset))));
    } else {
      LOG(FATAL) << "Invalid step";
    }
    return 0;
  }
};
}  // namespace std

#endif  // TVM_ANSOR_TRANSFORM_STEP_H_
