/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/transform_step.h
 * \brief  Data structures for loop transformations

 * Basically this is a simplified TVM IR with schedule primitives.
 * We don't use the existing TVM IR because
 * 1. We want fast incremental change to the loop structures
 * 2. We want serializable history for replay and backtracking
 * 3. We want simplified IR for easy and clean feature extraction
 * 4. We may create some Macro schedule primitives

 * After search is done, we will lower this IR to TVM IR and TVM schedule primitives.
 * Because we share a lot common objects during search,  the transformation is
 * implemented in copy on write style.  All objects are immutable, which is
 * similar to TVM IR.
 */

#ifndef TVM_ANSOR_TRANSFORM_STEP_H_
#define TVM_ANSOR_TRANSFORM_STEP_H_

#include <dmlc/common.h>
#include <string>
#include <vector>
#include "compute_dag.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

inline std::string CleanName(const std::string& str) {
  // to make the name valid in python code
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  return ret;
}

enum IteratorType {
  kSpace,     // spatial iterator
  kReduce,    // reduction iterator
  kMixed,     // fused spatial and reduction iterator
  kSpecial    // special iterator (e.g. virtual root iterator)
};

enum IteratorAnnotation {
  kNone, kUnroll, kVectorize, kParallel,
  kVThread, kBlockX, kThreadX, kBlockY, kThreadY
};

class Iterator;

/*!
 * \brief An for loop iterator
 * Similar to tvm::IterVar in `include/expr.h`
 */
class IteratorNode : public Object {
 public:
  std::string name;
  Range range;             // domain of for loop range
  IteratorType iter_type;
  IteratorAnnotation annotation;
  std::vector<Iterator> ori_iters;

  static Iterator make(std::string name, Range range,
                       IteratorType iter_type, IteratorAnnotation annotation,
                       const std::vector<Iterator>* ori_iters = nullptr);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("range", &range);
  }

  static constexpr const char *_type_key = "ansor.Iterator";
  TVM_DECLARE_FINAL_OBJECT_INFO(IteratorNode, Object);
};
TVM_DEFINE_COW_NODE_REF(Iterator, ObjectRef, IteratorNode);

/*! \brief The base class for a transformation step */
class StepNode: public Object {
 public:
  int stage_id;

  // Print step as equivalent python schedule API
  virtual std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                       StageToAxesMap *stage_to_axes,
                                       te::Schedule *schedule,
                                       const std::vector<Step>& transform_steps) const = 0;

  static constexpr const char* _type_key = "ansor.Step";
  TVM_DECLARE_BASE_OBJECT_INFO(StepNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(Step, StepNode);

/*
 * Note on how to add a new transform step
 *
 * Take fuse for example:
 * 1. Define class FuseStepNode, FuseStep in loop_state.h, and implement its make function
 *    in  FuseStepNode::make(...)  loop_state.cc
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

class ReorderStep; class SplitStep; class FollowSplitStep;
class FollowFusedSplitStep;
class FuseStep; class AnnotationStep;
class ComputeAtStep; class ComputeRootStep; class ComputeInlineStep;
class PackForVecStep; class CacheReadStep; class CacheWriteStep;
class PragmaStep; class RfactorStep; class StorageAlignStep;
class AttachMap;

class ReorderStepNode: public StepNode {
 public:
  std::vector<int> after_ids;

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
TVM_DEFINE_COW_NODE_REF(ReorderStep, Step, ReorderStepNode);


class SplitStepNode: public StepNode {
 public:
  int iter_id;
  PrimExpr extent;                // the extent of the axis to split
  std::vector<PrimExpr> lengths;  // The split factors
  bool inner_to_outer;

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
TVM_DEFINE_COW_NODE_REF(SplitStep, Step, SplitStepNode);

// Similar to SplitStepNode, but use split factor from another step
// (i.e. Follow another split step)
class FollowSplitStepNode: public StepNode {
 public:
  int iter_id;
  int src_step_id;
  int n_split;

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
TVM_DEFINE_COW_NODE_REF(FollowSplitStep, Step, FollowSplitStepNode);


// Similar to FollowSplitStep, but use split factors from multiple steps
// This can be used for the split in cooperative fetching.
class FollowFusedSplitStepNode: public StepNode {
 public:
  int iter_id;
  std::vector<int> src_step_ids;
  int level;              // Use the length in this split level
  bool factor_or_nparts;  // If this is true, use factor. Otherwise, use nparts

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
TVM_DEFINE_COW_NODE_REF(FollowFusedSplitStep, Step, FollowFusedSplitStepNode);


class FuseStepNode: public StepNode {
 public:
  std::vector<int> fused_ids;

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
TVM_DEFINE_COW_NODE_REF(FuseStep, Step, FuseStepNode);


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
TVM_DEFINE_COW_NODE_REF(AnnotationStep, Step, AnnotationStepNode);


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
TVM_DEFINE_COW_NODE_REF(ComputeAtStep, Step, ComputeAtStepNode);


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
TVM_DEFINE_COW_NODE_REF(ComputeRootStep, Step, ComputeRootStepNode);


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
TVM_DEFINE_COW_NODE_REF(ComputeInlineStep, Step, ComputeInlineStepNode);

class PackForVecStepNode: public StepNode {
 public:
  int iter_id;
  int vec_size;

  static PackForVecStep make(int stage_id, int iter_id, int vec_size);

  void ApplyToSchedule(std::vector<te::Stage> *stages,
      StageToAxesMap *stage_to_axes, te::Schedule *schedule) const;

  std::string PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const final;

  static constexpr const char* _type_key = "ansor.PackForVecStep";
  TVM_DECLARE_FINAL_OBJECT_INFO(PackForVecStepNode, Object);
};
TVM_DEFINE_COW_NODE_REF(PackForVecStep, Step, PackForVecStepNode);


/*! \brief Apply cache_read to a stage
 * TVM Api: te::Schedule::cache_read(tensor, scope, readers) */
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
TVM_DEFINE_COW_NODE_REF(CacheReadStep, Step, CacheReadStepNode);


/*! \brief Apply cache_write to a stage
 * TVM Api: te::Schedule::cache_write(tensor, scope)
 * This step will cache_write all output tensors of target stage */
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
TVM_DEFINE_COW_NODE_REF(CacheWriteStep, Step, CacheWriteStepNode);

/*! \brief Add pragma to a specific iterator */
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
TVM_DEFINE_COW_NODE_REF(PragmaStep, Step, PragmaStepNode);

/*! \brief Factor a reduction axis
 * TVM Api: te::Schedule::rfactor(tensor, axis, factor_axis) */
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
TVM_DEFINE_COW_NODE_REF(RfactorStep, Step, RfactorStepNode);

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
TVM_DEFINE_COW_NODE_REF(StorageAlignStep, Step, StorageAlignStepNode);

}  // namespace ansor
}  // namespace tvm

// Hash and equal function for State, Stage, Iterator and Step
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
    } else if (auto ps = step.as<::tvm::ansor::PackForVecStepNode>()) {
      return ::dmlc::HashCombine(10,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 ps->vec_size)));
    } else if (auto ps = step.as<::tvm::ansor::CacheReadStepNode>()) {
      return ::dmlc::HashCombine(11,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<std::string>()(ps->scope_name),
                                 ps->reader_stage_ids)));
    } else if (auto ps = step.as<::tvm::ansor::CacheWriteStepNode>()) {
      return ::dmlc::HashCombine(12,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
                                 ps->scope_name));
    } else if (auto ps = step.as<::tvm::ansor::PragmaStepNode>()) {
      return ::dmlc::HashCombine(13,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 ps->pragma_type)));
    } else if (auto ps = step.as<::tvm::ansor::RfactorStepNode>()) {
      return ::dmlc::HashCombine(14,
             ::dmlc::HashCombine(std::hash<int>()(ps->stage_id),
             ::dmlc::HashCombine(std::hash<int>()(ps->iter_id),
                                 ps->factor_iter_id)));
    } else if (auto ps = step.as<::tvm::ansor::StorageAlignStepNode>()) {
      return ::dmlc::HashCombine(15,
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
