/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/interfaces.h
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

#ifndef TVM_ANSOR_LOOP_STATE_H_
#define TVM_ANSOR_LOOP_STATE_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include "transform_step.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

enum StageType {
  kPlaceholder, kCompute
};

enum ComputeAtType {
  kRoot,     // compute at root
  kInlined,  // inlined
  kIter,     // compute at some iterator
};

class Stage; class State;

/*!
 * \brief A stage in the compute declaration
 * Similar to te::Stage in `include/schedule.h`
 */
class StageNode : public Object {
 public:
  te::Operation op;
  StageType op_type;
  std::vector<Iterator> iters;
  ComputeAtType compute_at;
  int16_t auto_unroll_max_step;
  int storage_offset;

  static Stage make(te::Operation op);
  static Stage make(te::Operation op, StageType op_type,
                    const std::vector<Iterator>& iters,
                    ComputeAtType compute_at, int16_t auto_unroll_max_step,
                    int storage_offset);
  static Stage make(te::Operation op, StageType op_type,
                    std::vector<Iterator>&& iters,
                    ComputeAtType compute_at, int16_t auto_unroll_max_step,
                    int storage_offset);

  static constexpr const char *_type_key = "ansor.Stage";
  TVM_DECLARE_FINAL_OBJECT_INFO(StageNode, Object);
};
TVM_DEFINE_COW_NODE_REF(Stage, ObjectRef, StageNode);

/*! \brief stores the compute_at relation between stages */
class AttachMapNode: public Object {
 public:
  using StageKey = int;
  using IterKey = std::pair<int, int>;  // stage_id and iter_id

  std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
  std::unordered_map<IterKey, std::vector<StageKey>> iter_to_attached_stages;

  static AttachMap make();

  static constexpr const char* _type_key = "ansor.AttachMap";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachMapNode, Object);
};

/*! \brief stores the compute_at relation between stages
 * This stores a bi-directional mapping from stages and iter:
 * 1. Stage to its attached iterator 2. Iterator to the stage attached to it 
 *
 * You can use AttachMapNode::stage_to_attach_iter and AttachMapNode::iter_to_attached_stages
 * to query the relations */
class AttachMap : public ObjectRef {
 public:
  using StageKey = int;
  using IterKey = std::pair<int, int>;  // stage_id and iter_id

  void SetComputeAtIter(int stage_id, int target_stage_id, int target_iter_id);
  void DeleteStage(int stage_id);
  void ReplaceIters(const std::vector<IterKey>& old_iters,
                    const std::vector<IterKey>& new_iters);
  AttachMap ApplyStageIdOfffset(int start_id, int offset) const;

  TVM_DEFINE_OBJECT_REF_METHODS(AttachMap, ObjectRef, AttachMapNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachMapNode);

 private:
  static void DeleteStageEntry(AttachMapNode* pnode, int stage_id);
};

/*! \brief The loop state and corresponding history steps to reach this state */
class StateNode: public Object {
 public:
  std::vector<Stage> stages;           // Current stages and loop structures
  std::vector<Step> transform_steps;   // History transformation steps
  bool complete;          // Indicate whether this state has unfilled tile sizes
  AttachMap attach_map;   // stores the compute_at relation between stages
  ObjectRef aux_info;     // Used to store any auxiliary info about this state
  ComputeDAG task_dag;    // The up-to-date ComputeDAG of this state.
                          // The default value is an empty NodeRef
                          // (means no modification to the DAG)

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("complete", &complete);
    v->Visit("aux_info", &aux_info);
  }

  static State make_empty_state();
  static State make(const Array<te::Operation>& ops);
  static State make(const std::vector<Stage>& stages,
                    const std::vector<Step>& transform_steps, bool complete,
                    ObjectRef aux_info);

  static constexpr const char* _type_key = "ansor.State";
  TVM_DECLARE_FINAL_OBJECT_INFO(StateNode, Object);
};

/*! \brief The loop state and corresponding history steps to reach this state */
class State : public ObjectRef {
 public:
  // Schedule primitives
  void reorder(int stage_id, const std::vector<Iterator>& order);
  std::vector<Iterator> split(int stage_id, const Iterator& it,
                              const std::vector<PrimExpr>& lengths,
                              bool inner_to_outer = true);
  std::vector<Iterator> follow_split(int stage_id, const Iterator& it,
                                     int src_step_id, int n_split);
  std::vector<Iterator> follow_fused_split(int stage_id, const Iterator& it,
                                           const std::vector<int>& src_step_ids,
                                           int level, bool factor_or_nparts);
  Iterator fuse(int stage_id, const std::vector<Iterator>& iters);
  Iterator vectorize(int stage_id, const Iterator& it);
  Iterator parallel(int stage_id, const Iterator& it);
  Iterator unroll(int stage_id, const Iterator& it, int max_unroll = -1);
  // Valide thread_type: kVThread, kBlockX, kThreadX, kThreadY
  Iterator bind_thread(int stage_id, const Iterator& it,
                       IteratorAnnotation thread_type);
  void compute_at(int stage_id, int target_stage_id,
                  const Iterator& target_iter);
  void compute_root(int stage_id);
  void compute_inline(int stage_id);
  void pack_for_vec(int stage_id, const Iterator& target_iter, int vec_size);
  int cache_read(int stage_id, const std::string& scope_name,
                 const std::vector<int>& reader_stage_ids,
                 const ComputeDAG& task_dag);
  int cache_write(int stage_id, const std::string& scope_name,
                  const ComputeDAG& task_dag);
  void pragma(int stage_id, const Iterator& it, const std::string& pragma_type);
  int rfactor(int stage_id, const Iterator& it, int factor_iter_id,
              const ComputeDAG& task_dag);
  void storage_align(int stage_id, const Iterator& it, int factor, int offset);

  // We separate these functions out,
  // so you can call them for replay easily given history steps
  void DoReorderStep(const ReorderStep& step);
  std::vector<Iterator> DoSplitStep(const SplitStep& step);
  std::vector<Iterator> DoFollowSplitStep(const FollowSplitStep& step);
  std::vector<Iterator> DoFollowFusedSplitStep(const FollowFusedSplitStep& step);
  Iterator DoFuseStep(const FuseStep& step);
  Iterator DoAnnotationStep(const AnnotationStep& step);
  void DoComputeAtStep(const ComputeAtStep& step);
  void DoComputeRootStep(const ComputeRootStep& step);
  void DoComputeInlineStep(const ComputeInlineStep& step);
  void DoPackForVecStep(const PackForVecStep& step);
  int DoCacheReadStep(const CacheReadStep& step, const ComputeDAG& dag);
  int DoCacheWriteStep(const CacheWriteStep& step, const ComputeDAG& dag);
  void DoPragmaStep(const PragmaStep& step);
  int DoRfactorStep(const RfactorStep& step, const ComputeDAG& dag);
  void DoStorageAlignStep(const StorageAlignStep& step);

  /* Do transform steps
   * Note: The following function only change loop state.
   *       They do not change transform_history.
   */
  void DoStep(const Step& step, const ComputeDAG& dag);
  void DoSteps(const std::vector<Step>& step, const ComputeDAG& dag);

  // Print to str
  std::string ToStr(bool delete_trivial_loop = true) const;

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);

 private:
  // common function for DoSplitStep and DoFollowSplitStep
  std::vector<Iterator> DoSplitStepCommon(int stage_id, int iter_id,
                                          const std::vector<PrimExpr>& lengths,
                                          bool inner_to_outer);
};

}  // namespace ansor
}  // namespace tvm


// Hash and equal function for State, Stage, Iterator and Step
namespace std {

template <>
struct hash<::tvm::ansor::State> {
  std::size_t operator()(const ::tvm::ansor::State& state) const {
    return std::hash<std::string>()(state.ToStr());
  }
};

template <>
struct equal_to<::tvm::ansor::State> {
  bool operator() (const ::tvm::ansor::State& lhs,
                   const ::tvm::ansor::State& rhs) const {
    return lhs.ToStr() == rhs.ToStr();
  }
};

}  // namespace std

#endif  // TVM_ANSOR_LOOP_STATE_H_
