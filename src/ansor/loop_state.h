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
 * \brief The definition of the "state" in search. A state consists a current loop structure
 * and the transform history to reach its current loop structure.
 * To enable flexible manipulation of the loop structure, we implemented a lightweight
 * loop structure IR (Intermediate Representation) specifically for search.
 *
 * Basically this is a simplified TVM IR with schedule primitives.
 * We don't use the existing TVM IR because
 * 1. We want fast incremental change to the loop structures
 * 2. We want serializable transformation history for replay, backtracking, and mutation.
 * 3. We may create some macro schedule primitives
 *
 * After the search is done, we will lower this IR to TVM IR with TVM schedule primitives.
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
#include "compute_dag.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

/*! \brief The type of a stage */
enum StageType {
  kPlaceholder,  // A placeholder stage
  kCompute       // A compute stage
};

/*! \brief The type of compute location */
enum ComputeAtType {
  kRoot,     // compute at root
  kInlined,  // inlined
  kIter,     // compute at some iterator
};

/*! \brief The type of an iterator */
enum IteratorType {
  kSpace,     // spatial iterator
  kReduce,    // reduction iterator
  kMixed,     // fused spatial and reduction iterator
  kSpecial    // special iterator (e.g. virtual root iterator)
};

/*! \brief The type of an iterator's annotation */
enum IteratorAnnotation {
  kNone, kUnroll, kVectorize, kParallel,
  kVThread, kBlockX, kThreadX, kBlockY, kThreadY,
  kTensorized
};

// forward declaration
class Iterator;

/*!
 * \brief A for loop iterator
 * Similar to tvm::IterVar in `include/tvm/tir/expr.h`
 */
class IteratorNode : public Object {
 public:
  std::string name;
  Range range;
  IteratorType iter_type;
  IteratorAnnotation annotation;
  std::vector<Iterator> ori_iters;  // The original iterators before fusion
  std::string attr;                 // Todo(jcf94): Document this

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("range", &range);
    v->Visit("attr", &attr);
  }

  static constexpr const char *_type_key = "ansor.Iterator";
  TVM_DECLARE_FINAL_OBJECT_INFO(IteratorNode, Object);
};

/*!
 * \brief Managed reference to IteratorNode.
 * \sa IteratorNode
 */
class Iterator : public ObjectRef {
 public:
  Iterator(std::string name, Range range, IteratorType iter_type,
           IteratorAnnotation annotation,
           const std::vector<Iterator>* ori_iters = nullptr,
           std::string attr = "");

  TVM_DEFINE_OBJECT_REF_METHODS(Iterator, ObjectRef, IteratorNode);
};

/*! \brief Stage-level attributes */
struct StageAttributes {
  int auto_unroll_max_step;  // The maximum steps for the pragma `auto_unroll_max_step`
  int storage_offset;        // The storage offset for the schedule primitive `storage_align`
};

/*!
 * \brief A stage in the compute declaration
 * Similar to te::Stage in `include/schedule.h`
 */
class StageNode : public Object {
 public:
  te::Operation op;              // The operator of this stage
  StageType op_type;             // The type of this stage
  std::vector<Iterator> iters;   // The iterators in this stage
  ComputeAtType compute_at;      // The compute location of this stage
  StageAttributes attrs;         // Other stage-level attributes

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
  }

  static constexpr const char *_type_key = "ansor.Stage";
  TVM_DECLARE_FINAL_OBJECT_INFO(StageNode, Object);
};

/*!
 * \brief Managed reference to StageNode.
 * \sa StageNode
 */
class Stage : public ObjectRef {
 public:
  explicit Stage(te::Operation op);
  Stage(te::Operation op, StageType op_type,
        const std::vector<Iterator>& iters,
        ComputeAtType compute_at, StageAttributes attrs);
  Stage(te::Operation op, StageType op_type,
        std::vector<Iterator>&& iters,
        ComputeAtType compute_at, StageAttributes attrs);

  TVM_DEFINE_OBJECT_REF_METHODS(Stage, ObjectRef, StageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StageNode);
};

/*! \brief stores the compute_at relation between stages
 * This stores a bi-directional mapping from stages and iter:
 * 1. Stage to its attached iterator 2. Iterator to the stage attached to it 
 *
 * You can use AttachMapNode::stage_to_attach_iter and AttachMapNode::iter_to_attached_stages
 * to query the relations */
class AttachMapNode: public Object {
 public:
  using StageKey = int;
  using IterKey = std::pair<int, int>;  // stage_id and iter_id

  std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
  std::unordered_map<IterKey, std::vector<StageKey>> iter_to_attached_stages;

  static constexpr const char* _type_key = "ansor.AttachMap";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachMapNode, Object);
};

/*!
 * \brief Managed reference to AttachMapNode.
 * \sa AttachMapNode
 */
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
TVM_DEFINE_MUTABLE_OBJECT_REF(Step, StepNode);

// Step forward decelerations
class ReorderStep; class SplitStep; class FuseStep;

/*! \brief A state in the search process.
 *  It consists of the current loop structure and the history steps to reach this state. */
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
  explicit State(const Array<te::Operation>& ops);
  State(const std::vector<Stage>& stages,
        const std::vector<Step>& transform_steps, bool complete,
        ObjectRef aux_info);

  // Schedule primitives
  void reorder(int stage_id, const std::vector<Iterator>& order);
  std::vector<Iterator> split(int stage_id, const Iterator& it,
                              const std::vector<PrimExpr>& lengths,
                              bool inner_to_outer = true);
  Iterator fuse(int stage_id, const std::vector<Iterator>& iters);

  /* Do transform steps
   * Note: The following functions only change loop state but do not change transform_history.
   * We separate these functions out,
   * so you can call them for replay easily given history steps */
  void DoReorderStep(const ReorderStep& step);
  std::vector<Iterator> DoSplitStep(const SplitStep& step);
  Iterator DoFuseStep(const FuseStep& step);

  // General do step functions with a runtime dynamic dispatcher
  void DoStep(const Step& step, const ComputeDAG& dag);
  void DoSteps(const std::vector<Step>& step, const ComputeDAG& dag);

  // Print the state to a string
  std::string ToStr(bool delete_trivial_loop = true) const;

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);

 private:
  // Common function for DoSplitStep and DoFollowSplitStep
  std::vector<Iterator> DoSplitStepCommon(int stage_id, int iter_id,
                                          const std::vector<PrimExpr>& lengths,
                                          bool inner_to_outer);
};

/*! \brief Clean the name of an iterator to make it valid in python code */
inline std::string CleanName(const std::string& str) {
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  return ret;
}

}  // namespace ansor
}  // namespace tvm


// Hash and equal function for State
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
