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
 * \file ansor/compute_dag.h
 * \brief Compute declaration graph and its related analysis tools
 */

#ifndef TVM_ANSOR_COMPUTE_DAG_H_
#define TVM_ANSOR_COMPUTE_DAG_H_

#include <tvm/node/node.h>
#include <tvm/te/schedule.h>
#include <utility>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "utils.h"

namespace tvm {
namespace ansor {

class ComputeDAG; class AccessAnalyzer;
class StateNode; class State; class Step;

/*! \brief Read/Write access static analysis result */
class AccessAnalyzerNode : public Object {
 public:
  template<class T>
  using OperationMap = std::unordered_map<te::Operation, T, ObjectHash, ObjectEqual>;

  OperationMap<OperationMap<std::vector<std::vector<PrimExpr> > > > read_from;
  OperationMap<OperationMap<std::vector<std::vector<PrimExpr> > > > read_by;
  OperationMap<bool> is_injective;
  OperationMap<bool> is_strict_inlineable;
  OperationMap<bool> needs_multi_level_tiling;
  OperationMap<bool> is_output;
  std::vector<te::Operation> ops_topo_order;

  static AccessAnalyzer make(const Array<te::Tensor>& tensors);

  static constexpr const char* _type_key = "ansor.AccessAnalyzer";
  TVM_DECLARE_FINAL_OBJECT_INFO(AccessAnalyzerNode, Object);
};

/*! \brief Read/Write access static analysis result */
class AccessAnalyzer : public ObjectRef {
 public:
  // read/write access analysis
  bool NeedsMultiLevelTiling(const te::Operation& op) const;
  bool IsInjective(const te::Operation& op) const;
  bool IsStrictInlineable(const te::Operation& op) const;
  bool IsOutput(const te::Operation& op) const;

  // Get all producers of an op
  void GetProducers(const State& state, const te::Operation& op,
      std::unordered_set<te::Operation, ObjectHash, ObjectEqual>* producers) const;

  // Get all consumers of an op. This func deals with inlined op correctly.
  void GetConsumers(const State& state, const te::Operation& op,
      std::unordered_set<te::Operation, ObjectHash, ObjectEqual>* consumers) const;

  // Check whether two ops are elementwise matched
  // (e.g. conv2d and relu are elementwise matched)
  bool ElementWiseMatch(const te::Operation& op,
                        const te::Operation& target_op) const;

  /*! \Note The current implementation follows these (rough) definitions.
   *
   * Definition of data-reuse : Exists axis in (op->axis union op->reduce_axis)
   *   and acc in read accesses, such that axis not in acc.
   *   (e.g. A[i][j] = B[i] has data reuse, while A[i][j] = B[i][j] does not)
   * Definition of NeedsMultiLevelTiling: Exists two acc, both of them make this op have data reuse.
   * Definition of injective : For all index expressions, they are single axis variable
   *  plus an optional const shift.
   *    (e.g. A[i][j] = B[i][j], A[i][j] = B[i+1][j] are injective, while A[i][j] = B[i*j] is not)
   * Definition of strict-inlineable : All read accesses are elementwise, and no branch in the body
   *    (e.g. A[i][j] = B[i][j] + C[i][j] is strict-inlineable,
   *          while A[i][j] = tvm_if_then_else(B[i][j] > 0, C[i][j], 0) is not
   */
  TVM_DEFINE_OBJECT_REF_METHODS(AccessAnalyzer, ObjectRef, AccessAnalyzerNode);
};

typedef std::unordered_map<tvm::te::Stage, std::vector<tir::IterVar>, ObjectHash, ObjectEqual>
    StageToAxesMap;

// Update StageToAxes Map during replay
void UpdateStageAxis(const tvm::te::Stage& stage, StageToAxesMap *stage_to_axes);


/*! \brief Computation declaration graph */
class ComputeDAGNode : public Object {
 public:
  Array<te::Tensor> tensors;       // Input and output tensors
  Array<te::Operation> ops;        // All related operations in topo order
  double flop_ct;                  // Number of float operations
  AccessAnalyzer access_analyzer;  // Read/Write accesss static analyzer
  ObjectRef init_state;            // The initial state

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensors", &tensors);
    v->Visit("ops", &ops);
    v->Visit("flop_ct", &flop_ct);
    v->Visit("access_analyzer", &access_analyzer);
  }

  static ComputeDAG make(Array<te::Tensor> tensors);
  static ComputeDAG make_by_workload_key(const std::string& workload_key);

  static constexpr const char* _type_key = "ansor.ComputeDAG";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeDAGNode, Object);
};

enum LayoutRewriteLevel {
  kNoRewrite = 0,           // No layout rewrite
  kPlaceholderRewrite = 1,  // Only rewrite layout of placeholder in the compute dag
  kComputeRewrite = 2,      // Only rewrite compute body for new layout in the compute dag
  kBothRewrite = 3,         // Rewrite both placeholder and compute body in the compute dag
};

/*! \brief Compute declaration graph */
class ComputeDAG: public ObjectRef {
 public:
  // Apply transform steps to the init state of this DAG, and get the equivalent tvm::schedule.
  // The return values can be used as arguments to tvm.build or tvm.lower
  std::pair<te::Schedule, Array<te::Tensor> > ApplySteps(
      const std::vector<Step>& transform_steps,
      LayoutRewriteLevel layout_rewrite_level = kNoRewrite) const;

  // Rewrite the the layout of "layout free" placeholders according to transform steps
  void RewriteLayout(const std::vector<Step>& transform_steps,
                     LayoutRewriteLevel layout_rewrite_level = kNoRewrite) const;

  // Print transform steps as equivalent python schedule API
  std::string PrintStepsAsPython(const std::vector<Step>& steps) const;

  // Replay the transform steps and call ir_pass::InferBound to fill correct bound information
  State ReplayAndInferBound(const std::vector<Step>& transform_steps) const;

  // Fill the correct bound information for a given state by calling ir_pass::InferBound
  State InferBound(const State& state) const;

  // Fill the correct bound information for a list of given states.
  // Return the new states inplace
  void InferBound(std::vector<State>* states) const;

  // Replay the transform steps and get the new DAG
  void ReplayAndGetDAG(const std::vector<Step>& steps, ComputeDAG* task_dag) const;

  // Get the init state
  State GetInitState() const;

  static constexpr const char* layout_free_placeholders_key = "layout_free_placeholders";

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);

 private:
  // Internal common parts for replaying steps
  std::pair<te::Schedule, Array<te::Tensor> > ReplaySteps(
      const std::vector<Step>& transform_steps, std::vector<te::Stage>* stages,
      StageToAxesMap* stage_to_axes) const;

  // Internal common parts for inferring bound
  void InferBoundCommon(StateNode* pstate) const;
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_COMPUTE_DAG_H_
