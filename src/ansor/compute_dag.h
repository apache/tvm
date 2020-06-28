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

class StateNode; class State; class Step;

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
  ObjectRef init_state;            // The initial state

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensors", &tensors);
    v->Visit("ops", &ops);
    v->Visit("flop_ct", &flop_ct);
  }

  static constexpr const char* _type_key = "ansor.ComputeDAG";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeDAGNode, Object);
};

/*!
 * \brief Managed reference to ComputeDAGNode.
 * \sa ComputeDAGNode
 */
class ComputeDAG: public ObjectRef {
 public:
  explicit ComputeDAG(Array<te::Tensor> tensors);
  explicit ComputeDAG(const std::string& workload_key);

  // Apply transform steps to the init state of this DAG, and get the equivalent tvm::schedule.
  // The return values can be used as arguments to tvm.build or tvm.lower
  std::pair<te::Schedule, Array<te::Tensor> > ApplySteps(
      const std::vector<Step>& transform_steps) const;

  // Print transform steps as equivalent python schedule API
  std::string PrintStepsAsPython(const std::vector<Step>& steps) const;

  // Replay the transform steps and call ir_pass::InferBound to fill correct bound information
  State ReplayAndInferBound(const std::vector<Step>& transform_steps) const;

  // Fill the correct bound information for a given state by calling ir_pass::InferBound
  State InferBound(const State& state) const;

  // Fill the correct bound information for a list of given states.
  // Return the new states inplace
  void InferBound(std::vector<State>* states) const;

  // Get the init state
  State GetInitState() const;

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
