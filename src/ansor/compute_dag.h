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
 * \brief Compute declaration graph and its related analysis tools.
 * ComputeDAG is also responsible for the interaction with the original TVM schedule system, to
 * apply state to a runable TVM schedule or provide the schedule Python code.
 */

#ifndef TVM_ANSOR_COMPUTE_DAG_H_
#define TVM_ANSOR_COMPUTE_DAG_H_

#include <tvm/te/schedule.h>

#include <utility>

#include "loop_state.h"

namespace tvm {
namespace ansor {

/*!
 * \brief Update stage and axes mapping during replay.
 * \param stage A `te::Stage`.
 * \param stage_to_axes A pointer to StageToAxesMap.
 */
void UpdateStageAxis(const tvm::te::Stage& stage, StageToAxesMap* stage_to_axes);

/*! \brief Computation declaration graph. */
class ComputeDAGNode : public Object {
 public:
  /*! \brief Input and output tensors. */
  Array<te::Tensor> tensors;
  /*! \brief All related operations in topo order. */
  Array<te::Operation> ops;
  /*! \brief Number of total float operations for this ComputeDAG. */
  double flop_ct;
  /*! \brief The initial state without any transform steps. */
  State init_state;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensors", &tensors);
    v->Visit("ops", &ops);
    v->Visit("flop_ct", &flop_ct);
    v->Visit("init_state", &init_state);
  }

  static constexpr const char* _type_key = "ansor.ComputeDAG";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeDAGNode, Object);
};

/*!
 * \brief Managed reference to ComputeDAGNode.
 * \sa ComputeDAGNode
 */
class ComputeDAG : public ObjectRef {
 public:
  /*! \brief The constructor.
   * \param tensors `te::Tensor`s for a compute declaration.
   */
  explicit ComputeDAG(Array<te::Tensor> tensors);

  /*!
   * \brief Apply transform steps to the init state of this DAG, and get the
   * equivalent `tvm::schedule`.
   * \param transform_steps Transform steps of the target state.
   * \return The return values can be used as arguments to `tvm.build` or `tvm.lower`.
   */
  std::pair<te::Schedule, Array<te::Tensor> > ApplySteps(const Array<Step>& transform_steps) const;
  /*!
   * \brief Print transform steps as equivalent python schedule API.
   * \param transform_steps Transform steps of the target state.
   * \return Python schedule code.
   */
  String PrintStepsAsPython(const Array<Step>& transform_steps) const;

  /*!
   * \brief Replay the transform steps and call ir_pass::InferBound to fill correct bound
   * information.
   * State api supports to define a split step with its split factor to be a blank placeholder,
   * so sometimes we may get a State will incomplete iterator extent information.
   * And another situation is after some steps (for exp. compute_at), it may be hard to track the
   * extent change of all iterators.
   * We perform infer bound using TVM schedule and fill the State with those informations. After
   * applying this methods, the State is guaranteed to have complete interator extent information.
   * \param transform_steps Transform steps of the target state.
   * \return The State after inferbound.
   */
  State ReplayAndInferBound(const Array<Step>& transform_steps) const;
  /*!
   * \brief Fill the correct bound information for a given state by calling ir_pass::InferBound.
   * \param state The target state.
   * \return The State after inferbound.
   */
  State InferBound(const State& state) const;
  /*!
   * \brief Fill the correct bound information for a list of given states.
   * Return the new states inplace.
   * \param states A pointer to a State vector, States are updated inplace.
   */
  void InferBound(Array<State>* states) const;

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);

 private:
  /*!
   * \brief Internal common parts for replaying steps. This is the key method to apply steps to
   * TVM schedule.
   * \param transform_steps Transform steps of the target state.
   * \param stages A pointer to `te::Stage` vector.
   * \param stage_to_axes A pointer to StageToAxesMap.
   * \return The return values can be used as arguments to `tvm.build` or `tvm.lower`.
   */
  std::pair<te::Schedule, Array<te::Tensor> > ReplaySteps(const Array<Step>& transform_steps,
                                                          Array<te::Stage>* stages,
                                                          StageToAxesMap* stage_to_axes) const;

  /*!
   * \brief Internal common parts for inferring bound.
   * \param pstate A pointer to StateNode, the target state will be updated with filled
   * bound information.
   */
  void InferBoundCommon(StateNode* pstate) const;
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_COMPUTE_DAG_H_
