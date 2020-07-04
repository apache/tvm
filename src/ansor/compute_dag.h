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
 * \brief The Ansor computational graph and related program analyses.
 *
 * We convert a compute declaration described by `tvm.compute` (could be a single operator or a
 * subgraph) to a ComputeDAG. It keeps the input/output tensors of the target compute declaration,
 * a list of all related operations in topo order as well as a set of analyses over each operation
 * stage (e.g. the total float operation count, consumer/producer relations of each operation
 * stage, whether a operation stage should be tiled/compute inlined ...). These analyses can
 * help the search policy to do some specific decisions during schedule search process.
 *
 * ComputeDAG is also responsible for the interaction between Ansor LoopState and TVM schedule
 * (e.g. applying the LoopState transform steps to TVM schedule, providing LoopState with extra
 * information get from TVM schedule ...).
 */

#ifndef TVM_ANSOR_COMPUTE_DAG_H_
#define TVM_ANSOR_COMPUTE_DAG_H_

#include <tvm/te/schedule.h>

#include <utility>

#include "loop_state.h"

namespace tvm {
namespace ansor {

/*! \brief The Ansor computational graph and related program analyses. */
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
  // TODO(merrymercy): Add more analyses later.

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
   * \param stages A pointer to a `te::Stage` Array, default to be nullptr.
   * Pass a valid pointer if these information needs to be used outside this function.
   * \param stage_to_axes A pointer to a StageToAxesMap, default to be nullptr.
   * Pass a valid pointer if these information needs to be used outside this function.
   * \return The return values can be used as arguments to `tvm.build` or `tvm.lower`.
   */
  std::pair<te::Schedule, Array<te::Tensor> > ApplySteps(
      const Array<Step>& transform_steps, Array<te::Stage>* stages = nullptr,
      StageToAxesMap* stage_to_axes = nullptr) const;

  /*!
   * \brief Print transform steps as equivalent python schedule API.
   * \param transform_steps Transform steps of the target state.
   * \return Python schedule code.
   */
  String PrintStepsAsPython(const Array<Step>& transform_steps) const;

  /*!
   * \brief Fill the correct bound information for a given state by calling ir_pass::InferBound.
   * \param state The target state.
   * \return The State after inferbound.
   */
  State InferBound(const State& state) const;
  /*!
   * \brief Fill the correct bound information for a list of given states.
   * Return the new states inplace.
   * \param states A pointer to a State Array, States are updated inplace.
   */
  void InferBound(Array<State>* states) const;

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_COMPUTE_DAG_H_
