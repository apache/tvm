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
 * \file auto_schedule/compute_dag.h
 * \brief The TVM Auto-scheduler computational graph and related program analyses.
 *
 * We convert a compute declaration described by `tvm.compute` (could be a single operator or a
 * subgraph) to a ComputeDAG. It keeps the input/output tensors of the compute declaration,
 * a list of all operations in the DAG as well as static analysis results for the DAG (e.g. the
 * total float operation count, consumer/producer relations of each operation stage, whether an
 * operation stage should be tiled/compute inlined ...). These analyses can help the search policy
 * to make decisions during search process.
 * ComputeDAG is also responsible for the interaction between TVM Auto-scheduler `LoopState` and
 * TVM schedule (e.g. applying the `LoopState` transform steps to TVM schedule, providing
 * `LoopState` with extra information got from TVM schedule ...).
 */

#ifndef TVM_AUTO_SCHEDULE_COMPUTE_DAG_H_
#define TVM_AUTO_SCHEDULE_COMPUTE_DAG_H_

#include <tvm/te/schedule.h>

#include <utility>

#include "loop_state.h"

namespace tvm {
namespace auto_schedule {

/*! \brief The TVM Auto-scheduler computational graph and related program analyses. */
class ComputeDAGNode : public Object {
 public:
  /*!
   * \brief Input and output tensors.
   * This is used as the input of `tvm.lower` or `tvm.build`.
   */
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

  static constexpr const char* _type_key = "auto_schedule.ComputeDAG";
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
   * \brief Apply the history transform steps from a State to get a TVM schedule.
   * \param transform_steps Transform steps of a state.
   * \param stages A pointer to a `te::Stage` Array, default to be nullptr.
   * Pass a valid pointer if these information needs to be used outside this function.
   * \param stage_to_axes A pointer to a StageToAxesMap, default to be nullptr.
   * Pass a valid pointer if these information needs to be used outside this function.
   * \return A `te.schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
   */
  std::pair<te::Schedule, Array<te::Tensor>> ApplySteps(
      const Array<Step>& transform_steps, Array<te::Stage>* stages = nullptr,
      StageToAxesMap* stage_to_axes = nullptr) const;

  /*!
   * \brief Print transform steps as equivalent python schedule API.
   * This can be used for debugging.
   * \param transform_steps Transform steps of a state.
   * \return The Python schedule code.
   */
  String PrintStepsAsPython(const Array<Step>& transform_steps) const;

  /*!
   * \brief Fill the correct bound information for a given state by calling ir_pass::InferBound.
   * The states can lose complete bound information after some transform steps (e.g., compute_at).
   * We can call this function to infer and fill all the bound information.
   * This function calls TVM InferBound pass internally to get the bound.
   * The returned state of this function is guaranteed to have complete iterator extent information.
   * \param state The state to.
   * \return The State after inferbound.
   */
  State InferBound(const State& state) const;

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);
};

}  // namespace auto_schedule
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULE_COMPUTE_DAG_H_
