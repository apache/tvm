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
 * \file auto_scheduler/cost_model.h
 * \brief Cost models that estimate the performance of programs
 */

#ifndef TVM_AUTO_SCHEDULER_COST_MODEL_H_
#define TVM_AUTO_SCHEDULER_COST_MODEL_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/node/node.h>
#include <tvm/runtime/packed_func.h>

#include <vector>

namespace tvm {
namespace auto_scheduler {

using runtime::PackedFunc;
using runtime::TypedPackedFunc;

/*! \brief The base class for cost model */
class CostModelNode : public Object {
 public:
  /*!
   * \brief Update the cost model according to new measurement results (training data).
   * \param inputs The measure inputs
   * \param results The measure results
   */
  virtual void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) = 0;

  /*!
   * \brief Predict the scores of states
   * \param task The search task of states
   * \param states The input states
   * \param scores The predicted scores for all states
   */
  virtual void Predict(const SearchTask& task, const Array<State>& states,
                       std::vector<float>* scores) = 0;

  /*!
   * \brief Predict the scores of all stages in states. This is the breakdown version of `Predict`
   * \param task The search task
   * \param states The input states
   * \param state_scores The predicted scores for all states
   * \param stage_scores The predicted scores for all stages in all stages
   */
  virtual void PredictStages(const SearchTask& task, const Array<State>& states,
                             std::vector<float>* state_scores,
                             std::vector<std::vector<float>>* stage_scores) {
    LOG(FATAL) << "Not implemented";
  }

  /*!
   * \brief Default virtual destructor
   */
  virtual ~CostModelNode() {}

  static constexpr const char* _type_key = "auto_scheduler.CostModel";
  TVM_DECLARE_BASE_OBJECT_INFO(CostModelNode, Object);
};

/*!
 * \brief Managed reference to CostModelNode.
 * \sa CostModelNode
 */
class CostModel : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CostModel, ObjectRef, CostModelNode);
};

/*! \brief The cost model returning random value for all predictions */
class RandomModelNode : public CostModelNode {
 public:
  /*! \brief Pointer to a random number generator function */
  const TypedPackedFunc<void(size_t, void*)>* random_number_func;

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;

  void Predict(const SearchTask& task, const Array<State>& states,
               std::vector<float>* scores) final;

  static constexpr const char* _type_key = "auto_scheduler.RandomModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandomModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to RandomModelNode.
 * \sa RandomModelNode
 */
class RandomModel : public CostModel {
 public:
  RandomModel();
  explicit RandomModel(::tvm::runtime::ObjectPtr<::tvm::runtime::Object> n) : CostModel(n) {}

  RandomModelNode* operator->() const { return static_cast<RandomModelNode*>(data_.get()); }

  TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(RandomModel);
  using ContainerType = RandomModelNode;
};

/*! \brief A wrapper for cost model defined by python code
 *  This class will call functions defined in the python */
class PythonBasedModelNode : public CostModelNode {
 public:
  /*! \brief Pointer to the update function in python */
  PackedFunc update_func;
  /*! \brief Pointer to the predict function in python */
  PackedFunc predict_func;
  /*! \brief Pointer to the predict function in python */
  PackedFunc predict_stage_func;

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;

  void Predict(const SearchTask& task, const Array<State>& states,
               std::vector<float>* scores) final;

  void PredictStages(const SearchTask& task, const Array<State>& states,
                     std::vector<float>* state_scores,
                     std::vector<std::vector<float>>* stage_scores) final;

  static constexpr const char* _type_key = "auto_scheduler.PythonBasedModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(PythonBasedModelNode, CostModelNode);
};

/*!
 * \brief Managed reference to PythonBasedModelNode.
 * \sa PythonBasedModelNode
 */
class PythonBasedModel : public CostModel {
 public:
  /*!
   * \brief The constructor.
   * \param update_func The pointer to the update function defined in python
   * \param predict_func The pointer to the prediction function defined in python
   * \param predict_stage_func The pointer to the prediction function defined in python
   */
  PythonBasedModel(PackedFunc update_func, PackedFunc predict_func, PackedFunc predict_stage_func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PythonBasedModel, CostModel, PythonBasedModelNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_COST_MODEL_H_
