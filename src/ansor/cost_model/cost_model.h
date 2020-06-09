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
 * \file ansor/cost_model.h
 * \brief Cost model that estimates the performance of programs
*/

#ifndef TVM_ANSOR_COST_MODEL_COST_MODEL_H_
#define TVM_ANSOR_COST_MODEL_COST_MODEL_H_

#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <tvm/runtime/packed_func.h>
#include <vector>
#include "../measure.h"

namespace tvm {
namespace ansor {

using runtime::PackedFunc;

class CostModel;

/*! \brief The base class for cost model */
class CostModelNode: public Object {
 public:
  // Update the cost model according to new measurement pairs
  virtual void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) = 0;

  // Predict the scores of states
  virtual void Predict(const SearchTask& task, const std::vector<State>& states,
      std::vector<float>* scores) = 0;

  // Predict the scores of all stages in states
  virtual void PredictStages(const SearchTask& task, const std::vector<State>& states,
                             std::vector<float>* state_scores,
                             std::vector<std::vector<float>>* stage_scores) {
    LOG(FATAL) << "Not Implemented";
  }

  static constexpr const char *_type_key = "ansor.CostModel";
  TVM_DECLARE_BASE_OBJECT_INFO(CostModelNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(CostModel, CostModelNode);

/*! \brief The cost model returns random value for all predictions */
class RandomModelNode: public CostModelNode {
 public:
  const PackedFunc* random_number_func;

  static CostModel make();

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;
  void Predict(const SearchTask& task, const std::vector<State>& states,
      std::vector<float>* scores) final;

  static constexpr const char *_type_key = "ansor.RandomModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandomModelNode, CostModelNode);
};

/*! \brief The cost model returns actual cost by measurement */
class MeasureModelNode : public CostModelNode {
 public:
  ProgramMeasurer measurer;

  static CostModel make(Builder builder, Runner runner);

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;
  void Predict(const SearchTask& task, const std::vector<State>& states,
               std::vector<float>* scores) final;

  static constexpr const char* _type_key = "ansor.MeasureModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureModelNode, CostModelNode);
};

/*! \brief  A wrapper for cost model defined by python code
 *  This class will call python's function */
class PythonBasedModelNode: public CostModelNode {
 public:
  PackedFunc update_func;
  PackedFunc predict_func;
  PackedFunc predict_stage_func;

  static CostModel make(PackedFunc update_func, PackedFunc predict_func,
                        PackedFunc predict_stage_func);

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;
  void Predict(const SearchTask& task, const std::vector<State>& states,
      std::vector<float>* scores) final;
  void PredictStages(const SearchTask& task, const std::vector<State>& states,
                     std::vector<float>* state_scores,
                     std::vector<std::vector<float>>* stage_scores) final;

  static constexpr const char *_type_key = "ansor.PythonBasedModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(PythonBasedModelNode, CostModelNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_COST_MODEL_COST_MODEL_H_
