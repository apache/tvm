/*!
 *  Copyright (c) 2020 by Contributors
 * \file ansor/cost_model.h
 * \brief Base class of cost model
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
  virtual void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) = 0;
  virtual void Predict(const SearchTask& task, const std::vector<State>& states,
      std::vector<float>* scores) = 0;
  virtual void PredictStages(const SearchTask& task, const std::vector<State>& states,
                             std::vector<float>* state_scores,
                             std::vector<std::vector<float>>* stage_scores) = 0;

  static constexpr const char *_type_key = "ansor.CostModel";
  TVM_DECLARE_BASE_OBJECT_INFO(CostModelNode, Object);
};
TVM_DEFINE_MUTABLE_NODE_REF(CostModel, CostModelNode);

/*! \brief The cost model returns random value for all predictions */
class RandomModelNode: public CostModelNode {
 public:
  const PackedFunc* random_number_func;

  static CostModel make();

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;
  void Predict(const SearchTask& task, const std::vector<State>& states,
      std::vector<float>* scores) final;
  void PredictStages(const SearchTask& task, const std::vector<State>& states,
                     std::vector<float>* state_scores,
                     std::vector<std::vector<float>>* stage_scores) { ; }

  static constexpr const char *_type_key = "ansor.RandomModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(RandomModelNode, CostModelNode);
};

class MeasureModelNode : public CostModelNode {
 public:
  ProgramMeasurer measurer;

  static CostModel make(Builder builder, Runner runner);

  void Update(const Array<MeasureInput>& inputs, const Array<MeasureResult>& results) final;
  void Predict(const SearchTask& task, const std::vector<State>& states,
               std::vector<float>* scores) final;
  void PredictStages(const SearchTask& task, const std::vector<State>& states,
                     std::vector<float>* state_scores,
                     std::vector<std::vector<float>>* stage_scores) { ; }

  static constexpr const char* _type_key = "ansor.MeasureModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureModelNode, CostModelNode);
};

/*! \brief  A wrapper for cost model defined by python code
 *  This class will call python's function */
class PythonBasedCostModelNode: public CostModelNode {
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

  static constexpr const char *_type_key = "ansor.PythonBasedCostModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(PythonBasedCostModelNode, CostModelNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_COST_MODEL_COST_MODEL_H_
