/*!
 *  Copyright (c) 2020 by Contributors
 */
#include "cost_model.h"
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <utility>

namespace tvm {
namespace ansor {

using ::tvm::runtime::NDArray;

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_OBJECT_TYPE(RandomModelNode);
TVM_REGISTER_OBJECT_TYPE(MeasureModelNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedCostModelNode);

void RandomNumber(TVMArgs args, TVMRetValue* rv) {
  int n = args[0];
  void* data = args[1];
  float* fdata = reinterpret_cast<float*>(data);
  for (int i = 0; i < n; i++) {
    fdata[i] = static_cast<float>(rand_r(0)) / (static_cast<float>(RAND_MAX));
  }
}

CostModel RandomModelNode::make() {
  ObjectPtr<RandomModelNode> node = make_object<RandomModelNode>();
  node->random_number_func =
      runtime::Registry::Get("ansor.cost_model.random_number");
  if (node->random_number_func == nullptr) {
    LOG(WARNING) << "ansor.cost_model.random_number is not registered, "
                 << "use C++ default random_number func instead.";
    static PackedFunc cost_model_random_number(RandomNumber);
    node->random_number_func = &cost_model_random_number;
  }
  return CostModel(node);
}

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
    const Array<MeasureResult>& results) {
}

void RandomModelNode::Predict(const SearchTask& task,
                              const std::vector<State>& states,
                              std::vector<float>* scores) {
  scores->resize(states.size());
  (*random_number_func)(states.size(), static_cast<void*>(scores->data()));
}

CostModel MeasureModelNode::make(Builder builder, Runner runner) {
  ObjectPtr<MeasureModelNode> node = make_object<MeasureModelNode>();
  node->measurer = ProgramMeasurerNode::make(std::move(builder), std::move(runner),
                                             Array<MeasureCallback>(), 0);
  return CostModel(node);
}

void MeasureModelNode::Update(const Array<MeasureInput>& inputs,
    const Array<MeasureResult>& results) {
}

void MeasureModelNode::Predict(const SearchTask& task,
                               const std::vector<State>& states,
                               std::vector<float>* scores) {
  std::vector<MeasureInput> inputs;
  std::vector<MeasureResult> results;

  inputs.clear(); inputs.reserve(states.size());
  for (const auto& state : states) {
    inputs.push_back(MeasureInputNode::make(task, state));
  }
  measurer->SilentMeasure(task, inputs, &results);

  scores->clear();
  scores->reserve(results.size());
  for (const auto& res : results) {
    scores->push_back(1.0 / FloatArrayMean(res->costs));
  }
}

CostModel PythonBasedCostModelNode::make(PackedFunc update_func, PackedFunc predict_func,
                                         PackedFunc predict_stage_func) {
  auto node = make_object<PythonBasedCostModelNode>();
  node->update_func = std::move(update_func);
  node->predict_func = std::move(predict_func);
  node->predict_stage_func = std::move(predict_stage_func);
  return CostModel(node);
}

void PythonBasedCostModelNode::Update(const Array<MeasureInput>& inputs,
                                      const Array<MeasureResult>& results)  {
  update_func(inputs, results);
}

void PythonBasedCostModelNode::Predict(const SearchTask& task,
                                       const std::vector<State>& states,
                                       std::vector<float>* scores) {
  scores->resize(states.size());
  predict_func(task, Array<State>(states.begin(), states.end()),
               static_cast<void*>(scores->data()));
}

void PythonBasedCostModelNode::PredictStages(const SearchTask& task,
                                             const std::vector<State>& states,
                                             std::vector<float>* state_scores,
                                             std::vector<std::vector<float>>* stage_scores) {
  int n_states = states.size();
  int n_stages = task->compute_dag.GetInitState()->stages.size();
  std::vector<float> flatten_scores;
  flatten_scores.resize(n_states * n_stages * 2);  // Allocate sufficient spaces.
  predict_stage_func(task, Array<State>(states.begin(), states.end()),
                     static_cast<void*>(flatten_scores.data()));

  // Unpack flatten scores.
  state_scores->clear();
  stage_scores->clear();

  // Score of each states.
  for (int i = 0; i < n_states; ++i) {
    state_scores->push_back(flatten_scores[i]);
  }

  // Score of each stage in each states.
  size_t idx = n_states;
  for (int i = 0; i < n_states; ++i) {
    CHECK_LE(idx, flatten_scores.size());

    // Number of scored stages of this state.
    int s_length = (int)flatten_scores[idx++];

    if (s_length > 0) {
      std::vector<float> scores;
      int offset = 0;

      if ((*state_scores)[i] > -INFINITY) {
        // If the score is valid. Copy scored stages and assign 0 to placeholder and inlined stages.
        // If the score is 0, meaning this state failed to be lowered. Just bypass to update offset.
        for (const Stage& stage : states[i]->stages) {
          if (stage->op_type == kPlaceholder) {
            scores.push_back(0);
            continue;
          }
          if (stage->compute_at == kInlined) {
            scores.push_back(0);
            continue;
          }
          scores.push_back(flatten_scores[idx + offset]);
          offset++;
        }
        CHECK_EQ(offset, s_length);
        stage_scores->push_back(std::move(scores));
      }
      idx += s_length;
    } else {
      // Cost model does not provide any stage score details.
      stage_scores->push_back({});
    }
  }
}

}  // namespace ansor
}  // namespace tvm
