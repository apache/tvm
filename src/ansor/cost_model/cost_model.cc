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

#include "cost_model.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <utility>

namespace tvm {
namespace ansor {

using ::tvm::runtime::NDArray;

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_OBJECT_TYPE(RandomModelNode);
TVM_REGISTER_OBJECT_TYPE(MeasureModelNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedModelNode);

void RandomNumber(TVMArgs args, TVMRetValue* rv) {
  int n = args[0];
  void* data = args[1];
  float* fdata = reinterpret_cast<float*>(data);
  for (int i = 0; i < n; i++) {
    fdata[i] = static_cast<float>(rand_r(nullptr)) / (static_cast<float>(RAND_MAX));
  }
}

RandomModel::RandomModel() {
  ObjectPtr<RandomModelNode> node = make_object<RandomModelNode>();
  node->random_number_func =
      runtime::Registry::Get("ansor.cost_model.random_number");
  if (node->random_number_func == nullptr) {
    LOG(WARNING) << "ansor.cost_model.random_number is not registered, "
                 << "use C++ default random_number func instead.";
    static PackedFunc cost_model_random_number(RandomNumber);
    node->random_number_func = &cost_model_random_number;
  }
  data_ = std::move(node);
}

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

void RandomModelNode::Predict(const SearchTask& task,
                              const std::vector<State>& states,
                              std::vector<float>* scores) {
  scores->resize(states.size());
  (*random_number_func)(states.size(), static_cast<void*>(scores->data()));
}

MeasureModel::MeasureModel(Builder builder, Runner runner) {
  ObjectPtr<MeasureModelNode> node = make_object<MeasureModelNode>();
  node->measurer = ProgramMeasurer(std::move(builder), std::move(runner),
                                   Array<MeasureCallback>(), 0);
  data_ = std::move(node);
}

void MeasureModelNode::Update(const Array<MeasureInput>& inputs,
                              const Array<MeasureResult>& results) {}

void MeasureModelNode::Predict(const SearchTask& task,
                               const std::vector<State>& states,
                               std::vector<float>* scores) {
  std::vector<MeasureInput> inputs;
  std::vector<MeasureResult> results;

  inputs.clear();
  inputs.reserve(states.size());
  for (const auto& state : states) {
    inputs.push_back(MeasureInput(task, state));
  }
  measurer->SilentMeasure(task, inputs, &results);

  scores->clear();
  scores->reserve(results.size());
  for (const auto& res : results) {
    scores->push_back(1.0 / FloatArrayMean(res->costs));
  }
}

PythonBasedModel::PythonBasedModel(PackedFunc update_func,
                                   PackedFunc predict_func,
                                   PackedFunc predict_stage_func) {
  auto node = make_object<PythonBasedModelNode>();
  node->update_func = std::move(update_func);
  node->predict_func = std::move(predict_func);
  node->predict_stage_func = std::move(predict_stage_func);
  data_ = std::move(node);
}

void PythonBasedModelNode::Update(const Array<MeasureInput>& inputs,
                                  const Array<MeasureResult>& results) {
  update_func(inputs, results);
}

void PythonBasedModelNode::Predict(const SearchTask& task,
                                   const std::vector<State>& states,
                                   std::vector<float>* scores) {
  scores->resize(states.size());
  predict_func(task, Array<State>(states.begin(), states.end()),
               static_cast<void*>(scores->data()));
}

void PythonBasedModelNode::PredictStages(const SearchTask& task,
    const std::vector<State>& states, std::vector<float>* state_scores,
    std::vector<std::vector<float>>* stage_scores) {
  int n_states = states.size();
  int n_stages = task->compute_dag.GetInitState()->stages.size();
  std::vector<float> flatten_scores;
  // Allocate sufficient spaces.
  flatten_scores.resize(n_states * n_stages * 2);
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
    int s_length = static_cast<int>(flatten_scores[idx++]);

    if (s_length > 0) {
      std::vector<float> scores;
      int offset = 0;

      if ((*state_scores)[i] > -INFINITY) {
        // If the score is valid. Copy scored stages and assign 0 to placeholder
        // and inlined stages. If the score is 0, meaning this state failed to
        // be lowered. Just bypass to update offset.
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

TVM_REGISTER_GLOBAL("ansor.RandomModel").set_body_typed([]() {
  return RandomModel();
});

TVM_REGISTER_GLOBAL("ansor.PythonBasedModel")
.set_body_typed([](PackedFunc update_func, PackedFunc predict_func,
                   PackedFunc predict_stage_func) {
  return PythonBasedModel(update_func, predict_func,
                          predict_stage_func);
});

}  // namespace ansor
}  // namespace tvm
