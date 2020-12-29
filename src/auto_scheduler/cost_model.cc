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
 * \file auto_scheduler/cost_model.cc
 * \brief Cost models that estimate the performance of programs
 */

#include <tvm/auto_scheduler/cost_model.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_OBJECT_TYPE(RandomModelNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedModelNode);

RandomModel::RandomModel() {
  ObjectPtr<RandomModelNode> node = make_object<RandomModelNode>();
  const auto* f = runtime::Registry::Get("auto_scheduler.cost_model.random_fill_float");
  ICHECK(f != nullptr);
  node->random_number_func = reinterpret_cast<const TypedPackedFunc<void(size_t, void*)>*>(f);
  data_ = std::move(node);
}

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

void RandomModelNode::Predict(const SearchTask& task, const Array<State>& states,
                              std::vector<float>* scores) {
  scores->resize(states.size());
  (*random_number_func)(states.size(), static_cast<void*>(scores->data()));
}

PythonBasedModel::PythonBasedModel(PackedFunc update_func, PackedFunc predict_func,
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

void PythonBasedModelNode::Predict(const SearchTask& task, const Array<State>& states,
                                   std::vector<float>* scores) {
  scores->resize(states.size());
  predict_func(task, states, static_cast<void*>(scores->data()));
}

void PythonBasedModelNode::PredictStages(const SearchTask& task, const Array<State>& states,
                                         std::vector<float>* state_scores,
                                         std::vector<std::vector<float>>* stage_scores) {
  size_t n_states = states.size();
  size_t n_stages = task->compute_dag->init_state->stages.size();
  std::vector<float> flatten_scores;
  // Allocate sufficient spaces.
  flatten_scores.resize(n_states * n_stages * 2);
  predict_stage_func(task, states, static_cast<void*>(flatten_scores.data()));

  /* For faster data copy between c++ and python, the python part returns scores in a
   * single flatten array using a packed format. The c++ part then unpacks the flatten array.
   *
   * The packed format is:
   * {
   *   float  scores[N];                 // scores[i] is the score for states[i].
   *   int    n_stage_0;                 // the number of stages in states[0]
   *   float  stage_scores_0[[n_stage_0] // the scores for all stages in states[0]
   *   int    n_stage_1;                 // the number of stages in states[1]
   *   float  stage_scores_1[n_stage_1]; // the scores for all stages in states[1]
   *   ...
   *   int    n_stage_i;                 // the number of stages in states[i]
   *   float  stage_scores_1[n_stage_i]; // the scores for all stages in states[i]
   *   ...  // until i == N - 1
   * }
   * To implement this format, we also store int as float, so we can store all numbers
   * into a single float array.
   */

  // Unpack flatten scores.
  state_scores->clear();
  stage_scores->clear();

  // Score of each states.
  for (size_t i = 0; i < n_states; ++i) {
    state_scores->push_back(flatten_scores[i]);
  }

  // Score of each stage in each states.
  size_t idx = n_states;
  for (size_t i = 0; i < n_states; ++i) {
    ICHECK_LE(idx, flatten_scores.size());

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
          if (stage->op_type == StageKind::kPlaceholder) {
            scores.push_back(0);
            continue;
          }
          if (stage->compute_at == ComputeAtKind::kInlined) {
            scores.push_back(0);
            continue;
          }
          scores.push_back(flatten_scores[idx + offset]);
          offset++;
        }
        ICHECK_EQ(offset, s_length);
        stage_scores->push_back(std::move(scores));
      }
      idx += s_length;
    } else {
      // Cost model does not provide any stage score details.
      stage_scores->push_back({});
    }
  }
}

TVM_REGISTER_GLOBAL("auto_scheduler.RandomModel").set_body_typed([]() { return RandomModel(); });

TVM_REGISTER_GLOBAL("auto_scheduler.PythonBasedModel")
    .set_body_typed([](PackedFunc update_func, PackedFunc predict_func,
                       PackedFunc predict_stage_func) {
      return PythonBasedModel(update_func, predict_func, predict_stage_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.CostModelUpdate")
    .set_body_typed([](CostModel model, Array<MeasureInput> inputs, Array<MeasureResult> results) {
      model->Update(inputs, results);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.CostModelPredict")
    .set_body_typed([](CostModel model, SearchTask task, Array<State> states) {
      std::vector<float> scores;
      model->Predict(task, states, &scores);
      Array<FloatImm> ret;
      for (auto x : scores) {
        ret.push_back(FloatImm(DataType::Float(32), x));
      }
      return ret;
    });

}  // namespace auto_scheduler
}  // namespace tvm
