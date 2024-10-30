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
 * \file auto_scheduler/search_policy/utils.h
 * \brief Common utilities for search policies.
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_UTILS_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_UTILS_H_

#include <dmlc/common.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/ir/expr.h>
#include <tvm/te/operation.h>

#include <algorithm>
#include <condition_variable>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace auto_scheduler {

/*! \brief Return whether the search task is targeting a CPU. */
inline bool IsCPUTask(const SearchTask& task) {
  return (task)->target->GetTargetDeviceType() == kDLCPU;
}

/*! \brief Return whether the search task is targeting a GPU. */
inline bool IsGPUTask(const SearchTask& task) {
  int device_type = (task)->target->GetTargetDeviceType();
  return device_type == kDLCUDA || device_type == kDLOpenCL || device_type == kDLVulkan ||
         device_type == kDLMetal || device_type == kDLROCM || device_type == kOpenGL;
}

/*! \brief Return whether the search task is targeting a Hexagon. */
inline bool IsHexagonTask(const SearchTask& task) {
  return (task)->target->GetTargetDeviceType() == kDLHexagon;
}

/*! \brief Return whether the search task is targeting a CUDA GPU. */
inline bool IsCUDATask(const SearchTask& task) {
  return (task)->target->GetTargetDeviceType() == kDLCUDA;
}

/*! \brief Return whether the search task is targeting a OpenCL GPU. */
inline bool IsOpenCLTask(const SearchTask& task) {
  return (task)->target->GetTargetDeviceType() == kDLOpenCL;
}

/*! \brief Argsort. Order: largest to smallest */
template <typename T>
inline std::vector<int> Argsort(const std::vector<T>& scores) {
  std::vector<int> index;
  index.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    index.push_back(i);
  }
  auto cmp = [&scores](int l, int r) { return scores[l] > scores[r]; };
  std::sort(index.begin(), index.end(), cmp);
  return index;
}

/*! \brief Convert operation to stage id. */
inline int OperationToStage(const te::Operation& op, const State& state) {
  for (size_t i = 0; i < state->stages.size(); ++i) {
    if (op == state->stages[i]->op) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find op: " << op;
}

/********** Get Parameters **********/

/*! \brief Get an integer from a tvm str Map. */
inline int GetIntParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  ICHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pint = attr_dict[key].as<runtime::Int::ContainerType>();
  ICHECK(pint != nullptr);
  return pint->value;
}

/*! \brief Get a double from a tvm str Map. */
inline double GetDoubleParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  ICHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pdouble = attr_dict[key].as<runtime::Float::ContainerType>();
  ICHECK(pdouble != nullptr);
  return pdouble->value;
}

/*! \brief Get a string from a tvm str Map. */
inline std::string GetStringParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  ICHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  const auto& target = attr_dict[key];
  if (auto pstr = target.as<StringImmNode>()) {
    return pstr->value;
  } else if (auto pstr = target.as<StringObj>()) {
    return pstr->data;
  } else {
    LOG(FATAL) << "Could not convert object " << target << " of type " << target->GetTypeKey()
               << " to string";
  }
}

/*! \brief Get a iterator name set from a tvm str Map. */
inline std::set<std::string> GetIterNameSetParam(const Map<String, ObjectRef>& attr_dict,
                                                 const std::string& key) {
  std::set<std::string> ret;
  ICHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto names = attr_dict[key].as<ArrayNode>();
  ICHECK(names != nullptr);
  for (const auto& name : *names) {
    ret.insert(name.as<StringObj>()->data);
  }
  return ret;
}

/********** Checks with ComputeDAG **********/

/*! \brief Return whether an op is strictly-inlineable. */
inline bool IsStrictlyInlineable(const SearchTask& task, const State& state, int stage_id) {
  if (state->current_compute_dag) {
    return state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.IsStrictlyInlineable(
        state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.IsStrictlyInlineable(state->stages[stage_id]->op);
  }
}

/*! \brief Return whether an op is an output op. */
inline bool IsOutputOp(const SearchTask& task, const State& state, int stage_id) {
  if (state->current_compute_dag) {
    return state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.IsOutput(
        state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.IsOutput(state->stages[stage_id]->op);
  }
}

/*! \brief Return whether an op needs multi level tiling. */
inline bool NeedsMultilevelTiling(const SearchTask& task, const State& state, int stage_id) {
  if (state->current_compute_dag) {
    return state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.NeedsMultiLevelTiling(
        state->stages[stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.NeedsMultiLevelTiling(state->stages[stage_id]->op);
  }
}

/*! \brief Get all consumers for a stage. This function propagates the relation for inlined ops. */
inline std::set<int> GetConsumers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
  std::set<int> ret;

  if (state->current_compute_dag) {
    consumers = state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.GetConsumers(
        state, state->stages[stage_id]->op);
  } else {
    consumers = task->compute_dag->access_analyzer.GetConsumers(state, state->stages[stage_id]->op);
  }

  for (const auto& op : consumers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

/*! \brief Check if a stage has single consumer or all of its consumers share a common root, return
 * the target consumer root or -1. */
inline int GetSingleConsumerId(const SearchTask& task, const State& state, int stage_id) {
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.empty()) {
    return -1;
  }

  if (consumers.size() == 1) {
    return *consumers.begin();
  } else {
    // Check all consumers share a common root
    int common_root_id = -1;
    bool mismatch = false;
    for (const auto& consumer_stage_id : consumers) {
      int root_id = -1;
      if (state->stages[consumer_stage_id]->compute_at == ComputeAtKind::kRoot) {
        root_id = consumer_stage_id;
      } else if (state->stages[consumer_stage_id]->compute_at == ComputeAtKind::kIter) {
        root_id = state->attach_map->stage_to_attach_iter.at(consumer_stage_id).first;
      } else {
        LOG(FATAL) << "Invalid case";
      }

      if (common_root_id == -1) {
        common_root_id = root_id;
      } else {
        if (common_root_id != root_id) {
          mismatch = true;
          break;
        }
      }
    }

    return mismatch ? -1 : common_root_id;
  }
}

/*! \brief Get all producers for a stage. This function propagates the relation for inlined ops. */
inline std::set<int> GetProducers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> producers;
  std::set<int> ret;

  if (state->current_compute_dag) {
    producers = state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.GetProducers(
        state, state->stages[stage_id]->op);
  } else {
    producers = task->compute_dag->access_analyzer.GetProducers(state, state->stages[stage_id]->op);
  }

  for (const auto& op : producers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

/*! \brief Get all producers for a stage. This function DOES NOT propagates the relation for
 * inlined ops. */
inline std::set<int> GetDirectProducers(const SearchTask& task, const State& state, int stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> producers;
  std::set<int> ret;

  if (state->current_compute_dag) {
    producers = state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.GetDirectProducers(
        state->stages[stage_id]->op);
  } else {
    producers = task->compute_dag->access_analyzer.GetDirectProducers(state->stages[stage_id]->op);
  }

  for (const auto& op : producers) {
    ret.insert(OperationToStage(op, state));
  }
  return ret;
}

/*! \brief Get the number of common outer iterators. This function propagates the relation for
 * chains with multiple ops. */
inline int GetNumCommonOuterIterator(const SearchTask& task, const State& state, int stage_id,
                                     int target_stage_id) {
  if (state->current_compute_dag) {
    return state->current_compute_dag.as<ComputeDAGNode>()
        ->access_analyzer.GetNumCommonOuterIterator(state->stages[stage_id]->op,
                                                    state->stages[target_stage_id]->op);
  } else {
    return task->compute_dag->access_analyzer.GetNumCommonOuterIterator(
        state->stages[stage_id]->op, state->stages[target_stage_id]->op);
  }
}

/*! \brief Return whether two ops are elementwise-matched. */
inline bool ElementwiseMatch(const SearchTask& task, const State& state, int stage_id,
                             int target_stage_id) {
  const auto& op = state->stages[stage_id]->op;
  const auto& target_op = state->stages[target_stage_id]->op;
  if (state->current_compute_dag) {
    return state->current_compute_dag.as<ComputeDAGNode>()->access_analyzer.ElementWiseMatch(
        op, target_op);
  } else {
    return task->compute_dag->access_analyzer.ElementWiseMatch(op, target_op);
  }
}

/********** Get informations from Stage/Iterator **********/

/*! \brief Return the extent of an iterator. */
inline int64_t GetExtent(const Iterator& it) {
  if (it->range.defined()) {
    if (auto pint = it->range->extent.as<IntImmNode>()) {
      return pint->value;
    }
  }
  return -1;
}

/*! \brief Compute the product of lengths of all space iters and all reduce iters, respectively. */
inline std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLength(const Stage& stage) {
  int64_t cum_space_len = 1, cum_reduce_len = 1;
  for (const auto& iter : stage->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      cum_space_len *= GetExtent(iter);
    } else if (iter->iter_kind == IteratorKind::kReduction) {
      cum_reduce_len *= GetExtent(iter);
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

/*! \brief Return whether this stage needs rfactor. */
inline bool NeedsRfactor(const SearchTask& task, const State& state, int stage_id) {
  const auto& op = state->stages[stage_id]->op;
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    int cum_space_len, cum_reduce_len;
    std::tie(cum_space_len, cum_reduce_len) =
        GetCumulativeSpaceAndReductionLength(state->stages[stage_id]);

    if (NeedsMultilevelTiling(task, state, stage_id)) {
      // Do not use rfactor if we have enough parallelism on space iters
      if (cum_space_len > cum_reduce_len || cum_space_len > task->hardware_params->num_cores * 16) {
        return false;
      } else {
        return true;
      }
    } else if (cum_reduce_len > 1) {
      // Always try rfactor for reduction ops
      return cum_reduce_len > task->hardware_params->num_cores;
    }
  }

  return false;
}

/*! \brief Return whether the stage has reduce iterators. */
inline bool HasReduceIter(const Stage& stage) {
  for (const auto& iter : stage->iters) {
    if (iter->iter_kind != IteratorKind::kSpatial) {
      return true;
    }
  }
  return false;
}

/*! \brief Return whether the stage has specific annotated iterators. */
inline bool HasAnnotatedIter(const Stage& stage, IteratorAnnotation type) {
  for (const auto& iter : stage->iters) {
    if (iter->annotation == type) {
      return true;
    }
  }
  return false;
}

/*! \brief Return whether the stage has only one consumer and they are elementwise-matched. */
inline bool HasSingleElementwiseMatchedConsumer(const SearchTask& task, const State& state,
                                                int stage_id, int* target_stage_id = nullptr) {
  // Temporal object to be used if the input pointer is nullptr
  int temp_target_stage_id;
  if (target_stage_id == nullptr) {
    target_stage_id = &temp_target_stage_id;
  }
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.size() == 1) {
    *target_stage_id = *consumers.begin();
    if (ElementwiseMatch(task, state, stage_id, *target_stage_id) &&
        (!(HasReduceIter(state->stages[stage_id]) &&
           HasReduceIter(state->stages[*target_stage_id]))) &&
        (!StrEndsWith(state->stages[*target_stage_id]->op->name, ".shared"))) {
      return true;
    }
  }
  return false;
}

/*! \brief Return whether the step changes the number of stages */
inline bool IsStageNumberChangingStep(const Step& step) {
  return step->IsInstance<CacheWriteStepNode>() || step->IsInstance<CacheReadStepNode>() ||
         step->IsInstance<RfactorStepNode>();
}

/*! \brief Return whether the state does cache_read for stage_id. */
inline bool HasCacheReadStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheReadStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

/*! \brief Return whether the state does cache_write for stage_id. */
inline bool HasCacheWriteStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheWriteStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

/*! \brief Return whether the state does rfactor for stage_id. */
inline bool HasRfactorStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<RfactorStepNode>()) {
      if (stage_id == ps->stage_id) {
        return true;
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

/*! \brief Return whether the stage does cross thread reduction. */
inline bool HasCrossThreadReduction(const State& state, int stage_id) {
  std::function<bool(const Stage&)> check_stage = [](const Stage& in_stage) {
    for (const auto& iter : in_stage->iters) {
      if (iter->annotation == IteratorAnnotation::kThreadX &&
          iter->iter_kind == IteratorKind::kReduction) {
        return true;
      }
    }
    return false;
  };

  // Check the stage itself
  if (check_stage(state->stages[stage_id])) {
    return true;
  }

  // Check the attached stages
  for (size_t iter_id = 0; iter_id < state->stages[stage_id]->iters.size(); iter_id++) {
    const auto& res =
        state->attach_map->iter_to_attached_stages.find(std::make_pair(stage_id, iter_id));
    if (res != state->attach_map->iter_to_attached_stages.end()) {
      for (int attached_stage_id : res->second) {
        if (check_stage(state->stages[attached_stage_id])) {
          return true;
        }
      }
    }
  }

  return false;
}

/*! \brief Return whether the stage has been tiled already. */
inline bool IsTiled(const Stage& stage) {
  auto op = stage->op.as<te::ComputeOpNode>();
  ICHECK(op != nullptr);
  return stage->iters.size() != op->axis.size() + op->reduce_axis.size();
}

/*! \brief Extract primitive iterators from a nested fused or splitted iterator's name. */
inline void ExtractOriginalIterators(const std::string& name, std::set<std::string>* rets) {
  size_t last_pos = 0;
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '@' || name[i] == '.') {  // '@' for fuse and '.' for split
      if (!isdigit(name[last_pos]) && name[last_pos] != '@' && name[last_pos] != '.') {
        rets->insert(name.substr(last_pos, i - last_pos));
      }
      last_pos = i + 1;
    }
  }

  if (last_pos < name.size() && !isdigit(name[last_pos]) && name[last_pos] != '@' &&
      name[last_pos] != '.') {
    rets->insert(name.substr(last_pos, name.size() - last_pos));
  }
}

/*! \brief Get the last reduce iterator in the outermost reduce tile. */
inline Iterator GetLastReduceIteratorInOutermostReduceTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  std::set<std::string> original_names;

  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  size_t reduce_axis_size = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_axis_size++;
    }
  }
  if (reduce_axis_size) {
    for (const auto& iter : stage->iters) {
      if (iter->iter_kind == IteratorKind::kReduction) {
        ExtractOriginalIterators(iter->name, &original_names);
        if (original_names.size() == reduce_axis_size) {
          return iter;
        }
      }
    }
  } else {
    // Return the first reduce iterator
    for (const auto& iter : stage->iters) {
      if (iter->iter_kind == IteratorKind::kReduction) {
        return iter;
      }
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
}

/*! \brief Get the target stage id of a history step in the new state.
 * We need this because the stage_id in the history may be stale due to later steps */
inline int GetTargetStageIDInState(const State& s, int step_id) {
  int stage_inc = 0;

  for (size_t i = step_id + 1; i < s->transform_steps.size(); ++i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (s->transform_steps[i]->stage_id <= s->transform_steps[step_id]->stage_id + stage_inc)
        stage_inc++;
    }
  }
  return s->transform_steps[step_id]->stage_id + stage_inc;
}

/*! \brief Get all split steps for one stage. */
inline void GetSplitStepIds(const State& s, int stage_id, std::vector<int>* split_step_ids) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        split_step_ids->push_back(i);
      }
    }

    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
}

/*! \brief Fuse all reduction iterators. */
inline State FuseAllReductionIterators(const State& state, int stage_id, Iterator* fused_iter,
                                       Array<Iterator>* space_iters,
                                       Array<Iterator>* reduce_iters) {
  space_iters->clear();
  reduce_iters->clear();

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      space_iters->push_back(iter);
    } else if (iter->iter_kind == IteratorKind::kReduction) {
      reduce_iters->push_back(iter);
    }
  }

  ICHECK(!reduce_iters->empty());
  State tmp_s = state;
  if (reduce_iters->size() > 1) {
    *fused_iter = tmp_s.fuse(stage_id, *reduce_iters);
  } else {
    *fused_iter = (*reduce_iters)[0];
  }
  return tmp_s;
}

/*! \brief Fuse all outer level space iterators. */
inline State FuseAllOuterSpaceIterators(const State& state, int stage_id, Iterator* fused_iter) {
  std::vector<Iterator> to_fuse;
  for (size_t iter_id = 0; iter_id < state->stages[stage_id]->iters.size(); ++iter_id) {
    const auto& it = state->stages[stage_id]->iters[iter_id];
    // Stop at reduce iterator or annotated iterator
    if (it->iter_kind == IteratorKind::kReduction || it->annotation != IteratorAnnotation::kNone) {
      break;
    }
    // Stop at compute_at attach point
    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id - 1))) {
      break;
    }
    to_fuse.push_back(it);
  }

  State tmp_s = state;
  if (to_fuse.size() == 1) {
    *fused_iter = to_fuse[0];
  } else {
    *fused_iter = tmp_s.fuse(stage_id, to_fuse);
  }
  return tmp_s;
}

/*! \brief Random sample states. */
inline Array<State> RandomSampleStates(const Array<State>& in_states, std::mt19937* random_gen,
                                       size_t out_size) {
  Array<State> out_states;
  for (size_t i = 0; i < out_size; i++) {
    out_states.push_back(in_states[(*random_gen)() % in_states.size()]);
  }
  return out_states;
}

/*! \brief Compute prefix-sum probability based on the given weights */
inline void ComputePrefixSumProb(const std::vector<float>& weights,
                                 std::vector<double>* prefix_sum_probs) {
  // Compute selection probabilities.
  float sum = 0.0;
  prefix_sum_probs->resize(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    sum += std::max(weights[i], 0.0f);
    (*prefix_sum_probs)[i] = sum;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    (*prefix_sum_probs)[i] /= sum;
  }
}

/*! \brief Random choose an index according to a prefix sum probability. */
inline int RandomChoose(const std::vector<double>& prefix_sum_probs, std::mt19937* random_gen) {
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double x = dis(*random_gen);

  ICHECK(!prefix_sum_probs.empty());

  return std::lower_bound(prefix_sum_probs.begin(), prefix_sum_probs.end(), x) -
         prefix_sum_probs.begin();
}

/*! \brief Print a title */
inline void PrintTitle(const std::string& title, int verbose) {
  StdCout(verbose) << Chars('-', 70) << "\n"
                   << Chars('-', 30) << "  [ " << title << " ]\n"
                   << Chars('-', 70) << std::endl;
}

/*!
 * \brief Enumerate all possible factorization schemes for splitting an axes.
 * \note This class will memorize the results for reuse.
 */
class SplitFactorizationMemo {
 public:
  using QueryKey = std::tuple<int, int, int>;

  const Array<Array<Integer>>& GetFactorizationSchemes(int extent, int n_lengths,
                                                       int max_innermost_factor);
  const std::vector<int>& GetFactors(int n);

 private:
  void DfsEnumerate(int now, int remaining_length, int max_innermost_factor);

  std::unordered_map<QueryKey, Array<Array<Integer>>> memory_;

  int n_lengths_;
  Array<Integer> tmp_stack_;
  Array<Array<Integer>>* results_;
  std::unordered_map<int, std::vector<int>> factor_memory_;
};

/*! \brief Get the indexes of SplitStep that processes on spatial iterator. */
Array<Integer> GetSpatialSplitStepIds(const State& s, int stage_id);

/*! \brief Get the possible compute locations for a stage. */
std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id);

// Apply multi-level tiling structure according to a string format,
// where "S" stands a space level, "R" stands for a reduction level.
// For example, if the format is "SSRSRS", then we will
// use tiling structure:  space_L0, space_L1, reduce_L0, space_L2, reduce_L1, space_L3
// For example, if apply "SSRSRS" to matrix multiplication,
// we have space iterators i and j, reduce iterator k.
// Then the tiling structure is : i0, j0, i1, j1, k0, i2, j2, k1, i3, j3
State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids = nullptr);

// Apply tiling structure: space, space, space, ..., with tile sizes from other SplitStep
State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split);

// Prune invalid states and return the results in-place.
void PruneInvalidState(const SearchTask& task, Array<State>* states);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_UTILS_H_
