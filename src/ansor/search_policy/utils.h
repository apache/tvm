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
 * \file ansor/search_policy/utils.cc
 * \brief Common utilities for search policies
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_UTILS_H_
#define TVM_ANSOR_SEARCH_POLICY_UTILS_H_

#include <tvm/te/operation.h>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../cost_model/cost_model.h"
#include "../utils.h"
#include "../loop_state.h"
#include "../transform_step.h"
#include "search_policy.h"

namespace tvm {
namespace ansor {

// Get an integer from a tvm str Map
inline int GetIntParam(const Map<String, ObjectRef>& attr_dict,
                       const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pint = attr_dict[key].as<IntImmNode>();
  CHECK(pint != nullptr);
  return pint->value;
}

// Get a double from a tvm str Map
inline double GetDoubleParam(const Map<String, ObjectRef>& attr_dict,
                             const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pdouble = attr_dict[key].as<FloatImmNode>();
  CHECK(pdouble != nullptr);
  return pdouble->value;
}

// Get a string from a tvm str Map
inline std::string GetStringParam(const Map<String, ObjectRef>& attr_dict,
                                  const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0)
      << "Cannot find key: \"" << key << "\" in " << attr_dict;
  const auto& target = attr_dict[key];
  if (auto pstr = target.as<StringImmNode>()) {
    return pstr->value;
  }
  auto pstr = target.as<StringObj>();
  CHECK(pstr != nullptr);
  return pstr->data;
}

// Get a iterator name set from a tvm str Map
inline std::set<std::string> GetIterNameSetParam(const Map<String, ObjectRef>& attr_dict,
                                                 const std::string& key) {
  std::set<std::string> ret;
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto names = attr_dict[key].as<ArrayNode>();
  CHECK(names != nullptr);
  for (const auto & name : *names) {
    ret.insert(name.as<StringImmNode>()->value);
  }
  return ret;
}

// Convert operation to stage id
inline int OperationToStage(const te::Operation& op, const State& state) {
  for (size_t i = 0; i < state->stages.size(); ++i) {
    if (op == state->stages[i]->op) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find op: " << op;
  return -1;
}

// Return the extent of an iterator
inline int64_t GetExtent(const Iterator& it) {
  if (it->range.defined()) {
    if (auto pint = it->range->extent.as<IntImmNode>()) {
      return pint->value;
    }
  }
  return -1;
}

// Return whether an op is strict inlineable
inline bool IsStrictInlineable(const SearchTask& task,
    const State& state, const te::Operation& op) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.IsStrictInlineable(op);
  } else {
    return task->compute_dag->access_analyzer.IsStrictInlineable(op);
  }
}

// Return whether an op is an output op
inline bool IsOutputOp(const SearchTask& task, const State& state, const te::Operation& op) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.IsOutput(op);
  } else {
    return task->compute_dag->access_analyzer.IsOutput(op);
  }
}

// Return whether the stage has an attribute flag
inline bool HasAttrsFlag(const State& state, int stage_id, const char* target) {
  if (state->stages[stage_id]->op->attrs.count(target)) {
    return GetStringParam(state->stages[stage_id]->op->attrs, target) == "True";
  }
  return false;
}

// Return whether the stage has reduce iterators
inline bool HasReduceIter(const Stage& stage) {
  for (const auto& iter : stage->iters) {
    if (iter->iter_type != kSpace) {
      return true;
    }
  }
  return false;
}

// Return whether the stage has specific annotated iterators
inline bool HasAnnotationIter(const Stage& stage, IteratorAnnotation type) {
  for (const auto& iter : stage->iters) {
    if (iter->annotation == type) {
      return true;
    }
  }
  return false;
}

// Return whether an op needs multi level tiling
inline bool NeedsMultilevelTiling(const SearchTask& task,
    const State& state, const te::Operation& op) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.NeedsMultiLevelTiling(op);
  } else {
    return task->compute_dag->access_analyzer.NeedsMultiLevelTiling(op);
  }
}

// Get all consumers for an op. This will take inline into consideration
inline void GetConsumers(const SearchTask& task, const State& state, const te::Operation& op,
    std::unordered_set<te::Operation, ObjectHash, ObjectEqual>* consumers) {
  if (state->task_dag.defined()) {
    state->task_dag->access_analyzer.GetConsumers(state, op, consumers);
  } else {
    task->compute_dag->access_analyzer.GetConsumers(state, op, consumers);
  }
}

inline void GetProducers(const SearchTask& task, const State& state, const te::Operation& op,
    std::unordered_set<te::Operation, ObjectHash, ObjectEqual>* producers) {
  if (state->task_dag.defined()) {
    state->task_dag->access_analyzer.GetProducers(state, op, producers);
  } else {
    task->compute_dag->access_analyzer.GetProducers(state, op, producers);
  }
}

// Return whether two ops are elementwise-matched
inline bool ElementwiseMatch(const SearchTask& task, const State& state, const te::Operation& op,
                             const te::Operation& target_op) {
  if (state->task_dag.defined()) {
    return state->task_dag->access_analyzer.ElementWiseMatch(op, target_op);
  } else {
    return task->compute_dag->access_analyzer.ElementWiseMatch(op, target_op);
  }
}

// Return whether the stage has only one consumer and they are elementwise-matched
inline bool HasSingleElementwiseMatchedConsumer(const SearchTask& task,
    const State& state, const Stage& stage, int* target_stage_id) {
  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;

  GetConsumers(task, state, stage->op, &consumers);
  if (consumers.size() == 1) {
    *target_stage_id = OperationToStage(*consumers.begin(), state);
    const Stage& target_stage = state->stages[*target_stage_id];
    if (ElementwiseMatch(task, state, stage->op, target_stage->op) &&
        (!(HasReduceIter(stage) && HasReduceIter(target_stage)))) {
      return true;
    }
  }
  return false;
}

// Return whether this stage needs rfactor
inline bool NeedsRfactor(const SearchTask& task, const State& state, const te::Operation& op) {
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    int64_t cum_space_len = 1, cum_reduce_len = 1;
    int stage_id = OperationToStage(op, state);
    for (const auto& iter : state->stages[stage_id]->iters) {
      if (iter->iter_type == kSpace) {
        cum_space_len *= GetExtent(iter);
      } else if (iter->iter_type == kReduce) {
        cum_reduce_len *= GetExtent(iter);
      }
    }

    if (NeedsMultilevelTiling(task, state, op)) {
      // Do not use rfactor if we have enough parallelism on space iters
      if (cum_space_len > cum_reduce_len ||
          cum_space_len > task->hardware_params->num_cores * 16) {
        return false;
      } else {
        return true;
      }
    } else if (cum_reduce_len > 1) {
      // Always try rfactor for reduction ops
      return true;
    }
  }

  return false;
}

// Return whether the state did cache_write for stage_id
inline bool HasCacheWriteStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheWriteStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      } else if (stage_id == ps->stage_id) {
        return true;
      }
    } else if (auto ps = s->transform_steps[i].as<CacheReadStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<RfactorStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

// Return whether the state did cache_read for stage_id
inline bool HasCacheReadStage(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (auto ps = s->transform_steps[i].as<CacheWriteStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<CacheReadStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      } else if (stage_id == ps->stage_id) {
        return true;
      }
    } else if (auto ps = s->transform_steps[i].as<RfactorStepNode>()) {
      if (stage_id > ps->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

// Return whether the state did split/follow_split/follow_fused_split in stage_id
inline bool HasSplitStep(const State& s, int stage_id) {
  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (s->transform_steps[i]->IsInstance<CacheWriteStepNode>() ||
        s->transform_steps[i]->IsInstance<CacheReadStepNode>() ||
        s->transform_steps[i]->IsInstance<RfactorStepNode>()) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (s->transform_steps[i]->IsInstance<SplitStepNode>() ||
        s->transform_steps[i]->IsInstance<FollowSplitStepNode>() ||
        s->transform_steps[i]->IsInstance<FollowFusedSplitStepNode>()) {
      if (stage_id == s->transform_steps[i]->stage_id) {
        return true;
      }
    }
  }
  return false;
}

// Return whether the stage has been tiled already
inline bool IsTiled(const Stage& stage) {
  auto op = stage->op.as<te::ComputeOpNode>();
  CHECK(op != nullptr);
  return stage->iters.size() != op->axis.size() + op->reduce_axis.size();
}

// Query axes that should not be splitted according to the attribute from tvm.compute
inline std::pair<std::set<std::string>, std::set<std::string> > QueryNoSplitAxis(
    const Stage& stage) {
  std::pair<std::set<std::string>, std::set<std::string> > ret;
  if (stage->op->attrs.count(SearchPolicyNode::no_split_at_inner_key)) {
    ret.first = GetIterNameSetParam(stage->op->attrs, SearchPolicyNode::no_split_at_inner_key);
  }
  if (stage->op->attrs.count(SearchPolicyNode::no_split_at_outer_key)) {
    ret.second = GetIterNameSetParam(stage->op->attrs, SearchPolicyNode::no_split_at_outer_key);
  }
  return ret;
}

// Query axes that last split is one
inline std::set<std::string> QueryLastSplitIsOneAxis(const Stage& stage) {
  std::set<std::string> ret;
  if (stage->op->attrs.count(SearchPolicyNode::last_split_is_one_key)) {
    ret = GetIterNameSetParam(stage->op->attrs, SearchPolicyNode::last_split_is_one_key);
  }
  return ret;
}

// Extract primitive iterators from a nested fused or splitted iterator's name
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

  if (last_pos < name.size() && !isdigit(name[last_pos]) &&
      name[last_pos] != '@' && name[last_pos] != '.') {
    rets->insert(name.substr(last_pos, name.size() - last_pos));
  }
}

// Get the last space iterator in the outer most tile
inline const Iterator& GetLastSpaceIteratorInOutermostTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  std::set<std::string> original_names;

  for (const auto& iter : stage->iters) {
    ExtractOriginalIterators(iter->name, &original_names);
    if (original_names.size() == pop->axis.size()) {
      return iter;
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
  return stage->iters[0];
}

// Get the last reduce iterator in the outermost reduce tile
inline const Iterator& GetLastReduceIteratorInOutermostReduceTile(const Stage& stage) {
  auto pop = stage->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  std::set<std::string> original_names;

  auto no_split_name_pair = QueryNoSplitAxis(stage);
  std::set<std::string> no_split_at_inner_name_set = no_split_name_pair.first;
  size_t axis_size = 0;
  for (const auto axis : pop->axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      axis_size++;
    }
  }
  size_t reduce_axis_size = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_axis_size++;
    }
  }

  if (reduce_axis_size) {
    for (const auto& iter : stage->iters) {
      ExtractOriginalIterators(iter->name, &original_names);
      if (original_names.size() == axis_size + reduce_axis_size) {
        return iter;
      }
    }
  } else {
    for (size_t i = 0; i < stage->iters.size(); i++) {
      ExtractOriginalIterators(stage->iters[i]->name, &original_names);
      if (original_names.size() == axis_size + 1) {
        return stage->iters[i-1];
      }
    }
  }

  LOG(FATAL) << "Cannot find the iterator.";
  return stage->iters[0];
}

// Random sample states
inline void RandomSampleStates(const std::vector<State>& in_states, std::mt19937* random_gen,
        size_t out_size, std::vector<State>* out_states) {
  out_states->clear();
  for (size_t i = 0; i < out_size; i++) {
    out_states->push_back(in_states[(*random_gen)() % in_states.size()]);
  }
}

// Random choose an index according to a prefix sum probability
inline int RandomChoose(const std::vector<double>& prefix_sum_probs, std::mt19937* random_gen) {
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double x = dis(*random_gen);

  CHECK(!prefix_sum_probs.empty());

  return std::lower_bound(prefix_sum_probs.begin(), prefix_sum_probs.end(), x) -
      prefix_sum_probs.begin();
}

// Print all states
inline void PrintAllStates(const std::vector<State>& states) {
  for (size_t i = 0; i < states.size(); ++i) {
    std::cerr << i << std::endl;
    std::cerr << states[i];
    std::cerr << "==============================================" << std::endl;
  }
}

// Get all split steps on spatial iterators for one stage
void GetSpaceSplitStepIds(const State& s, int stage_id, std::vector<int>* spatial_split_step_ids);

// Apply multi-level tiling structure according to a string format,
// where "S" stands a space level, "R" stands for a reudciton level.
// For example, if the format is "SSRSRS", the we will
// use tiling structure:  space_L0, space_L1, reduce_L0, space_L2, reduce_L1, space_L3
// For example, if apply "SSRSRS" to matrix multiplication,
// we have space iterators i and j, reduce iterator k.
// Then the tiling structure is : i0, j0, i1, j1, k0, i2, j2, k1, i3, j3
State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids);

// Apply tiling structure: space, space, space, ..., with tile sizes from other SplitStep
State FollowTiling(const State& state, int stage_id,
                   const std::vector<int>& split_step_ids, int n_split);

// Randomly mutate the tile size of one SplitStep
State RandomMutateTileSize(const State& old_state, SplitFactorizationMemo* split_memo,
                           std::mt19937* random_gen, int max_innermost_split_factor);

// Randomly mutate the value of one auto_unroll_max_step PragmaStep
State RandomMutateMaxUnrollStep(const State& old_state, std::mt19937* random_gen,
                                const std::vector<int>& auto_unroll_configs);

// Randomly mutate the parallel degree of one stage.
State RandomMutateParallel(const State& old_state, std::mt19937* random_gen,
                           const SearchTask& task, int verbose = 0);

// Randomly mutate the computation location of one stage.
State RandomMutateComputeLocation(const State& old_state, std::mt19937* random_gen,
                                  const SearchTask& task);

// GA: Crossover two states
State CrossOverState(const State& p1, const State& p2);

// Prune undefined states.
void PruneUndefined(std::vector<State>* states);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_UTILS_H_
