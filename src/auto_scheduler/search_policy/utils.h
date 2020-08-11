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
 * \file auto_scheduler/search_policy/utils.cc
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
  return -1;
}

/********** Get Parameters **********/

/*! \brief Get an integer from a tvm str Map. */
inline int GetIntParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pint = attr_dict[key].as<IntImmNode>();
  CHECK(pint != nullptr);
  return pint->value;
}

/*! \brief Get a double from a tvm str Map. */
inline double GetDoubleParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto pdouble = attr_dict[key].as<FloatImmNode>();
  CHECK(pdouble != nullptr);
  return pdouble->value;
}

/*! \brief Get a string from a tvm str Map. */
inline std::string GetStringParam(const Map<String, ObjectRef>& attr_dict, const std::string& key) {
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  const auto& target = attr_dict[key];
  if (auto pstr = target.as<StringImmNode>()) {
    return pstr->value;
  }
  auto pstr = target.as<StringObj>();
  CHECK(pstr != nullptr);
  return pstr->data;
}

/*! \brief Get a iterator name set from a tvm str Map. */
inline std::set<std::string> GetIterNameSetParam(const Map<String, ObjectRef>& attr_dict,
                                                 const std::string& key) {
  std::set<std::string> ret;
  CHECK_GT(attr_dict.count(key), 0) << "Cannot find key: \"" << key << "\" in " << attr_dict;
  auto names = attr_dict[key].as<ArrayNode>();
  CHECK(names != nullptr);
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
inline std::pair<int64_t, int64_t> GetCumulativeSpaceAndReductionLengh(const Stage& stage) {
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
        GetCumulativeSpaceAndReductionLengh(state->stages[stage_id]);

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
           HasReduceIter(state->stages[*target_stage_id])))) {
      return true;
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

    if (s->transform_steps[i]->IsInstance<CacheWriteStepNode>() ||
        s->transform_steps[i]->IsInstance<CacheReadStepNode>() ||
        s->transform_steps[i]->IsInstance<RfactorStepNode>()) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    }
  }
  return false;
}

/*! \brief Return whether the stage has been tiled already. */
inline bool IsTiled(const Stage& stage) {
  auto op = stage->op.as<te::ComputeOpNode>();
  CHECK(op != nullptr);
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

  CHECK(!reduce_iters->empty());
  State tmp_s = state;
  if (reduce_iters->size() > 1) {
    *fused_iter = tmp_s.fuse(stage_id, *reduce_iters);
  } else {
    *fused_iter = (*reduce_iters)[0];
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

/*! \brief Print a title */
inline void PrintTitle(const std::string& title, int verbose) {
  StdCout(verbose) << Chars('-', 60) << "\n"
                   << Chars('-', 25) << "  [ " << title << " ]\n"
                   << Chars('-', 60) << std::endl;
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
  void DfsEnumerate(int now, int remaining_lenght, int max_innermost_factor);

  std::unordered_map<QueryKey, Array<Array<Integer>>> memory_;

  int n_lengths_;
  Array<Integer> tmp_stack_;
  Array<Array<Integer>>* results_;
  std::unordered_map<int, std::vector<int>> factor_memory_;
};

// Apply multi-level tiling structure according to a string format,
// where "S" stands a space level, "R" stands for a reudciton level.
// For example, if the format is "SSRSRS", the we will
// use tiling structure:  space_L0, space_L1, reduce_L0, space_L2, reduce_L1, space_L3
// For example, if apply "SSRSRS" to matrix multiplication,
// we have space iterators i and j, reduce iterator k.
// Then the tiling structure is : i0, j0, i1, j1, k0, i2, j2, k1, i3, j3
State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids = nullptr);

// Apply tiling structure: space, space, space, ..., with tile sizes from other SplitStep
State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_UTILS_H_
