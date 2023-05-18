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
 * \brief Common utilities
 */

#include "utils.h"

#include <algorithm>

namespace tvm {
namespace auto_scheduler {

Array<Integer> GetSpatialSplitStepIds(const State& s, int stage_id) {
  const auto& stage = s->stages[stage_id];
  const auto& pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  Array<Integer> spatial_split_step_ids;
  for (int i = s->transform_steps.size() - 1; i >= 0; --i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids.push_back(i);
        }
      }
    }
  }

  return spatial_split_step_ids;
}

std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id) {
  int target_stage_id = GetSingleConsumerId(task, state, stage_id);
  if (target_stage_id < 0) {
    return {};
  }
  const Stage& target_stage = state->stages[target_stage_id];

  std::vector<std::pair<int, int>> candidates;
  bool target_compute_at_other = target_stage->compute_at == ComputeAtKind::kIter;
  bool target_is_tiled = IsTiled(target_stage);

  bool visited_reduce = false;
  // Enumerate compute_at location at target_stage
  // TODO(merrymercy): More analysis here to make smarter choices
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    const Iterator& target_iter = target_stage->iters[i];
    if (target_iter->iter_kind == IteratorKind::kReduction) {
      visited_reduce = true;
      if (!target_is_tiled) {  // Do not go into reduce iter
        break;
      }
    } else if (target_iter->iter_kind == IteratorKind::kSpatial) {
      if (visited_reduce) {  // Do not go into inner tile
        break;
      }
    }

    if (target_iter->annotation == IteratorAnnotation::kUnroll) {
      // Do not go into the unroll region of const tensor indices
      break;
    }

    if (GetExtent(target_iter) == 1) {
      // Skip iterators with length of 1
      continue;
    }
    if (target_compute_at_other && target_iter->iter_kind == IteratorKind::kSpatial &&
        StrEndsWith(target_iter->name, ".0")) {
      // Skip the first level iterators if target stage compute_at another stage
      // In this case, the lengths of first level iterators are always one
      continue;
    }
    candidates.emplace_back(target_stage_id, i);

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
      break;
    }
  }

  // if the target_stage is already compute_at another stage X, try also compute_at X
  // We call stage X as `target_target_stage`
  if (target_compute_at_other) {
    int target_target_stage_id;
    target_target_stage_id = state->attach_map->stage_to_attach_iter.at(target_stage_id).first;
    const Stage& target_target_stage = state->stages[target_target_stage_id];

    for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
      const Iterator& target_target_iter = target_target_stage->iters[i];
      if (target_target_iter->iter_kind == IteratorKind::kReduction ||
          state->attach_map->iter_to_attached_stages.count(
              std::make_pair(target_target_stage_id, i))) {
        break;
      }

      if (target_target_iter->annotation == IteratorAnnotation::kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }

      candidates.emplace_back(target_target_stage_id, i);
    }
  }

  return candidates;
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids) {
  // Temporal object to be used if the input pointer is nullptr
  std::vector<int> temp_split_step_ids;
  if (spatial_split_step_ids == nullptr) {
    spatial_split_step_ids = &temp_split_step_ids;
  }
  spatial_split_step_ids->clear();

  std::vector<std::vector<Iterator>> space_levels;
  std::vector<std::vector<Iterator>> reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;

  size_t n_space =
      std::count(format.begin(), format.end(), 's') + std::count(format.begin(), format.end(), 'S');
  size_t n_reduce =
      std::count(format.begin(), format.end(), 'r') + std::count(format.begin(), format.end(), 'R');
  if (n_space + n_reduce != format.size()) {
    LOG(FATAL) << "Invalid multi-level tiling format: " << format;
  }
  space_levels.resize(n_space);
  reduce_levels.resize(n_reduce);

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();

  auto sr_levels = [&](int size, const Iterator& iter, std::vector<std::vector<Iterator>>& levels) {
    ICHECK_GE(size, 1);
    if (size == 1) {
      levels[0].push_back(iter);
    } else {
      Array<Iterator> split_res =
          tmp_s.split(stage_id, iter, Array<Optional<Integer>>(size - 1, NullOpt));
      for (int i = 0; i < size; i++) {
        levels[i].push_back(split_res[i]);
      }
      if (iter->iter_kind == IteratorKind::kSpatial) {
        spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
      }
    }
  };

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (!no_split_at_inner_name_set.count(iter->name)) {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        sr_levels(n_space, iter, space_levels);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        sr_levels(n_reduce, iter, reduce_levels);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    } else {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        space_inner.push_back(iter);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        reduce_inner.push_back(iter);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    }
  }

  auto fill_levels = [&](std::vector<Iterator>& levels_iter, std::vector<Iterator>& fill) {
    if (!fill.empty()) {
      levels_iter.insert(levels_iter.begin(), std::make_move_iterator(fill.begin()),
                         std::make_move_iterator(fill.end()));
    }
  };
  if (!space_levels.empty()) {
    fill_levels(space_levels.front(), space_outer);
    fill_levels(space_levels.back(), space_inner);
  }
  if (!reduce_levels.empty()) {
    fill_levels(reduce_levels.front(), reduce_outer);
    fill_levels(reduce_levels.back(), reduce_inner);
  }

  Array<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (c == 's' || c == 'S') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
                   std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (c == 'r' || c == 'R') {
      order.insert(order.end(), std::make_move_iterator(reduce_levels[reduce_ct].begin()),
                   std::make_move_iterator(reduce_levels[reduce_ct].end()));
      reduce_ct++;
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }

  tmp_s.reorder(stage_id, order);
  return tmp_s;
}

State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3, tmp_order;
  Array<Iterator> split_res;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  int no_split_at_inner_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
  }

  ICHECK_EQ(state->stages[stage_id]->iters.size() - no_split_at_inner_name_in_stage_cnt,
            split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct], n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case IteratorAnnotation::kUnroll:
            split_res.Set(n_split, tmp_s.unroll(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kVectorize:
            split_res.Set(n_split, tmp_s.vectorize(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kParallel:
            split_res.Set(0, tmp_s.parallel(stage_id, split_res[0]));
            break;
          default:
            break;
        }

        space_0.push_back(split_res[0]);
        space_1.push_back(split_res[1]);
        if (n_split >= 2) {
          space_2.push_back(split_res[2]);
          if (n_split == 3) {
            space_3.push_back(split_res[3]);
          }
        }
        ct++;
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            ICHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
    }
  }

  if (n_split == 3) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2, &space_3);
  } else if (n_split == 2) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2);
  } else {
    ConcatenateMove(&tmp_order, &space_0, &space_1);
  }
  tmp_s.reorder(stage_id, tmp_order);
  return tmp_s;
}

// Return whether a state has nested parallel, which is invalid on CPUs
bool HasNestedParallel(const State& state) {
  std::function<void(int stage_id, size_t*)> count_parallel_ct;

  count_parallel_ct = [&state, &count_parallel_ct](int stage_id, size_t* parallel_ct) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      return;
    }

    for (size_t i = 0; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->annotation == IteratorAnnotation::kParallel) {
        (*parallel_ct)++;
      }

      IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          count_parallel_ct(attach_stage_id, parallel_ct);
        }
      }
    }
  };

  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    size_t parallel_ct = 0;

    if (state->stages[stage_id]->compute_at == ComputeAtKind::kRoot) {
      count_parallel_ct(stage_id, &parallel_ct);
      if (parallel_ct >= 2) {
        return true;
      }
    }
  }

  return false;
}

State CrossOverState(const SearchTask& task, std::mt19937* random_gen, const State& p1,
  const State& p2, std::vector<int>* fail_counters,float proportion){
  // An internal class that replays a parent state to make the stage ID consist.
  class SyncingState {
   public:
    int id;
    State sync_state;
    const std::vector<Step>& steps;
    int stage_change_cnt;
    size_t step_ptr;

    SyncingState(const SearchTask& task, int id, const State& ref_state)
        : steps(ref_state->transform_steps) {
      this->id = id;
      this->sync_state = task->compute_dag.GetInitState();
      this->stage_change_cnt = 0;
      this->step_ptr = 0;
    }

    // Indicate if the state is up-to-date (all steps are applied).
    bool IsSynced() { return step_ptr == steps.size(); }

    // Number of applied steps that changed stages.
    int StageChangeCount() { return stage_change_cnt; }

    // Get the target stage name of the step to be applied.
    std::string GetCurrStageName() {
      if (IsSynced()) {
        return "";
      }
      return sync_state->stages[steps[step_ptr]->stage_id]->op->name;
    }

    // Apply one step to the syncing state. Do nothing if all steps are applied already.
    void ApplyOneStep(const SearchTask& task) {
      if (IsSynced()) {
        return;
      }

      const Step& step = steps[this->step_ptr];
      this->sync_state.CopyOnWrite()->transform_steps.push_back(step);
      this->sync_state.DoStep(step, task->compute_dag);

      if (IsStageNumberChangingStep(step)) {
        this->stage_change_cnt++;
      }
      this->step_ptr++;
    }
  };

  // Don't do crossover when the stage numbers are different
  if (p1->stages.size() != p2->stages.size()) {
    (*fail_counters)[0]++;
    return State();
  }

  // Create sync states to match the stages.
  SyncingState sync_p1(task, 1, p1);
  SyncingState sync_p2(task, 2, p2);
  std::vector<SyncingState*> sync_states = {&sync_p1, &sync_p2};

  // Stage index to the selected state. Default to p1.
  std::unordered_map<std::string, int> stage_out_to_states;
  int p1_selected = 0, p2_selected = 0;

  //doublex----p2+p1+p2
  int length=static_cast<int>(p1->stages.size());
  int one_point=rand() % length;
  int two_point=rand() % length;
  while(two_point==one_point){
    two_point=rand() % length;
  }
  int cnt=0;
  for (int t=length-1; t >= 0; --t) {

    // Don't do crossover only when the stage names are different
    if (p1->stages[t]->op->name != p2->stages[t]->op->name) {
      (*fail_counters)[1]++;
      return State();
    }

    // This stage is already been assigned
    if (stage_out_to_states.count(p1->stages[t]->op->name)) {
      continue;
    }

    if (p1->stages[t]->op_type == kPlaceholder) {
      // Since CacheRead steps target to placeholder stage, we assign all placeholders to p1.
      stage_out_to_states[p1->stages[t]->op->name] = sync_p1.id;
      continue;
    } 

    if(t==one_point || t==two_point) ++cnt;

    if(cnt==0 || cnt==2){
      stage_out_to_states[p2->stages[t]->op->name] = sync_p2.id;
      if (p2->stages[t]->compute_at != kInlined) {
        p2_selected++;
      }
    }

    if(cnt==1){
      stage_out_to_states[p1->stages[t]->op->name] = sync_p1.id;
      if (p1->stages[t]->compute_at != kInlined) {
        p1_selected++;
      }
    }

    if (IsGPUTask(task)) {
      int id = stage_out_to_states[p1->stages[t]->op->name];
      const State& parent = (id == 1 ? p1 : p2);

      // On GPU, if we choose a root stage, all stages in this GPU kernel should also be chosen.
      // This can fix some fatal dependency problems.
      if (parent->stages[t]->compute_at == kRoot) {
        std::function<void(int)> assign_attached_stages;
        assign_attached_stages = [&assign_attached_stages, id, &parent, &stage_out_to_states](int stage_id) {
          const Stage& stage = parent->stages[stage_id];
          for (size_t i = 0; i < stage->iters.size(); ++i) {
            AttachMap::IterKey iter_key(stage_id, i);
            auto res = parent->attach_map->iter_to_attached_stages.find(iter_key);
            if (res != parent->attach_map->iter_to_attached_stages.end()) {
              for (const auto& attach_stage_id : res->second) {
                stage_out_to_states[parent->stages[attach_stage_id]->op->name] = id;
                assign_attached_stages(attach_stage_id);
              }
            }
          }
        };
        assign_attached_stages(t);
      }
    } else {
      // If a rfactor stage is chosen, all stages related to this rfactor should be chosen.
      // This can fix some fatal dependency problems.
      if (StrEndsWith(p1->stages[t]->op->name, ".repl")) {
        int id = stage_out_to_states[p1->stages[t]->op->name];
        std::string raw_name = p1->stages[t]->op->name.substr(0, p1->stages[t]->op->name.size() - 5);
        stage_out_to_states[raw_name] = id;
        stage_out_to_states[raw_name + ".rf"] = id;
      }
    }
  }

  // If all stages are coming from the same state, then no need to crossover.
  if (p1_selected == 0 || p2_selected == 0) {
    (*fail_counters)[2]++;
    return State();
  }

  // Create a new state.
  State tmp_s = task->compute_dag.GetInitState();

  // Apply steps. Meanwhile we also re-apply steps to p1 and p2 to make sure
  // the stage ID is matched.
  while (!sync_states[0]->IsSynced() && !sync_states[1]->IsSynced()) {
    SyncingState* sync_s = nullptr;

    // Determine which state we will focus on this round.
    // If a state has changed its stages more times than another state, we prior to another state to
    // make their stages synced. Otherwise we simply go for the one with smaller step pointer.
    if (sync_states[0]->StageChangeCount() < sync_states[1]->StageChangeCount()) {
      sync_s = sync_states[0];
    } else if (sync_states[0]->StageChangeCount() > sync_states[1]->StageChangeCount()) {
      sync_s = sync_states[1];
    } else {
      sync_s = sync_states[(sync_states[0]->step_ptr <= sync_states[1]->step_ptr) ? 0 : 1];
    }
    const std::string& curr_stage_name = sync_s->GetCurrStageName();

    // Check if we want to apply this step.
    std::string target_stage_name = curr_stage_name;
    if (auto ps = sync_s->steps[sync_s->step_ptr].as<ComputeAtStepNode>()) {
      // Whether to apply Compute_at step depends on the target stage instead of self stage.
      target_stage_name = sync_s->sync_state->stages[ps->target_stage_id]->op->name;
    }

    // If the target stage of the current state is selected, we apply this step to the new state.
    if (stage_out_to_states[target_stage_name] == sync_s->id) {
      tmp_s = ApplyStepToNewState(task, tmp_s, sync_s->sync_state, sync_s->steps[sync_s->step_ptr]);
      if (!tmp_s.defined()) {
        (*fail_counters)[3]++;
        return tmp_s;
      }
    }

    sync_s->ApplyOneStep(task);
  }

  // Process tails.
  for (size_t i = 0; i < sync_states.size(); ++i) {
    SyncingState* sync_s = sync_states[i];
    while (!sync_s->IsSynced()) {
      const std::string& stage_name = sync_s->GetCurrStageName();

      // Check if we want to apply this step.
      std::string target_stage_name = stage_name;
      if (auto ps = sync_s->steps[sync_s->step_ptr].as<ComputeAtStepNode>()) {
        // Whether to apply Compute_at step depends on the target stage instead of self stage.
        target_stage_name = sync_s->sync_state->stages[ps->target_stage_id]->op->name;
      }

      // If the target stage of the current state is selected, we apply this step to the new state.
      if (stage_out_to_states[target_stage_name] == sync_s->id) {
        tmp_s = ApplyStepToNewState(task, tmp_s, sync_s->sync_state, sync_s->steps[sync_s->step_ptr]);
        if (!tmp_s.defined()) {
          (*fail_counters)[4]++;
          return tmp_s;
        }
      }

      sync_s->ApplyOneStep(task);
    }
  }
  return tmp_s;  
}
  
void PruneInvalidState(const SearchTask& task, Array<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    if (!IsGPUTask(task) && HasNestedParallel((*states)[i])) {
      continue;
    }

    if (i != pt) {
      states->Set(pt, (*states)[i]);
    }
    pt++;
  }

  if (pt == 0) {
    LOG(FATAL) << "Internal error: All states are invalid.";
  } else {
    states->resize(pt);
  }
}

/********** SplitFactorizationMemo **********/
const Array<Array<Integer>>& SplitFactorizationMemo::GetFactorizationSchemes(
    int extent, int n_lengths, int max_innermost_factor) {
  QueryKey key = std::make_tuple(extent, n_lengths, max_innermost_factor);
  const auto& it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }

  tmp_stack_ = Array<Integer>(n_lengths, Integer());
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent, max_innermost_factor);

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_length, int max_innermost_factor) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <= max_innermost_factor) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_length)) {
      tmp_stack_.Set(now, Integer(f));
      DfsEnumerate(now + 1, remaining_length / f, max_innermost_factor);
    }
  }
}

const std::vector<int>& SplitFactorizationMemo::GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }

  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

/********** Utils interface API for ffi **********/

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsGetConsumers")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id) {
      const std::set<int>& consumers = GetConsumers(task, state, stage_id);
      tvm::Map<IntImm, IntImm> ret;
      for (const auto& i : consumers) {
        ret.Set(Integer(i), Integer(i));
      }
      return ret;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsElementwiseMatch")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id,
                       int target_stage_id) {
      return ElementwiseMatch(task, state, stage_id, target_stage_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsTiled")
    .set_body_typed([](const Stage& stage) { return IsTiled(stage); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheReadStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheReadStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheWriteStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheWriteStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasRfactorStage")
    .set_body_typed([](const State& s, int stage_id) { return HasRfactorStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction")
    .set_body_typed([](const State& s, int stage_id) {
      return HasCrossThreadReduction(s, stage_id);
    });

}  // namespace auto_scheduler
}  // namespace tvm
