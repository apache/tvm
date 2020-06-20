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

#include "utils.h"
#include "search_policy.h"

namespace tvm {
namespace ansor {

void GetSpaceSplitStepIds(const State& s, int stage_id, std::vector<int>* spatial_split_step_ids) {
  auto pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);

  auto no_split_name_pair = QueryNoSplitAxis(s->stages[stage_id]);
  std::set<std::string> no_split_at_inner_name_set = no_split_name_pair.first;
  std::set<std::string> no_split_at_outer_name_set = no_split_name_pair.second;
  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint) &&
        !no_split_at_outer_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  for (int i = static_cast<int>(s->transform_steps.size()) - 1; i >= 0; --i) {
    if (s->transform_steps[i]->IsInstance<CacheWriteStepNode>() ||
        s->transform_steps[i]->IsInstance<CacheReadStepNode>() ||
        s->transform_steps[i]->IsInstance<RfactorStepNode>()) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids->push_back(i);
        }
      }
    }
  }
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids) {
  std::vector<std::vector<Iterator> > space_levels;
  std::vector<std::vector<Iterator> > reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;
  std::vector<Iterator> split_res;

  for (const auto c : format) {
    if (tolower(c) == 's') {
      space_levels.emplace_back();
    } else if (tolower(c) == 'r') {
      reduce_levels.emplace_back();
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  auto no_split_name_pair = QueryNoSplitAxis(stage);  // handle special split strategy
  auto last_split_is_one_name_set = QueryLastSplitIsOneAxis(stage);
  std::set<std::string> no_split_at_inner_name_set = no_split_name_pair.first;
  std::set<std::string> no_split_at_outer_name_set = no_split_name_pair.second;

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_type == kSpace) {
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        CHECK_GE(n_space, 1);
        int tmp_n_space = n_space;

        if (last_split_is_one_name_set.count(iter->name)) {
          tmp_n_space--;
        }

        if (tmp_n_space == 1) {
          space_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, std::vector<PrimExpr>(tmp_n_space - 1));
          for (int i = 0; i < tmp_n_space; i++) {
            space_levels[i].push_back(std::move(split_res[i]));
          }
          spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
        }
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          space_inner.push_back(iter);
        }
        if (no_split_at_outer_name_set.count(iter->name)) {
          space_outer.push_back(iter);
        }
      }
    } else if (iter->iter_type == kReduce) {
      // for reduce iterator, split it into two iterators
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        CHECK_GE(n_reduce, 1);
        if (n_reduce == 1) {
          reduce_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter, std::vector<PrimExpr>(n_reduce - 1));
          for (size_t i = 0; i < n_reduce; i++) {
            reduce_levels[i].push_back(std::move(split_res[i]));
          }
        }
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          reduce_inner.push_back(iter);
        }
        if (no_split_at_outer_name_set.count(iter->name)) {
          reduce_outer.push_back(iter);
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << iter->iter_type;
    }
  }

  if (!space_outer.empty()) {
    CHECK(!space_levels.empty());
    space_levels.front().insert(space_levels.front().begin(),
            space_outer.begin(), space_outer.end());
  }
  if (!space_inner.empty()) {
    CHECK(!space_levels.empty());
    space_levels.back().insert(space_levels.back().begin(),
            space_inner.begin(), space_inner.end());
  }

  if (!reduce_outer.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.front().insert(reduce_levels.front().begin(),
            reduce_outer.begin(), reduce_outer.end());
  }
  if (!reduce_inner.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.back().insert(reduce_levels.back().begin(),
            reduce_inner.begin(), reduce_inner.end());
  }

  std::vector<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (tolower(c) == 's') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
              std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (tolower(c) == 'r') {
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

State FollowTiling(const State& state, int stage_id,
                   const std::vector<int>& split_step_ids, int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3;
  std::vector<Iterator> split_res, tmp_order;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  CHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  auto no_split_name_pair = QueryNoSplitAxis(stage);  // handle special split strategy
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;
  int no_split_at_inner_name_in_stage_cnt = 0;
  int no_split_at_outer_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
    no_split_at_outer_name_in_stage_cnt += no_split_at_outer_name_set.count(iter->name);
  }

  CHECK_EQ(state->stages[stage_id]->iters.size()
               - no_split_at_inner_name_in_stage_cnt
               - no_split_at_outer_name_in_stage_cnt,
           split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_type == kSpace) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct],
                                       n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case kUnroll:
            split_res[n_split] = tmp_s.unroll(stage_id, split_res[n_split]);
            break;
          case kVectorize:
            split_res[n_split] = tmp_s.vectorize(stage_id, split_res[n_split]);
            break;
          case kParallel:
            split_res[0] = tmp_s.parallel(stage_id, split_res[0]); break;
          default:
            break;
        }

        space_0.push_back(std::move(split_res[0]));
        space_1.push_back(std::move(split_res[1]));
        if (n_split >= 2) {
          space_2.push_back(std::move(split_res[2]));
          if (n_split == 3) {
            space_3.push_back(std::move(split_res[3]));
          }
        }
        ct++;
      } else {
        if (no_split_at_outer_name_set.count(iter->name)) {
          space_0.push_back(iter);
        }
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            CHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << iter->iter_type;
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

State RandomMutateTileSize(const State& old_state, SplitFactorizationMemo* split_memo,
                           std::mt19937* random_gen, int max_innermost_split_factor) {
  State tmp_s = old_state;

  // Extract all SplitStep
  std::vector<size_t> split_step_ids;
  for (size_t i = 0; i < tmp_s->transform_steps.size(); ++i) {
    if (auto ps = tmp_s->transform_steps[i].as<SplitStepNode>()) {
      if (ps->extent.defined() && ps->extent->IsInstance<IntImmNode>() &&
          GetIntImm(ps->lengths.back()) <= max_innermost_split_factor) {
        split_step_ids.push_back(i);
      }
    }
  }
  if (split_step_ids.empty()) {
    return State();
  }

  // Find a SplitStep with extent != 1
  int retry_ct = 0;
  int64_t extent = 1;
  int step_id;
  const SplitStepNode* ps;

  do {
    step_id = split_step_ids[(*random_gen)() % split_step_ids.size()];
    ps = tmp_s->transform_steps[step_id].as<SplitStepNode>();
    CHECK(ps != nullptr);
    extent = GetIntImm(ps->extent);
    retry_ct += 1;
  } while (retry_ct < static_cast<int>(split_step_ids.size()) << 2 &&
           (extent == 1 || extent == 0));

  if (extent == 0 || extent == 1) {
    return State();
  }

  // Mutate tile size
  std::vector<int> lengths(ps->lengths.size() + 1, 1);
  for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
    lengths[i + 1] = GetIntImm(ps->lengths[i]);
  }
  lengths[0] = extent / ElementProduct(lengths);

  std::vector<int> random_perm;
  RandomPermutation(lengths.size(), &random_perm, random_gen);

  for (size_t i = 0; i < random_perm.size(); ++i) {
    size_t src_idx = random_perm[i];
    int length = lengths[src_idx];

    if (length == 1) {
      continue;
    }

    // Divide one factor from lengths[src_idx] and multiply it to lengths[dst_idx]
    size_t dst_idx = random_perm[(i + 1) % random_perm.size()];

    const std::vector<int>& factors = split_memo->GetFactors(length);
    CHECK_GE(factors.size(), 1);

    int divide_factor;
    if (dst_idx == lengths.size() - 1) {
      // Maintain the restriction of hardware_params.max_innermost_split_factor
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * lengths[dst_idx] <= max_innermost_split_factor) {
          break;
        }
      }
      if (max_factor_index == 0) {
        // failed on this dst_idx, try next one
        continue;
      }
      divide_factor = factors[1 + (*random_gen)() % (max_factor_index)];
    } else {
      divide_factor = factors[1 + (*random_gen)() % (factors.size() - 1)];
    }

    std::vector<PrimExpr> new_lengths;
    for (size_t j = 1; j < lengths.size(); ++j) {
      if (j == src_idx) {
        new_lengths.emplace_back(lengths[j] / divide_factor);
      } else if (j == dst_idx) {
        new_lengths.emplace_back(lengths[j] * divide_factor);
      } else {
        new_lengths.emplace_back(lengths[j]);
      }
    }

    CHECK_LE(GetIntImm(new_lengths.back()), max_innermost_split_factor);

    auto pstate = tmp_s.CopyOnWrite();
    pstate->transform_steps[step_id] =
        SplitStepNode::make(ps->stage_id, ps->iter_id, ps->extent, new_lengths, ps->inner_to_outer);
    return tmp_s;
  }

  return State();
}

State RandomMutateMaxUnrollStep(const State& old_state, std::mt19937* random_gen,
    const std::vector<int>& auto_unroll_configs) {
  State tmp_s = old_state;

  // Extract all auto_unroll_max_step pragma steps.
  std::vector<int> annotate_steps;
  for (size_t i = 0; i < old_state->transform_steps.size(); ++i) {
    if (auto ps = tmp_s->transform_steps[i].as<PragmaStepNode>()) {
      if (ps->pragma_type.find("auto_unroll_max_step") != std::string::npos) {
        annotate_steps.push_back(i);
      }
    }
  }
  if (annotate_steps.empty()) {
    return State();
  }

  // Randomly pick one step.
  auto step_id = annotate_steps[(*random_gen)() % annotate_steps.size()];
  auto ps = tmp_s->transform_steps[step_id].as<PragmaStepNode>();
  auto val = std::to_string(auto_unroll_configs[(*random_gen)() % auto_unroll_configs.size()]);

  auto pstate = tmp_s.CopyOnWrite();
  pstate->transform_steps[step_id] = PragmaStepNode::make(
      ps->stage_id, ps->iter_id, std::string("auto_unroll_max_step") + "$" + val);
  return tmp_s;
}

void PruneUndefined(std::vector<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    (*states)[pt++] = std::move((*states)[i]);
  }

  if (pt == 0) {
    LOG(FATAL) << "All states are undefined.";
  } else {
    states->resize(pt);
  }
}

State CrossOverState(const State& p1, const State& p2) { return State(); }

}  // namespace ansor
}  // namespace tvm

