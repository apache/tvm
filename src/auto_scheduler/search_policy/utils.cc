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
 * \file auto_scheduler/utils.cc
 * \brief Common utilities
 */

#include "utils.h"

#include <algorithm>

namespace tvm {
namespace auto_scheduler {


State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids) {
  std::vector<std::vector<Iterator>> space_levels;
  std::vector<std::vector<Iterator>> reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;
  Array<Iterator> split_res;

  for (const auto c : format) {
    if (tolower(c) == 's') {
      space_levels.emplace_back();
    } else if (tolower(c) == 'r') {
      reduce_levels.emplace_back();
    } else {
      LOG(FATAL) << "Invalid multi-level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const auto& no_split_name_pair = GetNoSplitAxisAttr(stage);  // handle special split strategy
  const auto& last_split_is_one_name_set = GetLastSplitIsOneAxisAttr(stage);
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
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
          split_res = tmp_s.split(stage_id, iter,
                                  Array<Optional<Integer>>(tmp_n_space - 1, NullOpt));
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
    } else if (iter->iter_kind == IteratorKind::kReduction) {
      if (!no_split_at_inner_name_set.count(iter->name) &&
          !no_split_at_outer_name_set.count(iter->name)) {
        CHECK_GE(n_reduce, 1);

        if (n_reduce == 1) {
          reduce_levels[0].push_back(iter);
        } else {
          split_res = tmp_s.split(stage_id, iter,
                                  Array<Optional<Integer>>(n_reduce - 1, NullOpt));
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
      LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
    }
  }

  if (!space_outer.empty()) {
    CHECK(!space_levels.empty());
    space_levels.front().insert(space_levels.front().begin(),
            std::make_move_iterator(space_outer.begin()),
            std::make_move_iterator(space_outer.end()));
  }
  if (!space_inner.empty()) {
    CHECK(!space_levels.empty());
    space_levels.back().insert(space_levels.back().begin(),
            std::make_move_iterator(space_inner.begin()),
            std::make_move_iterator(space_inner.end()));
  }

  if (!reduce_outer.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.front().insert(reduce_levels.front().begin(),
        std::make_move_iterator(reduce_outer.begin()),
        std::make_move_iterator(reduce_outer.end()));
  }
  if (!reduce_inner.empty()) {
    CHECK(!reduce_levels.empty());
    reduce_levels.back().insert(reduce_levels.back().begin(),
        std::make_move_iterator(reduce_inner.begin()),
        std::make_move_iterator(reduce_inner.end()));
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

const Array<Array<Integer> >& SplitFactorizationMemo::GetFactorizationSchemes(int extent,
    int n_lengths, int max_innermost_factor) {
  QueryKey key = std::make_tuple(extent, n_lengths, max_innermost_factor);
  auto it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }

  tmp_stack_.clear();
  tmp_stack_.reserve(n_lengths);
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent, max_innermost_factor);

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_lenght, int max_innermost_factor) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <= max_innermost_factor) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_lenght)) {
      tmp_stack_.Set(now, Integer(f));
      DfsEnumerate(now + 1, remaining_lenght / f, max_innermost_factor);
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
        res.push_back(n/i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

}  // namespace auto_scheduler
}  // namespace tvm
