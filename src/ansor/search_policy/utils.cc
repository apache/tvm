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

  const auto& no_split_name_pair = QueryNoSplitAxis(s->stages[stage_id]);
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;

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
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        // TODO(jcf94): do not rely on this assumption
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
      LOG(FATAL) << "Invalid multi-level tiling format: " << format;
    }
  }
  size_t n_space = space_levels.size();
  size_t n_reduce = reduce_levels.size();

  spatial_split_step_ids->clear();

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const auto& no_split_name_pair = QueryNoSplitAxis(stage);  // handle special split strategy
  const auto& last_split_is_one_name_set = QueryLastSplitIsOneAxis(stage);
  const std::set<std::string>& no_split_at_inner_name_set = no_split_name_pair.first;
  const std::set<std::string>& no_split_at_outer_name_set = no_split_name_pair.second;

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
  const auto& no_split_name_pair = QueryNoSplitAxis(stage);  // handle special split strategy
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
        SplitStep(ps->stage_id, ps->iter_id, ps->extent, new_lengths, ps->inner_to_outer);
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
  pstate->transform_steps[step_id] = PragmaStep(
      ps->stage_id, ps->iter_id, std::string("auto_unroll_max_step") + "$" + val);
  return tmp_s;
}

State RandomMutateParallel(const State& old_state, std::mt19937* random_gen,
                           const SearchTask& task, int verbose) {
  // To make this mutation simple but promising, we only focus on a specific case that
  // parallel was added to the outermost loop and the loop is generated by fusing other loops.
  // In short, we mutate the step pattern of (fuse -> parallel).

  // Extract all parallel steps.
  std::vector<int> parallel_steps;
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    auto ps = old_state->transform_steps[s].as<AnnotationStepNode>();
    if (!ps || ps->annotation != kParallel) {
      continue;
    }
    parallel_steps.push_back(s);
  }
  if (parallel_steps.empty()) {
    StdCout(verbose) << "Parallel mutation failed: No parallel annotations" << std::endl;
    return State();
  }

  // Randomly pick one step.
  int retry_ct = 0;
  size_t step_id = 0;
  size_t stage_id = 0;
  do {
    step_id = parallel_steps[(*random_gen)() % parallel_steps.size()];
    auto step = old_state->transform_steps[step_id].as<AnnotationStepNode>();
    stage_id = step->stage_id;

    // Check assumptions.
    auto iter_id = step->iter_id;
    if (iter_id == 0 && step_id > 0 && old_state->transform_steps[step_id - 1].as<FuseStepNode>()) {
      break;
    }
    retry_ct++;
  } while (retry_ct <= 3);

  if (retry_ct > 3) {
    StdCout(verbose) << "Parallel mutation failed: No valid parallel annotations" << std::endl;
    return State();
  }

  // Replay a new state until the picked fuse step.
  State tmp_s = task->compute_dag.GetInitState();
  for (size_t s = 0; s < step_id - 1; ++s) {
    auto step = old_state->transform_steps[s];
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    tmp_s.DoStep(step, task->compute_dag);
  }

  // Determine the fuse direction.
  // 0: fuse less; 1: fuse more.
  auto fuse_step = old_state->transform_steps[step_id - 1].as<FuseStepNode>();
  std::vector<int> fused_ids = fuse_step->fused_ids;
  std::vector<double> fuse_dir = {0.5, 1.0};

  // The case we can only fuse more.
  if (fused_ids.size() == 1) {
    fuse_dir[0] = 0.0;
  }

  // The cases that we cannot fuse the next iters.
  if (old_state->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, 0)) > 0 ||
      tmp_s->stages[stage_id]->iters.size() == fused_ids.size() ||
      tmp_s->stages[stage_id]->iters[1]->iter_type == kReduce) {
    // In case we cannot fuse less neither, give up.
    if (fuse_dir[0] == 0.0) {
      StdCout(verbose) << "Parallel mutation failed: Cannot fuse more or less iters" << std::endl;
      return State();
    }
    fuse_dir[0] = 1.0;
  }

  int iter_offset = 0;
  if (RandomChoose(fuse_dir, random_gen) == 0) {
    StdCout(verbose) << "Parallel mutation: release iter " << fused_ids.back() << std::endl;
    fused_ids.pop_back();
    iter_offset = 1;
  } else {
    StdCout(verbose) << "Parallel mutation: include iter " << fused_ids.back() + 1 << std::endl;
    fused_ids.push_back(fused_ids.back() + 1);
    iter_offset = -1;
  }

  // Replay the mutated fused and annotation step.
  auto new_fuse_step = FuseStep(stage_id, fused_ids);
  tmp_s.CopyOnWrite()->transform_steps.push_back(new_fuse_step);
  tmp_s.DoStep(new_fuse_step, task->compute_dag);
  tmp_s.CopyOnWrite()->transform_steps.push_back(old_state->transform_steps[step_id]);
  tmp_s.DoStep(old_state->transform_steps[step_id], task->compute_dag);

  // Replay the rest steps.
  for (size_t s = step_id + 1; s < old_state->transform_steps.size(); ++s) {
    auto step = old_state->transform_steps[s];
    if (step->stage_id == static_cast<int>(stage_id)) {
      // Since we change the loop structure, iter ID in later steps to the same stage
      // has to be adjusted.
      auto ps = step.as<AnnotationStepNode>();
      if (ps) {
        if (ps->iter_id == 0) {
          step = AnnotationStep(ps->stage_id, 0, ps->annotation);
        } else {
          CHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
          step = AnnotationStep(ps->stage_id, ps->iter_id + iter_offset, ps->annotation);
        }
      } else {
        StdCout(verbose) << "Parallel mutation: Cannot apply " << step << " after fuse"
                         << std::endl;
        return State();
      }
    }
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    tmp_s.DoStep(step, task->compute_dag);
  }
  return tmp_s;
}


State RandomMutateComputeLocation(const State& old_state, std::mt19937* random_gen,
                                  const SearchTask& task) {
  // Extract all compute_at steps.
  std::vector<int> compute_at_steps;
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    if (auto ps = old_state->transform_steps[s].as<ComputeAtStepNode>()) {
      const Stage& stage = old_state->stages[ps->stage_id];
      if (IsTiled(stage)) {
        continue;
      }

      if (NeedsMultilevelTiling(task, old_state, stage->op)) {
        continue;
      }
      compute_at_steps.push_back(s);
    }
  }
  if (compute_at_steps.empty()) {
    return State();
  }

  // Randomly pick one step
  size_t step_id = compute_at_steps[(*random_gen)() % compute_at_steps.size()];
  auto ps = old_state->transform_steps[step_id].as<ComputeAtStepNode>();
  CHECK(ps != nullptr);
  const Stage& stage = old_state->stages[ps->stage_id];

  // Randomly pick one tile level
  int new_compute_at_stage_id;
  int new_compute_at_iter_id;

  // Copied from InitPopulationChangeComputeLocation
  {
    std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
    GetConsumers(task, old_state, stage->op, &consumers);
    if (consumers.empty()) {
      return State();
    }

    int target_stage_id;
    if (consumers.size() == 1) {
      target_stage_id = OperationToStage(*consumers.begin(), old_state);
    } else {
      // check all consumers share a common root
      int common_root_id = -1;
      bool mismatch = false;
      for (const auto& consumer : consumers) {
        int consumer_stage_id = OperationToStage(consumer, old_state);
        int root_id = -1;
        if ((old_state)->stages[consumer_stage_id]->compute_at == kRoot) {
          root_id = consumer_stage_id;
        } else if ((old_state)->stages[consumer_stage_id]->compute_at == kIter) {
          root_id = (old_state)->attach_map->stage_to_attach_iter.at(consumer_stage_id).first;
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

      if (mismatch) {
        return State();
      }
      target_stage_id = common_root_id;
    }

    const Stage& target_stage = old_state->stages[target_stage_id];
    std::set<std::string> to_unroll_name_set;
    if (target_stage->op->attrs.count(SearchPolicyNode::always_unroll_key)) {
      to_unroll_name_set = GetIterNameSetParam(target_stage->op->attrs,
                                               SearchPolicyNode::always_unroll_key);
    }

    std::vector<std::pair<int, int> > candidates;
    bool target_compute_at_other = target_stage->compute_at == kIter;
    bool target_is_tiled = IsTiled(target_stage);

    bool visited_reduce = false;
    // enumerate compute_at location at target_stage
    int ct = 0;
    for (size_t iter_id = 0; iter_id < target_stage->iters.size(); ++iter_id) {
      const auto& target_iter = target_stage->iters[iter_id];
      if (target_iter->iter_type == kReduce) {
        visited_reduce = true;
        if (!target_is_tiled) {  // do not go into reduce iter
          break;
        }
      } else if (target_iter->iter_type == kSpace) {
        if (visited_reduce) {  // do not go into inner tile
          break;
        }
      }

      if (to_unroll_name_set.count(target_iter->name)) {
        // Do not go into always unroll region
        break;
      }

      if (GetExtent(target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }
      if (target_compute_at_other && target_iter->iter_type == kSpace &&
          StrEndsWith(target_iter->name, ".0")) {
        // skip the first level iterators if target stage compute_at another stage
        // In this case, the lengths of first level iterators are always one
        continue;
      }
      candidates.emplace_back(target_stage_id, iter_id);

      if ((old_state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(target_stage_id, ct++))) {
        break;
      }
    }

    // if the target_stage is already compute_at another stage X, try also compute_at X
    // We call stage X as `target_target_stage`
    if (target_compute_at_other) {
      int target_target_stage_id;
      target_target_stage_id = (old_state)->attach_map->stage_to_attach_iter.at(
          target_stage_id).first;
      const Stage& target_target_stage = (old_state)->stages[target_target_stage_id];
      if (target_target_stage->op->attrs.count(SearchPolicyNode::always_unroll_key)) {
        to_unroll_name_set = GetIterNameSetParam(target_target_stage->op->attrs,
                                                 SearchPolicyNode::always_unroll_key);
      } else {
        to_unroll_name_set.clear();
      }

      int ct = 0;
      for (size_t iter_id = 0; iter_id < target_target_stage->iters.size(); ++iter_id) {
        const auto& target_target_iter = target_target_stage->iters[iter_id];
        if (target_target_iter->iter_type == kReduce ||
            (old_state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(target_target_stage_id, ct++))) {
          break;
        }

        if (to_unroll_name_set.count(target_target_iter->name)) {
          // Do not go into always unroll region
          break;
        }

        if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
          continue;
        }

        candidates.emplace_back(target_target_stage_id, iter_id);
      }
    }

    if (candidates.empty()) {
      return State();
    }

    int choice = (*random_gen)() % (candidates.size());
    new_compute_at_stage_id = candidates[choice].first;
    new_compute_at_iter_id = candidates[choice].second;
  }

  // Replay a new state.
  State tmp_s = task->compute_dag.GetInitState();
  for (size_t s = 0; s < old_state->transform_steps.size(); ++s) {
    if (s == step_id) {
      tmp_s.CopyOnWrite()->transform_steps.push_back(
          ComputeAtStep(ps->stage_id, new_compute_at_stage_id, new_compute_at_iter_id));
    } else {
      tmp_s.CopyOnWrite()->transform_steps.push_back(old_state->transform_steps[s]);
    }
    try {
      tmp_s.DoStep(tmp_s->transform_steps.back(), task->compute_dag);
    } catch (dmlc::Error &e) {
      return State();
    }
  }

  return tmp_s;
}

void PruneUndefined(std::vector<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    if (i != pt) {
      (*states)[pt++] = std::move((*states)[i]);
    }
    pt++;
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

