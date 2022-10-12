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
#include "default_schedule.h"

#include <tvm/meta_schedule/postproc.h>
#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "../printer/text_printer.h"
#include "../tir/schedule/analysis.h"

namespace tvm {
namespace meta_schedule {

static const std::unordered_set<std::string> gpu_targets{"cuda", "rocm", "vulkan", "metal"};

ScheduleRule GetDefaultAutoInline(const std::string& target_name) {
  if (target_name == "llvm" || target_name == "hexagon") {
    return ScheduleRule::AutoInline(
        /*into_producer=*/false,
        /*into_consumer=*/true,
        /*inline_const_tensor=*/true,
        /*disallow_if_then_else=*/true,
        /*require_injective=*/true,
        /*require_ordered=*/true,
        /*disallow_op=*/Array<String>{"tir.exp"});
  } else if (gpu_targets.count(target_name)) {
    return ScheduleRule::AutoInline(
        /*into_producer=*/true,
        /*into_consumer=*/true,
        /*inline_const_tensor=*/true,
        /*disallow_if_then_else=*/false,
        /*require_injective=*/false,
        /*require_ordered=*/false,
        /*disallow_op=*/Array<String>{});
  }
  LOG(FATAL) << "Unsupported target " << target_name;
  return ScheduleRule(nullptr);
}

std::set<std::string> GetBlockNames(const IRModule& mod) {
  struct BlockNameCollector : public tir::StmtVisitor {
    void VisitStmt_(const tir::BlockNode* block) override {
      if (block->name_hint == "root") {
        StmtVisitor::VisitStmt(block->body);
      } else {
        block_names.insert(block->name_hint);
      }
    }
    std::set<std::string> block_names;
  };

  auto prim_func = tir::FindEntryFunc(mod, nullptr);
  BlockNameCollector collector;
  collector(prim_func->body);
  return collector.block_names;
}

std::vector<tir::BlockRV> GetUnscheduledBlocks(const tir::Schedule& sch_orig,
                                               const tir::Schedule& sch) {
  auto block_names_orig = GetBlockNames(sch_orig->mod());
  auto block_names = GetBlockNames(sch->mod());

  std::vector<std::string> common_blocks;

  std::set_intersection(block_names_orig.begin(), block_names_orig.end(), block_names.begin(),
                        block_names.end(), std::back_inserter(common_blocks));

  auto is_scheduled = [=](const std::string& block_name) {
    auto loops = sch->GetLoops(sch->GetBlock(block_name));
    auto loops_orig = sch_orig->GetLoops(sch_orig->GetBlock(block_name));
    if (loops.size() != loops.size()) {
      return true;
    }
    for (size_t i = 0; i < loops.size(); ++i) {
      auto loop = sch->Get(loops[i]);
      auto loop_orig = sch_orig->Get(loops_orig[i]);
      if (loop->kind != loop_orig->kind) {
        return true;
      }
    }
    return false;
  };

  std::vector<tir::BlockRV> unscheduled_blocks;

  for (auto name : common_blocks) {
    if (!is_scheduled(name)) {
      unscheduled_blocks.push_back(sch->GetBlock(name));
    }
  }

  return unscheduled_blocks;
}

void ScheduleFusedBlocks(tir::Schedule sch, tir::Trace anchor_trace, tvm::Target target) {
  auto sch_orig = sch->Copy();
  anchor_trace->ApplyToSchedule(sch, /*remove_postproc=*/false);

  auto unscheduled_blocks = GetUnscheduledBlocks(sch_orig, sch);

  if (unscheduled_blocks.empty()) {
    // All blocks have already been scheduled.
    // e.g. Applying a trace from conv2d -> add to conv2d -> subtract
    return;
  }

  auto inline_rule = GetDefaultAutoInline(target->kind->name);
  Optional<tir::BlockRV> last_block;

  for (auto block : unscheduled_blocks) {
    auto sch_copy = sch->Copy();
    inline_rule->Apply(sch, block);
    if (tvm::StructuralEqual()(sch->mod(), sch_copy->mod())) {
      ICHECK(!last_block.defined());
      last_block = block;
    }
  }

  if (last_block.defined()) {
    sch->ReverseComputeInline(last_block.value());
  }
}

}  // namespace meta_schedule
}  // namespace tvm
