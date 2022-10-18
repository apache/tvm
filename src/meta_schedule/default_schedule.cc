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
#include "../tir/schedule/utils.h"

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
    if (loops.size() != loops_orig.size()) {
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

std::vector<tir::BlockRV> ApplyAnchorTrace(tir::Schedule sch, tir::Trace anchor_trace) {
  std::unordered_map<const Object*, const Object*> rv_map;
  static auto kind_get_child_blocks = tir::InstructionKind::Get("GetChildBlocks");
  static auto kind_get_block = tir::InstructionKind::Get("GetBlock");
  auto block_names_orig = GetBlockNames(sch->mod());
  block_names_orig.insert("root");  // todo
  std::unordered_set<tir::BlockRV, ObjectHash, ObjectEqual> unknown_blocks;
  std::unordered_set<tir::LoopRV, ObjectHash, ObjectEqual> unknown_loops;
  std::set<std::string> scheduled_blocks;

  for (const auto& inst : anchor_trace->insts) {
    bool ok = true;
    for (auto input : inst->inputs) {
      if (input->IsInstance<tir::BlockRVNode>() &&
          unknown_blocks.count(Downcast<tir::BlockRV>(input))) {
        ok = false;
      } else if (input->IsInstance<tir::LoopRVNode>() &&
                 unknown_loops.count(Downcast<tir::LoopRV>(input))) {
        ok = false;
      }
    }

    if (!ok) {
      for (auto output : inst->outputs) {
        if (output->IsInstance<tir::BlockRVNode>()) {
          unknown_blocks.insert(Downcast<tir::BlockRV>(output));
        } else if (output->IsInstance<tir::LoopRVNode>()) {
          unknown_loops.insert(Downcast<tir::LoopRV>(output));
        }
      }
      continue;
    }

    if (inst->kind.same_as(kind_get_block)) {
      std::string block_name = Downcast<String>(inst->attrs[0]);

      bool found_match = false;
      for (auto name : block_names_orig) {
        if (block_name.find(name) == 0) {
          found_match = true;
        }
      }

      if (!found_match) {
        auto block = Downcast<tir::BlockRV>(inst->outputs[0]);
        unknown_blocks.insert(block);
        LOG(INFO) << "adding unknown block " << block_name << ", " << block;
        continue;
      } else {
	scheduled_blocks.insert(block_name);
      }
    }

    Array<ObjectRef> inputs = tir::TranslateInputRVs(inst->inputs, rv_map);
    Optional<ObjectRef> decision = anchor_trace->GetDecision(inst);
    Array<ObjectRef> outputs = inst->kind->f_apply_to_schedule(sch, inputs, inst->attrs, decision);

    if (inst->kind.same_as(kind_get_child_blocks)) {
      // We want to allow a trace generated for a single conv2d block to be applied to
      // conv2d -> elemwise blocks, where two conv2d are the same workload.
      // GetChildBlocks returns a different number of blocks for the two cases above, which
      // violates the assumption made by TranslateAddOutputRVs: old_outputs.size() ==
      // new_outputs.size(). We workaround this problem by assuming that the prefix of the "new"
      // outputs matches with the "old" outputs, and truncating the new outputs accordingly.
      ICHECK(inst->outputs.size() <= outputs.size());
      tir::TranslateAddOutputRVs(
          inst->outputs, Array<ObjectRef>(outputs.begin(), outputs.begin() + inst->outputs.size()),
          &rv_map);
    } else {
      tir::TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
    }
  }

  std::vector<tir::BlockRV> unscheduled_blocks;

  for (auto name : block_names_orig) {
    if (!scheduled_blocks.count(name)) {
      LOG(INFO) << "Unscheduled " << name;
      unscheduled_blocks.push_back(sch->GetBlock(name));
    } else {
      LOG(INFO) << "Scheduled " << name;
    }
  }

  return unscheduled_blocks;
}

void ScheduleFusedBlocks(tir::Schedule sch, tir::Trace anchor_trace, tvm::Target target) {
  auto sch_orig = sch->Copy();
  // LOG(INFO) << tir::AsTVMScript(sch_orig->mod());
  // LOG(INFO) << anchor_trace;

  auto unscheduled_blocks = ApplyAnchorTrace(sch, anchor_trace);

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

  // LOG(INFO) << tir::AsTVMScript(sch->mod());
}

}  // namespace meta_schedule
}  // namespace tvm
