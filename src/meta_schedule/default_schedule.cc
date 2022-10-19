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
      block_names.insert(block->name_hint);
      StmtVisitor::VisitStmt(block->body);
    }
    std::set<std::string> block_names;
  };

  auto prim_func = tir::FindEntryFunc(mod, nullptr);
  BlockNameCollector collector;
  collector(prim_func->body);
  return collector.block_names;
}

std::vector<tir::BlockRV> ApplyAnchorTrace(tir::Schedule sch, tir::Trace anchor_trace) {
  using namespace tir;
  std::unordered_map<const Object*, const Object*> rv_map;
  static auto kind_get_child_blocks = InstructionKind::Get("GetChildBlocks");
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  const auto block_names_orig = GetBlockNames(sch->mod());
  std::unordered_set<BlockRV, ObjectHash, ObjectEqual> foreign_blocks;
  std::unordered_set<LoopRV, ObjectHash, ObjectEqual> foreign_loops;
  std::set<std::string> scheduled_blocks;

  const auto sch_orig = sch->Copy();

  auto is_inst_applicable = [&foreign_blocks, &foreign_loops](Instruction inst) {
    for (auto input : inst->inputs) {
      if (!input.defined()) continue;
      if ((input->IsInstance<BlockRVNode>() && foreign_blocks.count(Downcast<BlockRV>(input))) ||
          (input->IsInstance<LoopRVNode>() && foreign_loops.count(Downcast<LoopRV>(input)))) {
        return false;
      }
    }
    return true;
  };

  for (const auto& inst : anchor_trace->insts) {
    if (!is_inst_applicable(inst)) {
      for (auto output : inst->outputs) {
        if (output->IsInstance<BlockRVNode>()) {
          foreign_blocks.insert(Downcast<BlockRV>(output));
        } else if (output->IsInstance<LoopRVNode>()) {
          foreign_loops.insert(Downcast<LoopRV>(output));
        }
      }
      continue;
    }

    if (inst->kind.same_as(kind_get_block)) {
      auto find_prefix_any = [&block_names_orig](const std::string& block_name) {
        for (auto name : block_names_orig) {
          if (block_name.find(name) == 0) {
            return true;
          }
        }
        return false;
      };

      auto block_name = Downcast<String>(inst->attrs[0]);
      ICHECK(block_name.defined());

      if (!find_prefix_any(block_name)) {
        auto block = Downcast<BlockRV>(inst->outputs[0]);
        foreign_blocks.insert(block);
        continue;
      } else {
        scheduled_blocks.insert(block_name);
      }
    }

    Array<ObjectRef> inputs = TranslateInputRVs(inst->inputs, rv_map);
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
      TranslateAddOutputRVs(
          inst->outputs, Array<ObjectRef>(outputs.begin(), outputs.begin() + inst->outputs.size()),
          &rv_map);
    } else {
      TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
    }
  }

  const auto block_names_now = GetBlockNames(sch->mod());

  auto is_scheduled = [=, &scheduled_blocks](const std::string& block_name) {
    if (!block_names_now.count(block_name) || scheduled_blocks.count(block_name)) {
      return true;
    }
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

  std::vector<BlockRV> unscheduled_blocks;

  for (auto name : block_names_orig) {
    if (!is_scheduled(name)) {
      unscheduled_blocks.push_back(sch->GetBlock(name));
    }
  }

  return unscheduled_blocks;
}

void ScheduleFusedBlocks(tir::Schedule sch, tir::Trace anchor_trace, tvm::Target target) {
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
}

}  // namespace meta_schedule
}  // namespace tvm
