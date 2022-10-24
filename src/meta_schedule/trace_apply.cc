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
#include "trace_apply.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace tvm {
namespace meta_schedule {

std::unordered_set<std::string> GetBlockNames(const IRModule& mod) {
  struct BlockNameCollector : public tir::StmtVisitor {
    void VisitStmt_(const tir::BlockNode* block) override {
      block_names.insert(block->name_hint);
      StmtVisitor::VisitStmt(block->body);
    }
    std::unordered_set<std::string> block_names;
  };

  auto prim_func = tir::FindEntryFunc(mod, nullptr);
  BlockNameCollector collector;
  collector(prim_func->body);
  return collector.block_names;
}

using namespace tir;

bool IsAncestor(BlockRV b1, BlockRV b2, Schedule sch) {
  if (sch->Get(b1)->name_hint == sch->Get(b2)->name_hint) {
    return true;
  }
  for (auto prod : sch->GetProducers(b2)) {
    if (IsAncestor(b1, prod, sch)) return true;
  }
  return false;
}

void InlinePostBlocks(Schedule sch, Trace anchor_trace, Target target) {
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  std::unordered_set<std::string> get_block_names;
  for (const auto& inst : anchor_trace->insts) {
    if (inst->kind.same_as(kind_get_block)) {
      auto block_name = Downcast<String>(inst->attrs[0]);
      ICHECK(block_name.defined());
      get_block_names.insert(block_name);
    }
  }

  auto anchor_block = FindAnchorBlock(sch->mod());
  std::optional<BlockRV> anchor_block_rv = std::nullopt;
  if (anchor_block) {
    anchor_block_rv = sch->GetBlock(anchor_block->name_hint);
  }

  auto inline_rule = GetDefaultAutoInline(target->kind->name);

  for (auto name : GetBlockNames(sch->mod())) {
    auto block = sch->GetBlock(name);
    if (anchor_block_rv && IsAncestor(block, *anchor_block_rv, sch)) continue;
    if (IsSpatial(sch->GetSRef(block)) && !get_block_names.count(name)) {
      inline_rule->Apply(sch, block);
    }
  }
}

std::vector<BlockRV> ApplyAnchorTrace(Schedule sch, Trace anchor_trace) {
  static auto kind_get_child_blocks = InstructionKind::Get("GetChildBlocks");
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  static auto kind_compute_inline = InstructionKind::Get("ComputeInline");
  static auto kind_reverse_compute_inline = InstructionKind::Get("ReverseComputeInline");

  const auto block_names_orig = GetBlockNames(sch->mod());
  const auto sch_orig = sch->Copy();

  std::unordered_map<const Object*, const Object*> rv_map;
  std::unordered_set<BlockRV, ObjectHash, ObjectEqual> foreign_blocks;
  std::unordered_set<LoopRV, ObjectHash, ObjectEqual> foreign_loops;

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

    Array<ObjectRef> inputs = TranslateInputRVs(inst->inputs, rv_map);

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
      }
    } else if (inst->kind.same_as(kind_reverse_compute_inline)) {
      auto block = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block);
      if (!CanReverseComputeInline(sch->state(), block_sref)) {
        ICHECK(CanComputeInline(sch->state(), block_sref));
        sch->ComputeInline(block);
        continue;
      }
    } else if (inst->kind.same_as(kind_compute_inline)) {
      auto block = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block);
      if (!CanComputeInline(sch->state(), block_sref)) {
        ICHECK(CanReverseComputeInline(sch->state(), block_sref));
        sch->ReverseComputeInline(block);
        continue;
      }
    }

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

  const auto block_names_now = GetBlockNames(sch->mod());
  std::vector<BlockRV> unscheduled_blocks;

  for (auto name : block_names_orig) {
    if (block_names_now.count(name) && name != "root" && !is_scheduled(name)) {
      unscheduled_blocks.push_back(sch->GetBlock(name));
    }
  }

  return unscheduled_blocks;
}

void ScheduleUsingAnchorTrace(Schedule sch, const Trace& anchor_trace, const tvm::Target& target) {
  InlinePostBlocks(sch, anchor_trace, target);

  auto unscheduled_blocks = ApplyAnchorTrace(sch, anchor_trace);
  ICHECK(unscheduled_blocks.size() <= 1);

  if (unscheduled_blocks.empty()) {
    // All blocks have already been scheduled.
    // e.g. Applying a trace from conv2d -> add to
    //       - conv2d -> add -> add
    //       - conv2d -> subtract
    return;
  }

  auto last_block = unscheduled_blocks[0];
  auto last_block_producers = sch->GetProducers(last_block);

  if (last_block_producers.size() == 1 && IsSpatial(sch->GetSRef(last_block_producers[0]))) {
    // Inline into the cache write stage
    sch->ReverseComputeInline(last_block);
  } else if (target->kind->name == "llvm" || target->kind->name == "hexagon") {
    sch->Parallel(sch->Fuse(sch->GetLoops(last_block)));
  } else if (IsGPUTarget(target->kind->name)) {
    auto max_threads_per_block = target->GetAttr<Integer>("max_threads_per_block");
    ICHECK(max_threads_per_block.defined())
        << "ValueError: missing attribute `max_threads_per_block` in the target";

    auto auto_bind_rule =
        ScheduleRule::AutoBind(/*max_threadblocks=*/256,
                               /*thread_extents*/ Array<Integer>{32, 64, 128, 256, 512, 1024},
                               max_threads_per_block.value()->value);
    auto_bind_rule->Apply(sch, last_block);
  }
}

}  // namespace meta_schedule
}  // namespace tvm
