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

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../tir/schedule/analysis.h"
#include "utils.h"

namespace tvm {
namespace meta_schedule {

using namespace tir;

// Returns true if b1 is an ancestor of b2
bool IsAncestor(BlockRV b1, BlockRV b2, Schedule sch) {
  if (sch->Get(b1)->name_hint == sch->Get(b2)->name_hint) {
    return true;
  }
  for (auto prod : sch->GetProducers(b2)) {
    if (IsAncestor(b1, prod, sch)) return true;
  }
  return false;
}

// Inline or reverse inline spatial blocks after the anchor block
void InlinePostBlocks(Schedule sch, Trace anchor_trace, Target target) {
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  // We let blocks whose names are referenced in the anchor trace be scheduled by the anchor trace.
  // We record such block names to avoid inlining them here.
  std::unordered_set<std::string> get_block_names;
  for (const auto& inst : anchor_trace->insts) {
    if (inst->kind.same_as(kind_get_block)) {
      auto block_name = Downcast<String>(inst->attrs[0]);
      ICHECK(block_name.defined());
      get_block_names.insert(block_name);
    }
  }

  auto anchor_block = FindAnchorBlock(sch->mod());

  std::vector<std::string> inline_todos;
  std::optional<int> last_block_idx{std::nullopt};

  for (auto name : GetBlockNames(sch->mod())) {
    auto block = sch->GetBlock(name);
    if (anchor_block) {
      auto anchor_block_rv = sch->GetBlock(anchor_block->name_hint);
      if (IsAncestor(block, anchor_block_rv, sch)) continue;
    }
    // Spatial blocks which are not referenced in the anchor trace will be inlined here.
    auto block_sref = sch->GetSRef(block);
    if (IsSpatial(block_sref) && !get_block_names.count(name)) {
      if (IsOutputBlock(sch->state(), block_sref, GetScopeRoot(sch->state(), block_sref, false))) {
        last_block_idx = inline_todos.size();
      }
      inline_todos.push_back(name);
    }
  }

  if (last_block_idx) {
    // The last block can only be reverse compute inlined. We make sure to inline all
    // producer blocks of the last block beforehand so that reverse compute inline can succeed.
    std::swap(inline_todos[*last_block_idx], inline_todos.back());
  }

  auto inline_rule = GetDefaultAutoInline(target->kind->name);

  for (auto name : inline_todos) {
    inline_rule->Apply(sch, sch->GetBlock(name));
  }
}

// Apply instructions from the anchor trace to the target schedule, and returns blocks
// that remain unscheduled.
std::vector<BlockRV> ApplyAnchorTrace(Schedule sch, Trace anchor_trace) {
  static auto kind_get_child_blocks = InstructionKind::Get("GetChildBlocks");
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  static auto kind_compute_inline = InstructionKind::Get("ComputeInline");
  static auto kind_reverse_compute_inline = InstructionKind::Get("ReverseComputeInline");

  const auto block_names_orig = GetBlockNames(sch->mod());
  const auto sch_orig = sch->Copy();

  std::unordered_map<const Object*, const Object*> rv_map;
  // Blocks and loops that appear in the anchor trace but are not part of the target schedule.
  std::unordered_set<BlockRV, ObjectHash, ObjectEqual> foreign_blocks;
  std::unordered_set<LoopRV, ObjectHash, ObjectEqual> foreign_loops;

  // Instructions in the anchor trace can be applied only if all inputs are part of the target
  // schedule.
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
      // If we find an instruction that is not applicable, its outputs are recorded as "foreign"
      // to the target schedule.
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

    if (inst->kind.same_as(kind_get_block) && !HasBlock(sch, Downcast<String>(inst->attrs[0]))) {
      // The anchor trace does get_block on a block that is not part of the target schedule.
      auto block = Downcast<BlockRV>(inst->outputs[0]);
      foreign_blocks.insert(block);
      continue;
    } else if (inst->kind.same_as(kind_reverse_compute_inline)) {
      // The anchor trace does reverse_compute_inline on a block, but the block with the same name
      // in the target schedule cannot be reverse compute inline-ed.
      // In such cases, it should be possible to apply compute_inline instead.
      auto block = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block);
      if (!CanReverseComputeInline(sch->state(), block_sref)) {
        ICHECK(CanComputeInline(sch->state(), block_sref));
        sch->ComputeInline(block);
        continue;
      }
    } else if (inst->kind.same_as(kind_compute_inline)) {
      // Similar to the reverse_compute_inline case above.
      auto block = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block);
      auto state = sch->state();
      if (!CanComputeInline(state, block_sref)) {
        ICHECK(IsOutputBlock(state, block_sref, GetScopeRoot(state, block_sref, false)))
            << "If a spatial block cannot be inlined, it should be the output block";
        if (CanReverseComputeInline(sch->state(), block_sref)) {
          sch->ReverseComputeInline(block);
        }
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
  ICHECK(unscheduled_blocks.size() <= 1)
      << "All blocks should have been scheduled or only one (fused) spatial block can remain "
         "unscheduled at this point.";

  if (unscheduled_blocks.empty()) {
    // All blocks have already been scheduled.
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

TVM_REGISTER_GLOBAL("meta_schedule.ScheduleUsingAnchorTrace")
    .set_body_typed(ScheduleUsingAnchorTrace);

}  // namespace meta_schedule
}  // namespace tvm
