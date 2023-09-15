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
void InlinePostBlocks(Schedule sch, Trace anchor_trace, Target target,
                      Map<String, String> target2record_block_mapping) {
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
    // Spatial blocks which are not mapped between record mod and target mod will be inlined here.
    auto block_sref = sch->GetSRef(block);
    if (IsSpatial(block_sref) && !target2record_block_mapping.count(name)) {
      StmtSRef scopeRoot =
          (name != "root") ? GetScopeRoot(sch->state(), block_sref, false) : block_sref;
      if (IsOutputBlock(sch->state(), block_sref, scopeRoot)) {
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

// Same to Structural Equal except ignoring float constant difference
class SEqualHandlerIgnoreFloat : public SEqualHandlerDefault {
 public:
  SEqualHandlerIgnoreFloat() : SEqualHandlerDefault(false, nullptr, false) {}

 protected:
  bool DispatchSEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars,
                            const Optional<ObjectPathPair>& current_paths) {
    if (auto lhs_ptr = lhs.as<FloatImmNode>(), rhs_ptr = rhs.as<FloatImmNode>();
        lhs_ptr && rhs_ptr) {
      return true;
    }
    return SEqualHandlerDefault::DispatchSEqualReduce(lhs, rhs, map_free_vars, current_paths);
  }
};

// Extract all blocks from a module
class BlockExtractor : public StmtExprVisitor {
 public:
  static Array<Block> ExtractBlocks(Stmt stmt) {
    BlockExtractor extractor;
    extractor(stmt);
    return extractor.blocks;
  }

 private:
  void VisitStmt_(const BlockNode* block) final {
    if (block->name_hint == "root") {
      StmtExprVisitor::VisitStmt_(block);
      return;
    }
    ICHECK(!in_block_) << "Nested blocks are not supported";
    in_block_ = true;
    blocks.push_back(GetRef<Block>(block));
    StmtExprVisitor::VisitStmt_(block);
    in_block_ = false;
  }
  Array<Block> blocks;
  bool in_block_ = false;
};

// Map blocks between record mod and target mod
std::pair<Map<String, String>, Map<String, String>> GetBlockMapping(IRModule target_mod,
                                                                    IRModule record_mod) {
  PrimFunc record_func = Downcast<PrimFunc>(record_mod->Lookup("main"));
  Array<Block> blocks_from_record = BlockExtractor::ExtractBlocks(record_func->body);
  PrimFunc target_func = Downcast<PrimFunc>(target_mod->Lookup("main"));
  Array<Block> blocks_from_target = BlockExtractor::ExtractBlocks(target_func->body);
  // Map blocks between record mod and target mod
  Map<String, String> record2target{{"root", "root"}};
  Map<String, String> target2record{{"root", "root"}};

  for (int i = 0; i < static_cast<int>(blocks_from_record.size()); i++) {
    for (int j = 0; j < static_cast<int>(blocks_from_target.size()); j++) {
      if (SEqualHandlerIgnoreFloat().Equal(blocks_from_record[i], blocks_from_target[j], false) &&
          !target2record.count(blocks_from_target[j]->name_hint)) {
        record2target.Set(blocks_from_record[i]->name_hint, blocks_from_target[j]->name_hint);
        target2record.Set(blocks_from_target[j]->name_hint, blocks_from_record[i]->name_hint);
        break;
      }
    }
  }
  return std::make_pair(target2record, record2target);
}

// Apply instructions from the anchor trace to the target schedule, and returns blocks
// that remain unscheduled.
std::vector<BlockRV> ApplyAnchorTrace(Schedule sch, Trace anchor_trace, const IRModule& record_mod,
                                      Map<String, String> record2target,
                                      Map<String, String> target2record) {
  static auto kind_get_child_blocks = InstructionKind::Get("GetChildBlocks");
  static auto kind_get_block = InstructionKind::Get("GetBlock");
  static auto kind_compute_inline = InstructionKind::Get("ComputeInline");
  static auto kind_reverse_compute_inline = InstructionKind::Get("ReverseComputeInline");
  static auto kind_decompose_reduction = InstructionKind::Get("DecomposeReduction");
  static auto kind_blockize = InstructionKind::Get("Blockize");

  const auto block_names_orig = GetBlockNames(sch->mod());
  const auto sch_orig = sch->Copy();
  Schedule sch_for_record_mod =
      Schedule::Traced(record_mod,
                       /*rand_state=*/-1,
                       /*debug_mode=*/0,
                       /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
  std::unordered_map<const Object*, const Object*> rv_map_record_mod;
  std::unordered_map<const Object*, const Object*> rv_map_target_mod;
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
    // execute instruction on record module
    auto inputs_for_record_mod = TranslateInputRVs(inst->inputs, rv_map_record_mod);
    Optional<ObjectRef> decision = anchor_trace->GetDecision(inst);
    Array<ObjectRef> outputs_for_record_mod = inst->kind->f_apply_to_schedule(
        sch_for_record_mod, inputs_for_record_mod, inst->attrs, decision);
    TranslateAddOutputRVs(inst->outputs, outputs_for_record_mod, &rv_map_record_mod);

    // execute instruction on target module
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

    Array<ObjectRef> inputs = TranslateInputRVs(inst->inputs, rv_map_target_mod);
    Array<ObjectRef> attrs = inst->attrs;
    if (inst->kind.same_as(kind_get_block)) {
      BlockRV block_rv_record = Downcast<BlockRV>(outputs_for_record_mod[0]);
      Block block_record = Downcast<Block>(sch_for_record_mod->Get(block_rv_record));
      if (!record2target.count(block_record->name_hint)) {
        // The anchor trace does get_block on a block that is not part of the target schedule.
        auto block = Downcast<BlockRV>(inst->outputs[0]);
        foreign_blocks.insert(block);
        continue;
      }
      attrs.Set(0, record2target[block_record->name_hint]);
    } else if (inst->kind.same_as(kind_reverse_compute_inline)) {
      // The anchor trace does reverse_compute_inline on a block, but the block with the same name
      // in the target schedule cannot be reverse compute inline-ed.
      // In such cases, it should be possible to apply compute_inline instead.
      auto block_rv = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block_rv);
      Block block = sch->Get(block_rv);
      if (target2record.count(block->name_hint)) {
        String record_block_name = target2record[block->name_hint];
        target2record.erase(block->name_hint);
        record2target.erase(record_block_name);
      }
      if (!CanReverseComputeInline(sch->state(), block_sref)) {
        ICHECK(CanComputeInline(sch->state(), block_sref));
        sch->ComputeInline(block_rv);
        continue;
      }
    } else if (inst->kind.same_as(kind_compute_inline)) {
      // Similar to the reverse_compute_inline case above.
      auto block_rv = Downcast<BlockRV>(inputs[0]);
      auto block_sref = sch->GetSRef(block_rv);
      Block block = sch->Get(block_rv);
      if (target2record.count(block->name_hint)) {
        String record_block_name = target2record[block->name_hint];
        target2record.erase(block->name_hint);
        record2target.erase(record_block_name);
      }
      auto state = sch->state();
      if (!CanComputeInline(state, block_sref)) {
        ICHECK(IsOutputBlock(state, block_sref, GetScopeRoot(state, block_sref, false)))
            << "If a spatial block cannot be inlined, it should be the output block";
        if (CanReverseComputeInline(sch->state(), block_sref)) {
          sch->ReverseComputeInline(block_rv);
        }
        continue;
      }
    }

    Array<ObjectRef> outputs = inst->kind->f_apply_to_schedule(sch, inputs, attrs, decision);

    if (inst->kind.same_as(kind_get_child_blocks)) {
      BlockRV input_rv = Downcast<BlockRV>(inputs[0]);
      Block input_block = Downcast<Block>(sch->Get(input_rv));
      // only add the mapped blocks to the output rvs
      Array<ObjectRef> new_target_outputs;
      for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
        auto block_target = sch->Get(Downcast<BlockRV>(outputs[i]));
        if (target2record.count(block_target->name_hint)) {
          new_target_outputs.push_back(outputs[i]);
        }
      }
      Array<ObjectRef> new_inst_outputs;
      for (int i = 0; i < static_cast<int>(outputs_for_record_mod.size()); i++) {
        auto block_record = sch_for_record_mod->Get(Downcast<BlockRV>(outputs_for_record_mod[i]));
        if (record2target.count(block_record->name_hint)) {
          new_inst_outputs.push_back(inst->outputs[i]);
        } else {
          foreign_blocks.insert(Downcast<BlockRV>(inst->outputs[i]));
        }
      }
      TranslateAddOutputRVs(new_inst_outputs, new_target_outputs, &rv_map_target_mod);
      continue;
    }
    // maintain block mapping
    ICHECK(outputs.size() == outputs_for_record_mod.size());
    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
      if (outputs[i]->IsInstance<BlockRVNode>()) {
        ICHECK(outputs_for_record_mod[i]->IsInstance<BlockRVNode>());
        auto block_rv_record = Downcast<BlockRV>(outputs_for_record_mod[i]);
        auto block_record = Downcast<Block>(sch_for_record_mod->Get(block_rv_record));
        auto block_rv_target = Downcast<BlockRV>(outputs[i]);
        auto block_target = Downcast<Block>(sch->Get(block_rv_target));
        record2target.Set(block_record->name_hint, block_target->name_hint);
        target2record.Set(block_target->name_hint, block_record->name_hint);
      }
    }
    if (inst->kind.same_as(kind_decompose_reduction)) {
      // decompose_reduction will modify input block
      ICHECK(inputs.size() == inputs_for_record_mod.size());
      for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        if (inputs[i]->IsInstance<BlockRVNode>()) {
          auto block_rv_record = Downcast<BlockRV>(inputs_for_record_mod[i]);
          auto block_record = Downcast<Block>(sch_for_record_mod->Get(block_rv_record));
          auto block_rv_target = Downcast<BlockRV>(inputs[i]);
          auto block_target = Downcast<Block>(sch->Get(block_rv_target));
          record2target.Set(block_record->name_hint, block_target->name_hint);
          target2record.Set(block_target->name_hint, block_record->name_hint);
        }
      }
    } else if (inst->kind.same_as(kind_blockize)) {
      // blockize will generate a block under the init block
      Block target_output_block = Downcast<Block>(sch->Get(Downcast<BlockRV>(outputs[0])));
      if (target_output_block->init.defined()) {
        Array<Block> target_init_blocks =
            BlockExtractor::ExtractBlocks(target_output_block->init.value());
        Block record_output_block =
            Downcast<Block>(sch_for_record_mod->Get(Downcast<BlockRV>(outputs_for_record_mod[0])));
        ICHECK(record_output_block->init.defined());
        Array<Block> record_init_blocks =
            BlockExtractor::ExtractBlocks(record_output_block->init.value());
        ICHECK(target_init_blocks.size() == 1);
        ICHECK(record_init_blocks.size() == 1);
        record2target.Set(record_init_blocks[0]->name_hint, target_init_blocks[0]->name_hint);
        target2record.Set(target_init_blocks[0]->name_hint, record_init_blocks[0]->name_hint);
      }
    }
    TranslateAddOutputRVs(inst->outputs, outputs, &rv_map_target_mod);
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

void ScheduleUsingAnchorTrace(Schedule sch, const Trace& anchor_trace, const IRModule& record_mod,
                              const tvm::Target& target) {
  Map<String, String> record2target, target2record;
  std::tie(target2record, record2target) = GetBlockMapping(sch->mod(), record_mod);
  InlinePostBlocks(sch, anchor_trace, target, target2record);

  auto unscheduled_blocks =
      ApplyAnchorTrace(sch, anchor_trace, record_mod, record2target, target2record);
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
