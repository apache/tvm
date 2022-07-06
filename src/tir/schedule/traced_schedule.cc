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
#include "./traced_schedule.h"

namespace tvm {
namespace tir {

Schedule Schedule::Traced(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                          int debug_mask, ScheduleErrorRenderLevel error_render_level) {
  ObjectPtr<TracedScheduleNode> n = make_object<TracedScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mask);
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->trace_ = Trace();
  n->Seed(seed);
  GlobalVar gv = NullValue<GlobalVar>();
  if (FindEntryFunc(mod, &gv) != nullptr) {
    n->func_working_on_ = gv;
  } else {
    n->func_working_on_ = NullOpt;
  }
  return Schedule(std::move(n));
}

Schedule TracedScheduleNode::Copy() {
  ObjectPtr<TracedScheduleNode> n = make_object<TracedScheduleNode>();
  n->error_render_level_ = this->error_render_level_;
  ConcreteScheduleNode::Copy(&n->state_, &n->symbol_table_);
  n->func_working_on_ = this->func_working_on_;
  n->analyzer_ = std::make_unique<arith::Analyzer>();  // new analyzer needed because it is stateful
  n->rand_state_ = ForkSeed();
  n->trace_ = Trace(this->trace_->insts, this->trace_->decisions);
  return Schedule(std::move(n));
}

/******** Schedule: Sampling ********/

ExprRV TracedScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                             const Array<FloatImm>& probs,
                                             Optional<Integer> decision) {
  ExprRV result =
      CreateRV(tir::SampleCategorical(&this->rand_state_, candidates, probs, &decision));
  static const InstructionKind& kind = InstructionKind::Get("SampleCategorical");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{candidates, probs},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

Array<ExprRV> TracedScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                    int max_innermost_factor,
                                                    Optional<Array<Integer>> decision) {
  Array<ExprRV> results = CreateRV(tir::SamplePerfectTile(
      &this->rand_state_, this->GetSRef(loop_rv), n, max_innermost_factor, &decision));

  static const InstructionKind& kind = InstructionKind::Get("SamplePerfectTile");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{Integer(n), Integer(max_innermost_factor)},
                                      /*outputs=*/{results.begin(), results.end()}),
                 /*decision=*/decision);
  return results;
}

LoopRV TracedScheduleNode::SampleComputeLocation(const BlockRV& block_rv,
                                                 Optional<Integer> decision) {
  LoopRV result = CreateRV<LoopRV>(tir::SampleComputeLocation(this->state_, &this->rand_state_,
                                                              this->GetSRef(block_rv), &decision));

  static const InstructionKind& kind = InstructionKind::Get("SampleComputeLocation");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}),
                 /*decision=*/decision);
  return result;
}

/******** Schedule: Get blocks & loops ********/

BlockRV TracedScheduleNode::GetBlock(const String& name, const Optional<String>& func_name) {
  GlobalVar gv = NullValue<GlobalVar>();
  if (func_name.defined()) {
    gv = state_->mod->GetGlobalVar(func_name.value());
  } else if (func_working_on_.defined()) {
    gv = this->func_working_on_.value();
  } else {
    LOG(FATAL) << "ValueError: `get_block` does not know which function to be working on. Please "
                  "specify the function name explicitly, or call `work_on` to specify the function "
                  "before using `get_block`.";
  }
  BlockRV result = ConcreteScheduleNode::GetBlock(name, func_name);

  static const InstructionKind& kind = InstructionKind::Get("GetBlock");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{},
                                      /*attrs=*/{name, gv->name_hint},
                                      /*outputs=*/{result}));
  return result;
}

Array<LoopRV> TracedScheduleNode::GetLoops(const BlockRV& block_rv) {
  Array<LoopRV> results = ConcreteScheduleNode::GetLoops(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetLoops");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetChildBlocks(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetChildBlocks(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetChildBlocks");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetProducers(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetProducers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetProducers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

Array<BlockRV> TracedScheduleNode::GetConsumers(const BlockRV& block_rv) {
  Array<BlockRV> results = ConcreteScheduleNode::GetConsumers(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("GetConsumers");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,  //
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

/******** Schedule: Transform loops ********/

LoopRV TracedScheduleNode::Fuse(const Array<LoopRV>& loop_rvs, bool preserve_unit_loops) {
  LoopRV result = ConcreteScheduleNode::Fuse(loop_rvs, preserve_unit_loops);

  static const InstructionKind& kind = InstructionKind::Get("Fuse");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rvs.begin(), loop_rvs.end()},
                                      /*attrs=*/{Integer(preserve_unit_loops)},
                                      /*outputs=*/{result}));
  return result;
}

Array<LoopRV> TracedScheduleNode::Split(const LoopRV& loop_rv,
                                        const Array<Optional<ExprRV>>& factor_rvs,
                                        bool preserve_unit_iters) {
  Array<LoopRV> results = ConcreteScheduleNode::Split(loop_rv, factor_rvs, preserve_unit_iters);

  std::vector<ObjectRef> inputs;
  inputs.reserve(1 + factor_rvs.size());
  inputs.push_back(loop_rv);
  for (const ObjectRef& obj : factor_rvs) {
    inputs.push_back(obj);
  }

  static const InstructionKind& kind = InstructionKind::Get("Split");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/inputs,
                                      /*attrs=*/{Integer(preserve_unit_iters)},
                                      /*outputs=*/{results.begin(), results.end()}));
  return results;
}

void TracedScheduleNode::Reorder(const Array<LoopRV>& ordered_loop_rvs) {
  ConcreteScheduleNode::Reorder(ordered_loop_rvs);

  static const InstructionKind& kind = InstructionKind::Get("Reorder");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{ordered_loop_rvs.begin(), ordered_loop_rvs.end()},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

LoopRV TracedScheduleNode::AddUnitLoop(const BlockRV& block_rv) {
  LoopRV result = ConcreteScheduleNode::AddUnitLoop(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("AddUnitLoop");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

LoopRV TracedScheduleNode::AddUnitLoop(const LoopRV& loop_rv) {
  LoopRV result = ConcreteScheduleNode::AddUnitLoop(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("AddUnitLoop");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: Manipulate ForKind ********/

void TracedScheduleNode::Parallel(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Parallel(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Parallel");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Vectorize(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Vectorize(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Vectorize");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Bind(const LoopRV& loop_rv, const String& thread_axis) {
  ConcreteScheduleNode::Bind(loop_rv, thread_axis);

  static const InstructionKind& kind = InstructionKind::Get("Bind");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{thread_axis},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Unroll(const LoopRV& loop_rv) {
  ConcreteScheduleNode::Unroll(loop_rv);

  static const InstructionKind& kind = InstructionKind::Get("Unroll");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Insert cache stages ********/
BlockRV TracedScheduleNode::CacheRead(const BlockRV& block_rv, int read_buffer_index,
                                      const String& storage_scope) {
  BlockRV result = ConcreteScheduleNode::CacheRead(block_rv, read_buffer_index, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("CacheRead");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(read_buffer_index), storage_scope},
                                      /*outputs=*/{result}));
  return result;
}

BlockRV TracedScheduleNode::CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                                       const String& storage_scope) {
  BlockRV result = ConcreteScheduleNode::CacheWrite(block_rv, write_buffer_index, storage_scope);

  static const InstructionKind& kind = InstructionKind::Get("CacheWrite");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(write_buffer_index), storage_scope},
                                      /*outputs=*/{result}));
  return result;
}

BlockRV TracedScheduleNode::ReIndex(const BlockRV& block_rv, int buffer_index,
                                    BufferIndexType buffer_index_type) {
  BlockRV result = ConcreteScheduleNode::ReIndex(block_rv, buffer_index, buffer_index_type);

  static const InstructionKind& kind = InstructionKind::Get("ReIndex");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{Integer(buffer_index), Integer(buffer_index_type)},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: Compute location ********/

void TracedScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                   bool preserve_unit_loops) {
  ConcreteScheduleNode::ComputeAt(block_rv, loop_rv, preserve_unit_loops);

  static const InstructionKind& kind = InstructionKind::Get("ComputeAt");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{Integer(preserve_unit_loops)},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                          bool preserve_unit_loops) {
  ConcreteScheduleNode::ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeAt");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{Integer(preserve_unit_loops)},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ComputeInline(const BlockRV& block_rv) {
  ConcreteScheduleNode::ComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  ConcreteScheduleNode::ReverseComputeInline(block_rv);

  static const InstructionKind& kind = InstructionKind::Get("ReverseComputeInline");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

/******** Schedule: Reduction ********/

BlockRV TracedScheduleNode::DecomposeReduction(const BlockRV& block_rv, const LoopRV& loop_rv) {
  BlockRV result = ConcreteScheduleNode::DecomposeReduction(block_rv, loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("DecomposeReduction");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, loop_rv},
                                      /*attrs=*/{},
                                      /*outputs=*/{result}));
  return result;
}

BlockRV TracedScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  BlockRV result = ConcreteScheduleNode::RFactor(loop_rv, factor_axis);
  static const InstructionKind& kind = InstructionKind::Get("RFactor");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{Integer(factor_axis)},
                                      /*outputs=*/{result}));
  return result;
}

/******** Schedule: Block annotation ********/

void TracedScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis,
                                      int factor, int offset) {
  ConcreteScheduleNode::StorageAlign(block_rv, buffer_index, axis, factor, offset);
  static const InstructionKind& kind = InstructionKind::Get("StorageAlign");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{Integer(buffer_index), Integer(axis), Integer(factor), Integer(offset)},
      /*outputs=*/{}));
}

void TracedScheduleNode::SetScope(const BlockRV& block_rv, int buffer_index,
                                  const String& storage_scope) {
  ConcreteScheduleNode::SetScope(block_rv, buffer_index, storage_scope);
  static const InstructionKind& kind = InstructionKind::Get("SetScope");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{Integer(buffer_index), storage_scope},
      /*outputs=*/{}));
}

/******** Schedule: Blockize & Tensorize ********/

BlockRV TracedScheduleNode::Blockize(const LoopRV& loop_rv) {
  BlockRV new_block = ConcreteScheduleNode::Blockize(loop_rv);
  static const InstructionKind& kind = InstructionKind::Get("Blockize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{loop_rv},
      /*attrs=*/{},
      /*outputs=*/{new_block}));
  return new_block;
}

void TracedScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin) {
  ConcreteScheduleNode::Tensorize(loop_rv, intrin);
  static const InstructionKind& kind = InstructionKind::Get("Tensorize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{loop_rv},
      /*attrs=*/{intrin},
      /*outputs=*/{}));
}

void TracedScheduleNode::Tensorize(const BlockRV& block_rv, const String& intrin) {
  ConcreteScheduleNode::Tensorize(block_rv, intrin);
  static const InstructionKind& kind = InstructionKind::Get("Tensorize");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{intrin},
      /*outputs=*/{}));
}

/******** Schedule: Annotation ********/

void TracedScheduleNode::Annotate(const LoopRV& loop_rv, const String& ann_key,
                                  const ObjectRef& ann_val) {
  ConcreteScheduleNode::Annotate(loop_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("Annotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Annotate(const BlockRV& block_rv, const String& ann_key,
                                  const ObjectRef& ann_val) {
  ConcreteScheduleNode::Annotate(block_rv, ann_key, ann_val);
  static const InstructionKind& kind = InstructionKind::Get("Annotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv, ann_val},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Unannotate(const LoopRV& loop_rv, const String& ann_key) {
  ConcreteScheduleNode::Unannotate(loop_rv, ann_key);
  static const InstructionKind& kind = InstructionKind::Get("Unannotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{loop_rv},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

void TracedScheduleNode::Unannotate(const BlockRV& block_rv, const String& ann_key) {
  ConcreteScheduleNode::Unannotate(block_rv, ann_key);
  static const InstructionKind& kind = InstructionKind::Get("Unannotate");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{block_rv},
                                      /*attrs=*/{ann_key},
                                      /*outputs=*/{}));
}

/******** Schedule: Layout transformation ********/

void TracedScheduleNode::TransformLayout(const BlockRV& block_rv, int buffer_index,
                                         BufferIndexType buffer_index_type,
                                         const IndexMap& index_map) {
  ConcreteScheduleNode::TransformLayout(block_rv, buffer_index, buffer_index_type, index_map);
  static const InstructionKind& kind = InstructionKind::Get("TransformLayout");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv},
                           /*attrs=*/{Integer(buffer_index), Integer(buffer_index_type), index_map},
                           /*outputs=*/{}));
}

void TracedScheduleNode::TransformBlockLayout(const BlockRV& block_rv, const IndexMap& index_map) {
  ConcreteScheduleNode::TransformBlockLayout(block_rv, index_map);
  static const InstructionKind& kind = InstructionKind::Get("TransformBlockLayout");
  trace_->Append(
      /*inst=*/Instruction(/*kind=*/kind,
                           /*inputs=*/{block_rv},
                           /*attrs=*/{index_map},
                           /*outputs=*/{}));
}

void TracedScheduleNode::SetAxisSeparator(const BlockRV& block_rv, int buffer_index,
                                          BufferIndexType buffer_index_type,
                                          const Array<IntImm>& axis_separators) {
  ConcreteScheduleNode::SetAxisSeparator(block_rv, buffer_index, buffer_index_type,
                                         axis_separators);
  static const InstructionKind& kind = InstructionKind::Get("SetAxisSeparator");
  trace_->Append(/*inst=*/Instruction(
      /*kind=*/kind,
      /*inputs=*/{block_rv},
      /*attrs=*/{Integer(buffer_index), Integer(buffer_index_type), axis_separators},
      /*outputs=*/{}));
}

/******** Schedule: Misc ********/

void TracedScheduleNode::EnterPostproc() {
  ConcreteScheduleNode::EnterPostproc();
  static const InstructionKind& kind = InstructionKind::Get("EnterPostproc");
  trace_->Append(/*inst=*/Instruction(/*kind=*/kind,
                                      /*inputs=*/{},
                                      /*attrs=*/{},
                                      /*outputs=*/{}));
}

}  // namespace tir
}  // namespace tvm
