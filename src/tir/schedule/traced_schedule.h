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
#ifndef TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_

#include "./concrete_schedule.h"

namespace tvm {
namespace tir {

class TracedScheduleNode : public ConcreteScheduleNode {
  friend class Schedule;

 protected:
  Trace trace_;

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `state_` is not visited
    // `error_render_level_` is not visited
    // `symbol_table_` is not visited
    // `analyzer_` is not visitied
    // `trace_` is not visited
  }

  ~TracedScheduleNode() = default;

 public:
  Optional<Trace> trace() const final { return trace_; }
  Schedule Copy() final;

 public:
  /******** Schedule: Sampling ********/
  ExprRV SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                           Optional<Integer> decision = NullOpt) final;
  Array<ExprRV> SamplePerfectTile(const LoopRV& loop_rv, int n, int max_innermost_factor,
                                  Optional<Array<Integer>> decision = NullOpt) final;
  LoopRV SampleComputeLocation(const BlockRV& block_rv, Optional<Integer> decision = NullOpt) final;
  /******** Schedule: Get blocks & loops ********/
  BlockRV GetBlock(const String& name, const Optional<String>& func_name) final;
  Array<LoopRV> GetLoops(const BlockRV& block_rv) final;
  Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) final;
  Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) final;
  Array<BlockRV> GetProducers(const BlockRV& block_rv) final;
  Array<BlockRV> GetConsumers(const BlockRV& block_rv) final;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const Array<LoopRV>& loop_rvs, bool preserve_unit_iters) final;
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factor_rvs,
                      bool preserve_unit_iters) final;
  void Reorder(const Array<LoopRV>& ordered_loop_rvs) final;
  LoopRV AddUnitLoop(const BlockRV& block_rv) final;
  LoopRV AddUnitLoop(const LoopRV& loop_rv) final;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) final;
  void Vectorize(const LoopRV& loop_rv) final;
  void Bind(const LoopRV& loop_rv, const String& thread_axis) final;
  void Unroll(const LoopRV& loop_rv) final;
  /******** Schedule: Insert cache stages ********/
  BlockRV CacheRead(const BlockRV& block_rv, int read_buffer_index, const String& storage_scope,
                    const Array<BlockRV> consumer_blocks = {}) final;
  BlockRV CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                     const String& storage_scope) final;
  Array<BlockRV> CacheInplace(const BlockRV& block_rv, int read_buffer_index,
                              const String& storage_scope) final;
  BlockRV ReIndex(const BlockRV& block_rv, int buffer_index,
                  BufferIndexType buffer_index_type) final;
  /******** Schedule: Compute location ********/
  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                 int index = -1) final;
  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                        int index = -1) final;
  void ComputeInline(const BlockRV& block_rv) final;
  void ReverseComputeInline(const BlockRV& block_rv) final;
  /******** Schedule: Reduction ********/
  BlockRV DecomposeReduction(const BlockRV& block_rv, const LoopRV& loop_rv) final;
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  /******** Schedule: Block annotation ********/
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) final;
  void SetScope(const BlockRV& block_rv, int buffer_index, const String& storage_scope) final;
  /******** Schedule: Blockize & Tensorize ********/
  BlockRV Blockize(const LoopRV& loop_rv) final;
  void Tensorize(const BlockRV& block_rv, const String& intrin) final;
  void Tensorize(const LoopRV& loop_rv, const String& intrin) final;
  /******** Schedule: Annotation ********/
  void Annotate(const LoopRV& loop_rv, const String& ann_key, const ObjectRef& ann_val) override;
  void Unannotate(const LoopRV& loop_rv, const String& ann_key) override;
  void Annotate(const BlockRV& block_rv, const String& ann_key, const ObjectRef& ann_val) override;
  void Unannotate(const BlockRV& block_rv, const String& ann_key) override;
  /******** Schedule: Layout transformation ********/
  void TransformLayout(const BlockRV& block_rv, int buffer_index, BufferIndexType buffer_index_type,
                       const IndexMap& index_map, const Optional<IndexMap>& pad_value) override;
  void TransformBlockLayout(const BlockRV& block_rv, const IndexMap& index_map) override;
  void SetAxisSeparator(const BlockRV& block_rv, int buffer_index,
                        BufferIndexType buffer_index_type,
                        const Array<IntImm>& axis_separators) final;
  /******** Schedule: Padding ********/
  BlockRV DecomposePadding(const BlockRV& block_rv, const LoopRV& loop_rv) final;
  void PadEinsum(const BlockRV& block_rv, const Array<Integer>& padding) final;
  /******** Schedule: Misc ********/
  void EnterPostproc() final;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
