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
#include <tvm/ffi/container/array.h>

namespace tvm {
namespace tir {

class TracedScheduleNode : public ConcreteScheduleNode {
  friend class Schedule;

 protected:
  Trace trace_;

 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TracedScheduleNode>();
  }

  ~TracedScheduleNode() = default;

 public:
  ffi::Optional<Trace> trace() const final { return trace_; }
  Schedule Copy() final;

 public:
  /******** Schedule: Sampling ********/
  ExprRV SampleCategorical(const ffi::Array<Integer>& candidates, const ffi::Array<FloatImm>& probs,
                           ffi::Optional<Integer> decision = std::nullopt) final;
  ffi::Array<ExprRV> SamplePerfectTile(
      const LoopRV& loop_rv, int n, int max_innermost_factor,
      ffi::Optional<ffi::Array<Integer>> decision = std::nullopt) final;
  ffi::Array<ExprRV> SamplePartitionedTile(
      const LoopRV& loop_rv, int n, int partition_pos, int innerpart_factor,
      ffi::Optional<ffi::Array<Integer>> decision = std::nullopt) final;
  LoopRV SampleComputeLocation(const BlockRV& block_rv,
                               ffi::Optional<Integer> decision = std::nullopt) final;
  /******** Schedule: Get blocks & loops ********/
  BlockRV GetBlock(const ffi::String& name, const ffi::Optional<ffi::String>& func_name) final;
  ffi::Array<LoopRV> GetLoops(const BlockRV& block_rv) final;
  ffi::Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) final;
  ffi::Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) final;
  ffi::Array<BlockRV> GetProducers(const BlockRV& block_rv) final;
  ffi::Array<BlockRV> GetConsumers(const BlockRV& block_rv) final;
  ffi::Array<BlockRV> GetOutputBlocks(const BlockRV& scope_block_rv) final;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const ffi::Array<LoopRV>& loop_rvs, bool preserve_unit_iters) final;
  LoopRV Merge(const ffi::Array<LoopRV>& loop_rvs) final;
  ffi::Array<LoopRV> Split(const LoopRV& loop_rv,
                           const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                           bool preserve_unit_iters, bool disable_predication) final;
  ffi::Array<LoopRV> LoopPartition(const LoopRV& loop_rv,
                                   const ffi::Array<ffi::Optional<ExprRV>>& factor_rvs,
                                   bool preserve_unit_iters) final;
  void Reorder(const ffi::Array<LoopRV>& ordered_loop_rvs) final;
  void ReorderBlockIterVar(const BlockRV& block_rv, const ffi::Array<Integer> new_order) final;
  LoopRV AddUnitLoop(const BlockRV& block_rv) final;
  LoopRV AddUnitLoop(const LoopRV& loop_rv) final;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) final;
  void Vectorize(const LoopRV& loop_rv) final;
  void Bind(const LoopRV& loop_rv, const ffi::String& thread_axis) final;
  void Unroll(const LoopRV& loop_rv) final;
  /******** Schedule: Insert cache stages ********/
  BlockRV CacheRead(const BlockRV& block_rv, int read_buffer_index,
                    const ffi::String& storage_scope,
                    const ffi::Array<BlockRV> consumer_blocks = {}) final;
  BlockRV CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                     const ffi::String& storage_scope,
                     const ffi::Array<BlockRV> consumer_blocks = {}) final;
  BlockRV ReindexCacheRead(const BlockRV& block_rv, int read_buffer_index,
                           const ffi::String& storage_scope, const IndexMap& index_map) final;
  BlockRV ReindexCacheWrite(const BlockRV& block_rv, int write_buffer_index,
                            const ffi::String& storage_scope, const IndexMap& index_map) final;
  ffi::Array<BlockRV> CacheInplace(const BlockRV& block_rv, int read_buffer_index,
                                   const ffi::String& storage_scope) final;
  BlockRV ReIndex(const BlockRV& block_rv, int buffer_index,
                  BufferIndexType buffer_index_type, bool skip_simplify) final;
  ffi::Array<BlockRV> CacheIndex(const BlockRV& block_rv, const ffi::String& storage_scope,
                            int cse_thresh) final;
  /******** Schedule: Data movement ********/
  BlockRV ReadAt(const LoopRV& loop_rv, const BlockRV& block_rv, int read_buffer_index,
                 const ffi::String& storage_scope) final;
  BlockRV WriteAt(const LoopRV& loop_rv, const BlockRV& block_rv, int write_buffer_index,
                  const ffi::String& storage_scope) final;
  /******** Schedule: Compute location ********/
  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                 int index = -1) final;
  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                        int index = -1) final;
  void ComputeInline(const BlockRV& block_rv) final;
  void ReverseComputeInline(const BlockRV& block_rv) final;
  void FuseReductionEpilogue(const BlockRV& reduction_block, const BlockRV& epilogue_block) final;
  /******** Schedule: Reduction ********/
  BlockRV DecomposeReduction(const BlockRV& block_rv, const LoopRV& loop_rv) final;
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) final;
  /******** Schedule: Block annotation ********/
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) final;
  void SetScope(const BlockRV& block_rv, int buffer_index, const ffi::String& storage_scope) final;
  void UnsafeSetDType(const BlockRV& block_rv, int buffer_index, const ffi::String& dtype) final;
  /******** Schedule: Blockize & Tensorize ********/
  BlockRV Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) final;
  BlockRV Blockize(const ffi::Array<BlockRV>& blocks, bool preserve_unit_iters) final;
  void Tensorize(const BlockRV& block_rv, const ffi::String& intrin,
                 bool preserve_unit_iters) final;
  void Tensorize(const LoopRV& loop_rv, const ffi::String& intrin, bool preserve_unit_iters) final;
  /******** Schedule: Annotation ********/
  void Annotate(const LoopRV& loop_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const LoopRV& loop_rv, const ffi::String& ann_key) override;
  void Annotate(const BlockRV& block_rv, const ffi::String& ann_key, const Any& ann_val) override;
  void Unannotate(const BlockRV& block_rv, const ffi::String& ann_key) override;
  /******** Schedule: Layout transformation ********/
  void TransformLayout(const BlockRV& block_rv, int buffer_index, BufferIndexType buffer_index_type,
                       const IndexMap& index_map, const ffi::Optional<IndexMap>& pad_value,
                       bool assume_injective_transform) override;
  void TransformBlockLayout(const BlockRV& block_rv, const IndexMap& index_map) override;
  void SetAxisSeparator(const BlockRV& block_rv, int buffer_index,
                        BufferIndexType buffer_index_type,
                        const ffi::Array<IntImm>& axis_separators) final;
  /******** Schedule: Padding ********/
  BlockRV DecomposePadding(const BlockRV& block_rv, const LoopRV& loop_rv) final;
  void PadEinsum(const BlockRV& block_rv, const ffi::Array<Integer>& padding) final;
  /******** Schedule: Buffer transformation ********/
  void RollingBuffer(const BlockRV& block_rv, int write_buffer_index) final;
  /******** Schedule: Misc ********/
  void EnterPostproc() final;
  void UnsafeHideBufferAccess(const BlockRV& block_rv, const ffi::String& buf_type,
                              const ffi::Array<IntImm>& buf_index_array) final;
  void AnnotateBufferAccess(const BlockRV& block_rv, int buffer_index,
                            BufferIndexType buffer_index_type, const IndexMap& index_map) final;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRACED_SCHEDULE_H_
