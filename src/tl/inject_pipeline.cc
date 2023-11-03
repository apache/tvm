/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
  * \file inject_software_pipeline.cc
  * \brief Transform annotated loops into pipelined one that parallelize producers and consumers
  */
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../support/utils.h"
#include "../tir/schedule/utils.h"
#include "../tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;

/*!
 * \brief Create a block and infer the access region with the given body.
 *
 * The result is a opaque block that doesn't contain any block iter vars. In case the body is a
 * block realize without predicate, it is unnecessary to create a new block, the block of the block
 * realize will be returned.
 *
 * \param body The body of the block.
 * \param buffer_data_to_buffer The map from buffer data to buffer.
 * \return The result block.
 */
Block MakeBlock(const Stmt& body, const Map<Var, Buffer>& buffer_data_to_buffer) {
  if (const BlockRealizeNode* block_realize = body.as<BlockRealizeNode>()) {
    if (is_one(block_realize->predicate)) {
      // no need to create a new block
      return block_realize->block;
    }
  }
  Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"", /*body*/ body);
  Array<Array<BufferRegion>> access = GetBlockReadWriteRegion(block, buffer_data_to_buffer);
  BlockNode* n = block.CopyOnWrite();
  n->reads = access[0];
  n->writes = access[1];
  return block;
}

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async;
};

using PipelineInfo = std::unordered_map<Block, PipelineAnnotation, ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1;  // the defining stage of the buffer
  int use = -1;  // the last using stage of the buffer
};

/*!
 * \brief Rewriter for the body of the software pipeline. This pass inserts `floormod` to indices
 * of the remapped buffer to select the version corresponding to the pipeline stage.
 */
class PipelineBodyRewriter : public StmtExprMutator {
public:
  /*!
   * \brief Constructor of PipelineBodyRewriter.
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param buffer_remap The map from original buffer to the buffer with updated shape for
   *        multi-versioning in the software pipeline.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param access_all_versions Whether all versions the buffers in the software pipeline are
   *        accessed. This will be used to update block access region. In the prologue and epilogue
   *        of a two-stage software pipeline, only one version of these buffers are accessed.
   */
  PipelineBodyRewriter(const Map<Var, Buffer>& buffer_data_to_buffer,
    const Map<Buffer, Buffer>& buffer_remap, For pipeline_loop,
    bool access_all_versions)
    : buffer_data_to_buffer_(buffer_data_to_buffer),
    buffer_remap_(buffer_remap),
    pipeline_loop_(pipeline_loop),
    access_all_versions_(access_all_versions) {}

private:
  BufferRegion RewritePipelineBufferRegion(const BufferRegion& buffer_region) const {
    auto it = buffer_remap_.find(buffer_region->buffer);
    if (it != buffer_remap_.end()) {
      Region new_region = buffer_region->region;
      const Buffer& new_buffer = (*it).second;
      // For pipeline buffers, relax the access region of the first dimension to full extent
      // if access_all_versions == true
      Range accessed_version =
        access_all_versions_
        ? Range::FromMinExtent(0, new_buffer->shape[0])
        : Range::FromMinExtent(floormod((pipeline_loop_->loop_var - pipeline_loop_->min),
          new_buffer->shape[0]),
          Integer(1));
      new_region.insert(new_region.begin(), accessed_version);
      return BufferRegion(new_buffer, new_region);
    }
    return buffer_region;
  }

  PrimExpr RewriteBufferAccess(const Call& call, const std::vector<int> arg_indices) {
    auto product = [](const Array<PrimExpr>& input) {
      return foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
        make_const(DataType::Int(32), 1), input);
      };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      const Buffer& buffer = buffer_data_to_buffer_.at(Downcast<Var>(call->args[i]));
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        const Buffer& new_buffer = (*it).second;
        const PrimExpr& old_index = call->args[i + 1];
        PrimExpr offset;
        if (new_buffer->strides.empty()) {
          offset = product(buffer->shape);
        } else {
          offset = new_buffer->strides[0];
        }
        PrimExpr new_index =
          old_index + floormod(pipeline_loop_->loop_var, new_buffer->shape[0]) * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const Buffer& alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(alloc_buffer->data, alloc_buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* n = block.CopyOnWrite();
    n->reads.MutateByApply([this](const BufferRegion& buffer_region) {
      return RewritePipelineBufferRegion(buffer_region);
      });
    n->writes.MutateByApply([this](const BufferRegion& buffer_region) {
      return RewritePipelineBufferRegion(buffer_region);
      });
    for (const Buffer& alloc_buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(alloc_buffer->data);
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(store);
    }
    const Buffer& new_buffer = (*it).second;
    auto* n = store.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version =
      floormod((pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end()) {
      return std::move(load);
    }
    const Buffer& new_buffer = (*it).second;
    auto* n = load.CopyOnWrite();
    n->buffer = new_buffer;
    PrimExpr version =
      floormod((pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
    n->indices.insert(n->indices.begin(), version);
    return std::move(load);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, { 1 });
    }
    return call;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  For pipeline_loop_;
  bool access_all_versions_;
};

/*!
 * \brief Rewriter for the software pipeline that rewrite a loop into a pipelined one.
 */
class PipelineRewriter : public StmtExprMutator {
public:
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
    const Array<Buffer>& pipeline_allocs, const For& pipeline_loop,
    const PipelineInfo& pipeline_info)

    : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
    pipeline_allocs_(pipeline_allocs),
    pipeline_loop_(pipeline_loop),
    pipeline_info_(pipeline_info) {}

  Stmt BuildPipeline() {
    // Step 1: Analyze accesses to the buffers in the pipeline and compute the number of versions
    // need to maintain for each buffer.
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> infos =
      GetBufferAccessInfo();
    for (const Buffer& buffer : pipeline_allocs_) {
      int num_versions = ComputeBufferVersions(buffer, infos.at(buffer));
      if (num_versions > 1) {
        buffer_remap_.Set(buffer, RewriteAllocBuffer(buffer, num_versions));
      }
    }

    ordered_stmts_.resize(pipeline_info_.size());
    for (const auto& [block, anno] : pipeline_info_) {
      ordered_stmts_.Set(anno.order, block);
    }

    for (const Block& block : ordered_stmts_) {
      int stage = pipeline_info_[block].stage;
      if (pipeline_info_[block].async) {
        auto& state = async_states[stage];
        for (auto write_region : block->writes) {
          auto buffer = write_region->buffer;
          state.dst_buffers.insert(buffer.get());
          if (buffer_remap_.count(buffer))
            state.dst_buffers.insert(buffer_remap_[buffer].get());
        }
      }
    }
    std::unordered_set<int> consumed;
    for (const Block& block : ordered_stmts_) {
      int stage = pipeline_info_[block].stage;
      if (pipeline_info_[block].async) {
        auto& state = async_states[stage];
        if (state.commit_groups.empty() || consumed.count(stage)) {
          state.commit_groups.push_back({});
        }
        state.commit_groups.back().push_back(pipeline_info_[block].order);
        consumed.erase(stage);
        for (auto write_region : block->writes) {
          auto buffer = buffer_remap_.count(write_region->buffer) ? buffer_remap_[write_region->buffer] : write_region->buffer;
          state.buffer_to_commit_group_[buffer.get()] = state.commit_groups.size() - 1;
        }
      }

      for (auto read_region : block->reads) {
        for (const auto& [producer_stage_id, producer_state] : async_states) {
          if (producer_stage_id <= stage && producer_state.writes(read_region->buffer)) {
            consumed.insert(producer_stage_id);
          }
        }
      }
    }

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue = EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true);
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_,
      pipeline_loop_->min + pipeline_loop_->extent, false);
    Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
      pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true);

    SeqStmt stmt = SeqStmt({ prologue, body, epilogue });

    // Step 3: Make a new block that contains new buffer allocations after pipeline rewriting.
    Array<Buffer> alloc_buffers;
    for (const auto& alloc : pipeline_allocs_) {
      alloc_buffers.push_back(buffer_remap_.Get(alloc).value_or(alloc));
      buffer_data_to_buffer_.erase(alloc->data);
    }
    Block block = MakeBlock(stmt, buffer_data_to_buffer_);
    block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
    return BlockRealize({}, Bool(true), block);
  }

private:
  /*!
   * \brief Analyze accesses to the buffers in the software pipeline.
   *
   * This method check the 'define' and 'use' stage of the buffers in the software pipeline, which
   * can be used to compute the number of versions needed to maintain after rewriting.
   */
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
    GetBufferAccessInfo() {
    std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> infos;
    for (const auto& pair : pipeline_info_) {
      const Block& block = pair.first;
      int stage = pair.second.stage;
      max_stage_ = std::max(max_stage_, stage);

      for (const BufferRegion& write : block->writes) {
        if (!infos.count(write->buffer)) {
          infos.emplace(write->buffer, BufferAccessInfo{});
        }
        auto& info = infos.at(write->buffer);
        if (info.def == -1) {
          info.def = stage;
        } else {
          info.def = std::min(info.def, stage);
        }
      }

      for (const BufferRegion& read : block->reads) {
        if (!infos.count(read->buffer)) {
          infos.emplace(read->buffer, BufferAccessInfo{});
        }
        auto& info = infos.at(read->buffer);
        info.use = std::max(info.use, stage);
      }
    }
    return infos;
  }

  /*!
   * \brief Check whether two regions have intersections.
   * \param region1 The first region.
   * \param region2 The second region.
   * \return Whether region1 and region2 have intersections.
   */
  bool MayConflict(Region region1, Region region2) {
    ICHECK(region1.size() == region2.size());
    for (size_t i = 0; i < region1.size(); i++) {
      Range dim1 = region1[i];
      Range dim2 = region2[i];
      auto int_set1 = arith::IntSet::FromRange(dim1);
      auto int_set2 = arith::IntSet::FromRange(dim2);
      if (arith::Intersect({ int_set1, int_set2 }).IsNothing()) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Compute the number of versions need to maintain for buffer accessed in the software
   * pipeline.
   *
   * This method applies liveness analysis to the target buffer to compute the number of versions
   * need to maintain during the software pipeline.
   * Annotation `attr::double_buffer_scope` is handled here which provides a way to override the
   * result of the analysis. Additional double buffering in the software pipeline can be useful
   * to eliminate synchronizations in GPU devices.
   *
   * \param buffer The target buffer
   * \param buffer_info The access information of the target buffer.
   * \return The number of versions required for the target buffer.
   */
  int ComputeBufferVersions(const Buffer& buffer, const BufferAccessInfo& buffer_info) {
    if (buffer_info.def == -1) {
      // Keep the original number of versions as buffers defined outside the software pipeline
      // should not be mutated.
      return 1;
    }

    // `use - def + 1` is a upper bound of the needed versions
    // We optimize a few case where the number of versions can be smaller than the upper bound
    int num_versions = buffer_info.use - buffer_info.def + 1;
    if (num_versions >= 2) {
      // A special case when `use - def + 1 == 2`. Double buffering is only needed in this case when
      // these exists a reader block_i and a writer block_j such that
      // order(block_i) < order(block_j) and stage(block_i) < stage(block_j) and the access regions
      // of block_i and block_j overlap.
      bool need_multi_version = false;
      for (const auto& pair1 : pipeline_info_) {
        const Block& writer_block = pair1.first;
        const auto& writer_info = pair1.second;

        auto it1 = std::find_if(writer_block->writes.begin(), writer_block->writes.end(),
          [&](const BufferRegion& buffer_region) {
            return buffer_region->buffer.same_as(buffer);
          });
        if (it1 == writer_block->writes.end()) {
          continue;
        }

        for (const auto& pair2 : pipeline_info_) {
          const Block& reader_block = pair2.first;
          const auto& reader_info = pair2.second;
          auto it2 = std::find_if(reader_block->reads.begin(), reader_block->reads.end(),
            [&](const BufferRegion& buffer_region) {
              return buffer_region->buffer.same_as(buffer);
            });
          if (it2 == reader_block->reads.end()) {
            continue;
          }
          if (writer_info.order < reader_info.order && writer_info.stage < reader_info.stage &&
            MayConflict((*it1)->region, (*it2)->region)) {
            need_multi_version = true;
            break;
          }
        }
      }
      if (!need_multi_version) {
        num_versions--;
      }
    }
    return num_versions;
  }

  /*!
   * \brief Rewrite buffer allocation to keep multiple versions of original buffer for pipelined
   * accesses.
   * \param buffer The buffer to be resized.
   * \param num_versions The number of versions to keep.
   * \return The resized buffer.
   */
  Buffer RewriteAllocBuffer(const Buffer& buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
    if (new_buffer->strides.size()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  // Per-stage states that need to be tracked across pipeline prologue, body, and epilogue.
  struct AsyncStateGlobal {
    // Buffers that this stage asynchronously writes.
    std::unordered_set<const BufferNode*> dst_buffers;
    // An imaginary index that the latest async operation associated with this stage has written
    // into. Only valid if all associated predicates are true, so that we can count the number of
    // async invocations exactly. When it is valid, it is the "sum of extents of loops that have
    // been executed" - 1, e.g. for epilogue it is prologue extent + body extent - 1. This
    // is only needed to compute wait count for epilogue without async producers.
    Optional<PrimExpr> producer_head{ PrimExpr(-1) };
    std::vector<std::vector<int>> commit_groups;
    std::unordered_map<const BufferNode*, int> buffer_to_commit_group_;
    bool writes(Buffer buf) const { return dst_buffers.count(buf.get()) > 0; }
  };


  // Per-stage states that are local to each of pipeline prologue, body, and epilogue.
  struct AsyncStateLocal {
    struct PendingWait {
      // The index into a list of blocks, where async_wait_queue should be attached at the
      // beginning.
      int insert_before;
      // in_flight_count would be a more precise name, but the implementation uses wait_count for
      // brevity.
      PrimExpr wait_count{ nullptr };

      bool valid() const { return wait_count.defined(); }
    };

    std::vector<PendingWait> pending_waits;

    // A symbolic expression representing the index the latest async operation associated with this
    // stage has written into, at the "current" iteration.
    Optional<PrimExpr> producer_head;
  };

  /*! Structure holding intermediate information for pipeline loop rewriting. */
  struct RewrittenBlockInfo {
    int stage;
    int order;
    PrimExpr predicate;
    Block block;
    PrimExpr access_index;
    bool is_async;
  };

  void PopulateWaitCounts(const std::vector<RewrittenBlockInfo>& new_blocks,
    std::map<int, AsyncStateLocal>* async_states_local) {
    for (size_t i = 0; i < new_blocks.size(); ++i) {
      int producer_stage_idx = -1;
      for (auto read_region : new_blocks[i].block->reads) {
        for (const auto& [stage, state] : async_states) {
          if (stage <= new_blocks[i].stage && state.writes(read_region->buffer)) {
            // Found an earlier stage where read_region->buffer was asynchronously written
            ICHECK(producer_stage_idx == -1 || producer_stage_idx == stage)
              << "A dependency on multiple async stages is not supported";
            producer_stage_idx = stage;
          }
        }
      }
      if (producer_stage_idx == -1) continue;
      const auto& state = async_states[producer_stage_idx];
      auto& dep_local_state = (*async_states_local)[producer_stage_idx];
      PrimExpr in_flight_cnt = 0;
      for (const auto& group : state.commit_groups) {
        PrimExpr consumer_head = new_blocks[i].access_index;
        PrimExpr producer_head;
        if (dep_local_state.producer_head.defined()) {
          producer_head = dep_local_state.producer_head.value();
          // if the group is after the wait point, minus by 1
          if (group.front() > new_blocks[i].order) producer_head -= 1;
        } else {
          producer_head = state.producer_head.value();
        }
        in_flight_cnt += producer_head - consumer_head;
      }

      // We can relax the in-flight-count by the number of independent commit.
      std::unordered_set<int> dependent_groups;
      for (const auto& read_region: new_blocks[i].block->reads) {
        if (state.buffer_to_commit_group_.count(read_region->buffer.get()))
          dependent_groups.insert(state.buffer_to_commit_group_.at(read_region->buffer.get()));
      }
      for (int i = int(state.commit_groups.size()) - 1; i >= 0; i--) {
        if (dependent_groups.count(i) == 0) in_flight_cnt += 1;
        else break; // stop relaxing
      }
      in_flight_cnt = analyzer_.Simplify(in_flight_cnt);
      dep_local_state.pending_waits.push_back({ static_cast<int>(i), in_flight_cnt });
    }
  }

  // Given pipelined blocks and async-related information, generate final loop statements with async
  // scopes (if any).
  Array<Stmt> CompletePipelineLoopStatements(
    const std::vector<RewrittenBlockInfo>& blocks,
    const std::map<int, AsyncStateLocal>& async_states_local) const {
    std::vector<RewrittenBlockInfo> new_blocks = blocks;
    for (const auto& [stage_id, state] : async_states_local) {
      for (const auto& pw : state.pending_waits) {
        auto& block = new_blocks[pw.insert_before].block;
        BlockNode* n = block.CopyOnWrite();
        auto zero = make_zero(DataType::Int(32));
        n->body =
          AttrStmt(zero, tir::attr::async_wait_queue_scope, stage_id,
            AttrStmt(zero, tir::attr::async_wait_inflight_count, pw.wait_count, n->body));
      }
    }

    // mark the last async stmt as commit
    std::unordered_set<int> commit_group_indices;
    for (const auto& [stage_id, state] : async_states) {
      for (size_t i = 0; i < state.commit_groups.size(); ++i) {
        commit_group_indices.insert(state.commit_groups[i].back());
      }
    }

    Array<Stmt> stmts;

    for (size_t i = 0; i < new_blocks.size(); i++) {
      Block block = new_blocks[i].block;
      if (commit_group_indices.count(new_blocks[i].order)) {
        auto commit_queue_scope = AttrStmt(make_zero(DataType::Int(32)),
          tir::attr::async_commit_queue_scope, new_blocks[i].stage, block->body);
        block = MakeBlock(commit_queue_scope, buffer_data_to_buffer_);
      }
      stmts.push_back(BlockRealize({}, new_blocks[i].predicate, block));
    }

    return stmts;
  }

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \return The result loop.
   */
  Stmt EmitImpl(PrimExpr start, PrimExpr end, bool unroll_loop) {
    PrimExpr new_loop_var;
    PrimExpr extent = end - start;

    auto make_nop = []() { return BlockRealize({}, Bool(true), MakeBlock(Evaluate(0), {})); };

    bool is_unit_loop = analyzer_.CanProveEqual(extent, 1);
    if (is_unit_loop) {
      new_loop_var = start;  // use constants as the loop var for unit loops
    } else {
      new_loop_var = pipeline_loop_->loop_var.copy_with_suffix("");
      analyzer_.Bind(Downcast<Var>(new_loop_var), Range(start, end));
    }

    std::vector<RewrittenBlockInfo> new_blocks;

    // Async related
    std::map<int, AsyncStateLocal> async_states_local;

    for (const Block& block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
      int order = pipeline_info_.at(block).order;
      PrimExpr skewed_loop_var = new_loop_var - stage;
      PrimExpr inbound = analyzer_.Simplify(pipeline_loop_->min <= skewed_loop_var) &&
        (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent);
      if (analyzer_.CanProve(!inbound)) {
        continue;
      }
      Block new_block = Downcast<Block>(PipelineBodyRewriter(buffer_data_to_buffer_, buffer_remap_,
        pipeline_loop_, max_stage_ != 1)(block));

      PrimExpr delta = start - pipeline_loop_->min;
      // This variable corresponds to
      // - "producer_head" if this stage is an async producer
      // - "consumer_head" if this stage reads from asynchronously written buffers.
      PrimExpr normalized_access_index = is_unit_loop ? skewed_loop_var : skewed_loop_var + delta;

      // Adjust the block predicate and the body according to the final loop bound
      //  [pipeline_loop_->min, extent).
      if (!is_unit_loop) {
        Var loop_iter = Downcast<Var>(new_loop_var);
        inbound = Substitute(inbound, { {loop_iter, loop_iter + delta} });
      }

      new_block = Downcast<Block>(
        Substitute(new_block, { {pipeline_loop_->loop_var, normalized_access_index} }));

      if (pipeline_info_[block].async) {
        auto& local_state = async_states_local[stage];
        local_state.producer_head = normalized_access_index;
        BlockNode* n = new_block.CopyOnWrite();
        n->body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::async_scope, 1, n->body);
      }

      new_blocks.push_back(
        { stage, order, inbound, new_block, normalized_access_index, pipeline_info_[block].async });
    }

    PopulateWaitCounts(new_blocks, &async_states_local);
    auto stmts = CompletePipelineLoopStatements(new_blocks, async_states_local);

    Stmt new_loop{ nullptr };

    if (stmts.empty()) {
      return make_nop();
    }
    if (stmts.size() == 1) {
      new_loop = stmts[0];
    } else {
      new_loop = SeqStmt(stmts);
    }

    if (!is_unit_loop) {
      Map<String, ObjectRef> preserved_annotations;
      for (const auto& kv : pipeline_loop_->annotations) {
        const String& key = kv.first;
        if (kv.first != tir::attr::software_pipeline_stage && kv.first != tir::attr::software_pipeline_order &&
          kv.first != tir::attr::software_pipeline_async_stages) {
          preserved_annotations.Set(key, kv.second);
        }
      }
      new_loop = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, extent,
        unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind, std::move(new_loop),
        NullOpt, preserved_annotations);
    }

    // Update producer heads in the global async states.
    for (const auto& [stage_id, state] : async_states_local) {
      async_states[stage_id].producer_head = async_states[stage_id].producer_head.value() + extent;
    }

    return BlockRealize({}, Bool(true), MakeBlock(std::move(new_loop), buffer_data_to_buffer_));
  }

  arith::Analyzer analyzer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Array<Buffer> pipeline_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Array<Block> ordered_stmts_;
  std::map<int, AsyncStateGlobal> async_states;
};

/*!
 * \brief Build the dependency graph among a array of blocks.
 * \param[in] blocks The array of blocks.
 * \param[out] dep_src2dst Optional, a map to store dependency edges from the source to the
 * destination.
 * \param[out] dep_dst2src Optional, a map to store dependency edges from the
 * destination to the source.
 */
void BuildDependencyGraph(
  const Array<Block>& blocks,
  std::unordered_map<Block, Array<Block>, ObjectPtrHash, ObjectPtrEqual>* dep_src2dst,
  std::unordered_map<Block, Array<Block>, ObjectPtrHash, ObjectPtrEqual>* dep_dst2src) {
  std::unordered_map<Var, Array<Block>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;

  for (const Block& block : blocks) {
    for (const BufferRegion& read : block->reads) {
      auto it = buffer_writers.find(read->buffer->data);
      if (it != buffer_writers.end()) {
        for (const Block& writer : it->second) {
          if (dep_src2dst != nullptr) {
            (*dep_src2dst)[writer].push_back(block);
          }
          if (dep_dst2src != nullptr) {
            (*dep_dst2src)[block].push_back(writer);
          }
        }
      }
    }
    for (const BufferRegion& write : block->writes) {
      buffer_writers[write->buffer->data].push_back(block);
    }
  }
}

class PipelineInjector : private StmtExprMutator {
public:
  static Stmt Inject(const PrimFunc& func) {
    auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    PipelineInjector injector(global_symbol);
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      injector.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return injector(func->body);
  }

private:
  explicit PipelineInjector(Optional<String> global_symbol) : global_symbol_(global_symbol) {}

  /*!
   * \brief Check the pipeline satisfies the following conditions:
   * 1. No conflicting order: The order of each statement should be unique.
   * 2. Reordering of statements doesn't break buffer access dependencies. Specifically, for
   * dependency (e.g. read-after-write) from statement A to statement B, it requires:
   *   case 1: stage(A) < stage(B)
   *   case 2: stage(A) == stage(B) and order(A) < order(B)
   */
  void ValidatePipelineBody(const PipelineInfo& pipeline_info, const Array<Block>& original_order) {
    std::unordered_set<int> used_orders;
    std::unordered_map<int, int> stage_max_order;
    std::unordered_map<int, const Block*> order_to_block;
    std::unordered_map<const Block*, int> block_to_stage;
    for (const Block& block : original_order) {
      const auto& stmt_info = pipeline_info.at(block);
      int order = stmt_info.order;
      CHECK(!used_orders.count(order))
        << "ValueError: Two statements in the software pipeline cannot have the same order";
      used_orders.insert(order);
    }

    std::unordered_map<Block, Array<Block>, ObjectPtrHash, ObjectPtrEqual> dep_src2dst;
    BuildDependencyGraph(original_order, &dep_src2dst, nullptr);

    for (const auto& pair : dep_src2dst) {
      const Block& src = pair.first;
      const auto& src_info = pipeline_info.at(src);
      const Array<Block>& dsts = pair.second;
      for (const Block& dst : dsts) {
        const auto& dst_info = pipeline_info.at(dst);
        CHECK_LE(src_info.stage, dst_info.stage)
          << "ValueError: statement " << dst << " in stage " << dst_info.stage
          << " cannot depends on statement " << src << " in a later stage " << src_info.stage;
        if (src_info.stage == dst_info.stage) {
          CHECK_LT(src_info.order, dst_info.order) << "ValueError: two statements with buffer "
            "access dependency in the same stage of the "
            "software pipeline cannot be reordered";
        }
      }
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1: Recursively rewrite the children first.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (!HasPipelineAnnotation(op)) {
      return std::move(for_node);
    }
    // Step 2: Find the body and buffer allocations of the pipeline. The body can be direct child of
    // the for-loop. If the for-loop has BlockRealize as its child, the pipeline body will be the
    // child of the block.
    Stmt pipeline_body{ nullptr };
    Array<Buffer> pipeline_allocs;
    if (const auto* realize = for_node->body.as<BlockRealizeNode>()) {
      const auto& block = realize->block;
      for (const auto& buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body = block->body;
      pipeline_allocs = block->alloc_buffers;
    } else {
      pipeline_body = for_node->body;
    }

    const SeqStmtNode* pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
      << "ValueError: The body of the software pipeline should be SeqStmt, got "
      << pipeline_body->GetTypeKey();

    // Step 3: Blockize the components of the pipeline. Each child of the pipelined loop will be
    // converted into a block.
    PipelineInfo pipeline_info;
    Array<Block> original_order;  // pipeline body blocks in the original order

    auto f_add_child = [&](const Stmt& child) {
      original_order.push_back(MakeBlock(child, buffer_data_to_buffer_));
      };
    for (size_t i = 0; i < pipeline_body_seq->seq.size(); i++) {
      const auto* nested_block_realize = pipeline_body_seq->seq[i].as<BlockRealizeNode>();
      if (nested_block_realize && is_one(nested_block_realize->predicate) &&
        nested_block_realize->block->body->IsInstance<SeqStmtNode>()) {
        const Block& nested_pipeline_block = nested_block_realize->block;
        ICHECK(
          nested_pipeline_block->match_buffers.empty());  // match_buffer should have been lowered
        for (const auto& buffer : nested_pipeline_block->alloc_buffers) {
          pipeline_allocs.push_back(buffer);
          buffer_data_to_buffer_.Set(buffer->data, buffer);
        }
        const auto* nested_seq = nested_pipeline_block->body.as<SeqStmtNode>();
        for (size_t j = 0; j < nested_seq->seq.size(); j++) {
          f_add_child(nested_seq->seq[j]);
        }
      } else {
        f_add_child(pipeline_body_seq->seq[i]);
      }
    }

    auto pipeline_stages =
      Downcast<Array<Integer>>(op->annotations.at(tir::attr::software_pipeline_stage));
    auto pipeline_orders =
      Downcast<Array<Integer>>(op->annotations.at(tir::attr::software_pipeline_order));
    CHECK_EQ(pipeline_stages.size(), original_order.size())
      << "PrimFunc " << global_symbol_ << " has original order "
      << original_order.Map([](const auto& block) { return block->name_hint; })
      << ", but pipeline annotation is " << pipeline_stages << " with different size";
    CHECK_EQ(pipeline_orders.size(), original_order.size())
      << "PrimFunc " << global_symbol_ << " has original order "
      << original_order.Map([](const auto& block) { return block->name_hint; })
      << ", but pipeline annotation is " << pipeline_orders << " with different size";

    std::unordered_set<int> pipeline_async_stages;
    if (auto annot = op->annotations.Get(tir::attr::software_pipeline_async_stages)) {
      for (auto s : Downcast<Array<Integer>>(annot)) {
        pipeline_async_stages.insert(s->value);
      }
    }

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]->value);
      bool is_async = pipeline_async_stages.find(stage) != pipeline_async_stages.end();
      PipelineAnnotation stage_order{ stage,
        /*order=*/static_cast<int>(pipeline_orders[i]->value),
        is_async };
      pipeline_info.emplace(original_order[i], stage_order);
    }

    ValidatePipelineBody(pipeline_info, original_order);

    // Step 4: Rewrite the pipeline body.
    Stmt pipeline = PipelineRewriter(buffer_data_to_buffer_,
      pipeline_allocs, GetRef<For>(op), pipeline_info).BuildPipeline();

    if (const auto* realize = op->body.as<BlockRealizeNode>()) {
      const auto& block = realize->block;
      for (const auto& buffer : block->alloc_buffers) {
        buffer_data_to_buffer_.erase(buffer->data);
      }
    }
    return pipeline;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

  bool HasPipelineAnnotation(const ForNode* op) const {
    auto it1 = op->annotations.find(tir::attr::software_pipeline_stage);
    auto it2 = op->annotations.find(tir::attr::software_pipeline_order);
    bool has_stage = it1 != op->annotations.end();
    bool has_order = it2 != op->annotations.end();
    if (has_stage && has_order) {
      return true;
    }
    if (has_stage) {
      LOG(FATAL) << "ValueError: Order of the software pipeline is not defined.";
    }
    if (has_order) {
      LOG(FATAL) << "ValueError: Stage of the software pipeline is not defined.";
    }
    return false;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<String> global_symbol_;
};

/*!
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers.
 * \return The IR transform pass.
 */
tir::transform::Pass InjectSoftwarePipeline() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body = PipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
    };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tl.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace tl
}  // namespace tvm
