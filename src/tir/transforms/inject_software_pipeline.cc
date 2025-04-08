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

#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

namespace software_pipeline {

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

class PipelineOpaqueAccessRewriter {
 public:
  /*!
   * \brief Constructor
   * \param buffer_data_to_buffer The map from buffer data to buffer.
   * \param buffer_remap The map from original buffer to the buffer with updated shape for
   *        multi-versioning in the software pipeline.
   * \param pipeline_loop The original loop to be software pipelined.
   * \param fragment_info Information about tensor core fragment
   */
  PipelineOpaqueAccessRewriter(
      const Map<Var, Buffer>& buffer_data_to_buffer, const Map<Buffer, Buffer>& buffer_remap,
      const For& pipeline_loop,
      const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        buffer_remap_(buffer_remap),
        pipeline_loop_(pipeline_loop),
        fragment_info_(fragment_info) {}

  PrimExpr Rewrite(const Call& call) {
    // Intrinsic calls should be handled explicitly here as they are opaque accesses to
    // buffer.
    static const auto& load_matrix_sync = builtin::tvm_load_matrix_sync();
    static const auto& store_matrix_sync = builtin::tvm_store_matrix_sync();
    static const auto& mma_sync = builtin::tvm_mma_sync();
    static const auto& access_ptr = builtin::tvm_access_ptr();
    static const auto& ptx_ldmatrix = builtin::ptx_ldmatrix();
    static const auto& ptx_mma = builtin::ptx_mma();
    if (call->op.same_as(load_matrix_sync) || call->op.same_as(store_matrix_sync)) {
      const Buffer& buffer = buffer_data_to_buffer_.at(Downcast<Var>(call->args[0]));
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        Array<PrimExpr> new_args = call->args;
        const Buffer& new_buffer = (*it).second;
        new_args.Set(4, RewriteWmmaFragmentIndex(buffer, new_buffer, call->args[4]));
        return Call(call->dtype, call->op, new_args, call->span);
      }
    } else if (call->op.same_as(mma_sync)) {
      Array<PrimExpr> new_args = call->args;
      for (int i = 0; i < 4; i++) {
        const Var& buffer_var = Downcast<Var>(call->args[i * 2]);
        const PrimExpr& index = call->args[i * 2 + 1];
        const Buffer& buffer = buffer_data_to_buffer_.at(buffer_var);
        auto it = buffer_remap_.find(buffer);
        if (it != buffer_remap_.end()) {
          PrimExpr new_index = RewriteWmmaFragmentIndex(buffer, (*it).second, index);
          new_args.Set(i * 2 + 1, new_index);
        }
      }
      return Call(call->dtype, call->op, new_args, call->span);
    } else if (call->op.same_as(access_ptr)) {
      return RewriteBufferAccess(call, {1});
    } else if (call->op.same_as(ptx_mma)) {
      return RewriteBufferAccess(call, {6, 8, 10});
    } else if (call->op.same_as(ptx_ldmatrix)) {
      return RewriteBufferAccess(call, {3});
    }
    return call;
  }

 private:
  int GetWmmaFragmentSize(const Buffer& buffer) {
    auto it = fragment_info_.find(buffer->data.get());
    ICHECK(it != fragment_info_.end());
    const FragmentInfo& info = (*it).second;
    return info.GetSize();
  }

  PrimExpr RewriteWmmaFragmentIndex(const Buffer& old_buffer, const Buffer& new_buffer,
                                    const PrimExpr& old_index) {
    PrimExpr new_buffer_offset = old_index;

    int fragment_size = GetWmmaFragmentSize(old_buffer);
    PrimExpr offset =
        floordiv(foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                       make_const(DataType::Int(32), 1), old_buffer->shape),
                 fragment_size);
    new_buffer_offset +=
        floormod(pipeline_loop_->loop_var - pipeline_loop_->min, new_buffer->shape[0]) * offset;
    return new_buffer_offset;
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
        if (buffer.scope() == "m16n8k8.matrixA" || buffer.scope() == "m16n8k8.matrixB") {
          // mma scope size will shrink by warp size
          // @see transform_mma_buffer_layout
          ICHECK_EQ(Downcast<IntImm>(floormod(offset, 32))->value, 0)
              << "mma scope size should be multiple of warp size";
          offset = floordiv(offset, 32);
        }
        PrimExpr new_index =
            old_index + floormod(pipeline_loop_->loop_var, new_buffer->shape[0]) * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  const Map<Var, Buffer>& buffer_data_to_buffer_;
  const Map<Buffer, Buffer>& buffer_remap_;
  const For& pipeline_loop_;
  const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info_;
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
   * \param fragment_info Information about tensor core fragment
   */
  PipelineBodyRewriter(const Map<Var, Buffer>& buffer_data_to_buffer,
                       const Map<Buffer, Buffer>& buffer_remap, For pipeline_loop,
                       bool access_all_versions,
                       const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info)
      : buffer_data_to_buffer_(buffer_data_to_buffer),
        buffer_remap_(buffer_remap),
        pipeline_loop_(pipeline_loop),
        access_all_versions_(access_all_versions),
        opaque_access_rewriter_(buffer_data_to_buffer_, buffer_remap_, pipeline_loop_,
                                fragment_info) {}

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
    return opaque_access_rewriter_.Rewrite(call);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  For pipeline_loop_;
  bool access_all_versions_;
  PipelineOpaqueAccessRewriter opaque_access_rewriter_;
};

/*!
 * \brief Rewriter for the software pipeline that rewrite a loop into a pipelined one.
 */
class PipelineRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(
      Map<Var, Buffer> buffer_data_to_buffer,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers,
      const Array<Buffer> pipeline_allocs, const For& pipeline_loop,
      const PipelineInfo& pipeline_info,
      const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info,
      const Map<String, ObjectRef> preserved_annotations) {
    PipelineRewriter rewriter(buffer_data_to_buffer, double_buffers, pipeline_allocs, pipeline_loop,
                              pipeline_info, fragment_info, preserved_annotations);
    return rewriter.BuildPipeline();
  }

 private:
  PipelineRewriter(Map<Var, Buffer> buffer_data_to_buffer,
                   const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers,
                   const Array<Buffer>& pipeline_allocs, const For& pipeline_loop,
                   const PipelineInfo& pipeline_info,
                   const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info,
                   const Map<String, ObjectRef> preserved_annotations)

      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        double_buffers_(double_buffers),
        pipeline_allocs_(pipeline_allocs),
        pipeline_loop_(pipeline_loop),
        pipeline_info_(pipeline_info),
        fragment_info_(fragment_info),
        preserved_annotations_(preserved_annotations) {}

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
    for (const auto& pair : pipeline_info_) {
      const Block& block = pair.first;
      int order = pair.second.order;
      ordered_stmts_.Set(order, block);
    }

    // Step 2: Emit the pipeline prologue, body and epilogue.
    Stmt prologue = EmitImpl(pipeline_loop_->min, pipeline_loop_->min + max_stage_, true);
    Stmt body = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent, false);
    // introduce extra lowerbound when the loop length is smaller than num stages
    // to ensure the epilogue interval do not overlap the prologue interval.
    PrimExpr epigogue_start = pipeline_loop_->min + pipeline_loop_->extent;
    Optional<PrimExpr> extra_epilogue_lower_bound = NullOpt;
    if (max_stage_ > 1 && !analyzer_.CanProveGreaterEqual(pipeline_loop_->extent, max_stage_)) {
      if (is_const_int(epigogue_start)) {
        epigogue_start = max(epigogue_start, pipeline_loop_->min + max_stage_);
      } else {
        // for dynamic case, introduce extra lowerbound as loop predicate
        // to ensure the epilogue part unrollable.
        extra_epilogue_lower_bound = pipeline_loop_->min + max_stage_;
      }
    }
    Stmt epilogue =
        EmitImpl(epigogue_start, pipeline_loop_->min + pipeline_loop_->extent + max_stage_, true,
                 extra_epilogue_lower_bound);

    SeqStmt stmt = SeqStmt({prologue, body, epilogue});

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
      if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
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
    if (num_versions == 2) {
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
        num_versions = 1;
      }
    }
    if (num_versions == 1 && double_buffers_.count(buffer)) {
      num_versions = 2;
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
    Optional<PrimExpr> producer_head{PrimExpr(-1)};

    bool writes(Buffer buf) const { return dst_buffers.count(buf.get()) > 0; }
  };

  // Per-stage states that are local to each of pipeline prologue, body, and epilogue.
  struct AsyncStateLocal {
    struct {
      // The index into a list of blocks, where async_wait_queue should be attached at the
      // beginning.
      int insert_before;
      // in_flight_count would be a more precise name, but the implementation uses wait_count for
      // brevity.
      PrimExpr wait_count{nullptr};

      bool valid() const { return wait_count.defined(); }
    } pending_wait;

    // Destination buffers of async operations that have been encountered so far in the loop
    //
    // for (size_t i = 0; i < new_blocks.size(); ++i) {
    //    ...
    // }
    //
    // This is for tracking which async operations have been issued at the "current" iteration, up
    // until a point where we encounter a consumer of async result buffers. This is used to decide
    // if the producer_head of each buffer points to a copy written in the current or previous
    // iteration.
    std::unordered_set<const BufferNode*> seen;

    // A symbolic expression representing the index the latest async operation associated with this
    // stage has written into, at the "current" iteration.
    Optional<PrimExpr> producer_head;
    // The predicate of BlockRealize containing the async operation of this stage.
    Optional<PrimExpr> predicate;
    // Indices into a list of blocks, where async_commit_queue scope should be attached.
    // If multiple async producers are interleaved with their consumer in between, we need separate
    // async_commit_queue for each producer. Thus, we need multiple sets of indices.
    std::vector<std::vector<size_t>> commit_groups;

    // This is set to true when we reach a stage that consumes this async stage.
    bool consumed{false};
  };

  /*! Structure holding intermediate information for pipeline loop rewriting. */
  struct RewrittenBlockInfo {
    int stage;
    PrimExpr predicate;
    Block block;
    PrimExpr access_index;
    bool is_async;
  };

  // Determine where to insert async_wait and the corresponding wait count.
  void PopulateWaitCounts(const std::vector<RewrittenBlockInfo>& new_blocks,
                          arith::Analyzer* ana_normalized,
                          const std::unordered_map<const BufferNode*, int>& buffer_to_commit_group,
                          std::map<int, AsyncStateLocal>* async_states_local) {
    for (size_t i = 0; i < new_blocks.size(); ++i) {
      if (new_blocks[i].is_async) {
        // Record the fact that we have encountered these write buffers.
        for (auto write_region : new_blocks[i].block->writes) {
          (*async_states_local)[new_blocks[i].stage].seen.insert(write_region->buffer.get());
        }
      }

      int producer_stage_idx = -1;
      for (auto read_region : new_blocks[i].block->reads) {
        for (auto kv : async_states) {
          if (kv.first <= new_blocks[i].stage && kv.second.writes(read_region->buffer)) {
            // Found an earlier stage where read_region->buffer was asynchronously written
            ICHECK(producer_stage_idx == -1 || producer_stage_idx == kv.first)
                << "A dependency on multiple async stages is not supported";
            producer_stage_idx = kv.first;
          }
        }
      }

      if (producer_stage_idx == -1) continue;

      // The following logic has become complicated to handle case like this:
      //
      // for i in range(13):
      //     # Stage 0
      //     async_commit_queue(0):
      //        async_scope:
      //           A_shared[(i + 3) % 4] = A[...]
      //
      //
      //     # Stage 1
      //     async_wait_queue(0, 5):
      //        compute(A_shared[i], B_shared[i])
      //
      //     # Stage 0
      //     async_commit_queue(0)
      //        async_scope:
      //           B_shared[(i + 3) % 4] = B[...]
      //
      //
      // Here, multiple async producers in the same stage are interleaved with their consumer in
      // between. Since each buffer is associated with different commit groups, the wait_count
      // before the consumer should be bigger than the simpler case:
      //
      // for i in range(13):
      //     # Stage 0
      //     async_commit_queue(0):
      //        async_scope:
      //           A_shared[(i + 3) % 4] = A[...]
      //           B_shared[(i + 3) % 4] = B[...]
      //
      //     # Stage 1
      //     async_wait_queue(0, 3):
      //        compute(A_shared[i], B_shared[i])
      //
      // The correct wait_count can be determined by considering each commit group separately, and
      // summing "per-commit" wait_counts.
      //
      // From A_shared's perspective, it allows for (i + 3) - i async commit groups to be in
      // flight while from B_shared's perspective, the producer head at compute points to the copy
      // done by the previous iteration, so its wait_count is calculated as ((i - 1) + 3) - i. The
      // sum of the two wait_counts gives 5.

      auto& dep_local_state = (*async_states_local)[producer_stage_idx];
      const auto num_commit_group = dep_local_state.commit_groups.size();
      std::vector<Optional<PrimExpr>> producer_head_per_commit;

      if (num_commit_group == 0) {
        // Epilogue, no async producer. Since "local" producer_head is not available, use
        // "global" producer_head.
        ICHECK(!dep_local_state.producer_head);
        producer_head_per_commit.push_back(async_states[producer_stage_idx].producer_head);
      } else {
        ICHECK(dep_local_state.producer_head);
        std::vector<bool> need_wait_count(num_commit_group, true);

        for (auto read_region : new_blocks[i].block->reads) {
          if (!async_states[producer_stage_idx].writes(read_region->buffer)) continue;
          auto commit_group_id = buffer_to_commit_group.at(read_region->buffer.get());
          if (!need_wait_count[commit_group_id]) continue;

          if (!dep_local_state.seen.count(read_region->buffer.get())) {
            // Multiple async producers interleaved: The most recent async write is from the
            // previous iteration. This is the B_shared case above.
            producer_head_per_commit.push_back(dep_local_state.producer_head.value() - 1);
          } else {
            // Normal case
            producer_head_per_commit.push_back(dep_local_state.producer_head.value());
          }

          need_wait_count[commit_group_id] = false;
        }
      }

      auto wait_count = [=, &ana_normalized]() {
        auto sum = PrimExpr(0);
        for (auto producer_head : producer_head_per_commit) {
          if (producer_head && ana_normalized->CanProve(producer_head.value() >= 0)) {
            // Here, new_blocks[i].access_index corresponds to "consumer_head".
            // The difference of producer_head and consumer_head is precisely the number of
            // async commit groups that can still be in flight after this wait.
            sum += analyzer_.Simplify(producer_head.value() - new_blocks[i].access_index);
          } else {
            // The precise count cannot be determined, give up.
            return PrimExpr(0);
          }
        }
        return sum;
      }();

      auto& pending_wait = dep_local_state.pending_wait;

      if (!pending_wait.valid()) {
        pending_wait = {static_cast<int>(i), wait_count};
      } else if (analyzer_.CanProve(wait_count < pending_wait.wait_count)) {
        // Coalesce multiple wait_queue if the later one allows fewer in-flight ops.
        pending_wait = {pending_wait.insert_before, wait_count};
      }
    }
  }

  // Given pipelined blocks and async-related information, generate final loop statements with async
  // scopes (if any).
  Array<Stmt> CompletePipelineLoopStatements(
      const std::vector<RewrittenBlockInfo>& blocks,
      const std::map<int, AsyncStateLocal>& async_states_local,
      arith::Analyzer* ana_normalized) const {
    std::vector<RewrittenBlockInfo> new_blocks = blocks;
    std::vector<int> commit_group_indices(new_blocks.size(), -1);
    for (const auto& [stage_id, state] : async_states_local) {
      if (!state.commit_groups.empty()) {
        for (size_t i = 0; i < state.commit_groups.size(); ++i) {
          for (size_t j = 0; j < state.commit_groups[i].size(); ++j) {
            ICHECK(state.commit_groups[i][0] + j < new_blocks.size());
            commit_group_indices[state.commit_groups[i][0] + j] = stage_id;
          }
        }
      }

      if (state.pending_wait.valid()) {
        auto attach_wait_scope = [&new_blocks](int i, int stage_id, PrimExpr wait_count) {
          auto& block = new_blocks[i].block;
          BlockNode* n = block.CopyOnWrite();
          auto zero = make_zero(DataType::Int(32));
          n->body =
              AttrStmt(zero, tir::attr::async_wait_queue_scope, stage_id,
                       AttrStmt(zero, tir::attr::async_wait_inflight_count, wait_count, n->body));
        };

        if (state.predicate && !ana_normalized->CanProve(state.predicate.value())) {
          // If the async operation that this wait_queue is waiting on is predicated, and we cannot
          // prove that the predicate is always true, the precise wait count is only valid
          // at iterations where the predicate is true;
          auto wait_count = Call(DataType::Int(32), builtin::if_then_else(),
                                 {state.predicate.value(), state.pending_wait.wait_count, 0});
          attach_wait_scope(state.pending_wait.insert_before, stage_id, wait_count);
        } else {
          attach_wait_scope(state.pending_wait.insert_before, stage_id,
                            state.pending_wait.wait_count);
        }
      }
    }

    Array<Stmt> stmts;

    for (size_t i = 0; i < new_blocks.size();) {
      if (commit_group_indices[i] == -1) {
        // A synchrnous block, not part of any commit group
        stmts.push_back(BlockRealize({}, new_blocks[i].predicate, new_blocks[i].block));
        ++i;
      } else {
        Array<Stmt> group_bodies;
        auto stage_id = commit_group_indices[i];
        auto predicate = new_blocks[i].predicate;
        for (; i < commit_group_indices.size() && commit_group_indices[i] == stage_id; ++i) {
          ICHECK(tvm::StructuralEqual()(predicate, new_blocks[i].predicate))
              << "Predicates in the same stage are expected to be identical";
          group_bodies.push_back(new_blocks[i].block->body);
        }

        if (group_bodies.size() > 1) {
          auto merged_bodies = SeqStmt(group_bodies);
          group_bodies.clear();
          group_bodies.push_back(merged_bodies);
        }

        for (auto body : group_bodies) {
          auto commit_queue_scope = AttrStmt(make_zero(DataType::Int(32)),
                                             tir::attr::async_commit_queue_scope, stage_id, body);
          auto new_block = MakeBlock(commit_queue_scope, buffer_data_to_buffer_);
          stmts.push_back(BlockRealize({}, predicate, new_block));
        }
      }
    }

    return stmts;
  }

  /*!
   * \brief Emit the pipeline loop in the given range.
   * \param start The start of the range
   * \param end The end of the range
   * \param unroll_loop Whether the loop should be unrolled.
   * \param extra_loop_lower_bound Extra loop lower bound.
   * \return The result loop.
   */
  Stmt EmitImpl(PrimExpr start, PrimExpr end, bool unroll_loop,
                Optional<PrimExpr> extra_loop_lower_bound = NullOpt) {
    PrimExpr new_loop_var;
    PrimExpr extent = end - start;

    auto make_nop = []() { return BlockRealize({}, Bool(true), MakeBlock(Evaluate(0), {})); };

    if (analyzer_.CanProve(extent <= 0)) {
      return make_nop();
    }
    bool is_unit_loop = analyzer_.CanProveEqual(extent, 1);
    if (is_unit_loop) {
      new_loop_var = start;  // use constants as the loop var for unit loops
    } else {
      new_loop_var = pipeline_loop_->loop_var.copy_with_suffix("");
      analyzer_.Bind(Downcast<Var>(new_loop_var), Range(start, end));
    }

    // In contrast to analyzer_ which is bound to [start, end), this one is bound to
    // the "normalized" range, [pipeline_loop_->min, extent).
    arith::Analyzer ana_normalized;
    if (!is_unit_loop) {
      ana_normalized.Bind(Downcast<Var>(new_loop_var), Range(pipeline_loop_->min, extent));
    }

    std::vector<RewrittenBlockInfo> new_blocks;

    // Async related
    std::map<int, AsyncStateLocal> async_states_local;
    std::unordered_map<const BufferNode*, int> buffer_to_commit_group;

    for (const Block& block : ordered_stmts_) {
      int stage = pipeline_info_.at(block).stage;
      PrimExpr skewed_loop_var = new_loop_var - stage;
      PrimExpr inbound = analyzer_.Simplify(pipeline_loop_->min <= skewed_loop_var) &&
                         (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent);
      if (extra_loop_lower_bound.defined()) {
        inbound = analyzer_.Simplify(inbound && new_loop_var >= extra_loop_lower_bound.value());
      }
      if (analyzer_.CanProve(!inbound)) {
        continue;
      }
      Block new_block = Downcast<Block>(PipelineBodyRewriter(buffer_data_to_buffer_, buffer_remap_,
                                                             pipeline_loop_, max_stage_ != 1,
                                                             fragment_info_)(block));

      PrimExpr delta = start - pipeline_loop_->min;
      // This variable corresponds to
      // - "producer_head" if this stage is an async producer
      // - "consumer_head" if this stage reads from asynchronously written buffers.
      PrimExpr normalized_access_index = is_unit_loop ? skewed_loop_var : skewed_loop_var + delta;

      // Adjust the block predicate and the body according to the final loop bound
      //  [pipeline_loop_->min, extent).
      if (!is_unit_loop) {
        Var loop_iter = Downcast<Var>(new_loop_var);
        inbound = Substitute(inbound, {{loop_iter, loop_iter + delta}});
      }

      new_block = Downcast<Block>(
          Substitute(new_block, {{pipeline_loop_->loop_var, normalized_access_index}}));

      if (pipeline_info_[block].async) {
        auto& local_state = async_states_local[stage];

        int commit_group_id = -1;
        if (local_state.commit_groups.empty() || local_state.consumed) {
          // consumed == true means there is already a consumer stage waiting for an
          // eariler async operation of this stage. In such cases, we make multiple commit_queue
          // for this stage.
          commit_group_id = local_state.commit_groups.size();
          local_state.commit_groups.push_back({new_blocks.size()});
        } else {
          // This is the case when one commit_queue groups multiple async blocks.
          // with commit_queue(stage):
          //   async_scope:
          //     A_shared[...] = ...
          //   async_scope:
          //     B_shared[...] = ...

          commit_group_id = local_state.commit_groups.size() - 1;
          local_state.commit_groups.back().push_back(new_blocks.size());
        }

        for (auto write_region : new_block->writes) {
          async_states[stage].dst_buffers.insert(write_region->buffer.get());
          buffer_to_commit_group[write_region->buffer.get()] = commit_group_id;
        }

        local_state.producer_head = normalized_access_index;

        if (!local_state.predicate || ana_normalized.CanProve(local_state.predicate.value())) {
          local_state.predicate = inbound;
        } else if (local_state.predicate) {
          local_state.predicate = ana_normalized.Simplify(local_state.predicate.value() & inbound);
        }

        BlockNode* n = new_block.CopyOnWrite();
        n->body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::async_scope, 1, n->body);
      }

      new_blocks.push_back(
          {stage, inbound, new_block, normalized_access_index, pipeline_info_[block].async});

      for (auto read_region : new_block->reads) {
        for (auto kv : async_states) {
          int producer_stage_id = kv.first;
          if (producer_stage_id <= stage && kv.second.writes(read_region->buffer)) {
            async_states_local[producer_stage_id].consumed = true;
          }
        }
      }
    }

    PopulateWaitCounts(new_blocks, &ana_normalized, buffer_to_commit_group, &async_states_local);
    auto stmts = CompletePipelineLoopStatements(new_blocks, async_states_local, &ana_normalized);

    Stmt new_loop{nullptr};

    if (stmts.empty()) {
      return make_nop();
    }
    if (stmts.size() == 1) {
      new_loop = stmts[0];
    } else {
      new_loop = SeqStmt(stmts);
    }

    if (!is_unit_loop) {
      new_loop = For(Downcast<Var>(new_loop_var), pipeline_loop_->min, extent,
                     unroll_loop ? ForKind::kUnrolled : pipeline_loop_->kind, std::move(new_loop),
                     NullOpt, preserved_annotations_);
    }

    // Update producer heads in the global async states.
    for (const auto& kv : async_states_local) {
      const int stage_id = kv.first;
      const AsyncStateLocal& state = kv.second;

      if (state.predicate && ana_normalized.CanProve(state.predicate.value()) &&
          async_states[stage_id].producer_head) {
        // Advance the "global" producer head if it is still valid and we know exactly how much we
        // can increment
        async_states[stage_id].producer_head =
            async_states[stage_id].producer_head.value() + extent;
      } else {
        // Otherwise, invalidate the global producer head
        async_states[stage_id].producer_head = NullOpt;
      }
    }

    return BlockRealize({}, Bool(true), MakeBlock(std::move(new_loop), buffer_data_to_buffer_));
  }

  arith::Analyzer analyzer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& double_buffers_;
  Array<Buffer> pipeline_allocs_;
  For pipeline_loop_;
  PipelineInfo pipeline_info_;
  const std::unordered_map<const VarNode*, FragmentInfo>& fragment_info_;
  int max_stage_ = -1;
  Map<Buffer, Buffer> buffer_remap_;
  Array<Block> ordered_stmts_;
  std::map<int, AsyncStateGlobal> async_states;
  Map<String, ObjectRef> preserved_annotations_;
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
  std::unordered_map<Var, Array<Block>> buffer_writers;

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
    injector.fragment_info_ = GetTensorCoreFragmentInfo(func->body);
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
    Stmt pipeline_body{nullptr};
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
        Downcast<Array<Integer>>(op->annotations.at(attr::software_pipeline_stage));
    auto pipeline_orders =
        Downcast<Array<Integer>>(op->annotations.at(attr::software_pipeline_order));
    CHECK_EQ(pipeline_stages.size(), original_order.size())
        << "PrimFunc " << global_symbol_ << " has original order "
        << original_order.Map([](const auto& block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_stages << " with different size";
    CHECK_EQ(pipeline_orders.size(), original_order.size())
        << "PrimFunc " << global_symbol_ << " has original order "
        << original_order.Map([](const auto& block) { return block->name_hint; })
        << ", but pipeline annotation is " << pipeline_orders << " with different size";

    std::unordered_set<int> pipeline_async_stages;
    if (auto annot = op->annotations.Get(attr::software_pipeline_async_stages)) {
      for (auto s : Downcast<Array<Integer>>(annot)) {
        pipeline_async_stages.insert(s->value);
      }
    }

    Map<String, ObjectRef> preserved_annotations;
    for (const auto& kv : op->annotations) {
      const String& key = kv.first;
      if (kv.first != attr::software_pipeline_stage && kv.first != attr::software_pipeline_order &&
          kv.first != attr::software_pipeline_async_stages) {
        preserved_annotations.Set(key, kv.second);
      }
    }

    for (size_t i = 0; i < pipeline_stages.size(); i++) {
      int stage = static_cast<int>(pipeline_stages[i]->value);
      bool is_async = pipeline_async_stages.find(stage) != pipeline_async_stages.end();
      PipelineAnnotation stage_order{stage,
                                     /*order=*/static_cast<int>(pipeline_orders[i]->value),
                                     is_async};
      pipeline_info.emplace(original_order[i], stage_order);
    }

    ValidatePipelineBody(pipeline_info, original_order);

    // Step 4: Rewrite the pipeline body.
    Stmt pipeline = PipelineRewriter::Rewrite(buffer_data_to_buffer_, double_buffers,
                                              pipeline_allocs, GetRef<For>(op), pipeline_info,
                                              fragment_info_, preserved_annotations);

    if (const auto* realize = op->body.as<BlockRealizeNode>()) {
      const auto& block = realize->block;
      for (const auto& buffer : block->alloc_buffers) {
        buffer_data_to_buffer_.erase(buffer->data);
      }
    }
    return pipeline;
  }

  /*!
   * \brief Add buffer allocations to a block and update the write region of the block.
   * \param n The block pointer to which the buffer allocations are added.
   * \param alloc_buffers The buffer allocations to be added.
   */
  void AddAllocBuffers(BlockNode* n, const Array<Buffer> alloc_buffers) {
    for (const Buffer& alloc_buffer : alloc_buffers) {
      n->alloc_buffers.push_back(alloc_buffer);
      Region region;
      region.reserve(alloc_buffer->shape.size());
      for (const PrimExpr& dim : alloc_buffer->shape) {
        region.push_back(Range::FromMinExtent(0, dim));
      }
      n->writes.push_back(BufferRegion(alloc_buffer, region));
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    auto it = op->annotations.find(attr::double_buffer_scope);
    if (it != op->annotations.end()) {
      int buffer_index = Downcast<Integer>((*it).second).IntValue();
      CHECK(buffer_index >= 0 && static_cast<size_t>(buffer_index) < op->writes.size())
          << "ValueError: Index of the buffer exceeds the size of the write regions of the block. ("
          << buffer_index << " vs. " << op->writes.size() << ")";
      double_buffers.insert(op->writes[buffer_index]->buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    for (const auto& buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

  bool HasPipelineAnnotation(const ForNode* op) const {
    auto it1 = op->annotations.find(attr::software_pipeline_stage);
    auto it2 = op->annotations.find(attr::software_pipeline_order);
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
  std::unordered_map<const VarNode*, FragmentInfo> fragment_info_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> double_buffers;
  Optional<String> global_symbol_;
};

}  // namespace software_pipeline

namespace transform {

/*!
 * \brief Transform annotated loops into pipelined one that parallelize producers and consumers.
 * \return The IR transform pass.
 */
Pass InjectSoftwarePipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    fptr->body = software_pipeline::PipelineInjector::Inject(f);
    fptr->body = ConvertSSA(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectSoftwarePipeline", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectSoftwarePipeline").set_body_typed(InjectSoftwarePipeline);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
