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
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/bulk_copy.h"

namespace tvm {
namespace tl {

using namespace tir;

template <typename T>
static bool ArrayIntersect(Array<T> A, Array<T> B) {
  for (const auto& a : A) {
    for (const auto& b : B) {
      if (a == b) return true;
    }
  }
  return false;
}

struct PipelineStageInfo {
  bool is_producer = false;
  Array<Buffer> reads, writes;

  // Indicate this stmt's completion is tracked by a barrier
  // -1 means not
  int release_barrier_id = -1;

  // Indicate this atmt need to acquire a barrier before execution
  // -1 means not
  int acquire_barrier_id = -1;

  // If true, means that this statement is the last statement of this release barrier
  // release_barrier_id will be released after this statement
  bool release_after = false;
};

struct SyncPattern {
  int release_idx, acquire_idx;
};

struct PipelineInfo {
  std::vector<PipelineStageInfo> pinfos;
  Array<Buffer> versioned_buffers;
  size_t num_barriers;
};

void debug_pipelineinfo(PipelineInfo& info) {
  std::cout << "stages " << std::endl;
  for (auto pinfo : info.pinfos) {
    std::cout << pinfo.is_producer << " " << pinfo.release_barrier_id << " "
              << pinfo.acquire_barrier_id << std::endl;
  }
}

static PipelineStageInfo MakePipelineStageInfo_impl(
    Stmt stmt, const std::unordered_set<const BufferNode*>& scoped_buffers,
    Map<Var, Buffer> buffer_data_to_buffer) {
  Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"", /*body*/ stmt);
  auto access = GetBlockAccessRegion(block, buffer_data_to_buffer);
  PipelineStageInfo pinfo;
  for (auto buffer_region : access[0]) {
    if (scoped_buffers.count(buffer_region->buffer.get()))
      pinfo.reads.push_back(buffer_region->buffer);
  }
  for (auto buffer_region : access[1]) {
    if (scoped_buffers.count(buffer_region->buffer.get()))
      pinfo.writes.push_back(buffer_region->buffer);
  }

  size_t cnt_global_read = 0, cnt_shared_write = 0;
  for (auto region : access[0]) {
    if (region->buffer.scope() == "global") ++cnt_global_read;
  }
  for (auto region : access[1]) {
    if (region->buffer.scope() == "shared" || region->buffer.scope() == "shared.dyn")
      ++cnt_shared_write;
  }
  if (cnt_global_read == access[0].size() && cnt_shared_write == access[1].size())
    pinfo.is_producer = true;
  return pinfo;
}

static std::vector<PipelineStageInfo> MakePipelineStageInfo(
    Array<Stmt> seq_stmt, Array<Buffer> scoped_buffers, Map<Var, Buffer> buffer_data_to_buffer) {
  std::unordered_set<const BufferNode*> set;
  set.reserve(scoped_buffers.size());
  for (auto buffer : scoped_buffers) set.insert(buffer.get());

  std::vector<PipelineStageInfo> pinfos;
  for (auto stmt : seq_stmt)
    pinfos.push_back(MakePipelineStageInfo_impl(stmt, set, buffer_data_to_buffer));
  return pinfos;
}

static std::vector<SyncPattern> MakeSyncPattern(Array<Stmt> seq_stmt, bool is_loop,
                                                const std::vector<PipelineStageInfo>& pinfos) {
  int n = seq_stmt.size();
  std::vector<SyncPattern> sync_patterns;
  // producer_release consumer_acquire,
  // inject before the first consumer stmt for each producer
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (pinfos[i].is_producer && !pinfos[j].is_producer &&
          ArrayIntersect(pinfos[i].writes, pinfos[j].reads)) {
        sync_patterns.push_back({i, j});
        break;
      }
    }
  }

  // consumer_release producer_acquire
  // valid when is_loop is true
  // inject before the earlest producer stmt for each consumer
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      if (!pinfos[i].is_producer && pinfos[j].is_producer &&
          ArrayIntersect(pinfos[i].reads, pinfos[j].writes) && is_loop) {
        sync_patterns.push_back({i, j});
        break;
      }
    }
  }

  /*
    Simplify multiple release-acquire pairs into one
    ------------------
      Produce(A)
      Produce(B)
      Consume(A, B)
    ------------------
    [(0, 2), (1, 2), (2, 0)] -> [(1, 2), (2, 0)]

    Or
    ------------------
      Produce(A, B)
      Consume(A)
      Consume(B)
    ------------------
    [(0, 1), (1, 0), (2, 0)] -> [(0, 1), (2, 0)]
  */
  int M = sync_patterns.size();
  std::vector<bool> removed(M, false);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      if (pinfos[i].is_producer == pinfos[j].is_producer &&
          sync_patterns[i].acquire_idx >= sync_patterns[j].acquire_idx &&
          sync_patterns[i].release_idx < sync_patterns[j].release_idx)
        removed[i] = true;
    }
  }

  std::vector<SyncPattern> sync_pattern_cleaned;
  sync_pattern_cleaned.reserve(M);
  for (int i = 0; i < M; i++)
    if (!removed[i]) sync_pattern_cleaned.push_back(sync_patterns[i]);

  // std::cout << n << std::endl;
  // for (auto pattern : sync_pattern_cleaned) {
  //   std::cout << pattern.release_idx << " " << pattern.acquire_idx << std::endl;
  // }

  return sync_pattern_cleaned;
}

PipelineInfo ExtractPipelineInfo(Array<Stmt> seq_stmt, Array<Buffer> scoped_buffers,
                                 Map<Var, Buffer> buffer_data_to_buffer) {
  PipelineInfo info;
  info.pinfos = MakePipelineStageInfo(seq_stmt, scoped_buffers, buffer_data_to_buffer);
  auto patterns = MakeSyncPattern(seq_stmt, true, info.pinfos);

  info.num_barriers = patterns.size();
  for (size_t i = 0; i < patterns.size(); i++) {
    info.pinfos[patterns[i].acquire_idx].acquire_barrier_id = i;
    info.pinfos[patterns[i].release_idx].release_barrier_id = i;
    info.pinfos[patterns[i].release_idx].release_after = true;
  }

  int cur_consumer_barrier = -1, cur_producer_barrier = -1;
  for (int i = seq_stmt.size(); i >= 0; i--) {
    if (info.pinfos[i].is_producer) {
      if (info.pinfos[i].release_barrier_id == -1) {
        info.pinfos[i].release_barrier_id = cur_producer_barrier;
      } else {
        cur_producer_barrier = info.pinfos[i].release_barrier_id;
      }
    } else {
      if (info.pinfos[i].release_barrier_id == -1) {
        info.pinfos[i].release_barrier_id = cur_consumer_barrier;
      } else {
        cur_consumer_barrier = info.pinfos[i].release_barrier_id;
      }
    }
  }

  std::unordered_set<const BufferNode*> consumer_used, producer_used;
  for (const auto& pinfo : info.pinfos) {
    if (pinfo.is_producer) {
      for (Buffer buffer : pinfo.writes) producer_used.insert(buffer.get());
    } else {
      for (Buffer buffer : pinfo.reads) consumer_used.insert(buffer.get());
    }
  }

  info.versioned_buffers.reserve(scoped_buffers.size());
  for (Buffer buffer : scoped_buffers) {
    if (consumer_used.count(buffer.get()) && producer_used.count(buffer.get())) {
      info.versioned_buffers.push_back(buffer);
    }
  }
  return info;
}

class ProducerTraitsCollector : public StmtExprMutator {
 public:
  ProducerTraitsCollector() { Clear(); }

  void Clear() {
    bulk_copy_bytes = 0;
    loop_extents = 1;
    has_simt_copy = false;
  }

  Stmt Rewrite(Stmt stmt, PrimExpr producer_barrier_idx) {
    producer_barrier_idx_ = producer_barrier_idx;
    return VisitStmt(stmt);
  }

  bool HasSimtCopy() { return has_simt_copy; }

  PrimExpr BulkCopyBytes() { return bulk_copy_bytes; }

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op == TMACopyOp()) {
      Call access_ptr = Downcast<Call>(call->args[2]);
      ICHECK(access_ptr->op.same_as(builtin::tvm_access_ptr()));
      int type_bytes = access_ptr->args[0]->dtype.bytes();
      bulk_copy_bytes += access_ptr->args[3] * loop_extents * type_bytes;
      call.CopyOnWrite()->args.Set(1, producer_barrier_idx_);
    }
    return call;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr old_loop_evtents = loop_extents;
    loop_extents *= op->extent;
    auto stmt = StmtExprMutator::VisitStmt_(op);
    loop_extents = old_loop_evtents;
    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    has_simt_copy = true;
    return StmtExprMutator::VisitExpr_(op);
  }

  bool has_simt_copy;
  PrimExpr bulk_copy_bytes;
  PrimExpr loop_extents;
  PrimExpr producer_barrier_idx_;
};

class MultiVersionBufferRewriter : public StmtExprMutator {
 public:
  static PrimFunc Rewrite(PrimFunc& f) {
    auto rewriter = MultiVersionBufferRewriter();
    rewriter.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : rewriter.buffer_lca_) {
      Var buffer_var = buffer->data;
      rewriter.buffer_data_to_buffer_.Set(buffer_var, buffer);
    }
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

 private:
  MultiVersionBufferRewriter() = default;

  static Buffer RewriteAllocBuffer(const Buffer& buffer, int num_versions) {
    ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
    new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
    if (new_buffer->strides.size()) {
      ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
      PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
      new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
    }
    return Buffer(new_buffer);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    AttrStmt attr_node = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
    if (attr_node->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(attr_node->node)->thread_tag == "threadIdx.x") {
      auto block = Downcast<BlockRealize>(attr_node->body)->block;
      auto thread_iv = Downcast<IterVar>(attr_node->node);
      Array<Buffer> alloc_buffers;
      for (auto buffer : block->alloc_buffers) {
        if (buffer_remap_.count(buffer)) {
          Buffer new_buffer = buffer_remap_[buffer];
          alloc_buffers.push_back(new_buffer);
        } else {
          alloc_buffers.push_back(buffer);
        }
      }
      block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
      return AttrStmt(attr_node->node, attr_node->attr_key, attr_node->value,
                      BlockRealize({}, Bool(true), block));
    }
    return attr_node;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (!num_stages_anno.defined()) return StmtExprMutator::VisitStmt_(op);

    ICHECK(num_stages_anno.as<IntImmNode>());
    int num_stages = static_cast<int>(num_stages_anno.as<IntImmNode>()->value);

    const SeqStmtNode* pipeline_body_seq = op->body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
        << "ValueError: The body of the software pipeline should be SeqStmt, got "
        << op->body->GetTypeKey();

    Array<Buffer> scoped_buffers = {};
    for (auto [buffer, stmt] : buffer_lca_) {
      if (stmt.defined() && stmt.value().get() == op) scoped_buffers.push_back(buffer);
    }

    PipelineInfo info =
        ExtractPipelineInfo(pipeline_body_seq->seq, scoped_buffers, buffer_data_to_buffer_);

    // Map<Buffer, Buffer> buffer_remap;
    for (auto buffer : info.versioned_buffers) {
      Var buffer_var = buffer->data;
      Buffer new_buffer = RewriteAllocBuffer(buffer, num_stages);
      buffer_remap_.Set(buffer, new_buffer);
    }
    version_index_ = FloorMod(op->loop_var - op->min, num_stages);
    auto for_node = StmtExprMutator::VisitStmt_(op);

    return for_node;
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
    n->indices.insert(n->indices.begin(), version_index_);
    return std::move(load);
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
    n->indices.insert(n->indices.begin(), version_index_);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      return RewriteBufferAccess(call, {1});
    }
    return call;
  }

  PrimExpr RewriteBufferAccess(const Call& call, const std::vector<int> arg_indices) {
    auto product = [](const Array<PrimExpr>& input) {
      return foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                   make_const(DataType::Int(32), 1), input);
    };
    Array<PrimExpr> new_args = call->args;
    for (int i : arg_indices) {
      auto buffer_var = Downcast<Var>(call->args[i]);
      if (!buffer_data_to_buffer_.count(buffer_var)) continue;
      const Buffer& buffer = buffer_data_to_buffer_[buffer_var];
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
        PrimExpr new_index = old_index + version_index_ * offset;
        new_args.Set(i + 1, new_index);
      }
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  PrimExpr version_index_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Buffer, Buffer> buffer_remap_;
};

class WarpSpecializedCodeEmitter : public StmtMutator {
 public:
  WarpSpecializedCodeEmitter(bool is_emitting_producer, IterVar thread_iv,
                             Map<Var, Buffer> buffer_data_to_buffer,
                             Map<Buffer, Optional<Stmt>> buffer_lca)
      : is_emitting_producer_(is_emitting_producer) {
    buffer_data_to_buffer_ = buffer_data_to_buffer;
    buffer_lca_ = buffer_lca;
    null_stmt_ = Evaluate(0);
    num_barriers_ = 0;
    if (is_emitting_producer)
      required_thread_extent_ = 1;  // TODO: fix the case for SIMT copy
    else
      required_thread_extent_ = thread_iv->dom->extent;
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (!num_stages_anno.defined()) return StmtMutator::VisitStmt_(op);
    ICHECK(num_stages_anno.as<IntImmNode>());
    int num_stages = static_cast<int>(num_stages_anno.as<IntImmNode>()->value);

    const SeqStmtNode* pipeline_body_seq = op->body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
        << "ValueError: The body of the software pipeline should be SeqStmt, got "
        << op->body->GetTypeKey();

    Array<Buffer> scoped_buffers = {};
    for (auto [buffer, stmt] : buffer_lca_) {
      if (stmt.defined() && stmt.value().get() == op) scoped_buffers.push_back(buffer);
    }

    PipelineInfo info =
        ExtractPipelineInfo(pipeline_body_seq->seq, scoped_buffers, buffer_data_to_buffer_);
    PrimExpr buffer_version = FloorMod(op->loop_var - op->min, num_stages);
    PrimExpr parity_bit = is_emitting_producer_
                              ? bitwise_and(FloorDiv(op->loop_var - op->min, num_stages) + 1, 1)
                              : bitwise_and(FloorDiv(op->loop_var - op->min, num_stages), 1);

    Array<Stmt> new_body;

    // Track all arrived barriers
    for (int i = 0; i < static_cast<int>(pipeline_body_seq->seq.size()); i++) {
      if (info.pinfos[i].is_producer != is_emitting_producer_) continue;
      if (!info.pinfos[i].release_after) continue;
      for (int j = 0; j < num_stages; j++) {
        released_barrier_.insert(num_barriers_ + j +
                                 num_stages * info.pinfos[i].release_barrier_id);
      }
    }

    if (is_emitting_producer_) {  // producer case
      ProducerTraitsCollector collector;
      for (int i = 0; i < static_cast<int>(pipeline_body_seq->seq.size()); i++) {
        if (!info.pinfos[i].is_producer) continue;
        if (info.pinfos[i].acquire_barrier_id != -1) {
          PrimExpr acquire_barrier_id =
              buffer_version + num_barriers_ + num_stages * info.pinfos[i].acquire_barrier_id;
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), MBarrierWaitParity(), {acquire_barrier_id, parity_bit})));
        }
        ICHECK(info.pinfos[i].release_barrier_id >= 0);
        PrimExpr release_barrier_id =
            buffer_version + num_barriers_ + num_stages * info.pinfos[i].release_barrier_id;
        auto stmt = collector.Rewrite(pipeline_body_seq->seq[i], release_barrier_id);
        if (!is_zero(collector.BulkCopyBytes())) {
          new_body.push_back(
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier_expect_tx(),
                            {release_barrier_id, collector.BulkCopyBytes()})));
        }
        new_body.push_back(stmt);
        if (collector.HasSimtCopy() > 0) {
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_cp_async_barrier(), {release_barrier_id})));
        }
        if (info.pinfos[i].release_after) {
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {release_barrier_id})));
        }
        collector.Clear();
      }
    } else {  // consumer case
      for (int i = 0; i < static_cast<int>(pipeline_body_seq->seq.size()); i++) {
        if (info.pinfos[i].is_producer) continue;
        if (info.pinfos[i].acquire_barrier_id != -1) {
          PrimExpr acquire_barrier_id =
              buffer_version + num_barriers_ + num_stages * info.pinfos[i].acquire_barrier_id;
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), MBarrierWaitParity(), {acquire_barrier_id, parity_bit})));
        }
        new_body.push_back(pipeline_body_seq->seq[i]);
        if (info.pinfos[i].release_after) {
          PrimExpr release_barrier_id =
              buffer_version + num_barriers_ + num_stages * info.pinfos[i].release_barrier_id;
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {release_barrier_id})));
        }
      }
    }

    ICHECK(new_body.size() > 0);

    num_barriers_ += num_stages * info.num_barriers;

    auto for_node = GetRef<For>(op);
    auto ptr = for_node.CopyOnWrite();
    ptr->annotations.erase("num_stage");
    ptr->body = new_body.size() == 1 ? new_body[0] : SeqStmt(new_body);
    return for_node;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    AttrStmt attr_stmt = Downcast<AttrStmt>(StmtMutator::VisitStmt_(op));
    if (attr_stmt->body == null_stmt_) return null_stmt_;
    return attr_stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore buffer_store = Downcast<BufferStore>(StmtMutator::VisitStmt_(op));
    // TODO: fix this
    if (buffer_store->buffer.scope() != "shared.dyn" && is_emitting_producer_) return null_stmt_;
    return buffer_store;
  }

  int num_barriers_;
  Stmt null_stmt_;
  const bool is_emitting_producer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  PrimExpr required_thread_extent_;
  std::unordered_set<int> released_barrier_;
  friend class WarpSpecializedPipeline;
};

class WarpSpecializedPipeline : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    f = MultiVersionBufferRewriter::Rewrite(f);
    auto T = WarpSpecializedPipeline();
    T.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : T.buffer_lca_) T.buffer_data_to_buffer_.Set(buffer->data, buffer);
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    AttrStmt attr_node = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
    if (attr_node->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(attr_node->node)->thread_tag == "threadIdx.x") {
      auto block = Downcast<BlockRealize>(attr_node->body)->block;
      auto thread_iv = Downcast<IterVar>(attr_node->node);

      WarpSpecializedCodeEmitter producer_emitter{true, thread_iv, buffer_data_to_buffer_,
                                                  buffer_lca_};
      WarpSpecializedCodeEmitter consumer_emitter{false, thread_iv, buffer_data_to_buffer_,
                                                  buffer_lca_};
      Stmt producer_code = producer_emitter(block->body);
      Stmt consumer_code = consumer_emitter(block->body);

      PrimExpr consumer_thread_extent = consumer_emitter.required_thread_extent_;
      PrimExpr producer_thread_extent = producer_emitter.required_thread_extent_;
      PrimExpr new_thread_extent = consumer_thread_extent + producer_thread_extent;
      thread_iv.CopyOnWrite()->dom = {0, new_thread_extent};

      ICHECK(producer_emitter.num_barriers_ == consumer_emitter.num_barriers_)
          << producer_emitter.num_barriers_ << " " << consumer_emitter.num_barriers_;
      int num_barriers = producer_emitter.num_barriers_;
      Stmt create_barrier =
          Evaluate(Call(DataType::Handle(), builtin::create_barriers(), {num_barriers}));

      Array<Stmt> barrier_init_seq;
      barrier_init_seq.reserve(num_barriers);
      for (int i = 0; i < num_barriers; i++) {
        PrimExpr arrive_thread_count = producer_emitter.released_barrier_.count(i)
                                           ? producer_emitter.required_thread_extent_
                                           : consumer_emitter.required_thread_extent_;
        Stmt init_barrier =
            Evaluate(Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                          {i, arrive_thread_count}));
        barrier_init_seq.push_back(init_barrier);
      }

      Stmt init_barrier = IfThenElse(EQ(thread_iv->var, 0), SeqStmt(barrier_init_seq));
      Stmt mem_sync =
          Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(), {StringImm("shared")}));
      Stmt body =
          IfThenElse(GE(thread_iv->var, consumer_thread_extent), producer_code, consumer_code);
      block.CopyOnWrite()->body = SeqStmt({create_barrier, init_barrier, mem_sync, body});

      return AttrStmt(thread_iv, attr_node->attr_key, new_thread_extent,
                      BlockRealize({}, Bool(true), block));
    }
    return attr_node;
  }

  WarpSpecializedPipeline() = default;

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Buffer, Buffer> buffer_remap_;
};

using namespace tir::transform;

tvm::transform::Pass WarpSpecializedPipeline() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return WarpSpecializedPipeline::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.WarpSpecializedPipeline", {});
}

TVM_REGISTER_GLOBAL("tl.WarpSpecializedPipeline").set_body_typed(WarpSpecializedPipeline);

}  // namespace tl
}  // namespace tvm
