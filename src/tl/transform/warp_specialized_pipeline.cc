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

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

struct PipelineStageInfo {
  bool is_producer = false;
  Array<Buffer> reads, writes;
};

struct PipelineInfo {
  std::vector<PipelineStageInfo> pinfos;
  Array<Buffer> versioned_buffers;
};

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

PipelineInfo ExtractPipelineInfo(Array<Stmt> seq_stmt, Array<Buffer> scoped_buffers,
                                 Map<Var, Buffer> buffer_data_to_buffer) {
  PipelineInfo info;
  info.pinfos = MakePipelineStageInfo(seq_stmt, scoped_buffers, buffer_data_to_buffer);

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

enum class Role { kConsumer, kProducer, kBoth };

class WarpSpecializedRoleMarker : public StmtVisitor {
 public:
  WarpSpecializedRoleMarker(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  Role GetRole(const StmtNode* stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end());
    return it->second;
  }

  Role GetRole(const Stmt& stmt) const { return GetRole(stmt.get()); }

  void VisitStmt_(const EvaluateNode* op) final {
    Role role = Role::kConsumer;
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(TMALoadOp())) {
        role = Role::kProducer;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    bool is_shared_store = op->buffer.scope() == "shared.dyn" || op->buffer.scope() == "shared";
    if (!is_shared_store) {
      SetRole(op, Role::kConsumer);
      return;
    }

    // Check reads from global
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ GetRef<Stmt>(op));
    auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto reads = access[0];
    Role role = Role::kProducer;
    for (auto read : reads) {
      if (read->buffer.scope() != "global") {
        role = Role::kConsumer;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (auto stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined()) {
      auto role_else = GetRole(op->else_case.value());
      if (role != role_else) role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType>
  void HandleBodyStmt(const NodeType* op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }

 private:
  void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const StmtNode*, Role> map_;
};

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
    if (call->op == TMALoadOp()) {
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

class ThreadIdxRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced) {
    auto rewriter = ThreadIdxRewriter(thread_var, replaced);
    return rewriter(stmt);
  }

 private:
  ThreadIdxRewriter(Var thread_var, PrimExpr replaced)
      : thread_var_(thread_var), replaced_(replaced) {}

  PrimExpr VisitExpr_(const VarNode* var) final {
    if (var == thread_var_.get()) {
      return replaced_;
    } else {
      return StmtExprMutator::VisitExpr_(var);
    }
  }

  Var thread_var_;
  PrimExpr replaced_;
};

class WSCodeEmitter : public StmtMutator {
 public:
  WSCodeEmitter(bool is_emitting_producer, IterVar thread_iv,
                Map<Var, Buffer> buffer_data_to_buffer, const WarpSpecializedRoleMarker& marker)
      : is_emitting_producer_(is_emitting_producer),
        buffer_data_to_buffer_(buffer_data_to_buffer),
        marker_(marker),
        thread_var_(thread_iv->var) {}

 private:
  template <typename NodeType>
  Stmt FilterByRole(const NodeType* op) {
    Role role = marker_.GetRole(op);
    if (role == Role::kBoth)
      return StmtMutator::VisitStmt_(op);
    else if ((role == Role::kProducer) == is_emitting_producer_)
      return GetRef<Stmt>(op);
    else
      return Evaluate(0);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    bool has_producer = false;
    for (auto stmt : op->seq) {
      if (marker_.GetRole(stmt) == Role::kProducer) {
        has_producer = true;
        break;
      }
    }
    bool need_producer_sync = has_producer && marker_.GetRole(op) == Role::kBoth;
    if (!need_producer_sync) return FilterByRole(op);

    auto seq_transformed = op->seq.Map([&](Stmt stmt) { return VisitStmt(stmt); });

    auto map = ExtractSyncPattern(op->seq);
    Array<Stmt> new_body;

    if (is_emitting_producer_) {  // producer case
      ProducerTraitsCollector collector;
      for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
        if (marker_.GetRole(op->seq[i]) == Role::kConsumer) continue;
        if (marker_.GetRole(op->seq[i]) == Role::kBoth) {
          new_body.push_back(seq_transformed[i]);
          continue;
        }
        if (map.acquire[i] != -1) {
          PrimExpr acquire_barrier_id = stage_ + num_barriers_ + num_stages_ * map.acquire[i];
          new_body.push_back(Evaluate(Call(DataType::Handle(), MBarrierWaitParity(),
                                           {acquire_barrier_id, bitwise_xor(parity_, 1)})));
        }
        ICHECK(map.release[i] >= 0);
        PrimExpr release_barrier_id = stage_ + num_barriers_ + num_stages_ * map.release[i];
        auto stmt = collector.Rewrite(seq_transformed[i], release_barrier_id);
        if (!is_zero(collector.BulkCopyBytes())) {
          auto expect_tx_call = Call(DataType::Handle(), MBarrierExpectTX(),
                                {release_barrier_id, collector.BulkCopyBytes()});
          new_body.push_back(IfThenElse(EQ(thread_var_, 0), Evaluate(expect_tx_call)));
        }
        new_body.push_back(stmt);
        if (collector.HasSimtCopy() > 0) {
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_cp_async_barrier(), {release_barrier_id})));
        }
        if (map.release_after[i]) {
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {release_barrier_id})));
          for (int j = 0; j < num_stages_; j++) {
            released_barrier_.insert(j + num_barriers_ + num_stages_ * map.release[i]);
          }
        }
        collector.Clear();
      }
    } else {  // consumer case
      for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
        if (marker_.GetRole(op->seq[i]) == Role::kProducer) continue;
        if (map.acquire[i] != -1) {
          PrimExpr acquire_barrier_id = stage_ + num_barriers_ + num_stages_ * map.acquire[i];
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), MBarrierWaitParity(), {acquire_barrier_id, parity_})));
        }
        new_body.push_back(seq_transformed[i]);
        if (map.release_after[i]) {
          PrimExpr release_barrier_id = stage_ + num_barriers_ + num_stages_ * map.release[i];
          new_body.push_back(Evaluate(
              Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {release_barrier_id})));
          for (int j = 0; j < num_stages_; j++) {
            released_barrier_.insert(j + num_barriers_ + num_stages_ * map.release[i]);
          }
        }
      }
    }

    num_barriers_ += map.num_barriers * num_stages_;

    ICHECK(new_body.size() > 0);
    return new_body.size() == 1 ? new_body[0] : SeqStmt(std::move(new_body));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    int num_stages = 1;
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (num_stages_anno.defined()) {
      ICHECK(num_stages_anno.as<IntImmNode>());
      num_stages = static_cast<int>(num_stages_anno.as<IntImmNode>()->value);
      ICHECK(num_stages_ == 1) << "Nested pipeline not supported.";
    }

    PrimExpr parity_before = std::move(parity_);
    PrimExpr stage_before = std::move(stage_);
    int num_stages_before = num_stages_;

    num_stages_ = num_stages;
    stage_ = FloorMod(op->loop_var - op->min, num_stages);
    parity_ =
        FloorMod(parity_before * op->extent + FloorDiv(op->loop_var - op->min, num_stages), 2);

    auto result = FilterByRole(op);

    parity_ = std::move(parity_before);
    stage_ = std::move(stage_before);
    num_stages_ = num_stages_before;

    // remove pipeline annotation
    auto for_node = result.as<For>();
    if (result.as<ForNode>()) {
      auto for_node = Downcast<For>(result);
      for_node.CopyOnWrite()->annotations.erase("num_stages");
      return for_node;
    }
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const EvaluateNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const AttrStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const BufferStoreNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const LetStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const AssertStmtNode* op) final { return FilterByRole(op); }
  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK(0);
    return Stmt();
  }
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    ICHECK(0);
    return Stmt();
  }

  struct SyncPattern {
    int release_idx, acquire_idx;
  };

  struct SyncPatternMap {
    std::vector<int> acquire;
    std::vector<int> release;
    std::vector<bool> release_after;
    int num_barriers;
  };

  static SyncPatternMap SyncPatternToMap(const std::vector<SyncPattern>& sync_patterns,
                                         const std::vector<bool>& is_producer) {
    size_t num_barriers = sync_patterns.size();
    size_t num_stmts = is_producer.size();
    SyncPatternMap map;
    map.acquire.resize(num_stmts, -1);
    map.release.resize(num_stmts, -1);
    map.release_after.resize(num_stmts, false);
    map.num_barriers = num_barriers;
    for (size_t i = 0; i < num_barriers; i++) {
      map.acquire[sync_patterns[i].acquire_idx] = i;
      map.release[sync_patterns[i].release_idx] = i;
      map.release_after[sync_patterns[i].release_idx] = true;
    }

    int cur_consumer_barrier = -1, cur_producer_barrier = -1;
    for (int i = num_stmts - 1; i >= 0; i--) {
      if (is_producer[i]) {
        if (map.release[i] == -1) {
          map.release[i] = cur_producer_barrier;
        } else {
          cur_producer_barrier = map.release[i];
        }
      } else {
        if (map.release[i] == -1) {
          map.release[i] = cur_consumer_barrier;
        } else {
          cur_consumer_barrier = map.release[i];
        }
      }
    }
    return map;
  }

  SyncPatternMap ExtractSyncPattern(Array<Stmt> seq_stmt) {
    std::vector<std::set<const BufferNode*>> reads, writes;
    std::vector<bool> is_producer;
    int n = seq_stmt.size();
    reads.reserve(n);
    writes.reserve(n);
    is_producer.reserve(n);
    for (int i = 0; i < n; i++) {
      Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                  /*body*/ seq_stmt[i]);
      auto access = GetBlockAccessRegion(block, buffer_data_to_buffer_);
      std::set<const BufferNode*> read_set, write_set;
      for (auto region : access[0]) read_set.insert(region->buffer.get());
      for (auto region : access[1]) write_set.insert(region->buffer.get());
      reads.push_back(std::move(read_set));
      writes.push_back(std::move(write_set));
      is_producer.push_back(marker_.GetRole(seq_stmt[i]) == Role::kProducer);
    }

    auto intersect_fn = [](const std::set<const BufferNode*>& lhs,
                           const std::set<const BufferNode*>& rhs) {
      for (auto ptr : lhs)
        if (rhs.count(ptr)) return true;
      return false;
    };

    std::vector<SyncPattern> sync_patterns;
    // producer_release consumer_acquire,
    // inject before the first consumer stmt for each producer
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if (is_producer[i] && !is_producer[j] && intersect_fn(writes[i], reads[j])) {
          sync_patterns.push_back({i, j});
          break;
        }
      }
    }

    // consumer_release producer_acquire
    // valid when is_loop is true
    // inject before the earlest producer stmt for each consumer
    bool in_loop = !is_zero(parity_);
    if (in_loop) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          if (!is_producer[i] && is_producer[j] && intersect_fn(reads[i], writes[j])) {
            sync_patterns.push_back({i, j});
            break;
          }
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
        if (is_producer[sync_patterns[i].acquire_idx] ==
                is_producer[sync_patterns[j].acquire_idx] &&
            sync_patterns[i].acquire_idx >= sync_patterns[j].acquire_idx &&
            sync_patterns[i].release_idx < sync_patterns[j].release_idx)
          removed[i] = true;
      }
    }

    std::vector<SyncPattern> sync_pattern_cleaned;
    sync_pattern_cleaned.reserve(M);
    for (int i = 0; i < M; i++)
      if (!removed[i]) sync_pattern_cleaned.push_back(sync_patterns[i]);

    // for (auto pattern : sync_pattern_cleaned) {
    //   std::cout << pattern.release_idx << " " << pattern.acquire_idx << std::endl;
    // }

    return SyncPatternToMap(sync_pattern_cleaned, is_producer);
  }

  const bool is_emitting_producer_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_set<int> released_barrier_;
  const WarpSpecializedRoleMarker& marker_;

  int num_barriers_ = 0;
  PrimExpr parity_ = 0;
  PrimExpr stage_ = 0;
  int num_stages_ = 1;
  Var thread_var_;
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

      WarpSpecializedRoleMarker marker(buffer_data_to_buffer_);
      marker(block);
      WSCodeEmitter producer(true, thread_iv, buffer_data_to_buffer_, marker);
      WSCodeEmitter consumer(false, thread_iv, buffer_data_to_buffer_, marker);
      Stmt producer_code = producer(block->body);
      Stmt consumer_code = consumer(block->body);

      PrimExpr consumer_thread_extent = thread_iv->dom->extent;
      PrimExpr producer_thread_extent = 1;

      producer_code = ThreadIdxRewriter::Rewrite(producer_code, thread_iv->var,
                                                 thread_iv->var - consumer_thread_extent);
      PrimExpr new_thread_extent = consumer_thread_extent + producer_thread_extent;
      thread_iv.CopyOnWrite()->dom = {0, new_thread_extent};

      ICHECK(producer.num_barriers_ == consumer.num_barriers_)
          << producer.num_barriers_ << " " << consumer.num_barriers_;
      int num_barriers = consumer.num_barriers_;
      Stmt create_barrier =
          Evaluate(Call(DataType::Handle(), builtin::create_barriers(), {num_barriers}));

      Array<Stmt> barrier_init_seq;
      barrier_init_seq.reserve(num_barriers);
      for (int i = 0; i < num_barriers; i++) {
        PrimExpr arrive_thread_count =
            producer.released_barrier_.count(i) ? producer_thread_extent : consumer_thread_extent;
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
