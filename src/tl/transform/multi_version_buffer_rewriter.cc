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

enum class Role { kConsumer, kProducer, kBoth };

class WarpSpecializedRoleMarker_ : public StmtVisitor {
 public:
  WarpSpecializedRoleMarker_(Map<Var, Buffer> buffer_data_to_buffer)
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
      if (call->op.same_as(TMALoadOp()) || call->op.same_as(TMALoadIm2ColOp())) {
        role = Role::kProducer;
        has_bulk_copy_ = true;
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
    if (role == Role::kProducer) has_simt_copy_ = true;
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

  bool HasProducer() { return has_simt_copy_ || has_bulk_copy_; }

  bool HasSimtCopy() { return has_simt_copy_; }

 private:
  void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const StmtNode*, Role> map_;
  bool has_simt_copy_ = false;
  bool has_bulk_copy_ = false;
};

class MultiVersionBufferRewriter : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc& f) {
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

  Array<Buffer> GetVersionedBuffers(Array<Stmt> seq_stmt, Array<Buffer> scoped_buffers) {
    std::vector<Role> roles;
    Array<Array<BufferRegion>> reads, writes;
    auto marker = WarpSpecializedRoleMarker_(buffer_data_to_buffer_);
    for (auto stmt : seq_stmt) {
      marker(stmt);
      Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"", /*body*/ stmt);
      auto access = GetBlockAccessRegion(block, buffer_data_to_buffer_);
      reads.push_back(std::move(access[0]));
      writes.push_back(std::move(access[1]));
      roles.push_back(marker.GetRole(stmt));
    }

    std::unordered_set<const BufferNode*> consumer_used, producer_used;
    for (size_t i = 0; i < seq_stmt.size(); i++) {
      if (roles[i] == Role::kProducer) {
        for (BufferRegion br : writes[i]) producer_used.insert(br->buffer.get());
      } else {
        for (BufferRegion br : reads[i]) consumer_used.insert(br->buffer.get());
      }
    }
    Array<Buffer> versioned_buffers;
    for (Buffer buffer : scoped_buffers) {
      if (consumer_used.count(buffer.get()) && producer_used.count(buffer.get())) {
        versioned_buffers.push_back(buffer);
      }
    }
    return versioned_buffers;
  }

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

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize block_realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
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
    block_realize.CopyOnWrite()->block = block;
    return block_realize;
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

    Array<Buffer> versioned_buffers = GetVersionedBuffers(pipeline_body_seq->seq, scoped_buffers);

    for (auto buffer : versioned_buffers) {
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

using namespace tir::transform;

tvm::transform::Pass MultiVersionBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return MultiVersionBufferRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MultiVersionBuffer", {});
}

TVM_REGISTER_GLOBAL("tl.MultiVersionBuffer").set_body_typed(MultiVersionBuffer);

}  // namespace tl
}  // namespace tvm
