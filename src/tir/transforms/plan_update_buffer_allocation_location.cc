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

/*!
 * \brief Planning where buffers to be allocated and update the AST.
 * \file plan_update_buffer_allocation_location.cc
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class BufferAllocationLocator : public StmtExprMutator {
 public:
  explicit BufferAllocationLocator(const PrimFunc& func) {
    Map<Buffer, Optional<Stmt>> buffer_lca = DetectBufferAccessLCA(func);
    std::unordered_set<const BufferNode*> arg_buffers;
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      arg_buffers.emplace(buffer.get());
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    // create buffers to be allocated at each stmts
    for (const auto& kv : buffer_lca) {
      const Buffer& buffer = kv.first;
      const StmtNode* stmt = kv.second.defined() ? kv.second.value().get() : nullptr;
      if (arg_buffers.count(buffer.get())) {
        continue;
      }
      alloc_buffers_[stmt].push_back(buffer);
    }
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    auto it = alloc_buffers_.find(op);
    if (it == alloc_buffers_.end()) {
      return StmtMutator::VisitStmt_(op);
    }
    for (const Buffer& buf : it->second) {
      buffer_data_to_buffer_.Set(buf->data, buf);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    ICHECK(op != nullptr);
    for (const Buffer& buf : it->second) {
      buffer_data_to_buffer_.erase(buf->data);
    }
    Stmt body = InjectOpaqueBlock(op->body, it->second);
    ObjectPtr<ForNode> n = CopyOnWrite(op);
    n->body = std::move(body);
    return Stmt(n);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK(!op->init.defined());
    bool is_root = is_root_;
    is_root_ = false;
    Array<Buffer> alloc_buffers;
    auto it = alloc_buffers_.find(op);
    if (it != alloc_buffers_.end()) {
      alloc_buffers = it->second;
      for (const Buffer& buf : it->second) {
        buffer_data_to_buffer_.Set(buf->data, buf);
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    // Ignore buffer allocated inside the block when getting access region.
    if (it != alloc_buffers_.end()) {
      for (const Buffer& buf : it->second) {
        buffer_data_to_buffer_.erase(buf->data);
      }
    }

    ObjectPtr<BlockNode> n = CopyOnWrite(op);
    n->alloc_buffers = std::move(alloc_buffers);
    // The read/write regions of root block are always empty.
    if (!is_root) {
      // Recalculate block access region
      CollectReadWrite(GetRef<Block>(op), &n->reads, &n->writes);
    }

    return Stmt(n);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    ICHECK(false) << "Internal Error: BufferRealizeNode is not allowed in TensorIR.";
    throw;
  }

  Stmt InjectOpaqueBlock(Stmt body, const Array<Buffer>& alloc_buffers) {
    ICHECK(!alloc_buffers.empty());
    Block opaque_block(/*iter_vars=*/{},
                       /*reads=*/{},
                       /*writes=*/{},
                       /*name_hint=*/"",
                       /*body=*/std::move(body),
                       /*init=*/NullOpt,
                       /*alloc_buffers=*/alloc_buffers);
    ObjectPtr<BlockNode> n = CopyOnWrite(opaque_block.get());
    CollectReadWrite(opaque_block, &n->reads, &n->writes);
    BlockRealize realize({}, Bool(true), Block(n));
    return std::move(realize);
  }

  void CollectReadWrite(const Block& block, Array<BufferRegion>* reads,
                        Array<BufferRegion>* writes) {
    Array<Array<BufferRegion>> access = GetBlockAccessRegion(block, buffer_data_to_buffer_);
    *reads = access[0];
    *writes = access[1];
    for (const auto& opaque_access : access[2]) {
      reads->push_back(opaque_access);
      writes->push_back(opaque_access);
    }
  }

  /*! \brief The map from stmt to the buffers to be allocated under it. */
  std::unordered_map<const StmtNode*, Array<Buffer>> alloc_buffers_;
  /*! \brief The buffer already allocated during recursive visiting. */
  Map<Var, Buffer> buffer_data_to_buffer_;
  /*! \brief indicate the whether the block is root. */
  bool is_root_{true};
};

PrimFunc PlanAndUpdateBufferAllocationLocation(PrimFunc func) {
  // Only apply this pass to TIR that is not from TE schedules
  Optional<Bool> from_legacy_te_schedule = func->GetAttr("from_legacy_te_schedule", Bool(false));
  if (!from_legacy_te_schedule.value()) {
    auto fptr = func.CopyOnWrite();
    BufferAllocationLocator locator(func);
    fptr->body = locator(fptr->body);
    return func;
  } else {
    return func;
  }
}

namespace transform {

Pass PlanAndUpdateBufferAllocationLocation() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return PlanAndUpdateBufferAllocationLocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.PlanAndUpdateBufferAllocationLocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.PlanAndUpdateBufferAllocationLocation")
    .set_body_typed(PlanAndUpdateBufferAllocationLocation);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
