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
 * \file lower_async_dma.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class AsyncDMALowerer : public StmtExprMutator {
 public:
  explicit AsyncDMALowerer(bool dma_bypass_cache) : dma_bypass_cache_(dma_bypass_cache) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // Convert this, for example:
    // attr [0] "async_wait_queue_scope" = 0;
    // attr [0] "async_wait_inflight_count" = 0;
    //
    // To this:
    // @tir.dma_wait(
    //   0, /* queue id */
    //   0, /* in flight count */
    //   dtype=int32
    // )
    if (op->attr_key == tir::attr::async_wait_queue_scope) {
      // get queue ID
      auto queue_id_node = op->value.as<IntImmNode>();
      ICHECK(queue_id_node);
      int queue_id = queue_id_node->value;

      // abort if we have not seen this queue ID in `copy` transform
      if (queue_ids_.find(queue_id) == queue_ids_.end()) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the queue ID observed in the "
                      "`async_wait_queue_scope` transform has not been previously observed in the "
                      "`async_commit_queue_scope` transform";
        return StmtExprMutator::VisitStmt_(op);
      }

      auto async_wait = op->body.as<AttrStmtNode>();
      if (!async_wait || async_wait->attr_key != tir::attr::async_wait_inflight_count) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the body of the `AttrStmtNode` with key "
                      "`async_wait_queue_scope` does not contain an `AttrStmtNode` with key "
                      "`async_wait_inflight_count`";
        return StmtExprMutator::VisitStmt_(op);
      }

      auto call_dma_wait =
          Evaluate(Call(DataType::Int(32), builtin::dma_wait(), {queue_id, async_wait->value}));

      // concatenate the call with the body and return
      return SeqStmt({call_dma_wait, StmtExprMutator::VisitStmt(async_wait->body)});

      // Convert this, for example:
      // attr [0] "async_commit_queue_scope" = 0;
      // attr [0] "async_scope" = 1;
      // for (ax0: int32, 0, 128) {
      //   A_global[ax0] = A[ax0]
      // }
      //
      // To this:
      // @tir.dma_copy(
      //   0, /* queue id */
      //   @tir.address_of(A_global[0], dtype=handle),
      //   @tir.address_of(A[0], dtype=handle),
      //   128, /* size */
      //   dtype=int32
      // )
    } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
      // get queue ID
      auto queue_id_node = op->value.as<IntImmNode>();
      ICHECK(queue_id_node);
      int queue_id = queue_id_node->value;

      // walk the graph to verify this is a mem copy ...
      // 1) async_commit_queue_scope contains async_scope
      auto async_scope = op->body.as<AttrStmtNode>();
      if (!async_scope || async_scope->attr_key != tir::attr::async_scope) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the body of the `AttrStmtNode` with key "
                      "`async_commit_queue_scope` does not contain an `AttrStmtNode` with key "
                      "`async_scope`";
        return StmtExprMutator::VisitStmt_(op);
      }

      // 2) async_scope contains single for loop
      auto for_loop = async_scope->body.as<ForNode>();
      if (!for_loop) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the body of the `AttrStmtNode` with key "
                      "`async_scope` does not contain a single `ForNode`";
        return StmtExprMutator::VisitStmt_(op);
      }

      // 3) for loop contains buffer store with single index
      auto bufferstorenode = for_loop->body.as<BufferStoreNode>();
      if (!bufferstorenode || bufferstorenode->indices.size() != 1) {
        DLOG(INFO)
            << "AsyncDMALowerer exiting because the body of the `ForNode` does not contain a "
               "single `BufferStoreNode` with a single index variable";
        return StmtExprMutator::VisitStmt_(op);
      }

      // 4) buffer store value is a buffer load with single index
      auto bufferloadnode = bufferstorenode->value.as<BufferLoadNode>();
      if (!bufferloadnode || bufferloadnode->indices.size() != 1) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the value of the `BufferStoreNode` is not a "
                      "single `BufferLoadNode` with a single index variable";
        return StmtExprMutator::VisitStmt_(op);
      }

      // get store buffer; assert it exists and is contiguous given it uses a single index
      auto bufferstore = bufferstorenode->buffer.as<BufferNode>();
      ICHECK(bufferstore && bufferstore->strides.empty());

      // get load buffer; assert it exists and is contiguous given it uses a single index
      auto bufferload = bufferloadnode->buffer.as<BufferNode>();
      ICHECK(bufferload && bufferload->strides.empty());

      // we will be replacing the entire for loop including its index
      // with a DMA copy instrinsic that spans the entire index space of the for loop
      // so we will need to replace the for loop index with value zero in the buffer indices
      // thus we eliminate the index from the expression so the DMA copy receives the buffer range
      // base address
      Map<Var, PrimExpr> loop_var_remap = {{for_loop->loop_var, IntImm(DataType::Int(32), 0)}};

      // map loop variable to zero for the store index & simplify
      Array<PrimExpr> store_index = bufferstorenode->indices;
      store_index.MutateByApply([&](PrimExpr expr) {
        arith::Analyzer analyzer;
        return analyzer.Simplify(Substitute(std::move(expr), loop_var_remap));
      });

      // map loop variable to zero for the load index & simplify
      Array<PrimExpr> load_index = bufferloadnode->indices;
      load_index.MutateByApply([&](PrimExpr expr) {
        arith::Analyzer analyzer;
        return analyzer.Simplify(Substitute(std::move(expr), loop_var_remap));
      });

      // now that we are about to perform the `copy` transform
      // save queue ID for inspection in `wait` transform
      queue_ids_.insert(queue_id);

      return Evaluate(Call(DataType::Int(32), builtin::dma_copy(),
                           {queue_id,
                            Call(DataType::Handle(), builtin::address_of(),
                                 {BufferLoad(bufferstorenode->buffer, store_index)}),
                            Call(DataType::Handle(), builtin::address_of(),
                                 {BufferLoad(bufferloadnode->buffer, load_index)}),
                            for_loop->extent * bufferloadnode->dtype.bytes(), dma_bypass_cache_}));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  std::set<int> queue_ids_;
  bool dma_bypass_cache_;
};

namespace transform {

Pass LowerAsyncDMA() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    bool dma_bypass_cache = ctx->GetConfig<Bool>("tir.dma_bypass_cache", Bool(false)).value();
    fptr->body = AsyncDMALowerer(dma_bypass_cache)(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAsyncDMA", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAsyncDMA").set_body_typed(LowerAsyncDMA);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
