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
#include <tvm/arith/bound.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class AsyncDMALowerer : public arith::IRMutatorWithAnalyzer {
 public:
  explicit AsyncDMALowerer(bool dma_bypass_cache, arith::Analyzer* analyzer)
      : IRMutatorWithAnalyzer(analyzer), dma_bypass_cache_(dma_bypass_cache) {}

  // TODO(leiwang1999): split lower async DMA support for CUDA and Hexagon Backend
  Stmt VisitStmt_(const ForNode* loop) final {
    // if for loop is not within async_commit_queue_scope
    if (!async_queue_id_.has_value()) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(loop);
    }

    // if for loop is not a memcpy of a contiguous region, it might be a cuda cp.async behavior
    std::optional<tvm::tir::MemCpyDetails> mem_copy = IdentifyMemCpy(GetRef<For>(loop), analyzer_);
    if (!mem_copy.has_value() || mem_copy->dest->region.size() != 1 ||
        mem_copy->source->region.size() != 1) {
      return arith::IRMutatorWithAnalyzer::VisitStmt_(loop);
    }

    // now that we are about to perform the `copy` transform
    // save queue ID for inspection in `wait` transform
    // and, increment the number of DMA copies in the group
    queue_ids_.insert(async_queue_id_.value());
    dmas_in_group_++;

    tvm::PrimExpr src_min = mem_copy->source->region[0]->min;
    tvm::PrimExpr dst_min = mem_copy->dest->region[0]->min;
    tvm::PrimExpr dst_extent = mem_copy->dest->region[0]->extent;

    auto src = BufferLoad(mem_copy->source->buffer, {src_min});
    auto dst = BufferLoad(mem_copy->dest->buffer, {dst_min});
    return Evaluate(
        Call(DataType::Int(32), builtin::dma_copy(),
             {async_queue_id_.value(), Call(DataType::Handle(), builtin::address_of(), {dst}),
              Call(DataType::Handle(), builtin::address_of(), {src}),
              dst_extent * src->dtype.bytes(), dma_bypass_cache_}));
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // populate analyzer knowledge of loop iterators
    auto previsit = arith::IRMutatorWithAnalyzer::VisitStmt_(op);

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
        return previsit;
      }

      auto async_wait = op->body.as<AttrStmtNode>();
      if (!async_wait || async_wait->attr_key != tir::attr::async_wait_inflight_count) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the body of the `AttrStmtNode` with key "
                      "`async_wait_queue_scope` does not contain an `AttrStmtNode` with key "
                      "`async_wait_inflight_count`";
        return previsit;
      }
      auto call_dma_wait =
          Evaluate(Call(DataType::Int(32), builtin::dma_wait(), {queue_id, async_wait->value}));

      // concatenate the call with the body and return
      return SeqStmt({call_dma_wait, arith::IRMutatorWithAnalyzer::VisitStmt(async_wait->body)});

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
      async_queue_id_ = queue_id_node->value;
      auto result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
      if (dmas_in_group_ > 1) {
        auto call_dma_start_group = Evaluate(
            Call(DataType::Int(32), builtin::dma_start_group(), {async_queue_id_.value()}));
        auto call_dma_end_group =
            Evaluate(Call(DataType::Int(32), builtin::dma_end_group(), {async_queue_id_.value()}));
        result = SeqStmt({call_dma_start_group, result, call_dma_end_group});
      }

      async_queue_id_ = std::nullopt;
      dmas_in_group_ = 0;
      return result;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

 private:
  int dmas_in_group_ = 0;
  std::set<int> queue_ids_;
  std::optional<int> async_queue_id_ = std::nullopt;
  bool dma_bypass_cache_;
  Map<Var, Range> input_iters = Map<Var, Range>();
};

namespace transform {

Pass LowerAsyncDMA() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    arith::Analyzer analyzer;
    bool dma_bypass_cache =
        ctx->GetConfig<Bool>("tir.experimental_dma_bypass_cache", Bool(false)).value();
    fptr->body = AsyncDMALowerer(dma_bypass_cache, &analyzer)(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAsyncDMA", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAsyncDMA").set_body_typed(LowerAsyncDMA);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
