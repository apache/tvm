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
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <tvm/arith/bound.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include <optional>


#include "ir_utils.h"

namespace tvm {
namespace tir {


class AsyncDMALowerer : public arith::IRMutatorWithAnalyzer {
 public:

  explicit AsyncDMALowerer(bool dma_bypass_cache, arith::Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer), dma_bypass_cache_(dma_bypass_cache) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    if (!async_queue_id_.has_value()) return arith::IRMutatorWithAnalyzer::VisitStmt_(loop);

    std::optional<tvm::tir::MemCpyDetails> mem_copy = IdentifyMemCpy(GetRef<For>(loop), analyzer_);

    // if memcpy is not replacable with DMA copy
    if (!mem_copy.has_value() || mem_copy->dest->region.size() != 1 || mem_copy->source->region.size() != 1) {
      // if we subsequently find a memcpy that is replacable with DMA copy
      // then create a DMA group to contain the outer for loop as a group of DMAs
      start_dma_group_ = true;
      return arith::IRMutatorWithAnalyzer::VisitStmt_(loop);
    }

    // now that we are about to perform the `copy` transform
    // save queue ID for inspection in `wait` transform
    queue_ids_.insert(async_queue_id_.value());
    
    tvm::PrimExpr src_min = mem_copy->source->region[0]->min;
    tvm::PrimExpr dst_min = mem_copy->dest->region[0]->min;
    tvm::PrimExpr dst_extent = mem_copy->dest->region[0]->extent;

    //FIXME(nverke): Choose one of these. 
    if (analyzer_->CanProve(dst_extent <= 0)) return arith::IRMutatorWithAnalyzer::VisitStmt_(loop);
    // if (analyzer_->CanProve(dst_extent <= 0)) return Evaluate(0);
    return Evaluate(
      Call(
        DataType::Int(32),
        builtin::dma_copy(),
        {
          async_queue_id_.value(),
          Call(DataType::Handle(), builtin::address_of(), {BufferLoad(mem_copy->dest->buffer, {dst_min})}),
          Call(DataType::Handle(), builtin::address_of(), {BufferLoad(mem_copy->source->buffer, {src_min})}),
          dst_extent,
          dma_bypass_cache_
        }
      )
    );
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    arith::IRMutatorWithAnalyzer::VisitStmt_(op);
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
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
      }

      auto async_wait = op->body.as<AttrStmtNode>();
      if (!async_wait || async_wait->attr_key != tir::attr::async_wait_inflight_count) {
        DLOG(INFO) << "AsyncDMALowerer exiting because the body of the `AttrStmtNode` with key "
                      "`async_wait_queue_scope` does not contain an `AttrStmtNode` with key "
                      "`async_wait_inflight_count`";
        return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
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
      int queue_id = queue_id_node->value;

      async_queue_id_ = queue_id;
      auto result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
      async_queue_id_ = std::nullopt;
      if (start_dma_group_) {
        auto call_dma_start_group = Evaluate(Call(DataType::Int(32), builtin::dma_start_group(), {queue_id}));
        auto call_dma_end_group = Evaluate(Call(DataType::Int(32), builtin::dma_end_group(), {queue_id}));
        start_dma_group_ = false;
        auto out = SeqStmt({call_dma_start_group, result, call_dma_end_group});
        return out;
      } else {
        return result;
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

 private:
  bool start_dma_group_ = false;
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
