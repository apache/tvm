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
  AsyncDMALowerer() {}

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
      auto async_wait = op->body.as<AttrStmtNode>();
      ICHECK(async_wait && async_wait->attr_key == tir::attr::async_wait_inflight_count);

      auto call_dma_wait =
          Evaluate(Call(DataType::Int(32), builtin::dma_wait(), {op->value, async_wait->value}));

      // concatenate the call with the body and return
      return SeqStmt({call_dma_wait, async_wait->body});

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
      auto async_scope = op->body.as<AttrStmtNode>();
      ICHECK(async_scope && async_scope->attr_key == tir::attr::async_scope);

      auto for_loop = async_scope->body.as<ForNode>();
      if (!for_loop) {
        return StmtExprMutator::VisitStmt_(op);
      }

      auto bufferstorenode = for_loop->body.as<BufferStoreNode>();
      if (!bufferstorenode) {
        return StmtExprMutator::VisitStmt_(op);
      }

      ICHECK(bufferstorenode->indices.size() == 1);

      auto bufferloadnode = bufferstorenode->value.as<BufferLoadNode>();
      if (!bufferloadnode) {
        return StmtExprMutator::VisitStmt_(op);
      }

      ICHECK(bufferloadnode->indices.size() == 1);

      auto bufferstore = bufferstorenode->buffer.as<BufferNode>();
      ICHECK(bufferstore && bufferstore->strides.empty());

      auto bufferload = bufferloadnode->buffer.as<BufferNode>();
      ICHECK(bufferload && bufferload->strides.empty());

      // map loop variable to zero
      Map<Var, PrimExpr> loop_var_remap = {{for_loop->loop_var, IntImm(DataType::Int(32), 0)}};

      Array<PrimExpr> store_indices = bufferstorenode->indices;
      store_indices.MutateByApply([&](PrimExpr expr) {
        arith::Analyzer analyzer;
        return analyzer.Simplify(Substitute(std::move(expr), loop_var_remap));
      });

      Array<PrimExpr> load_indices = bufferloadnode->indices;
      load_indices.MutateByApply([&](PrimExpr expr) {
        arith::Analyzer analyzer;
        return analyzer.Simplify(Substitute(std::move(expr), loop_var_remap));
      });

      return Evaluate(Call(DataType::Int(32), builtin::dma_copy(),
                           {op->value,
                            Call(DataType::Handle(), builtin::address_of(),
                                 {BufferLoad(bufferstorenode->buffer, store_indices)}),
                            Call(DataType::Handle(), builtin::address_of(),
                                 {BufferLoad(bufferloadnode->buffer, load_indices)}),
                            for_loop->extent * bufferloadnode->dtype.bytes()}));
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

namespace transform {

Pass LowerAsyncDMA() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = AsyncDMALowerer()(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAsyncDMA", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAsyncDMA").set_body_typed(LowerAsyncDMA);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
