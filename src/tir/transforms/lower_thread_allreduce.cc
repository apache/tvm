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
 *  Lower allreduce to device implementable ir.
 * \file lower_thread_allreduce.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>

#include "../pass/ir_util.h"
#include "../../arith/compute_expr.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

class ThreadAllreduceBuilder final : public StmtExprMutator {
 public:
  explicit ThreadAllreduceBuilder(int warp_size)
      : warp_size_(warp_size) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::thread_extent) {
      thread_extents_.push_back(op);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      thread_extents_.pop_back();
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      op = ret.as<AttrStmtNode>();
      const VarNode* v = op->node.as<VarNode>();
      if (alloc_remap_.count(v)) {
        return op->body;
      } else {
        return ret;
      }
    } else if (op->attr_key == attr::reduce_scope) {
      const CommReducerNode *combiner = op->node.as<CommReducerNode>();
      CHECK(combiner);
      reduce_combiner_.push_back(combiner);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      reduce_combiner_.pop_back();
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<EvaluateNode>();
    const CallNode* call = op->value.as<CallNode>();
    if (call && call->is_intrinsic(intrinsic::tvm_thread_allreduce)) {
      return MakeAllreduce(call);
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    auto it = alloc_remap_.find(op->buffer_var.get());
    if (it != alloc_remap_.end()) {
      const AllocateNode* repl = it->second.as<AllocateNode>();
      // use volatile access to shared buffer.
      stmt = AttrStmtNode::make(
          repl->buffer_var, attr::volatile_scope, 1, op->body);
      stmt = AllocateNode::make(
          repl->buffer_var, repl->dtype,
          repl->extents, repl->condition, stmt);
      stmt = AttrStmtNode::make(
          repl->buffer_var, attr::storage_scope,
          StringImmNode::make("shared"), stmt);
      return stmt;
    } else {
      return stmt;
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    auto it = load_remap_.find(op->buffer_var.get());
    if (it != load_remap_.end()) {
      CHECK(is_zero(op->index));
      return it->second;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  // Thread entry
  struct ThreadEntry {
    runtime::ThreadScope scope;
    IterVar iv;
    int extent;
    // comparator
    bool operator<(const ThreadEntry& other) const {
      return scope.dim_index < other.scope.dim_index;
    }
  };
  // make allreduce.
  Stmt MakeAllreduce(const CallNode* call) {
    CHECK(!reduce_combiner_.empty());
    const CommReducerNode *combiner = reduce_combiner_.back();
    size_t size = combiner->result.size();

    const IntImmNode *size_of_args = call->args[0].as<IntImmNode>();
    CHECK(size_of_args) << call->args[0]->GetTypeKey();
    CHECK_EQ(size, size_of_args->value);
    Array<PrimExpr> inits = combiner->identity_element;
    std::vector<PrimExpr> values(size);
    std::vector<DataType> types(size);
    PrimExpr cond  = call->args[size+1];
    for (size_t idx = 0; idx < size; ++idx) {
      values[idx] = call->args[1+idx];
      if (!is_one(cond)) {
        values[idx] = SelectNode::make(cond, values[idx], inits[idx]);
      }
      types[idx] = values[idx].dtype();
    }
    std::vector<const VarNode*> buffers(size);
    for (size_t idx = 0; idx < size; ++idx) {
      const VarNode* buffer = call->args[2+size+idx].as<VarNode>();
      CHECK(buffer);
      buffers[idx] = buffer;
    }

    std::unordered_set<const VarNode*> reduce_set;
    for (size_t i = 2 + 2 * size; i < call->args.size(); ++i) {
      const VarNode* v = call->args[i].as<VarNode>();
      CHECK(v);
      reduce_set.insert(v);
    }
    size_t nmatch = 0;
    std::vector<ThreadEntry> vred, vpar;
    for (const AttrStmtNode* attr : thread_extents_) {
      ThreadEntry e;
      IterVar iv = Downcast<IterVar>(attr->node);
      e.scope = runtime::ThreadScope::make(iv->thread_tag);
      e.iv = iv;
      CHECK_LE(e.scope.rank, 1);
      CHECK_GE(e.scope.dim_index, 0)
          << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        const auto* ptr = attr->value.as<IntImmNode>();
        CHECK(ptr)
            << "Need constant extent for reduce set " << iv;
        e.extent = static_cast<int>(ptr->value);
        if (reduce_set.count(iv->var.get())) {
          vred.push_back(e);
          ++nmatch;
        } else {
          vpar.push_back(e);
        }
      }
    }
    CHECK_EQ(nmatch, reduce_set.size())
        << "Not all reduce index are presented in the context";
    std::sort(vred.begin(), vred.end());
    std::sort(vpar.begin(), vpar.end());
    // the size of each index.
    int reduce_extent, group_extent;
    int threadx_extent = 1;
    PrimExpr reduce_index = FlattenThread(vred, &reduce_extent);
    PrimExpr group_index = FlattenThread(vpar, &group_extent);
    if (reduce_extent == 1) {
      // special case, no reduction is needed.
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        PrimExpr pred = const_true(types[i].lanes());
        Var buffer_var = Downcast<Var>(call->args[2+size+i]);
        stores[i] = StoreNode::make(buffer_var, values[i], 0, pred);
      }
      return SeqStmt::Flatten(stores);
    }
    // Whether the threadIdx.x is involved in reduction.
    if (vred[0].scope.dim_index == 0) {
      threadx_extent = vred[0].extent;
    }
    std::vector<Stmt> seq;
    std::vector<Var> shared_bufs(size);
    // This sync is necessary because there might be incomplete read of
    // previous iteration on the same buffer.
    seq.emplace_back(SyncThread("shared"));
    for (size_t idx = 0; idx < size; ++idx) {
      shared_bufs[idx] = Var("red_buf"+std::to_string(idx), DataType::Handle());
      PrimExpr pred = const_true(types[idx].lanes());
      seq.emplace_back(StoreNode::make(
          shared_bufs[idx], values[idx],
          BufIndex(reduce_index, group_index, reduce_extent), pred));
    }
    seq.emplace_back(SyncThread("shared"));
    seq.emplace_back(MakeBufAllreduce(
        combiner, types, shared_bufs,
        reduce_index, group_index, reduce_extent, threadx_extent));
    for (size_t idx = 0; idx < size; ++idx) {
      CHECK(!load_remap_.count(buffers[idx]));
      PrimExpr pred = const_true(types[idx].lanes());
      load_remap_[buffers[idx]] = LoadNode::make(
        types[idx], shared_bufs[idx],
        BufIndex(make_zero(reduce_index.dtype()), group_index, reduce_extent), pred);
      alloc_remap_[buffers[idx]] = AllocateNode::make(
        shared_bufs[idx], types[idx],
        {PrimExpr(group_extent), PrimExpr(reduce_extent)},
        pred, EvaluateNode::make(0));
    }
    return SeqStmt::Flatten(seq);
  }
  // make allreduce.
  Stmt MakeBufAllreduce(const CommReducerNode *combiner,
                        const std::vector<DataType>& types,
                        const Array<Var>& shared_bufs,
                        PrimExpr reduce_index,
                        PrimExpr group_index,
                        int reduce_extent,
                        int threadx_extent) {
    // Get next power of two
    int reduce_align = 1;
    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    CHECK_GT(reduce_align, 1);
    std::vector<Stmt> seq;

    size_t size = shared_bufs.size();
    PrimExpr buf_index = BufIndex(reduce_index, group_index, reduce_extent);
    // make reduction
    auto freduce = [&](int offset) {
      Array<PrimExpr> a, b;
      for (size_t i = 0; i < size; ++i) {
        b.push_back(LoadNode::make(types[i], shared_bufs[i],
          BufIndex(reduce_index + offset, group_index, reduce_extent),
          const_true()));
        a.push_back(LoadNode::make(types[i], shared_bufs[i], buf_index, const_true()));
      }
      Array<PrimExpr> ret = (*combiner)(a, b);
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        stores[i] = StoreNode::make(shared_bufs[i], ret[i], buf_index, const_true());
      }
      return SeqStmt::Flatten(stores);
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElseNode::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    CHECK(threadx_extent >= 1 && warp_size_ >= 1);
    // normal synchronization
    while (reduce_align > threadx_extent ||
           reduce_align > warp_size_) {
      reduce_align =  reduce_align >> 1;
      PrimExpr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElseNode::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    // in warp synchronization.
    std::vector<Stmt> in_warp_seq;
    PrimExpr in_warp_cond = reduce_index < (reduce_align >> 1);
    while (reduce_align > 1) {
      reduce_align = reduce_align >> 1;
      in_warp_seq.emplace_back(freduce(reduce_align));
      seq.emplace_back(SyncThread("warp"));
    }
    if (in_warp_seq.size() != 0) {
      Stmt warp_body = SeqStmt::Flatten(in_warp_seq);
      seq.emplace_back(IfThenElseNode::make(in_warp_cond, warp_body));
      seq.emplace_back(SyncThread("shared"));
    }
    return SeqStmt::Flatten(seq);
  }
  // Flatten the thread index.
  // Also return a warp number,
  PrimExpr FlattenThread(const std::vector<ThreadEntry>& tvec,
                     int* out_total_extent) {
    int& total_extent = *out_total_extent;
    total_extent = 1;
    if (tvec.size() == 0) {
      return make_zero(DataType::Int(32));
    }

    PrimExpr ret;
    for (const ThreadEntry& e : tvec) {
      if (ret.defined()) {
        ret = ret + e.iv->var * total_extent;
      } else {
        CHECK_EQ(total_extent, 1);
        ret = e.iv->var;
      }
      total_extent *= e.extent;
    }
    return ret;
  }
  // The local buffer index.
  PrimExpr BufIndex(PrimExpr reduce_index, PrimExpr group_index, int reduce_extent) {
    if (!is_zero(group_index)) {
      return analyzer_.Simplify(group_index * reduce_extent + reduce_index);
    } else {
      return reduce_index;
    }
  }
  // sync thread op.
  static Stmt SyncThread(const std::string& sync) {
    return EvaluateNode::make(
        CallNode::make(DataType::Int(32), intrinsic::tvm_storage_sync,
                   {StringImmNode::make(sync)},
                   CallNode::Intrinsic));
  }
  // The warp size of the device.
  int warp_size_{1};

  // surrounding scope of thread extent.
  std::vector<const AttrStmtNode*> thread_extents_;
  std::vector<const CommReducerNode*> reduce_combiner_;
  // The load remap
  std::unordered_map<const VarNode *, PrimExpr> load_remap_;
  // Allocate remap
  std::unordered_map<const VarNode *, Stmt> alloc_remap_;
  // Internal analyzer
  arith::Analyzer analyzer_;
};

namespace transform {

Pass LowerThreadAllreduce() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    CHECK(target.defined())
        << "LowerThreadAllreduce: Require the target attribute";
    n->body = ThreadAllreduceBuilder(target.value()->thread_warp_size)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerThreadAllreduce", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerThreadAllreduce")
.set_body_typed(LowerThreadAllreduce);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
