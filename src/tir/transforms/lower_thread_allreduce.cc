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
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

class ThreadAllreduceBuilder final : public StmtExprMutator {
 public:
  explicit ThreadAllreduceBuilder(const TargetNode* target)
      : target_(target), warp_size_(target->GetAttr<Integer>("thread_warp_size", 1).value()) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
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
      const CommReducerNode* combiner = op->node.as<CommReducerNode>();
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
    if (call && call->op.same_as(builtin::tvm_thread_allreduce())) {
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
      if (warp_allocs_.count(repl)) {
        stmt = Allocate(repl->buffer_var, repl->dtype, repl->extents, repl->condition, op->body);
        stmt = AttrStmt(repl->buffer_var, attr::storage_scope, StringImm("local"), stmt);
      } else {
        // use volatile access to shared buffer.
        stmt = AttrStmt(repl->buffer_var, attr::volatile_scope, 1, op->body);
        stmt = Allocate(repl->buffer_var, repl->dtype, repl->extents, repl->condition, stmt);
        stmt = AttrStmt(repl->buffer_var, attr::storage_scope, StringImm("shared"), stmt);
      }
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
    const CommReducerNode* combiner = reduce_combiner_.back();
    size_t size = combiner->result.size();

    const IntImmNode* size_of_args = call->args[0].as<IntImmNode>();
    CHECK(size_of_args) << call->args[0]->GetTypeKey();
    CHECK_EQ(size, size_of_args->value);
    Array<PrimExpr> inits = combiner->identity_element;
    std::vector<PrimExpr> values(size);
    std::vector<DataType> types(size);
    PrimExpr cond = call->args[size + 1];
    for (size_t idx = 0; idx < size; ++idx) {
      values[idx] = call->args[1 + idx];
      if (!is_one(cond)) {
        values[idx] = Select(cond, values[idx], inits[idx]);
      }
      types[idx] = values[idx].dtype();
    }
    std::vector<const VarNode*> buffers(size);
    for (size_t idx = 0; idx < size; ++idx) {
      const VarNode* buffer = call->args[2 + size + idx].as<VarNode>();
      CHECK(buffer);
      buffers[idx] = buffer;
    }

    std::unordered_set<const VarNode*> reduce_set;
    for (size_t i = 2 + 2 * size; i < call->args.size(); ++i) {
      const VarNode* v = call->args[i].as<VarNode>();
      // The simply optimization replace a iteration variable with a constant
      // when extent of the iteration is 1. As threaded IterVar always started from 0,
      // we can just ignore this variable in this case.
      if (v) {
        reduce_set.insert(v);
      } else {
        CHECK(call->args[i].as<IntImmNode>() && call->args[i].as<IntImmNode>()->value == 0)
            << "arg" << i << "should be a VarNode or IntImmNode";
      }
    }

    size_t nmatch = 0;
    std::vector<ThreadEntry> vred, vpar;
    for (const AttrStmtNode* attr : thread_extents_) {
      ThreadEntry e;
      IterVar iv = Downcast<IterVar>(attr->node);
      e.scope = runtime::ThreadScope::Create(iv->thread_tag);
      e.iv = iv;
      CHECK_LE(e.scope.rank, 1);
      CHECK_GE(e.scope.dim_index, 0) << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        const auto* ptr = attr->value.as<IntImmNode>();
        CHECK(ptr) << "Need constant extent for reduce set " << iv;
        e.extent = static_cast<int>(ptr->value);
        // ignore variables equal to 0
        if (e.extent == 1) {
          continue;
        }

        if (reduce_set.count(iv->var.get())) {
          vred.push_back(e);
          ++nmatch;
        } else {
          vpar.push_back(e);
        }
      }
    }
    CHECK_EQ(nmatch, reduce_set.size()) << "Not all reduce index are presented in the context";
    std::sort(vred.begin(), vred.end());
    std::sort(vpar.begin(), vpar.end());
    // the size of each index.
    int reduce_extent, group_extent;
    PrimExpr reduce_index = FlattenThread(vred, &reduce_extent);
    PrimExpr group_index = FlattenThread(vpar, &group_extent);
    std::vector<Stmt> seq;
    std::vector<Var> shared_bufs(size);
    std::vector<Stmt> local_vars;
    //
    // This is an optimization. For small reduction sizes, it may be beneficial
    // for a single warp to performance the entire reduction. No trips to shared
    // memory and no cross warp synchronizations are required.
    // The following code emits the reduction as follows:
    //
    // Allocate reduction vars v[i], i = 0..size-1
    //
    // for offset from WARP_SIZE to 1 by 2
    //
    //   a    <- load(v[i])
    //   b    <- shuffle_down(load(v[i], offset))
    //   v[i] <- reduction(a, b)
    //
    // broadcast results from lane 0 to all other lanes and store
    // the final reduction result to the proper location.
    //
    if (is_warp_reduction(types)) {
      // TODO(tvm-team) sub-warp reduction support.
      CHECK_EQ(reduce_extent, warp_size_) << "not a warp reduction";
      //
      // This is the index to the reduction variable, one reduction
      // variable per warp. Local scope seems easier to reason without
      // relying on a pattern match pass to fix it later.
      PrimExpr index(0);

      for (size_t idx = 0; idx < size; ++idx) {
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), DataType::Handle());
        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(Store(shared_bufs[idx], values[idx], index, pred));

        // Uses a local variable to store the shuffled data.
        // Later on, this allocation will be properly attached to this statement.
        Var var("t" + std::to_string(idx), types[idx]);
        Stmt s = Allocate(var, var.dtype(), {PrimExpr(1)}, pred, Evaluate(0));
        local_vars.push_back(s);
      }

      // The mask for this reducer, as this reducer may sit inside
      // a divergent control flow. Here it uses a variable to cache the current
      // active channels.
      //
      Var mask_var("mask", DataType::UInt(32));
      {
        PrimExpr pred = const_true(1);
        PrimExpr mask = Call(DataType::UInt(32), builtin::tvm_warp_activemask(), {});
        seq.emplace_back(Store(mask_var, mask, index, pred));
        // Push allocation with an empty body. Later this will be fixed
        // when the entire body is ready.
        auto stmt = Allocate(mask_var, mask_var->dtype, {PrimExpr(1)}, pred, Evaluate(0));
        local_vars.push_back(stmt);
      }

      // Emit reductions within a warp.
      for (int offset = warp_size_ / 2; offset > 0; offset /= 2) {
        // Load reduction values, no synchronization needed.
        Array<PrimExpr> a, b;
        for (size_t i = 0; i < size; ++i) {
          Var var = shared_bufs[i];
          PrimExpr pred = const_true(types[i].lanes());
          PrimExpr val = Load(types[i], var, index, pred);
          a.push_back(val);

          // __shfl_*sync calls shall not appear in if_then_else expressions
          // as this is causing extra divergency. E.g.
          //
          // v1 = (v2 < v3) ? v3 : __shfl_sync(mask, v1, 0);
          //
          // behaves differently from
          //
          // int t = __shfl_sync(mask, v1, 0);
          // v1 = (v2 < v3) ? v3 : t;
          //
          // The former may cause dead lock as there is a divergent
          // branch with a warp sync call inside.
          //
          PrimExpr other = WarpShuffle(builtin::tvm_warp_shuffle_down(), mask_var, val, offset);
          const AllocateNode* repl = local_vars[i].as<AllocateNode>();
          Stmt s = Store(repl->buffer_var, other, index, pred);
          seq.push_back(s);

          PrimExpr load = Load(types[i], repl->buffer_var, index, pred);
          b.push_back(load);
        }

        // Do reductions.
        Array<PrimExpr> ret = (*combiner)(a, b);

        // Store the reduction result to itself.
        std::vector<Stmt> stores(size);
        for (size_t i = 0; i < size; ++i) {
          Var var = shared_bufs[i];
          PrimExpr pred = const_true(types[i].lanes());
          stores[i] = Store(var, ret[i], index, pred);
        }
        seq.push_back(SeqStmt::Flatten(stores));
      }

      // Broadcast the reduction result from lane 0 to all other lanes.
      // This avoids to emit predicated stores, as all threads are
      // uniformmly writting the same result.
      //
      for (size_t i = 0; i < size; ++i) {
        Var var = shared_bufs[i];
        PrimExpr pred = const_true(types[i].lanes());
        PrimExpr val = Load(types[i], var, index, pred);
        PrimExpr splat = WarpShuffle(builtin::tvm_warp_shuffle(), mask_var, val, 0);
        seq.push_back(Store(var, splat, index, pred));
      }

      // Update existing allocations.
      for (size_t i = 0; i < size; ++i) {
        CHECK(!load_remap_.count(buffers[i]));
        PrimExpr pred = const_true(types[i].lanes());
        Var var = shared_bufs[i];
        load_remap_[buffers[i]] = Load(types[i], var, index, pred);
        Array<PrimExpr> extents{PrimExpr(1)};
        auto node = Allocate(var, types[i], extents, pred, Evaluate(0));
        alloc_remap_[buffers[i]] = node;
        warp_allocs_.insert(node.get());
      }
    } else {
      int threadx_extent = 1;
      if (reduce_extent == 1) {
        // special case, no reduction is needed.
        std::vector<Stmt> stores(size);
        for (size_t i = 0; i < size; ++i) {
          PrimExpr pred = const_true(types[i].lanes());
          Var buffer_var = Downcast<Var>(call->args[2 + size + i]);
          stores[i] = Store(buffer_var, values[i], 0, pred);
        }
        return SeqStmt::Flatten(stores);
      }
      // Whether the threadIdx.x is involved in reduction.
      if (vred[0].scope.dim_index == 0) {
        threadx_extent = vred[0].extent;
      }
      // This sync is necessary because there might be incomplete read of
      // previous iteration on the same buffer.
      seq.emplace_back(SyncThread("shared"));
      for (size_t idx = 0; idx < size; ++idx) {
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), DataType::Handle());
        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(Store(shared_bufs[idx], values[idx],
                               BufIndex(reduce_index, group_index, reduce_extent), pred));
      }
      seq.emplace_back(SyncThread("shared"));
      seq.emplace_back(MakeBufAllreduce(combiner, types, shared_bufs, reduce_index, group_index,
                                        reduce_extent, threadx_extent));
      for (size_t idx = 0; idx < size; ++idx) {
        CHECK(!load_remap_.count(buffers[idx]));
        PrimExpr pred = const_true(types[idx].lanes());
        load_remap_[buffers[idx]] =
            Load(types[idx], shared_bufs[idx],
                 BufIndex(make_zero(reduce_index.dtype()), group_index, reduce_extent), pred);
        alloc_remap_[buffers[idx]] =
            Allocate(shared_bufs[idx], types[idx],
                     {PrimExpr(group_extent), PrimExpr(reduce_extent)}, pred, Evaluate(0));
      }
    }

    // Fix all local allocations as all statements are built.
    Stmt body = SeqStmt::Flatten(seq);
    for (auto var : local_vars) {
      const AllocateNode* repl = var.as<AllocateNode>();
      if (repl) {
        body = Allocate(repl->buffer_var, repl->dtype, repl->extents, repl->condition, body);
        body = AttrStmt(repl->buffer_var, attr::storage_scope, StringImm("local"), body);
      }
    }

    return body;
  }

  // make allreduce.
  Stmt MakeBufAllreduce(const CommReducerNode* combiner, const std::vector<DataType>& types,
                        const Array<Var>& shared_bufs, PrimExpr reduce_index, PrimExpr group_index,
                        int reduce_extent, int threadx_extent) {
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
        b.push_back(Load(types[i], shared_bufs[i],
                         BufIndex(reduce_index + offset, group_index, reduce_extent),
                         const_true()));
        a.push_back(Load(types[i], shared_bufs[i], buf_index, const_true()));
      }
      Array<PrimExpr> ret = (*combiner)(a, b);
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        stores[i] = Store(shared_bufs[i], ret[i], buf_index, const_true());
      }
      return SeqStmt::Flatten(stores);
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    CHECK(threadx_extent >= 1 && warp_size_ >= 1);
    // normal synchronization
    while (reduce_align > threadx_extent || reduce_align > warp_size_) {
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
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
      seq.emplace_back(IfThenElse(in_warp_cond, warp_body));
      seq.emplace_back(SyncThread("shared"));
    }
    return SeqStmt::Flatten(seq);
  }
  // Flatten the thread index.
  // Also return a warp number,
  PrimExpr FlattenThread(const std::vector<ThreadEntry>& tvec, int* out_total_extent) {
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
    return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(), {StringImm(sync)}));
  }

  // Emit warp shuffle  calls.
  PrimExpr WarpShuffle(const Op& op, Var mask_var, PrimExpr val, int delta_or_lane) {
    PrimExpr pred = const_true(1);
    PrimExpr index(0);
    PrimExpr mask = Load(DataType::UInt(32), mask_var, index, pred);
    PrimExpr width = IntImm(DataType::Int(32), warp_size_);
    Array<PrimExpr> args{mask, val, IntImm(DataType::Int(32), delta_or_lane), width, width};
    return Call(val.dtype(), op, args);
  }

  // Check if this is a reduction on threadIdx.x and its extent matches
  // the warp size.
  //
  // TODO(tvm-team) reduction with a sub-warp of 8 or 16 threads.
  // Note: The ROCm backend will only have warp reductions for now.
  // Also, the warp/wavefront size differs (64 on rocm, 32 on cuda).
  bool is_warp_reduction(const std::vector<DataType>& types) const {
    // Only cuda target supports warp reductions.
    if ((target_->kind->name != "cuda") && (target_->kind->name != "rocm")) return false;

    // rocm only supports 32 bit operands for shuffling at the moment
    if ((target_->kind->name == "rocm") &&
        (std::any_of(types.begin(), types.end(), [](DataType ty) {
          if (ty.is_vector()) return true;
          return ty.bits() != 32;
        }))) {
      return false;
    }

    // Supported types:
    // {u}int, {u}long, {u}long long, float, double, half/half2
    if (std::any_of(types.begin(), types.end(), [](DataType ty) {
          if (ty.is_float16()) return ty.lanes() > 2;
          if (ty.is_vector()) return true;
          return ty.bytes() < 4 || ty.bytes() > 8;
        })) {
      return false;
    }
    if (thread_extents_.empty()) {
      return false;
    }

    const AttrStmtNode* op = thread_extents_.back();
    DCHECK_EQ(op->attr_key, attr::thread_extent);

    IterVar iv = Downcast<IterVar>(op->node);
    ThreadEntry e;
    e.scope = runtime::ThreadScope::Create(iv->thread_tag);
    e.extent = 0;
    if (auto ptr = op->value.as<IntImmNode>()) {
      e.extent = static_cast<int>(ptr->value);
    }

    return e.extent == warp_size_ && e.scope.dim_index == 0 && e.scope.rank == 1;
  }

  // The target.
  const TargetNode* target_ = nullptr;

  // The warp size of the device.
  int warp_size_{1};

  // surrounding scope of thread extent.
  std::vector<const AttrStmtNode*> thread_extents_;
  std::vector<const CommReducerNode*> reduce_combiner_;
  // The load remap
  std::unordered_map<const VarNode*, PrimExpr> load_remap_;
  // Allocate remap
  std::unordered_map<const VarNode*, Stmt> alloc_remap_;
  // Allocate from warp reductions
  std::unordered_set<const void*> warp_allocs_;
  // Internal analyzer
  arith::Analyzer analyzer_;
};

namespace transform {

Pass LowerThreadAllreduce() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    CHECK(target.defined()) << "LowerThreadAllreduce: Require the target attribute";
    const TargetNode* target_node = target.as<TargetNode>();
    n->body = ThreadAllreduceBuilder(target_node)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerThreadAllreduce", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerThreadAllreduce").set_body_typed(LowerThreadAllreduce);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
