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
#include "ir_utils.h"
#include "update_pointer_storage_scope.h"

namespace tvm {
namespace tir {

class UpdatePointerStorageScopeAllReduce final : public UpdatePointerStorageScope {
 public:
  explicit UpdatePointerStorageScopeAllReduce(
      const std::unordered_map<const VarNode*, String>& new_storage_scopes)
      : UpdatePointerStorageScope(new_storage_scopes) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    auto remapped = Downcast<Var>(StmtExprMutator::VisitExpr(op->buffer_var));
    auto new_scope = GetPtrStorageScope(remapped);
    if (new_scope != GetPtrStorageScope(op->buffer_var)) {
      Stmt body = StmtExprMutator::VisitStmt(op->body);
      if (new_scope == "shared") {
        // use volatile access to shared buffer.
        body = AttrStmt(remapped, attr::volatile_scope, 1, body);
      }
      body = Allocate(remapped, op->dtype, op->extents, op->condition, body);
      return AttrStmt(remapped, attr::storage_scope, StringImm(new_scope), body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

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
      ICHECK(combiner);
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
        new_storage_scopes_[repl->buffer_var.get()] = "local";
      } else {
        stmt = Allocate(repl->buffer_var, repl->dtype, repl->extents, repl->condition, op->body);
        new_storage_scopes_[repl->buffer_var.get()] = "shared";
      }
      return stmt;
    } else {
      return stmt;
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    auto it = load_remap_.find(op->buffer_var.get());
    if (it != load_remap_.end()) {
      ICHECK(is_zero(op->index));
      return it->second;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  std::unordered_map<const VarNode*, String> new_storage_scopes_;

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
    ICHECK(!reduce_combiner_.empty());
    const CommReducerNode* combiner = reduce_combiner_.back();
    size_t size = combiner->result.size();

    const IntImmNode* size_of_args = call->args[0].as<IntImmNode>();
    ICHECK(size_of_args) << call->args[0]->GetTypeKey();
    ICHECK_EQ(size, size_of_args->value);
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
      ICHECK(buffer);
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
        ICHECK(call->args[i].as<IntImmNode>() && call->args[i].as<IntImmNode>()->value == 0)
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
      ICHECK_LE(e.scope.rank, 1);
      ICHECK_GE(e.scope.dim_index, 0) << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        const auto* ptr = attr->value.as<IntImmNode>();
        ICHECK(ptr) << "Need constant extent for reduce set " << iv;
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
    ICHECK_EQ(nmatch, reduce_set.size()) << "Not all reduce index are presented in the context";
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
      ICHECK_EQ(reduce_extent, warp_size_) << "not a warp reduction";
      //
      // This is the index to the reduction variable, one reduction
      // variable per warp. Local scope seems easier to reason without
      // relying on a pattern match pass to fix it later.
      PrimExpr index(0);

      for (size_t idx = 0; idx < size; ++idx) {
        Type ptr_type = PointerType(PrimType(types[idx]));
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), ptr_type);
        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(Store(shared_bufs[idx], values[idx], index, pred));

        // Uses a local variable to store the shuffled data.
        // Later on, this allocation will be properly attached to this statement.
        Var var("t" + std::to_string(idx), ptr_type);
        Stmt s = Allocate(var, types[idx], {PrimExpr(1)}, pred, Evaluate(0));
        local_vars.push_back(s);
      }

      // The mask for this reducer, as this reducer may sit inside
      // a divergent control flow. Here it uses a variable to cache the current
      // active channels.
      //
      DataType mask_dtype = DataType::UInt(32);
      Var mask_var("mask", PointerType(PrimType(mask_dtype)));
      {
        PrimExpr pred = const_true(1);
        PrimExpr mask = Call(mask_dtype, builtin::tvm_warp_activemask(), {});
        seq.emplace_back(Store(mask_var, mask, index, pred));
        // Push allocation with an empty body. Later this will be fixed
        // when the entire body is ready.
        auto stmt = Allocate(mask_var, mask_dtype, {PrimExpr(1)}, pred, Evaluate(0));
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
        ICHECK(!load_remap_.count(buffers[i]));
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
        shared_bufs[idx] = Var("red_buf" + std::to_string(idx), PointerType(PrimType(types[idx])));
        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(Store(shared_bufs[idx], values[idx],
                               BufIndex(reduce_index, group_index, reduce_extent), pred));
      }
      seq.emplace_back(SyncThread("shared"));
      seq.emplace_back(MakeBufAllreduce(combiner, types, shared_bufs, reduce_index, group_index,
                                        reduce_extent, threadx_extent));
      for (size_t idx = 0; idx < size; ++idx) {
        ICHECK(!load_remap_.count(buffers[idx]));
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
        new_storage_scopes_[repl->buffer_var.get()] = "local";
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
    ICHECK_GT(reduce_align, 1);
    std::vector<Stmt> seq;

    size_t size = shared_bufs.size();
    PrimExpr buf_index = BufIndex(reduce_index, group_index, reduce_extent);
    // make reduction
    auto fload = [&](int offset) {
      Array<PrimExpr> a, b;
      for (size_t i = 0; i < size; ++i) {
        b.push_back(Load(types[i], shared_bufs[i],
                         BufIndex(reduce_index + offset, group_index, reduce_extent),
                         const_true()));
        a.push_back(Load(types[i], shared_bufs[i], buf_index, const_true()));
      }
      Array<PrimExpr> ret = (*combiner)(a, b);
      return ret;
    };
    auto fstore = [&](const Array<PrimExpr>& ret) {
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        stores[i] = Store(shared_bufs[i], ret[i], buf_index, const_true());
      }
      return SeqStmt::Flatten(stores);
    };
    auto freduce = [&](int offset) {
      auto ret = fload(offset);
      return fstore(ret);
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    ICHECK(threadx_extent >= 1 && warp_size_ >= 1);
    // normal synchronization
    while (reduce_align > threadx_extent || reduce_align > warp_size_) {
      reduce_align = reduce_align >> 1;
      PrimExpr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElse(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    // in warp synchronization.
    if (reduce_align > 1) {
      PrimExpr in_warp_cond = reduce_index < (reduce_align >> 1);

      std::vector<Stmt> in_warp_seq;

      while (reduce_align > 1) {
        reduce_align = reduce_align >> 1;

        // freduce can read/write to the same memory location.  For
        // example, with reduce_align of 4, threadIdx 3 reads from
        // memory location 7 as threadIdx 7 is writing to it.
        // Therefore, we need to separate out the load from the store
        // with a memory barrier in-between.  This isn't necessary for
        // the earlier normal synchronization, because those are each
        // protected by an if-statement.  The if-statement is avoided
        // here to reduce thread divergence.
        auto loads = fload(reduce_align);

        Array<Var> in_warp_local_vars;
        for (auto expr : loads) {
          Var var(
              "w_" + std::to_string(reduce_align) + "_" + std::to_string(in_warp_local_vars.size()),
              expr->dtype);
          in_warp_local_vars.push_back(var);
        }

        std::vector<Stmt> in_let_statement;
        in_let_statement.emplace_back(SyncThread("warp"));
        in_let_statement.emplace_back(
            fstore({in_warp_local_vars.begin(), in_warp_local_vars.end()}));
        in_let_statement.emplace_back(SyncThread("warp"));

        Stmt body = SeqStmt::Flatten(in_let_statement);
        for (size_t i = 0; i < size; i++) {
          body = LetStmt(in_warp_local_vars[i], loads[i], body);
        }
        in_warp_seq.push_back(body);
      }

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
        ICHECK_EQ(total_extent, 1);
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
    ICHECK(target.defined()) << "LowerThreadAllreduce: Require the target attribute";
    const TargetNode* target_node = target.as<TargetNode>();
    ThreadAllreduceBuilder thread_all_reduce(target_node);
    auto reduce_body = thread_all_reduce(n->body);
    n->body =
        UpdatePointerStorageScopeAllReduce(thread_all_reduce.new_storage_scopes_)(reduce_body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerThreadAllreduce", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerThreadAllreduce").set_body_typed(LowerThreadAllreduce);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
