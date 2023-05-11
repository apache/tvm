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
      return Allocate(remapped, op->dtype, op->extents, op->condition, body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class ThreadAllreduceBuilder final : public StmtExprMutator {
 public:
  explicit ThreadAllreduceBuilder(const TargetNode* target)
      : target_(target),
        warp_size_(target->GetAttr<Integer>("thread_warp_size", 1).value().IntValue()) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      thread_extents_.push_back(op);
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      thread_extents_.pop_back();
      return ret;
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
        new_storage_scopes_[repl->buffer_var.get()] = "local";
      } else {
        new_storage_scopes_[repl->buffer_var.get()] = "shared";
      }
      return Allocate(repl->buffer_var, repl->dtype, repl->extents, repl->condition, op->body);
    } else {
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    {
      auto it = load_remap_.find(op->buffer->data.get());
      if (it != load_remap_.end()) {
        for (const auto& index : op->indices) {
          ICHECK(is_zero(index));
        }
        return it->second;
      }
    }

    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    op = load.get();

    {
      auto it = buf_remap_.find(op->buffer.get());
      if (it != buf_remap_.end()) {
        return BufferLoad(it->second, op->indices, op->span);
      }
    }

    {
      auto it = var_remap_.find(op->buffer->data.get());
      if (it != var_remap_.end()) {
        Buffer remapped_buffer(it->second, op->buffer->dtype, op->buffer->shape,
                               op->buffer->strides, op->buffer->elem_offset, op->buffer->name,
                               op->buffer->data_alignment, op->buffer->offset_factor,
                               op->buffer->buffer_type, op->buffer->axis_separators,
                               op->buffer->span);
        buf_remap_[op->buffer.get()] = remapped_buffer;
        return BufferLoad(remapped_buffer, op->indices, op->span);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    auto it = store_remap_.find(store->buffer.get());
    if (it != store_remap_.end()) {
      for (const auto& index : op->indices) {
        ICHECK(is_zero(index));
      }

      auto writer = store.CopyOnWrite();
      writer->buffer = it->second;
      return std::move(store);
    }

    {
      auto it = buf_remap_.find(store->buffer.get());
      if (it != buf_remap_.end()) {
        return BufferStore(it->second, store->value, store->indices, store->span);
      }
    }

    {
      auto it = var_remap_.find(store->buffer->data.get());
      if (it != var_remap_.end()) {
        Buffer remapped_buffer(it->second, store->buffer->dtype, store->buffer->shape,
                               store->buffer->strides, store->buffer->elem_offset,
                               store->buffer->name, store->buffer->data_alignment,
                               store->buffer->offset_factor, store->buffer->buffer_type,
                               store->buffer->axis_separators, store->buffer->span);
        buf_remap_[store->buffer.get()] = remapped_buffer;
        return BufferStore(remapped_buffer, store->value, store->indices, store->span);
      }
    }

    return std::move(store);
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
    std::vector<Buffer> buffers(size);
    for (size_t idx = 0; idx < size; ++idx) {
      PrimExpr arg = call->args[2 + size + idx];
      // Loads from boolean buffers may have cast nodes inserted by
      // earlier passes.
      if (auto cast = arg.as<CastNode>()) {
        arg = cast->value;
      }
      buffers[idx] = Downcast<BufferLoad>(arg)->buffer;
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

    // the longest contiguous reduce extent after flattening
    int contiguous_reduce_extent = 1;
    std::vector<std::tuple<int, int, bool>> block_threads;  // tuple(dim_index, extent, is_reduce)
    for (const ThreadEntry& thr : vred) {
      if (thr.scope.rank == 1) {  // threadIdx
        block_threads.emplace_back(thr.scope.dim_index, thr.extent, true);
      }
    }
    for (const ThreadEntry& thr : vpar) {
      if (thr.scope.rank == 1) {  // threadIdx
        block_threads.emplace_back(thr.scope.dim_index, thr.extent, false);
      }
    }
    // sort according to dim_index
    std::sort(block_threads.begin(), block_threads.end());
    for (auto&& thr_attr : block_threads) {
      auto [dim_index, extent, is_reduce] = thr_attr;
      (void)dim_index;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
      if (is_reduce) {
        contiguous_reduce_extent *= extent;
      } else {
        break;
      }
    }

    std::vector<Stmt> seq;
    std::vector<Var> shared_buffer_vars(size);
    std::vector<Buffer> shared_bufs(size);
    std::vector<Buffer> local_bufs;
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
    if (is_warp_reduction(types, group_extent, reduce_extent, contiguous_reduce_extent)) {
      ICHECK_LE(reduce_extent, warp_size_) << "not a warp reduction";
      //
      // This is the index to the reduction variable, one reduction
      // variable per warp. Local scope seems easier to reason without
      // relying on a pattern match pass to fix it later.
      Array<PrimExpr> zero_indices = {0};

      for (size_t idx = 0; idx < size; ++idx) {
        Array<PrimExpr> shape = {1};

        Buffer buffer = decl_buffer(shape, types[idx], "red_buf" + std::to_string(idx));
        Var buffer_var = buffer->data;

        shared_buffer_vars[idx] = buffer_var;
        shared_bufs[idx] = buffer;

        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(BufferStore(shared_bufs[idx], values[idx], zero_indices));

        // Uses a local variable to store the shuffled data.  Later
        // on, an allocation will be built for this local variable.
        local_bufs.push_back(decl_buffer(shape, types[idx], "t" + std::to_string(idx)));
      }

      // The mask for this reducer, as this reducer may sit inside
      // a divergent control flow. Here it uses a variable to cache the current
      // active channels.
      //
      DataType mask_dtype = DataType::UInt(32);
      Buffer mask_buffer = decl_buffer({1}, mask_dtype, "mask");
      {
        PrimExpr mask = Call(mask_dtype, builtin::tvm_warp_activemask(), {});
        if (group_extent > 1) {
          mask = mask & (((1 << reduce_extent) - 1) << (reduce_extent * group_index));
        }
        seq.emplace_back(BufferStore(mask_buffer, mask, zero_indices));
        // Push the buffer description.  Later this will have an
        // allocation built for it.
        local_bufs.push_back(mask_buffer);
      }

      // Emit reductions within a warp.
      int start_offset = 1;
      while (start_offset * 2 < reduce_extent) {
        start_offset *= 2;
      }
      for (int offset = start_offset; offset > 0; offset /= 2) {
        // Load reduction values, no synchronization needed.
        Array<PrimExpr> a, b;
        for (size_t i = 0; i < size; ++i) {
          Buffer shared_buf = shared_bufs[i];
          BufferLoad val(shared_buf, zero_indices);
          ICHECK_EQ(val->dtype, types[i]);
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
          PrimExpr other = WarpShuffle(builtin::tvm_warp_shuffle_down(), mask_buffer, val, offset);
          Buffer local_buf = local_bufs[i];
          Stmt s = BufferStore(local_buf, other, zero_indices);
          seq.push_back(s);

          BufferLoad load = BufferLoad(local_buf, zero_indices);
          ICHECK_EQ(load->dtype, types[i]);
          b.push_back(load);
        }

        // Do reductions.
        Array<PrimExpr> ret = (*combiner)(a, b);

        // Store the reduction result to itself.
        std::vector<Stmt> stores(size);
        for (size_t i = 0; i < size; ++i) {
          Buffer buf = shared_bufs[i];
          stores[i] = BufferStore(buf, ret[i], zero_indices);
        }

        // During the sub-warp reduction, values from inactive threads could be read,
        // which is an undefined behavior according to the cuda document.
        //
        // In practise, the return value are usually 0, which does no harm to sum reduction.
        // However, the result can be incorrect in max or prod reduction.
        // Therefore an additional range check has to be performed to ensure the correctness.
        if (offset * 2 > reduce_extent) {
          PrimExpr cond = reduce_index + offset < reduce_extent;
          seq.push_back(IfThenElse(cond, SeqStmt::Flatten(stores)));
        } else {
          seq.push_back(SeqStmt::Flatten(stores));
        }
      }

      // Broadcast the reduction result from lane 0 to all other lanes.
      // This avoids to emit predicated stores, as all threads are
      // uniformly writting the same result.
      //
      for (size_t i = 0; i < size; ++i) {
        Buffer buf = shared_bufs[i];
        PrimExpr val = BufferLoad(buf, zero_indices);
        ICHECK_EQ(val->dtype, types[i]);
        PrimExpr splat =
            WarpShuffle(builtin::tvm_warp_shuffle(), mask_buffer, val, reduce_extent * group_index);
        seq.push_back(BufferStore(buf, splat, zero_indices));
      }

      // Update existing allocations.
      for (size_t i = 0; i < size; ++i) {
        ICHECK(!load_remap_.count(buffers[i]->data.get()));
        PrimExpr pred = const_true(types[i].lanes());
        Buffer buf = shared_bufs[i];
        PrimExpr val = BufferLoad(buf, zero_indices);
        ICHECK_EQ(val->dtype, types[i]);
        load_remap_[buffers[i]->data.get()] = val;
        store_remap_[buffers[i].get()] = buf;
        Array<PrimExpr> extents{PrimExpr(1)};
        auto node = Allocate(buf->data, types[i], extents, pred, Evaluate(0));
        alloc_remap_[buffers[i]->data.get()] = node;
        var_remap_[buffers[i]->data.get()] = buf->data;
        warp_allocs_.insert(node.get());
      }
    } else {
      if (reduce_extent == 1) {
        // special case, no reduction is needed.
        std::vector<Stmt> stores;
        for (size_t i = 0; i < size; ++i) {
          stores.push_back(BufferStore(buffers[i], values[i], {0}));
        }
        return SeqStmt::Flatten(stores);
      }
      // This sync is necessary because there might be incomplete read of
      // previous iteration on the same buffer.
      seq.emplace_back(SyncThread("shared"));
      for (size_t idx = 0; idx < size; ++idx) {
        Buffer buffer = decl_buffer({1}, types[idx], "red_buf" + std::to_string(idx));

        shared_bufs[idx] = buffer;
        shared_buffer_vars[idx] = buffer->data;

        PrimExpr pred = const_true(types[idx].lanes());
        seq.emplace_back(BufferStore(shared_bufs[idx], values[idx],
                                     {BufIndex(reduce_index, group_index, reduce_extent)}));
      }
      seq.emplace_back(SyncThread("shared"));
      seq.emplace_back(MakeBufAllreduce(combiner, types, shared_bufs, reduce_index, group_index,
                                        reduce_extent, group_extent, contiguous_reduce_extent));
      for (size_t idx = 0; idx < size; ++idx) {
        ICHECK(!load_remap_.count(buffers[idx]->data.get()));
        PrimExpr pred = const_true(types[idx].lanes());
        BufferLoad load(shared_bufs[idx],
                        {BufIndex(make_zero(reduce_index.dtype()), group_index, reduce_extent)});
        ICHECK_EQ(load->dtype, types[idx]);
        load_remap_[buffers[idx]->data.get()] = load;
        alloc_remap_[buffers[idx]->data.get()] =
            Allocate(shared_bufs[idx]->data, types[idx],
                     {PrimExpr(group_extent), PrimExpr(reduce_extent)}, pred, Evaluate(0));
        var_remap_[buffers[idx]->data.get()] = shared_bufs[idx]->data;
        store_remap_[buffers[idx].get()] = shared_bufs[idx];
      }
    }

    // Fix all local allocations as all statements are built.
    Stmt body = SeqStmt::Flatten(seq);
    for (Buffer buf : local_bufs) {
      body = Allocate(buf->data, buf->dtype, buf->shape, const_true(buf->dtype.lanes()), body);
      new_storage_scopes_[buf->data.get()] = "local";
    }

    return body;
  }

  // make allreduce.
  Stmt MakeBufAllreduce(const CommReducerNode* combiner, const std::vector<DataType>& types,
                        const Array<Buffer>& shared_bufs, PrimExpr reduce_index,
                        PrimExpr group_index, int reduce_extent, int group_extent,
                        int contiguous_reduce_extent) {
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
        BufferLoad b_load(shared_bufs[i],
                          {BufIndex(reduce_index + offset, group_index, reduce_extent)});
        ICHECK_EQ(b_load->dtype, types[i]);
        b.push_back(b_load);

        BufferLoad a_load(shared_bufs[i], {buf_index});
        ICHECK_EQ(a_load->dtype, types[i]);
        a.push_back(a_load);
      }
      Array<PrimExpr> ret = (*combiner)(a, b);
      return ret;
    };
    auto fstore = [&](const Array<PrimExpr>& ret) {
      std::vector<Stmt> stores(size);
      for (size_t i = 0; i < size; ++i) {
        stores[i] = BufferStore(shared_bufs[i], ret[i], {buf_index});
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

    // normal synchronization
    bool warp_align = group_extent == 1 || contiguous_reduce_extent % warp_size_ == 0;
    while (reduce_align > contiguous_reduce_extent || reduce_align > warp_size_ || !warp_align) {
      if (reduce_align == 1) {
        break;
      }
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
  PrimExpr WarpShuffle(const Op& op, Buffer mask_buffer, PrimExpr val, PrimExpr delta_or_lane) {
    Array<PrimExpr> indices = {0};
    PrimExpr mask = BufferLoad(mask_buffer, indices);
    PrimExpr width = IntImm(DataType::Int(32), warp_size_);
    Array<PrimExpr> args{mask, val, delta_or_lane, width, width};
    return Call(val.dtype(), op, args);
  }

  // Check if we can use warp level reduction.
  //
  // Note: The ROCm backend will only have warp reductions for now.
  // Also, the warp/wavefront size differs (64 on rocm, 32 on cuda).
  bool is_warp_reduction(const std::vector<DataType>& types, int group_extent, int reduce_extent,
                         int contiguous_reduce_extent) const {
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

    // reduce region must be contiguous.
    if (contiguous_reduce_extent != reduce_extent) {
      return false;
    }

    // whether reduce_extent and group_extent are vaild for warp reduction.
    if (target_->kind->name == "rocm") {
      return reduce_extent == warp_size_;
    } else {  // target_->kind->name == "cuda"
      if (reduce_extent == 1) {
        return false;  // no need to warp reduce
      } else {
        if (warp_size_ % reduce_extent == 0) {
          return true;  // warp size is multiple of reduce extent
        } else {
          return group_extent == 1 && reduce_extent <= warp_size_;
        }
      }
    }
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
  // The store remap
  std::unordered_map<const BufferNode*, Buffer> store_remap_;
  // Allocate remap
  std::unordered_map<const VarNode*, Stmt> alloc_remap_;
  // BufferVar remap
  std::unordered_map<const VarNode*, Var> var_remap_;
  // Buffer remap
  std::unordered_map<const BufferNode*, Buffer> buf_remap_;
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
