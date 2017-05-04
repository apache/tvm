/*!
 *  Copyright (c) 2017 by Contributors
 *  Lower allreduce to device implementable ir.
 * \file lower_thread_allreduce.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

class ThreadAllreduceBuilder : public IRMutator {
 public:
  explicit ThreadAllreduceBuilder(int warp_size)
      : warp_size_(warp_size) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent) {
      thread_extents_.push_back(op);
      Stmt ret = IRMutator::Mutate_(op, s);
      thread_extents_.pop_back();
      return ret;
    } else if (op->attr_key == attr::storage_scope) {
      Stmt ret = IRMutator::Mutate_(op, s);
      op = ret.as<AttrStmt>();
      const Variable* v = op->node.as<Variable>();
      if (alloc_remap_.count(v)) {
        return op->body;
      } else {
        return ret;
      }
    } else if (op->attr_key == attr::reduce_scope) {
      const CommReducerNode *combiner = op->node.as<CommReducerNode>();
      CHECK(combiner);
      reduce_combiner_.push_back(combiner);
      Stmt ret = IRMutator::Mutate_(op, s);
      reduce_combiner_.pop_back();
      return ret;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    const Call* call = op->value.as<Call>();
    if (call && call->is_intrinsic(intrinsic::tvm_thread_allreduce)) {
      return MakeAllreduce(op, call);
    } else {
      return stmt;
    }
  }
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    auto it = alloc_remap_.find(op->buffer_var.get());
    if (it != alloc_remap_.end()) {
      const Allocate* repl = it->second.as<Allocate>();
      // use volatile access to shared buffer.
      stmt = AttrStmt::make(
          repl->buffer_var, attr::volatile_scope, 1, op->body);
      stmt = Allocate::make(
          repl->buffer_var, repl->type,
          repl->extents, repl->condition, stmt);
      stmt = AttrStmt::make(
          repl->buffer_var, attr::storage_scope,
          StringImm::make("shared"), stmt);
      return stmt;
    } else {
      return stmt;
    }
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    auto it = load_remap_.find(op->buffer_var.get());
    if (it != load_remap_.end()) {
      CHECK(is_zero(op->index));
      return it->second;
    } else {
      return IRMutator::Mutate_(op, e);
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
  Stmt MakeAllreduce(const Store* op, const Call* call) {
    CHECK(!reduce_combiner_.empty());
    const CommReducerNode *combiner = reduce_combiner_.back();
    Expr init  = combiner->identity_element;
    Expr value = call->args[0];
    Expr cond  = call->args[1];
    if (!is_one(cond)) {
      value = Select::make(cond, value, init);
    }

    std::unordered_set<const Variable*> reduce_set;
    for (size_t i = 2; i < call->args.size(); ++i) {
      const Variable* v = call->args[i].as<Variable>();
      CHECK(v);
      reduce_set.insert(v);
    }
    size_t nmatch = 0;
    std::vector<ThreadEntry> vred, vpar;
    for (const AttrStmt* attr : thread_extents_) {
      ThreadEntry e;
      IterVar iv(attr->node.node_);
      e.scope = runtime::ThreadScope::make(iv->thread_tag);
      e.iv = iv;
      CHECK_LE(e.scope.rank, 1);
      CHECK_GE(e.scope.dim_index, 0)
          << "vthread do not work with cross thread reduction";
      if (e.scope.rank == 1) {
        CHECK(arith::GetConstInt(attr->value, &(e.extent)))
            << "Need constant extent for reduce set " << iv;
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
    Expr reduce_index = FlattenThread(vred, &reduce_extent);
    Expr group_index = FlattenThread(vpar, &group_extent);
    Expr pred = const_true(value.type().lanes());
    if (reduce_extent == 1) {
      // special case, no reduction is needed.
      return Store::make(op->buffer_var, value, 0, pred);
    }
    // Whether the threadIdx.x is involved in reduction.
    if (vred[0].scope.dim_index == 0) {
      threadx_extent = vred[0].extent;
    }
    Var shared_buf("red_buf", Handle());
    std::vector<Stmt> seq;
    seq.emplace_back(Store::make(
        shared_buf, value,
        BufIndex(reduce_index, group_index, reduce_extent), pred));
    seq.emplace_back(SyncThread("shared"));
    seq.emplace_back(MakeBufAllreduce(
        combiner, value.type(), shared_buf,
        reduce_index, group_index, reduce_extent, threadx_extent));
    CHECK(!load_remap_.count(op->buffer_var.get()));
    load_remap_[op->buffer_var.get()] =
        Load::make(
            value.type(), shared_buf,
            BufIndex(make_zero(reduce_index.type()), group_index, reduce_extent),
            pred);
    alloc_remap_[op->buffer_var.get()] =
        Allocate::make(shared_buf, value.type(),
                       {Expr(group_extent), Expr(reduce_extent)},
                       pred, Evaluate::make(0));
    return MergeSeq(seq);
  }
  // make allreduce.
  Stmt MakeBufAllreduce(const CommReducerNode *combiner,
                        Type type,
                        Var shared_buf,
                        Expr reduce_index,
                        Expr group_index,
                        int reduce_extent,
                        int threadx_extent) {
    // Get next power of two
    int reduce_align = 1;
    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    CHECK_GT(reduce_align, 1);
    std::vector<Stmt> seq;

    Expr buf_index = BufIndex(reduce_index, group_index, reduce_extent);
    // make reduction
    auto freduce = [&](int offset) {
      Expr b = Load::make(
          type, shared_buf,
          BufIndex(reduce_index + offset, group_index, reduce_extent), const_true());
      Expr a = Load::make(type, shared_buf, buf_index, const_true());
      return Store::make(shared_buf, (*combiner)(a, b), buf_index, const_true());
    };
    // Step one, check for
    if (reduce_align > reduce_extent) {
      // reduction with the boundary condition
      reduce_align = reduce_align >> 1;
      Expr cond = reduce_index < (reduce_extent - reduce_align);
      seq.emplace_back(IfThenElse::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    CHECK(threadx_extent >= 1 && warp_size_ >= 1);
    // normal synchronization
    while (reduce_align > threadx_extent ||
           reduce_align > warp_size_) {
      reduce_align =  reduce_align >> 1;
      Expr cond = reduce_index < reduce_align;
      seq.emplace_back(IfThenElse::make(cond, freduce(reduce_align)));
      seq.emplace_back(SyncThread("shared"));
    }
    // in warp synchronization.
    std::vector<Stmt> in_warp_seq;
    Expr in_warp_cond = reduce_index < (reduce_align >> 1);
    while (reduce_align > 1) {
      reduce_align = reduce_align >> 1;
      in_warp_seq.emplace_back(freduce(reduce_align));
      seq.emplace_back(SyncThread("warp"));
    }
    if (in_warp_seq.size() != 0) {
      Stmt warp_body = MergeSeq(in_warp_seq);
      seq.emplace_back(IfThenElse::make(in_warp_cond, warp_body));
    }
    return MergeSeq(seq);
  }
  // Flatten the thread index.
  // Also return a warp number,
  Expr FlattenThread(const std::vector<ThreadEntry>& tvec,
                     int* out_total_extent) {
    int& total_extent = *out_total_extent;
    total_extent = 1;
    if (tvec.size() == 0) {
      return make_zero(Int(32));
    }

    Expr ret;
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
  // sync thread op.
  static Stmt SyncThread(const std::string& sync) {
    return Evaluate::make(
        Call::make(Int(32), intrinsic::tvm_storage_sync,
                   {StringImm::make(sync)},
                   Call::Intrinsic));
  }
  // The local buffer index.
  static Expr BufIndex(Expr reduce_index, Expr group_index, int reduce_extent) {
    if (!is_zero(group_index)) {
      return ir::Simplify(group_index * reduce_extent + reduce_index);
    } else {
      return reduce_index;
    }
  }
  // The warp size of the device.
  int warp_size_{1};

  // surrounding scope of thread extent.
  std::vector<const AttrStmt*> thread_extents_;
  std::vector<const CommReducerNode*> reduce_combiner_;
  // The load remap
  std::unordered_map<const Variable *, Expr> load_remap_;
  // Allocate remap
  std::unordered_map<const Variable *, Stmt> alloc_remap_;
};

LoweredFunc
LowerThreadAllreduce(LoweredFunc f, int warp_size) {
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = ThreadAllreduceBuilder(warp_size).Mutate(n->body);
  return LoweredFunc(n);
}
}  // namespace ir
}  // namespace tvm
