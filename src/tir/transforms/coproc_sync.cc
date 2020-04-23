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
 * \file coproc_sync.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include "ir_util.h"
#include "storage_access.h"

namespace tvm {
namespace tir {

// Visitor to find touched set by co-processor scope.
class CoProcTouchedBuffer : public StmtExprVisitor {
 public:
  void VisitExpr_(const LoadNode* op) final {
    if (in_scope_) {
      touched_[op->buffer_var.get()].coproc = true;
    } else {
      touched_[op->buffer_var.get()].normal = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const StoreNode* op) final {
    if (in_scope_) {
      touched_[op->buffer_var.get()].coproc = true;
    } else {
      touched_[op->buffer_var.get()].normal = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
      const VarNode* buffer = op->args[1].as<VarNode>();
      if (in_scope_) {
        touched_[buffer].coproc = true;
      } else {
        touched_[buffer].normal = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::coproc_scope && !in_scope_) {
      in_scope_ = true;
      IterVar iv = Downcast<IterVar>(op->node);
      coproc_.insert(iv);
      StmtExprVisitor::VisitStmt_(op);
      in_scope_ = false;
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  // Touch Entry
  struct TouchEntry {
    bool normal{false};
    bool coproc{false};
  };
  std::unordered_map<const VarNode*, TouchEntry> touched_;
  std::unordered_set<IterVar> coproc_;

 private:
  bool in_scope_{false};
};

// Synchronization planning with co-processor.
class CoProcSyncPlanner : public StorageAccessVisitor {
 public:
  explicit CoProcSyncPlanner(
      const std::unordered_set<const VarNode*>& touched,
      const std::string& coproc_name)
      : touched_(touched), coproc_name_(coproc_name) {
  }

  void Plan(const Stmt& stmt) {
    this->VisitStmt(stmt);
    PlanSync(scope_.back(), nullptr, true);
    if (sync_.size() == 0) {
      sync_[stmt.get()] = GetSync(coproc_name_ + ".coproc_sync");
    }
  }

  // Write synchronization to be inserted before or after stmt.
  std::unordered_map<const Object*, std::vector<Stmt> > sync_;

 protected:
  bool Enabled(const VarNode* buf,
               const StorageScope& scope) const final {
    return touched_.count(buf);
  }

  // Plan the sync
  std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const ForNode* loop) final {
    return PlanSync(seq, loop, false);
  }

 private:
  // Plan write synchronization if write is not coherent
  std::vector<AccessEntry> PlanSync(
      std::vector<StmtEntry> seq, const ForNode* loop,
      bool force_sync_at_end) {
    // detect write barriers
    // access by the co-processor.
    std::vector<AccessEntry> co_access;
    bool contain_sync = false;

    auto find_conflict = [&](const AccessEntry& acc) {
      for (const AccessEntry& x : co_access) {
        if (x.buffer.same_as(acc.buffer) &&
            ((acc.type == kRead && x.type == kWrite) ||
             acc.type == kWrite)) {
          return true;
        }
      }
      return false;
    };
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      bool sync_write = false;
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() == 0 && find_conflict(acc)) {
          sync_write = true; break;
        }
        if (acc.type == kSync) {
          co_access.clear();
          contain_sync = true;
        }
      }
      if (sync_write) {
        CHECK_NE(i, 0U);
        sync_[seq[i - 1].stmt] = GetSync(co_access);
        co_access.clear();
        contain_sync = true;
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() != 0) {
          co_access.push_back(acc);
        }
      }
    }
    bool sync_at_end = force_sync_at_end;
    if (loop != nullptr && !sync_at_end) {
      // loop carray dependency
      for (size_t i = 0; i < seq.size(); ++i) {
        const StmtEntry& s = seq[i];
        for (const AccessEntry& acc : s.access) {
          if (acc.threads.size() == 0 && find_conflict(acc)) {
            sync_at_end = true; break;
          }
        }
        if (sync_.count(s.stmt) || sync_at_end) break;
      }
    }
    if (sync_at_end && co_access.size() != 0) {
      CHECK_NE(seq.size(), 0);
      contain_sync = true;
      sync_[seq.back().stmt] = GetSync(co_access);
      co_access.clear();
    }
    if (contain_sync) {
      AccessEntry e;
      e.type = kSync;
      co_access.insert(co_access.begin(), e);
    }
    return co_access;
  }
  // Add write Synchronization
  std::vector<Stmt> GetSync(const std::vector<AccessEntry>& co_access) {
    // Does not consider memory coherence, need runtime.
    CHECK_NE(co_access.size(), 0U);
    CHECK_EQ(co_access[0].threads.size(), 1U);
    return GetSync(coproc_name_ + ".coproc_sync");
  }

  std::vector<Stmt> GetSync(std::string sync_name) {
    return {EvaluateNode::make(CallNode::make(
        DataType::Int(32),
        sync_name,
        {}, CallNode::Intrinsic))};
  }

  const std::unordered_set<const VarNode*>& touched_;
  std::string coproc_name_;
};

// Detect memory barriers when coproc read/write memory
class CoProcBarrierDetector : public StorageAccessVisitor {
 public:
  explicit CoProcBarrierDetector(
      const std::unordered_set<const VarNode*>& touched,
      const std::string& coproc_name)
      : touched_(touched) {
    read_barrier_name_ = coproc_name + ".coproc_read_barrier";
    write_barrier_name_ = coproc_name + ".coproc_write_barrier";
  }

  void PlanReadBarrier(const Stmt& stmt) {
    read_barrier_ = true;
    this->VisitStmt(stmt);
    PlanReadBarrier(scope_.back(), nullptr);
  }
  void PlanWriteBarrier(const Stmt& stmt) {
    read_barrier_ = false;
    this->VisitStmt(stmt);
    PlanWriteBarrier(scope_.back(), nullptr);
  }

  std::unordered_map<const Object*, std::vector<Stmt> > barrier_before_;
  std::unordered_map<const Object*, std::vector<Stmt> > barrier_after_;

 protected:
  bool Enabled(const VarNode* buf,
               const StorageScope& scope) const final {
    return touched_.count(buf);
  }

  // Plan the sync
  std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const ForNode* loop) final {
    if (read_barrier_) {
      return PlanReadBarrier(seq, loop);
    } else {
      return PlanWriteBarrier(seq, loop);
    }
  }

 private:
  // Plan write barrier at Read after write point.
  std::vector<AccessEntry> PlanWriteBarrier(
      std::vector<StmtEntry> seq, const ForNode* loop) {
    std::vector<AccessEntry> read_seq;
    std::unordered_map<const VarNode*, std::vector<AccessEntry> > write_set;

    auto fupdate = [&](size_t i, const AccessEntry& acc) {
      auto it  = write_set.find(acc.buffer.get());
      if (it != write_set.end()) {
        CHECK_NE(i, 0U);
        barrier_after_[seq[i - 1].stmt].push_back(
            MakeBarrier(write_barrier_name_, it->second));
        write_set.erase(it);
      }
    };
    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() == 0 && acc.type == kRead) {
          fupdate(i, acc);
          read_seq.push_back(acc);
        }
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() != 0 && acc.type == kWrite) {
          write_set[acc.buffer.get()].push_back(acc);
        }
      }
    }
    // loop carry
    if (loop != nullptr) {
      for (const AccessEntry& acc : read_seq) {
        fupdate(seq.size(), acc);
      }
    }
    for (const auto &kv : write_set) {
      read_seq.insert(read_seq.end(), kv.second.begin(), kv.second.end());
    }
    return read_seq;
  }

  std::vector<AccessEntry> PlanReadBarrier(
      std::vector<StmtEntry> seq, const ForNode* loop) {
    std::vector<AccessEntry> write_seq;
    std::unordered_map<const VarNode*, std::vector<AccessEntry> > read_set;

    auto fupdate = [&](size_t i, const AccessEntry& acc) {
      auto it  = read_set.find(acc.buffer.get());
      if (it != read_set.end()) {
        CHECK_NE(i, seq.size());
        barrier_before_[seq[i].stmt].push_back(
            MakeBarrier(read_barrier_name_, it->second));
        read_set.erase(it);
      }
    };

    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() == 0 && acc.type == kWrite) {
          fupdate(i, acc);
          write_seq.push_back(acc);
        }
      }
      for (const AccessEntry& acc : s.access) {
        if (acc.threads.size() != 0 && acc.type == kRead) {
          read_set[acc.buffer.get()].push_back(acc);
        }
      }
    }
    // loop carry
    if (loop != nullptr) {
      for (const AccessEntry& acc : write_seq) {
        fupdate(0, acc);
      }
    }
    for (const auto &kv : read_set) {
      write_seq.insert(write_seq.end(), kv.second.begin(), kv.second.end());
    }
    return write_seq;
  }

  Stmt MakeBarrier(const std::string& func, const std::vector<AccessEntry>& wvec) {
    // insert write point
    Array<arith::IntSet> wset;
    for (const AccessEntry& acc : wvec) {
      CHECK(acc.dtype == wvec[0].dtype);
      wset.push_back(acc.touched);
    }
    Range none;
    Range r = arith::Union(wset).cover_range(none);
    CHECK(r.defined())
        << "Cannot deduce write range of " << wvec[0].buffer;
    PrimExpr min = r->min;
    PrimExpr extent = r->extent;
    return EvaluateNode::make(CallNode::make(
        DataType::Int(32), func,
        {wvec[0].buffer, wvec[0].dtype.bits(), r->min, r->extent}, CallNode::Intrinsic));
  }
  // Write barrier name
  bool read_barrier_{false};
  std::string read_barrier_name_;
  std::string write_barrier_name_;
  const std::unordered_set<const VarNode*>& touched_;
};


class CoProcInstDepDetector : public StmtVisitor {
 public:
  explicit CoProcInstDepDetector(
      const IterVar& coproc_axis,
      const std::string& coproc_name)
      : coproc_axis_(coproc_axis) {
    sync_push_name_ = coproc_name + ".coproc_dep_push";
    sync_pop_name_ = coproc_name + ".coproc_dep_pop";
  }

  void Plan(const Stmt& stmt) {
    this->VisitStmt(stmt);
    if (last_state_.node != nullptr) {
      MatchFixEnterPop(first_state_);
      MatchFixExitPush(last_state_);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::coproc_scope &&
        op->node.same_as(coproc_axis_)) {
      const IntImmNode* ctx_id = op->value.as<IntImmNode>();
      CHECK(ctx_id != nullptr);
      curr_state_.clear();
      curr_state_.node = op->body.get();
      curr_state_.enter_ctx.insert(ctx_id->value);
      curr_state_.exit_ctx.insert(ctx_id->value);
      UpdateState();
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    SyncState temp_first, temp_last;
    std::swap(first_state_, temp_first);
    std::swap(last_state_, temp_last);
    this->VisitStmt(op->body);
    curr_state_.clear();
    if (last_state_.node != nullptr) {
      curr_state_.node = op;
      CHECK(first_state_.node != nullptr);
      // loop carry dependency
      InjectSync(last_state_, first_state_,
                 &(curr_state_.exit_push),
                 &(curr_state_.enter_pop));
      curr_state_.enter_ctx = first_state_.enter_ctx;
      curr_state_.exit_ctx = last_state_.exit_ctx;
    }
    std::swap(first_state_, temp_first);
    std::swap(last_state_, temp_last);
    if (curr_state_.node != nullptr) {
      UpdateState();
    }
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    SyncState temp_first, temp_last, curr_state;
    std::swap(first_state_, temp_first);
    std::swap(last_state_, temp_last);
    {
      // then stmt
      this->VisitStmt(op->then_case);
      if (last_state_.node != nullptr) {
        curr_state.node = op;
        MatchFixEnterPop(first_state_);
        MatchFixExitPush(last_state_);
        curr_state.enter_ctx.insert(
            first_state_.enter_ctx.begin(),
            first_state_.enter_ctx.end());
        curr_state.exit_ctx.insert(
            last_state_.exit_ctx.begin(),
            last_state_.exit_ctx.end());
      }
      first_state_.clear();
      last_state_.clear();
    }
    if (op->else_case.defined()) {
      this->VisitStmt(op->else_case);
      if (last_state_.node != nullptr) {
        curr_state.node = op;
        MatchFixEnterPop(first_state_);
        MatchFixExitPush(last_state_);
        curr_state.enter_ctx.insert(
            first_state_.enter_ctx.begin(),
            first_state_.enter_ctx.end());
        curr_state.exit_ctx.insert(
            last_state_.exit_ctx.begin(),
            last_state_.exit_ctx.end());
      }
    }
    // update in the trace.
    std::swap(first_state_, temp_first);
    std::swap(last_state_, temp_last);
    std::swap(curr_state_, curr_state);
    if (curr_state_.node != nullptr) {
      UpdateState();
    }
  }

  // insert before is stored in reverse order
  // the first element is closest to the node.
  std::unordered_map<const Object*, std::vector<Stmt> > insert_before_;
  std::unordered_map<const Object*, std::vector<Stmt> > insert_after_;

 private:
  // state in the sync entry
  struct SyncState {
    // The statement of the state.
    const Object* node{nullptr};
    // Set of all possible contexts in the entering moment.
    std::unordered_set<int> enter_ctx;
    // Set of all possible contexts in the exit moment.
    std::unordered_set<int> exit_ctx;
    // existing pop performed at enter
    std::vector<std::pair<int, int> > enter_pop;
    // existing push peformed at exit
    std::vector<std::pair<int, int> > exit_push;
    // clear the state
    void clear() {
      node = nullptr;
      enter_ctx.clear();
      exit_ctx.clear();
      enter_pop.clear();
      exit_push.clear();
    }
  };
  // inject proper sync into the pair
  // record the push/pop sequence that could be possibly un-matched.
  // return the push/pop message at enter/exit of the Block
  // after considering the existing unmatcheded events and added events
  void InjectSync(const SyncState& prev,
                  const SyncState& next,
                  std::vector<std::pair<int, int> >* prev_exit_push,
                  std::vector<std::pair<int, int> >* next_enter_pop) {
    prev_exit_push->clear();
    next_enter_pop->clear();
    // quick path
    if (prev.exit_push.size() == 0 && next.enter_pop.size() == 0 &&
        prev.exit_ctx.size() == 1 && next.enter_ctx.size() == 1) {
      int from = *prev.exit_ctx.begin();
      int to = *next.enter_ctx.begin();
      if (from != to) {
        insert_after_[prev.node].emplace_back(MakePush(from, to));
        insert_before_[next.node].emplace_back(MakePop(from, to));
        prev_exit_push->emplace_back(std::make_pair(from, to));
        next_enter_pop->emplace_back(std::make_pair(from, to));
      }
      return;
    }
    // complicate path.
    std::vector<std::pair<int, int> > vpush = prev.exit_push;
    std::vector<std::pair<int, int> > vpop = next.enter_pop;
    std::vector<std::pair<int, int> > pending;
    for (int from : prev.exit_ctx) {
      for (int to : next.enter_ctx) {
        if (from != to) {
          pending.emplace_back(std::make_pair(from, to));
        }
      }
    }
    // policy 1
    std::vector<Stmt> prev_after, next_before;
    for (const std::pair<int, int>& p : pending) {
      if (std::find(prev.exit_push.begin(),
                    prev.exit_push.end(), p) ==
          prev.exit_push.end()) {
        vpush.push_back(p);
        prev_after.emplace_back(MakePush(p.first, p.second));
      }
      if (std::find(next.enter_pop.begin(),
                    next.enter_pop.end(), p) ==
          next.enter_pop.end()) {
        vpop.push_back(p);
        next_before.emplace_back(MakePop(p.first, p.second));
      }
    }
    // fix pending
    for (const std::pair<int, int>& p : vpush) {
      if (std::find(vpop.begin(), vpop.end(), p) == vpop.end()) {
        prev_after.emplace_back(MakePop(p.first, p.second));
      } else {
        prev_exit_push->push_back(p);
      }
    }
    for (const std::pair<int, int>& p : vpop) {
      if (std::find(vpush.begin(), vpush.end(), p) == vpush.end()) {
        next_before.emplace_back(MakePush(p.first, p.second));
      } else {
        next_enter_pop->push_back(p);
      }
    }
    if (prev_after.size() != 0) {
      auto &v1 = insert_after_[prev.node];
      v1.insert(v1.end(), prev_after.begin(), prev_after.end());
    }
    if (next_before.size() != 0) {
      auto &v2 = insert_before_[next.node];
      v2.insert(v2.end(), next_before.begin(), next_before.end());
    }
  }

  void MatchFixEnterPop(const SyncState& state) {
    if (state.enter_pop.size() == 0) return;
    auto &vec = insert_before_[state.node];
    for (const std::pair<int, int>& p : state.enter_pop) {
      vec.push_back(MakePush(p.first, p.second));
    }
  }

  void MatchFixExitPush(const SyncState& state) {
    if (state.exit_push.size() == 0) return;
    auto &vec = insert_after_[state.node];
    for (const std::pair<int, int>& p : state.exit_push) {
      vec.push_back(MakePop(p.first, p.second));
    }
  }

  void UpdateState() {
    if (last_state_.node != nullptr) {
      std::vector<std::pair<int, int> > t1, t2;
      InjectSync(last_state_, curr_state_, &t1, &t2);
      std::swap(last_state_, curr_state_);
    } else {
      CHECK(first_state_.node == nullptr);
      first_state_ = curr_state_;
      last_state_ = curr_state_;
    }
  }

  Stmt MakePush(int from, int to) {
    return EvaluateNode::make(CallNode::make(
        DataType::Int(32), sync_push_name_,
        {make_const(DataType::Int(32), from), make_const(DataType::Int(32), to)},
        CallNode::Intrinsic));
  }
  Stmt MakePop(int from, int to) {
    return EvaluateNode::make(CallNode::make(
        DataType::Int(32), sync_pop_name_,
        {make_const(DataType::Int(32), from), make_const(DataType::Int(32), to)},
        CallNode::Intrinsic));
  }
  // sync states.
  SyncState first_state_, last_state_, curr_state_;
  // Variables
  IterVar coproc_axis_;
  std::string sync_push_name_, sync_pop_name_;
};


class CoProcSyncInserter : public StmtMutator {
 public:
  Stmt Insert(Stmt stmt) {
    CoProcTouchedBuffer visitor;
    visitor(stmt);
    if (visitor.coproc_.size() == 0) return stmt;
    std::unordered_set<const VarNode*> touched;

    for (const auto &kv : visitor.touched_) {
      if (kv.second.normal && kv.second.coproc) {
        touched.insert(kv.first);
      }
    }
    CHECK_EQ(visitor.coproc_.size(), 1U);
    std::string coproc_name = (*visitor.coproc_.begin())->var->name_hint;
    // plan sync.
    CoProcSyncPlanner sync_planner(touched, coproc_name);
    sync_planner.Plan(stmt);
    for (const auto& kv : sync_planner.sync_) {
      auto& vec = insert_after_[kv.first];
      vec.insert(vec.end(), kv.second.begin(), kv.second.end());
    }
    // Detect barrier
    CoProcBarrierDetector barrier_detector(touched, coproc_name);
    barrier_detector.PlanReadBarrier(stmt);
    barrier_detector.PlanWriteBarrier(stmt);
    for (const auto& kv : barrier_detector.barrier_before_) {
      auto& vec = insert_before_[kv.first];
      vec.insert(vec.end(), kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : barrier_detector.barrier_after_) {
      auto& vec = insert_after_[kv.first];
      vec.insert(vec.end(), kv.second.begin(), kv.second.end());
    }
    // Detect barrier
    CoProcInstDepDetector sync_detector(
        *visitor.coproc_.begin(), coproc_name);
    sync_detector.Plan(stmt);
    for (const auto& kv : sync_detector.insert_before_) {
      auto& vec = insert_before_[kv.first];
      vec.insert(vec.end(), kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : sync_detector.insert_after_) {
      auto& vec = insert_after_[kv.first];
      vec.insert(vec.end(), kv.second.begin(), kv.second.end());
    }
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    auto it_before = insert_before_.find(stmt.get());
    auto it_after = insert_after_.find(stmt.get());
    Stmt new_stmt = StmtMutator::VisitStmt(stmt);

    return SeqStmt::Flatten(
      it_before != insert_before_.end() ? it_before->second : std::vector<Stmt>(),
      new_stmt,
      it_after != insert_after_.end() ? it_after->second : std::vector<Stmt>());
  }

 private:
  // insert before is stored in reverse order
  // the first element is closest to the node.
  std::unordered_map<const Object*, std::vector<Stmt> > insert_before_;
  std::unordered_map<const Object*, std::vector<Stmt> > insert_after_;
};


Stmt CoProcSync(Stmt stmt) {
  return CoProcSyncInserter().Insert(std::move(stmt));
}

namespace transform {

Pass CoProcSync() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = CoProcSyncInserter().Insert(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CoProcSync", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CoProcSync")
.set_body_typed(CoProcSync);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
