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
 * \file tir/analysis/usmp/extract_buffer_info.cc
 *
 * \brief This analysis pass consumes a TIR IRModule with a main function
 * that defines a ordering in the calles to operators and produces BufferInfo
 * objects that contains information about tir.allocate nodes and liveness
 * conflicts between other tir.allocate nodes.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>

namespace tvm {
namespace tir {
namespace usmp {

/*! \brief This class takes a TIR IRModule and a main PrimFunc that contains
 * that defines a ordering in the calles to operators and produces BufferInfo
 * objects that contains information about tir.allocate nodes and liveness
 * conflicts between other tir.allocate nodes.
 */

class BufferInfoExtractor : public StmtExprVisitor {
 public:
  explicit BufferInfoExtractor(const IRModule& module) : module_(module) {
    for (const auto& gv_func : module_->functions) {
      functions_.Set(gv_func.first->name_hint, Downcast<PrimFunc>(gv_func.second));
    }
    // Pushing a scope info for the initial body of the main function
    scope_stack_.push(ScopeInfo());
  }
  Map<BufferInfo, tir::Stmt> operator()(const PrimFunc& func);

 private:
  void VisitStmt(const Stmt& n) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const LoadNode* op) override;
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;

  void UpdateAliases(const Array<PrimExpr>& args, const PrimFunc& func);

  Map<BufferInfo, tir::Stmt> buffer_info_map_;
  Map<tir::Stmt, Integer> buffer_info_start_stmt_idx_;
  Map<tir::Stmt, Integer> buffer_info_end_stmt_idx_;
  Map<tir::Var, tir::Stmt> allocate_var_to_stmt_map_;

  std::unordered_set<Stmt, ObjectPtrHash, ObjectPtrEqual> currently_live_allocates;
  int current_stmt_idx_ = 0;
  // This structure is supposed to contain information
  // around the scope the visitor is currently in.
  // We only check whether the current scope belong to
  // a Serial ForKind. We are not planning for Parallel
  // ForKind just yet.
  struct ScopeInfo {
    For for_loop;
  };
  std::stack<ScopeInfo> scope_stack_;

  Map<String, PrimFunc> functions_;
  IRModule module_;
};

void BufferInfoExtractor::VisitStmt(const Stmt& n) {
  current_stmt_idx_ += 1;
  StmtExprVisitor::VisitStmt(n);
}

static Integer CalculateExtentsSize(const AllocateNode* op) {
  size_t element_size_bytes = op->dtype.bytes();
  size_t num_elements = 1;
  for (const auto& ext : op->extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    } else {
      // We can't statically calculate workspace for dynamic shapes
      return Integer();
    }
  }
  return Integer(num_elements * element_size_bytes);
}

void BufferInfoExtractor::VisitStmt_(const AllocateNode* op) {
  const auto& currect_scope_info = scope_stack_.top();
  const auto& type = Downcast<PointerType>(op->buffer_var->type_annotation);
  const auto& storage_scope = type->storage_scope;

  // If the allocate is in a for loop, USMP currently only looks at serial for loops.
  // If its not a serial for loop, then memory planner will omit them in the current memory planning
  // process leaving them to as tir.allocate nodes for codegen. Additionally, the USMP can only work
  // with buffers that have global storage_scope
  if ((!currect_scope_info.for_loop.defined()) ||
      (currect_scope_info.for_loop.defined() &&
       currect_scope_info.for_loop->kind == ForKind::kSerial && storage_scope == "global")) {
    auto size_bytes = CalculateExtentsSize(op);
    // We only statically memory plan only allocates with known
    // compile time sizes.
    if (size_bytes.defined()) {
      // By default, the core compiler is assumed to attach the a default pool to each allocate.
      ICHECK(op->annotations.count(kPoolCandidatesAllocateAttr))
          << "Every statically sized allocate node needs an pool candidate attribute";
      auto pool_candidates =
          Downcast<Array<PoolInfo>>(op->annotations[kPoolCandidatesAllocateAttr]);

      // TODO(@manupa-arm): improve the error when the responsible component for attaching a single
      // pool is added
      ICHECK(pool_candidates.size() > 0)
          << "The core compiler should at least attach a single PoolInfo. If there were no "
             "user-given arguments for memory pools, the default behaviour is a single size "
             "un-restricted pool is assigned";
      auto buffer_info = BufferInfo(op->buffer_var->name_hint, size_bytes, pool_candidates);
      auto allocate = GetRef<Allocate>(op);
      allocate_var_to_stmt_map_.Set(op->buffer_var, allocate);
      buffer_info_map_.Set(buffer_info, allocate);
    }
  }
  StmtExprVisitor::VisitStmt(op->body);
}

void BufferInfoExtractor::VisitStmt_(const ForNode* op) {
  ScopeInfo si{
      GetRef<For>(op),
  };
  scope_stack_.push(si);
  StmtExprVisitor::VisitStmt_(op);
  scope_stack_.pop();
}

void BufferInfoExtractor::VisitExpr_(const LoadNode* op) {
  this->VisitExpr(op->buffer_var);
  StmtExprVisitor::VisitExpr_(op);
}

void BufferInfoExtractor::VisitStmt_(const StoreNode* op) {
  this->VisitExpr(op->buffer_var);
  StmtExprVisitor::VisitStmt_(op);
}

void BufferInfoExtractor::VisitExpr_(const VarNode* op) {
  auto var = GetRef<Var>(op);
  if (allocate_var_to_stmt_map_.count(var)) {
    auto allocate = allocate_var_to_stmt_map_[var];
    if (buffer_info_start_stmt_idx_.count(allocate) == 0) {
      buffer_info_start_stmt_idx_.Set(allocate, current_stmt_idx_);
    }
    buffer_info_end_stmt_idx_.Set(allocate, current_stmt_idx_);
  }
  StmtExprVisitor::VisitExpr_(op);
}

Array<Var> static GetMatchedBuffers(const PrimFunc& func) {
  Array<Var> buffer_vars;
  for (const auto& param : func->params) {
    buffer_vars.push_back(func->buffer_map[param]->data);
  }
  return buffer_vars;
}

void BufferInfoExtractor::UpdateAliases(const Array<PrimExpr>& args, const PrimFunc& func) {
  auto param_buffers = GetMatchedBuffers(func);
  ICHECK(args.size() == param_buffers.size());
  for (size_t i = 0; i < args.size(); i++) {
    auto arg = args[i];
    auto param_buf = param_buffers[i];
    // If tir.allocates are passed in to functions
    // The function params are re-directed to point
    // to the original allocate
    if (arg->IsInstance<LoadNode>()) {
      auto load = Downcast<Load>(arg);
      if (allocate_var_to_stmt_map_.count(load->buffer_var)) {
        allocate_var_to_stmt_map_.Set(param_buf, allocate_var_to_stmt_map_[load->buffer_var]);
      }
    } else if (arg->IsInstance<VarNode>()) {
      auto var = Downcast<Var>(arg);
      if (allocate_var_to_stmt_map_.count(var)) {
        allocate_var_to_stmt_map_.Set(param_buf, allocate_var_to_stmt_map_[var]);
      }
    }
  }
}

void BufferInfoExtractor::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern())) {
    auto func = functions_.at(Downcast<StringImm>(op->args[0])->value);
    auto actual_args = Array<PrimExpr>(op->args.begin() + 1, op->args.end());
    this->UpdateAliases(actual_args, func);
    this->VisitStmt(func->body);
  } else if (op->op->IsInstance<PrimFuncNode>()) {
    auto func = Downcast<PrimFunc>(op->op);
    this->UpdateAliases(op->args, func);
    this->VisitStmt(func->body);
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

Map<BufferInfo, tir::Stmt> BufferInfoExtractor::operator()(const PrimFunc& main_func) {
  this->VisitStmt(main_func->body);

  // A liveness event is an event that when
  // traversing the tir.Stmts where tir.allocate node
  // begins or ceases to be Live. This particular struct
  // is used to solve interval overlap problem using
  // a sweep-line algorithm. For that, we need to record
  // where the liveness event occurred in a chronological
  // order.
  enum LivenessEventType { START = 0, END = 1 };
  struct LivenessEvent {
    size_t tick;
    LivenessEventType le_type;
    BufferInfo buffer_info;
    bool operator==(const LivenessEvent& other) {
      if (tick == other.tick && le_type == other.le_type && buffer_info == other.buffer_info) {
        return true;
      }
      return false;
    }
  };

  // Create a vector of liveness events
  // associated with each BufferNodes.
  std::vector<LivenessEvent> le_events;
  for (const auto& kv : buffer_info_map_) {
    if (!kv.second->IsInstance<AllocateNode>()) {
      continue;
    }
    auto allocate = Downcast<Allocate>(kv.second);
    auto buffer_info = Downcast<BufferInfo>(kv.first);
    // If the allocate is not used; we remove it from the analysis
    if (buffer_info_start_stmt_idx_.count(allocate) == 0) {
      continue;
    }
    LivenessEvent le_event_start;
    le_event_start.buffer_info = buffer_info;
    le_event_start.le_type = START;
    le_event_start.tick = buffer_info_start_stmt_idx_[allocate];
    le_events.push_back(le_event_start);

    LivenessEvent le_event_end;
    le_event_end.buffer_info = buffer_info;
    le_event_end.le_type = END;
    le_event_end.tick = buffer_info_end_stmt_idx_[allocate];
    le_events.push_back(le_event_end);
  }

  // Sort the liveness events based on the chronological
  // ordering. For events that are simultaneous, START event
  // takes precedence.
  std::sort(le_events.begin(), le_events.end(),
            [](const LivenessEvent& lhs, const LivenessEvent& rhs) {
              if (lhs.tick < rhs.tick) {
                return true;
              } else if (lhs.tick == rhs.tick && lhs.le_type == START && rhs.le_type == END) {
                return true;
              }
              return false;
            });

  // Traverse the liveness events using a open set to track what
  // is live while updating the conflicts through out the linear traversal
  std::unordered_set<BufferInfo, ObjectPtrHash, ObjectPtrEqual> open_set;
  for (const auto& le_event : le_events) {
    if (le_event.le_type == START) {
      for (const auto& open_buffer_info : open_set) {
        open_buffer_info->conflicts.push_back(le_event.buffer_info);
        le_event.buffer_info->conflicts.push_back(open_buffer_info);
      }
      open_set.insert(le_event.buffer_info);
    } else {
      open_set.erase(le_event.buffer_info);
    }
  }
  return this->buffer_info_map_;
}

Map<BufferInfo, tir::Stmt> ExtractBufferInfo(const PrimFunc& main_func, const IRModule& mod) {
  return BufferInfoExtractor(mod)(main_func);
}

TVM_REGISTER_GLOBAL("tir.usmp.analysis.extract_buffer_info")
    .set_body_typed([](PrimFunc main_func, IRModule mod) {
      return (ExtractBufferInfo(main_func, mod));
    });

}  // namespace usmp
}  // namespace tir
}  // namespace tvm
