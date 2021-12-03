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
 * that defines a ordering in the callees to operators and produces BufferInfo
 * objects that contains information about tir.allocate nodes and liveness
 * conflicts between other tir.allocate nodes.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relay/executor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>

namespace tvm {
namespace tir {
namespace usmp {

/*!
 * \brief The visitor class to obtain buffer information
 *
 * The visitor would initiate the traversal from the main
 * function and visits into the operator PrimFuncs. It will
 * crate unique BufferInfo objects for each Allocate node.
 *
 * Every time the buffer variable of the allocate node is referenced
 * it will be recorded using the stmt index. However, note that
 * the same buffer variable could be references multiple times
 * from different calls. Thereafter, a sweep is done on all the
 * BufferInfo objects using the per-call liveness events. In the sweep,
 * The BufferInfo objects that are live together will be recorded as
 * mutual conflicts of each other.
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
  void RecordAllocateNodeInfo(const AllocateNode* op);
  void VisitPrimFunc(const PrimFunc& func, const Call& call);

  /*!
   * \brief Maintains the mapping of BufferInfo to their associated TIR Statements.
   */
  Map<BufferInfo, tir::Stmt> buffer_info_map_;
  /*!
   * \brief Records the order of calls in the main for stability.
   */
  std::set<Call> call_order_;
  /*!
   * \brief Records first access in-terms of Stmts to each buffer per call
   *
   * This is because multiple calls could happen to the same PrimFunc.
   */
  std::unordered_map<Call, Map<tir::Stmt, Integer>, ObjectPtrHash, ObjectPtrEqual>
      buffer_info_start_stmt_idx_;
  /*!
   * \brief Records last access in-terms of Stmts to each buffer per call
   *
   * This is because multiple calls could happen to the same PrimFunc.
   */
  std::unordered_map<Call, Map<tir::Stmt, Integer>, ObjectPtrHash, ObjectPtrEqual>
      buffer_info_end_stmt_idx_;

  /*!
   * \brief This structure contains information regarding a Allocate node.
   */
  struct AllocateInfo {
    tir::Stmt Allocate;
    PrimFunc prim_func;
    Call call;
  };

  /*!
   * \brief Maintains the mapping of buffer variable to their allocate nodes to ensure
   * that only one BufferInfo object is created.
   */
  std::unordered_map<tir::Var, AllocateInfo, ObjectPtrHash, ObjectPtrEqual> allocate_infos;
  /*!
   * \brief Indicates a count of stmts visited so far to use as a metric of liveness
   */
  int current_stmt_idx_ = 0;
  /*!
   * \brief This structure is supposed to contain information around the scope
   * the visitor is currently in.
   */
  struct ScopeInfo {
    /*!
     * \brief We need to record access per call
     */
    Call call;
    /*!
     * \brief Having access to PrimFunc metadata is useful
     */
    PrimFunc func;
    /*!
     * \brief We currently support only serial for loops. Therefore
     * need to know what kind of for loop the visitor is in.
     */
    For for_loop;
    /*!
     * \brief We record the live allocate_nodes because once in loops
     * the liveness range has to be extended to the whole of the nested
     * loops structure.
     */
    std::unordered_set<Allocate, ObjectPtrHash, ObjectPtrEqual> allocate_nodes;
    /*!
     * \brief This is recorded to extend the liveness of all allocates within
     * nested loop structure.
     */
    Integer initial_stmt_of_the_nested_loops;
  };
  std::stack<ScopeInfo> scope_stack_;

  /*!
   * \brief A liveness event is an event that when
   * traversing the tir.Stmts where tir.allocate node
   * begins or ceases to be Live. This particular struct
   * is used to solve interval overlap problem using
   * a sweep-line algorithm. For that, we need to record
   * where the liveness event occurred in a chronological
   * order.
   */
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
  /*!
   * \brief We need to create unique buffer name is the same name is used in
   * two allocate nodes for clarity for memory planning algorithms.
   */
  std::string GetUniqueBufferName(std::string name);

  /*!
   * \brief This is per buffer name counter to aid the generating the above
   * unique name.
   */
  std::unordered_map<std::string, int> buffer_names;
  /*!
   * \brief The TIR main function calls by name to PrimFuncs to be able to
   * support BYOC. Therefore, this Map records functions that are present
   * in the IRModule by name/
   */
  Map<String, PrimFunc> functions_;
  /*!
   * \brief The IRModule being analyzed.
   */
  IRModule module_;
};

std::string BufferInfoExtractor::GetUniqueBufferName(std::string name) {
  if (buffer_names.find(name) == buffer_names.end()) {
    buffer_names[name] = 1;
    return name;
  } else {
    buffer_names[name] = buffer_names[name] + 1;
    return name + std::to_string(buffer_names[name]);
  }
}

void BufferInfoExtractor::VisitStmt(const Stmt& n) {
  current_stmt_idx_ += 1;
  StmtExprVisitor::VisitStmt(n);
}

void BufferInfoExtractor::RecordAllocateNodeInfo(const AllocateNode* op) {
  auto size_bytes = CalculateExtentsSize(op);
  // We only statically memory plan only allocates with known
  // compile time sizes.
  if (size_bytes.defined()) {
    if (allocate_infos.find(op->buffer_var) == allocate_infos.end()) {
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
      PrimFunc func = scope_stack_.top().func;
      Optional<tvm::relay::Executor> executor_config =
          module_->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor);
      Integer workspace_alignment = 16;
      if (executor_config) {
        workspace_alignment =
            executor_config.value()->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
      }
      auto buffer_info = BufferInfo(GetUniqueBufferName(op->buffer_var->name_hint), size_bytes,
                                    pool_candidates, workspace_alignment);
      auto allocate = GetRef<Allocate>(op);
      allocate_infos[op->buffer_var] =
          AllocateInfo{allocate, scope_stack_.top().func, scope_stack_.top().call};
      buffer_info_map_.Set(buffer_info, allocate);
    } else {
      // Update the allocate info with the latest call
      AllocateInfo ai = allocate_infos[op->buffer_var];
      ai.call = scope_stack_.top().call;
      allocate_infos[op->buffer_var] = ai;
    }
  }
}

void BufferInfoExtractor::VisitStmt_(const AllocateNode* op) {
  ScopeInfo& current_scope_info = scope_stack_.top();
  const auto& type = Downcast<PointerType>(op->buffer_var->type_annotation);
  const auto& storage_scope = type->storage_scope;

  // If the allocate is in a for loop, USMP currently only looks at serial for loops.
  // If its not a serial for loop, then memory planner will omit them in the current memory planning
  // process leaving them to as tir.allocate nodes for codegen. Additionally, the USMP can only work
  // with buffers that have global storage_scope

  if (!current_scope_info.for_loop.defined()) {
    RecordAllocateNodeInfo(op);
  } else if (current_scope_info.for_loop.defined() &&
             current_scope_info.for_loop->kind == ForKind::kSerial && storage_scope == "global") {
    RecordAllocateNodeInfo(op);
  }
  StmtExprVisitor::VisitStmt(op->body);
  current_scope_info.allocate_nodes.erase(GetRef<Allocate>(op));
}

void BufferInfoExtractor::VisitStmt_(const ForNode* op) {
  ScopeInfo si{scope_stack_.top().call, scope_stack_.top().func, GetRef<For>(op),
               scope_stack_.top().allocate_nodes,
               scope_stack_.top().initial_stmt_of_the_nested_loops};
  if (!scope_stack_.top().initial_stmt_of_the_nested_loops.defined()) {
    si.initial_stmt_of_the_nested_loops = Integer(current_stmt_idx_);
  }
  Call current_call = scope_stack_.top().call;
  PrimFunc current_primfunc = scope_stack_.top().func;
  scope_stack_.push(si);
  StmtExprVisitor::VisitStmt_(op);
  // Extending the liveness to beginning of for-loop next and end of the current for-loop
  for (const Allocate& allocate : scope_stack_.top().allocate_nodes) {
    AllocateInfo ai = allocate_infos[allocate->buffer_var];
    Call update_call = current_call;
    // If the allocate does not belong to current prim func
    // We need to update the call to which the allocate belong to
    if (ai.prim_func != current_primfunc) {
      update_call = ai.call;
    }
    if (scope_stack_.top().initial_stmt_of_the_nested_loops->value <
        buffer_info_start_stmt_idx_[update_call][allocate]) {
      buffer_info_start_stmt_idx_[update_call].Set(
          allocate, scope_stack_.top().initial_stmt_of_the_nested_loops->value);
    }
    if (current_stmt_idx_ > buffer_info_end_stmt_idx_[update_call][allocate]) {
      buffer_info_end_stmt_idx_[update_call].Set(allocate, current_stmt_idx_);
    }
  }
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
  Call current_call = scope_stack_.top().call;
  PrimFunc current_primfunc = scope_stack_.top().func;
  if (allocate_infos.count(var)) {
    auto allocate = allocate_infos[var].Allocate;
    auto allocate_primfunc = allocate_infos[var].prim_func;
    Call update_call = current_call;
    if (allocate_primfunc != current_primfunc) {
      // If the allocate node does not belong to the current primfunc.
      // It's access should be reported to the call to PrimFunc that
      // Allocate belong to.
      update_call = allocate_infos[var].call;
    }
    if (buffer_info_start_stmt_idx_[update_call].count(allocate) == 0) {
      buffer_info_start_stmt_idx_[update_call].Set(allocate, current_stmt_idx_);
    }
    buffer_info_end_stmt_idx_[update_call].Set(allocate, current_stmt_idx_);

    ScopeInfo& currect_scope_info = scope_stack_.top();
    if (currect_scope_info.for_loop.defined()) {
      currect_scope_info.allocate_nodes.insert(Downcast<Allocate>(allocate));
    }
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
      if (allocate_infos.count(load->buffer_var)) {
        allocate_infos[param_buf] = allocate_infos[load->buffer_var];
      }
    } else if (arg->IsInstance<VarNode>()) {
      auto var = Downcast<Var>(arg);
      if (allocate_infos.count(var)) {
        allocate_infos[param_buf] = allocate_infos[var];
      }
    }
  }
}

void BufferInfoExtractor::VisitPrimFunc(const PrimFunc& func, const Call& call) {
  ScopeInfo si{call, func, scope_stack_.top().for_loop, scope_stack_.top().allocate_nodes,
               scope_stack_.top().initial_stmt_of_the_nested_loops};
  call_order_.insert(call);
  scope_stack_.push(si);
  this->VisitStmt(func->body);
  scope_stack_.pop();
}

void BufferInfoExtractor::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
    StringImm func_name = Downcast<StringImm>(op->args[0])->value;
    if (functions_.find(func_name->value) != functions_.end()) {
      auto func = functions_.at(func_name->value);
      auto actual_args = Array<PrimExpr>(op->args.begin() + 1, op->args.end());
      this->UpdateAliases(actual_args, func);
      VisitPrimFunc(func, GetRef<Call>(op));
      return;
    }
  }
  if (op->op->IsInstance<PrimFuncNode>()) {
    auto func = Downcast<PrimFunc>(op->op);
    this->UpdateAliases(op->args, func);
    VisitPrimFunc(func, GetRef<Call>(op));
    return;
  }
  StmtExprVisitor::VisitExpr_(op);
}

Map<BufferInfo, tir::Stmt> BufferInfoExtractor::operator()(const PrimFunc& main_func) {
  VisitPrimFunc(main_func, Call());

  // Create a vector of liveness events
  // associated with each BufferNodes.
  std::vector<LivenessEvent> le_events_timeline;
  for (const auto& kv1 : buffer_info_map_) {
    if (!kv1.second->IsInstance<AllocateNode>()) {
      continue;
    }
    auto allocate = Downcast<Allocate>(kv1.second);
    auto buffer_info = Downcast<BufferInfo>(kv1.first);

    ICHECK(call_order_.size() >= buffer_info_end_stmt_idx_.size());
    ICHECK(call_order_.size() >= buffer_info_end_stmt_idx_.size());

    for (const Call& call : call_order_) {
      Map<Stmt, Integer> buffer_info_starts = buffer_info_start_stmt_idx_[call];
      if (buffer_info_starts.find(allocate) != buffer_info_starts.end()) {
        LivenessEvent le_event_start;
        le_event_start.buffer_info = buffer_info;
        le_event_start.le_type = START;
        le_event_start.tick = buffer_info_starts[allocate];
        le_events_timeline.push_back(le_event_start);
      }
    }

    for (const Call& call : call_order_) {
      Map<Stmt, Integer> buffer_info_ends = buffer_info_end_stmt_idx_[call];
      if (buffer_info_ends.find(allocate) != buffer_info_ends.end()) {
        LivenessEvent le_event_end;
        le_event_end.buffer_info = buffer_info;
        le_event_end.le_type = END;
        le_event_end.tick = buffer_info_ends[allocate];
        le_events_timeline.push_back(le_event_end);
      }
    }
  }

  // Sort the liveness events based on the chronological
  // ordering. For events that are simultaneous, START event
  // takes precedence.
  std::sort(le_events_timeline.begin(), le_events_timeline.end(),
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
  std::unordered_map<BufferInfo, int, ObjectPtrHash, ObjectPtrEqual> open_set;
  for (const auto& le_event : le_events_timeline) {
    if (le_event.le_type == START) {
      for (const auto& kv : open_set) {
        BufferInfo open_buffer_info = kv.first;
        open_buffer_info->conflicts.push_back(le_event.buffer_info);
        if (le_event.buffer_info != open_buffer_info) {
          le_event.buffer_info->conflicts.push_back(open_buffer_info);
        }
      }
      if (open_set.find(le_event.buffer_info) == open_set.end()) {
        open_set[le_event.buffer_info] = 1;
      } else {
        open_set[le_event.buffer_info] += 1;
      }
    } else {
      if (open_set[le_event.buffer_info] == 1) {
        open_set.erase(le_event.buffer_info);
      } else {
        open_set[le_event.buffer_info] -= 1;
      }
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
