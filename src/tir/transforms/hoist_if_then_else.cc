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
 * \file hoist_if_then_else.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

struct HoistIfThenElseConfigNode : public tvm::AttrsNode<HoistIfThenElseConfigNode> {
  bool support_block_scope_hosting;

  TVM_DECLARE_ATTRS(HoistIfThenElseConfigNode, "tir.transform.HoistIfThenElseConfig") {
    TVM_ATTR_FIELD(support_block_scope_hosting)
        .describe("Hoist if cond with block scope variables")
        .set_default(false);
  }
};

class HoistIfThenElseConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(HoistIfThenElseConfig, Attrs,
                                            HoistIfThenElseConfigNode);
};

TVM_REGISTER_NODE_TYPE(HoistIfThenElseConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.HoistIfThenElse", HoistIfThenElseConfig);

using VarForMap = std::unordered_map<const VarNode*, const ForNode*>;
using HoistForIfTuple = std::tuple<bool, const ForNode*, const IfThenElseNode*>;

/*
 * This pass tries to hoist IfThenElse stmt out of For loop if condition is loop invariant.
 * For example, given the following block:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 4; j++)
 *        for (k = 0; k < 5; k++)
 *            if (likely(i*2 < 4))
 *                A[3*i+2j+k] = B[7*i+3j+k]
 *
 * We first detect all IfThenElse stmt and find the corresponding loop invariant For stmt.
 * Then we hoist IfThenElse stmt by one For stmt each step:
 *
 * Step 1:
 * for (i = 0; i < 3; i++)
 *     for (j = 0; j < 4; j++)
 *         if (likely(i*2 < 4))
 *             for (k = 0; k < 5; k++)
 *                 A[3*i+2j+k] = B[7*i+3j+k]
 *
 * Step 2:
 * for (i = 0; i < 3; i++)
 *     if (likely(i*2 < 4))
 *         for (j = 0; j < 4; j++)
 *             for (k = 0; k < 5; k++)
 *                 A[3*i+2j+k] = B[7*i+3j+k]
 *
 * In this pass, we only continue detecting possible hoisting chance when visiting For,
 * IfThenElse or AttrStmt Node. For example, for the following block:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 4; j++)
 *        A[i + j] = A[i + j] - 1
 *        for (k = 0; k < 5; k++)
 *            if (likely(i*2 < 4))
 *                A[3*i+2j+k] = B[7*i+3j+k]
 *
 * Only the For with k variable will be considered and the resulting stmt would be:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 4; j++)
 *        A[i + j] = A[i + j] - 1
 *        if (likely(i*2 < 4))
 *            for (k = 0; k < 5; k++)
 *                A[3*i+2j+k] = B[7*i+3j+k]
 *
 * This pass doesn't do hoisting for consecutive IfThenElse stmt. The following
 * block won't be optimized:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 4; j++)
 *        for (k = 0; k < 5; k++)
 *            if (likely(i*2 < 4))
 *                A[3*i+2j+k] = B[7*i+3j+k]
 *            if (likely(j > 2))
 *                A[i+j+k] = B[i+j+k]
 *
 *
 * This pass do hoisting for Block scope variables also.
 * As below:
 * Attr(IterVar: threadIdx.x)
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 4; j++)
 *        for (k = 0; k < 5; k++)
 *            if (likely(threadIdx.x < 3))
 *                A[3*i+2j+k] = B[7*i+3j+k]
 *
 * Will be transformed to as below:
 * Attr(IterVar: threadIdx.x)
 * if (likely(threadIdx.x < 3))
 *     for (i = 0; i < 3; i++)
 *         for (j = 0; j < 4; j++)
 *             for (k = 0; k < 5; k++)
 *                 A[3*i+2j+k] = B[7*i+3j+k]
 *
 */

// Select potential candidate IRs that can be hoisted.
class HoistCandidateSelector final : public StmtExprVisitor {
 public:
  explicit HoistCandidateSelector(bool support_block_scope_hosting)
      : support_block_scope_hosting_(support_block_scope_hosting) {
    InitRecorder();
  }
  HoistCandidateSelector() { InitRecorder(); }

  void VisitStmt_(const ForNode* op) final {
    // If already recording complete,
    // then stop tracing
    if (RecordingComplete()) {
      return;
    }

    // Check if it is first for loop, then start the recorder
    StartOrAddRecord(GetRef<ObjectRef>(op));
    StmtExprVisitor::VisitStmt_(op);
    RemoveRecord(GetRef<ObjectRef>(op));
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    // If SeqStmt is encountered in the middle of recording
    //  then need to purge all, as it can not be hoisted
    if (IsRecordingOn()) {
      ResetRecorderInternal();
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // Maintain list of all vars in AttrStmt
    // To stop hoisting if any of the block variables are used.
    //
    // In case we want to use hoisting in between certain passes
    // which have interdependencies of the postioning of if nodes with scope var
    // it is better to disable this section
    if (support_block_scope_hosting_) {
      if (IsRecordingOn()) {
        StartOrAddRecord(GetRef<ObjectRef>(op));
        StmtExprVisitor::VisitStmt_(op);
        RemoveRecord(GetRef<ObjectRef>(op));
        return;
      } else {
        return StmtExprVisitor::VisitStmt_(op);
      }
    }
    UpdateAttrVarList(op);
    StmtExprVisitor::VisitStmt_(op);
    RemoveAttrVarList(op);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    if (!IsRecordingOn()) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    is_if_cond_ = true;
    StmtExprVisitor::VisitExpr(op->condition);
    is_if_cond_ = false;

    if (CheckValidIf()) {
      // Check corresponding for loop
      int match_for_loop_pos = -1;
      for (auto var : if_var_list_) {
        for (int i = 0; i < static_cast<int>(ordered_list_.size()); ++i) {
          if ((ordered_list_[i] == var_for_map_[var]) || (ordered_list_[i] == var)) {
            if (match_for_loop_pos < i) {
              match_for_loop_pos = i;
            }
          }
        }
      }
      // If none of the for loop has the matching loop variable as if condition,
      // then the if node need to be hoisted on top of all, provided no parent loop exists.
      int target_for_pos = GetNextLoopPos(match_for_loop_pos);

      // Check if valid position
      if (target_for_pos >= 0) {
        StopAndAddRecord(static_cast<const ForNode*>(ordered_list_[target_for_pos]), op);
        if_var_list_.clear();
        return;
      }
    }

    if_var_list_.clear();
    StmtExprVisitor::VisitStmt_(op);
    StopRecording();
  }

  void VisitExpr_(const VarNode* op) final {
    if (is_if_cond_) {
      if_var_list_.emplace_back(op);
    }
  }

  HoistForIfTuple hoist_for_if_recorder;

  void ResetRecorder() {
    ResetRecorderInternal();

    // Reset Block scope vars also here
    attr_var_list_.clear();
  }

  bool RecordingComplete() { return std::get<0>(hoist_for_if_recorder); }

  const ForNode* GetTargetForNode() { return std::get<1>(hoist_for_if_recorder); }

  const IfThenElseNode* GetTargetIfNode() { return std::get<2>(hoist_for_if_recorder); }

 private:
  void ResetRecorderInternal() {
    if (is_recorder_on_) {
      CHECK_GT(ordered_list_.size(), 0);
      is_recorder_on_ = false;
    }
    ordered_list_.clear();
    var_for_map_.clear();
    hoist_for_if_recorder = std::make_tuple(false, nullptr, nullptr);
  }
  bool CheckValidIf() {
    // If no if var list is present, then all the condition vars are possibly from AttrStmt, so stop
    // hoisting
    return ((!if_var_list_.empty()) && (!CheckAttrVar()));
  }

  int GetNextLoopPos(int cur_pos) {
    for (size_t i = cur_pos + 1; i < ordered_list_.size(); ++i) {
      if (ordered_list_[i]->IsInstance<ForNode>()) {
        return i;
      }
    }
    return -1;
  }

  void InitRecorder() { hoist_for_if_recorder = std::make_tuple(false, nullptr, nullptr); }

  void StopRecording() { is_recorder_on_ = false; }

  bool IsRecordingOn() { return is_recorder_on_; }

  void StartOrAddRecord(const ObjectRef& op) {
    is_recorder_on_ = true;
    if (const auto* node = op.as<ForNode>()) {
      if (!var_for_map_.count(node->loop_var.get()))
        var_for_map_.insert({node->loop_var.get(), node});
      ordered_list_.emplace_back(op.get());
    } else if (const auto* node = op.as<AttrStmtNode>()) {
      if (const auto* iv = node->node.as<IterVarNode>()) {
        ordered_list_.emplace_back(iv->var.get());
      } else if (const auto* iv = node->node.as<VarNode>()) {
        ordered_list_.emplace_back(iv);
      }
    }
  }

  void RemoveRecord(const ObjectRef& op) {
    StopRecording();
    if (const auto* node = op.as<ForNode>()) var_for_map_.erase(node->loop_var.get());
    if (ordered_list_.size() > 0) ordered_list_.pop_back();
  }

  void StopAndAddRecord(const ForNode* for_node, const IfThenElseNode* if_node) {
    hoist_for_if_recorder = std::make_tuple(true, for_node, if_node);
    StopRecording();
  }

  void UpdateAttrVarList(const AttrStmtNode* op) {
    if (const auto* iv = op->node.as<IterVarNode>()) {
      attr_var_list_.insert(iv->var.get());
    } else if (const auto* iv = op->node.as<VarNode>()) {
      attr_var_list_.insert(iv);
    }
  }

  void RemoveAttrVarList(const AttrStmtNode* op) {
    if (const auto* iv = op->node.as<IterVarNode>()) {
      attr_var_list_.erase(iv->var.get());
    } else if (const auto* iv = op->node.as<VarNode>()) {
      attr_var_list_.erase(iv);
    }
  }

  bool CheckAttrVar() {
    for (auto var : if_var_list_) {
      if (attr_var_list_.count(var)) {
        return true;
      }
    }
    return false;
  }

  // Ordered List maintains all ForNodes & AttrStmtNodes encountered in sequence
  std::vector<const Object*> ordered_list_;
  std::vector<const VarNode*> if_var_list_;
  std::unordered_set<const VarNode*> attr_var_list_;
  VarForMap var_for_map_;

  bool is_if_cond_{false};
  bool is_recorder_on_{false};
  bool support_block_scope_hosting_{false};
};

class IfThenElseHoister : public StmtMutator {
 public:
  IfThenElseHoister() : hoist_selector_(HoistCandidateSelector()) {}
  explicit IfThenElseHoister(bool support_block_scope_hosting)
      : hoist_selector_(HoistCandidateSelector(support_block_scope_hosting)) {}

  Stmt VisitAndMutate(Stmt stmt) {
    hoist_selector_(stmt);
    Stmt stmt_copy = std::move(stmt);

    while (hoist_selector_.RecordingComplete()) {
      target_for_ = hoist_selector_.GetTargetForNode();
      target_if_ = hoist_selector_.GetTargetIfNode();

      stmt_copy = operator()(stmt_copy);

      hoist_selector_.ResetRecorder();
      hoist_selector_(stmt_copy);
    }

    // Support SSA Form
    stmt_copy = ConvertSSA(stmt_copy);
    return stmt_copy;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if ((!is_updating_) && (target_for_ == op)) {
      is_updating_ = true;
      is_then_case_ = true;
      Stmt then_case = StmtMutator::VisitStmt_(op);
      is_then_case_ = false;
      Stmt else_case = Stmt();
      if (target_if_->else_case.defined()) {
        else_case = StmtMutator::VisitStmt_(op);
      }
      is_updating_ = false;
      return IfThenElse(target_if_->condition, then_case, else_case);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    if (is_updating_ && (target_if_ == op)) {
      if (is_then_case_) {
        return StmtMutator::VisitStmt(op->then_case);
      } else if (op->else_case.defined()) {
        return StmtMutator::VisitStmt(op->else_case);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  bool is_updating_{false};
  bool is_then_case_{false};
  HoistCandidateSelector hoist_selector_;
  const ForNode* target_for_;
  const IfThenElseNode* target_if_;
};

Stmt HoistIfThenElse(Stmt stmt, bool support_block_scope_hosting) {
  return IfThenElseHoister(support_block_scope_hosting).VisitAndMutate(stmt);
}
Stmt HoistIfThenElse(Stmt stmt) { return IfThenElseHoister().VisitAndMutate(stmt); }

namespace transform {

Pass HoistIfThenElse() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<HoistIfThenElseConfig>("tir.HoistIfThenElse");

    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<HoistIfThenElseConfig>();
    }
    n->body = HoistIfThenElse(std::move(n->body), cfg.value()->support_block_scope_hosting);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HoistIfThenElse", {});
}

Pass HoistIfThenElseBasic() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = HoistIfThenElse(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HoistIfThenElseBasic", {});
}

TVM_REGISTER_GLOBAL("tir.transform.HoistIfThenElse").set_body_typed(HoistIfThenElse);

TVM_REGISTER_GLOBAL("tir.transform.HoistIfThenElseBasic").set_body_typed(HoistIfThenElseBasic);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
