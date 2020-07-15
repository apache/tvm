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
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

using VarForMap = std::unordered_map<const Object*, const Object*>;
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
 */


// Select potential candidate IRs that can be hoisted.
class HoistCandidateSelector final : public StmtExprVisitor {
 public:
  explicit HoistCandidateSelector() {
    InitRecorder();
  }
  void VisitStmt_(const ForNode* op) final {
  // Check if it is first for loop, then start the recorder
  if (!RecordingComplete()) {
    StartOrAddRecord(op);
  }

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    // If SeqStmt is encountered in the middle of recording
    //  then need to purge all, as it can not be hoisted
    if (is_recorder_on) {
      ResetRecorder();
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
	if (is_recorder_on) {
	      	is_if_cond = true;
	      	  StmtExprVisitor::VisitExpr(op->condition);
	      	is_if_cond = false;
	      	if (!CheckValidIf()) {
	      	  ResetRecorder();
	      	}
	      	else {
	      	  // Check corresponding for loop
		     bool match_found = false;
		     size_t match_for_loop_pos = 0;
		     for (auto var : if_var_list_) {
          		     for (size_t i = 0; i < ordered_for_list_.size() - 1; ++i) {
          		       if (ordered_for_list_[i] == var_for_map_[var]) {
          		           if (match_for_loop_pos < i) {match_for_loop_pos = i;}
          			  match_found = true;
          			  break;
          		       }
          		     }
		     }
			 // If none of the for loop has the matching loop variable as if condition,
			 // then the if node need to be hoisted on top of all, provided no parent loop exists.
			 int target_for_pos = match_found ? match_for_loop_pos + 1 : 0;
			 
			 // Check if target for loop is not the parent of current if node
			 if (!IsParentForLoop(target_for_pos)) {
			   StopAndAddRecord(ordered_for_list_[target_for_pos], op);
			 }
	      	}
	      	if_var_list_.clear();
	}
	StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode* op) final {
     if (is_if_cond) {
     	if_var_list_.emplace_back(op);
     }
    StmtExprVisitor::VisitExpr_(op);
  }

  HoistForIfTuple hoist_for_if_recorder;

  void ResetRecorder() {
   CHECK_GT(ordered_for_list_.size(), 0);
   is_recorder_on = false;
   ordered_for_list_.clear();
   var_for_map_.clear();
   hoist_for_if_recorder = std::make_tuple(false, nullptr, nullptr);
 }

  bool RecordingComplete() {
   if (std::get<0>(hoist_for_if_recorder)) return true;
   return false;
 }

 const ForNode* GetTargetForNode() {
   return std::get<1>(hoist_for_if_recorder);
 }

 const IfThenElseNode* GetTargetIfNode() {
	return std::get<2>(hoist_for_if_recorder);
  }

 private:

bool CheckValidIf() {
  if (if_var_list_.size() > ordered_for_list_.size()) {
  	return false;
  }
  return true;
}

 bool IsParentForLoop(int loop_pos) {
   // Check if the loop position is higher than the parent loop position
   for (auto var : if_var_list_) {
   	   if (GetParentLoopPos(var_for_map_[var]) >= loop_pos) {
	   	return true;
   	   }
   	}
   return false;
 }

 int GetParentLoopPos(const Object* node) {
 	for (size_t i = 0; i < ordered_for_list_.size(); ++i) {
		if (ordered_for_list_[i] == node)
			{
			  return i;
			}
 	}
	return -1;
 }

void InitRecorder() {
  hoist_for_if_recorder = std::make_tuple(false, nullptr, nullptr);
}

 void StartOrAddRecord(const ForNode* op) {
   if (!is_recorder_on) is_recorder_on = true;
   if (!var_for_map_.count(op->loop_var.get())) {
          var_for_map_.insert({op->loop_var.get(), op});
    }
   ordered_for_list_.emplace_back(op);
 }

 void StopAndAddRecord(const ForNode* for_node, const IfThenElseNode* if_node) {
	hoist_for_if_recorder = std::make_tuple(true, for_node, if_node);
	is_recorder_on = false;
  }

  std::vector<const ForNode*> ordered_for_list_;

  std::vector<const VarNode*> if_var_list_;

  VarForMap var_for_map_;

  bool is_if_cond{false};
  bool is_recorder_on{false};
};

class IfThenElseHoister : public StmtMutator {
 public:
  explicit IfThenElseHoister()
      : hoist_selector(HoistCandidateSelector()) {}

  Stmt VisitAndMutate(Stmt stmt) {
    hoist_selector(stmt);
    Stmt stmt_copy = std::move(stmt);
    while (hoist_selector.RecordingComplete()) {
	        target_for =  hoist_selector.GetTargetForNode();
	        target_if =  hoist_selector.GetTargetIfNode();

	        stmt_copy = operator()(stmt_copy);
		hoist_selector.ResetRecorder();
		hoist_selector(stmt_copy);
    }

    // Support SSA Form
    stmt_copy = ConvertSSA(stmt_copy);
    return stmt_copy;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if ((!is_updating) && (target_for == op)) {
      is_updating = true;
      is_then_case = true;
      Stmt then_case = StmtMutator::VisitStmt_(op);
      is_then_case = false;
      Stmt else_case = Stmt();
      if (target_if->else_case.defined()) {
        else_case = StmtMutator::VisitStmt_(op);
      }
      is_updating = false;
      return IfThenElse(target_if->condition, then_case, else_case);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    if (is_updating && (target_if == op)) {
	if (is_then_case) {
	  return StmtMutator::VisitStmt(op->then_case);
	} else if (op->else_case.defined()) {
	  return StmtMutator::VisitStmt(op->else_case);
	}
    }
    return StmtMutator::VisitStmt_(op);
  }

  const ForNode* target_for;
  const IfThenElseNode* target_if;

 private:

  bool is_updating{false};
  bool is_then_case{false};
  HoistCandidateSelector hoist_selector;

};

Stmt HoistIfThenElse(Stmt stmt) { return IfThenElseHoister().VisitAndMutate(stmt); }

namespace transform {

Pass HoistIfThenElse() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = HoistIfThenElse(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HoistIfThenElse", {});
}

TVM_REGISTER_GLOBAL("tir.transform.HoistIfThenElse").set_body_typed(HoistIfThenElse);

}  // namespace transform


}  // namespace tir
}  // namespace tvm

