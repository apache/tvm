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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {

using HoistMap = std::unordered_map<const Object*, std::vector<Stmt>>;
using VarMap = std::unordered_map<const Object*, std::unordered_set<const Object*>>;

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
class IfThenElseHoist {
 public:
  Stmt VisitAndMutate(const Stmt& stmt) {
    SelectCandidates(stmt);
    LocateTopFor();
    return PostOrderMutate(stmt);
  }

 private:
  void SelectCandidates(const Stmt& stmt);
  void LocateTopFor();
  Stmt PostOrderMutate(const Stmt& stmt);
  size_t GetUpdatedFor(const Stmt& for_stmt, const Stmt& if_stmt);
  Stmt HoistIf(const Stmt& if_stmt);

  // Map of all For nodes to all child IfThenElse nodes.
  HoistMap for2if_map_;
  // Map of all IfThenElse nodes to all For nodes which are loop invariant.
  HoistMap if2for_map_;
  // Map of highest loop invariant For to child IfThenElse.
  HoistMap top_for_var_map_;
  // Map of original For to list of update For nodes.
  HoistMap for_tracking_map_;
  // Map of all IfThenElse nodes to condition variable nodes.
  VarMap cond_var_map_;
  // List of For nodes added in post order DFS visiting.
  std::vector<Stmt> ordered_for_list_;
};

// Check whether a given IfThenElse stmt is the first one appearing
// in a For stmt.
bool is_first_if(const Stmt& for_stmt, const Stmt& if_stmt) {
  std::vector<const Object*> if_node_list;
  const ForNode* for_node = for_stmt.as<ForNode>();
  CHECK(for_node);
  CHECK(if_stmt.as<IfThenElseNode>());

  PostOrderVisit(for_node->body, [&](const ObjectRef& node) {
    if (node.as<IfThenElseNode>()) {
      if_node_list.push_back(node.get());
    }
  });
  return if_node_list.empty() ? false : if_stmt.get() == if_node_list.back();
}

// Update upper level For node when current For node is modified.
// With this function we only need to visit and mutate top level For node
// in the main VisitAndMutate function.
Stmt update_for(const Stmt& parent_for_stmt, const Stmt& new_if_stmt) {
  const Object* top_for_node;
  const ForNode* parent_for_node = parent_for_stmt.as<ForNode>();
  CHECK(parent_for_node);
  CHECK(new_if_stmt.as<IfThenElseNode>());

  PostOrderVisit(parent_for_node->body, [&](const ObjectRef& node) {
    if (node.as<ForNode>()) {
      top_for_node = node.get();
    }
  });

  PackedFunc replace_target_for = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const ObjectRef& current_for = args[0];
      if (current_for.get() == top_for_node) {
        *ret = new_if_stmt;
      }
    });

  return IRTransform(parent_for_stmt, nullptr, replace_target_for, Array<String>{"For"});
}

// Remove IfThenElse node from a For node.
// A pair of For nodes will be generated.
std::pair<Stmt, Stmt> RemoveIf(const Stmt& for_stmt, const Stmt& if_stmt) {
  Stmt then_for;
  Stmt else_for;
  CHECK(if_stmt.as<IfThenElseNode>());

  PackedFunc replace_then_case = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const ObjectRef& node  = args[0];
      if (node == if_stmt) {
        *ret = node.as<IfThenElseNode>()->then_case;
      }
    });

  PackedFunc replace_else_case = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const ObjectRef& node  = args[0];
      if (node == if_stmt) {
        *ret = node.as<IfThenElseNode>()->else_case;
      }
    });

  then_for = IRTransform(for_stmt, nullptr, replace_then_case, Array<String>{"IfThenElse"});
  if (if_stmt.as<IfThenElseNode>()->else_case.defined()) {
    else_for = IRTransform(for_stmt, nullptr, replace_else_case, Array<String>{"IfThenElse"});
  }

  return std::make_pair(then_for, else_for);
}

// Locate all For nodes and capture child IfThenElse nodes.
void IfThenElseHoist::SelectCandidates(const Stmt& stmt) {
  PostOrderVisit(stmt, [&](const ObjectRef& node){
    const ForNode* for_node = node.as<ForNode>();
    if (!for_node) return;

    std::queue<Stmt> tracker;
    tracker.push(for_node->body);
    Stmt for_stmt = Downcast<Stmt, ObjectRef>(node);
    for2if_map_.insert({for_stmt.get(), std::vector<Stmt>()});
    while (!tracker.empty()) {
      Stmt head = tracker.front();
      tracker.pop();
      if (head->IsInstance<ForNode>()) {
        for (const auto& if_stmt : for2if_map_.at(head.get())) {
          for2if_map_[for_stmt.get()].push_back(if_stmt);
        }
      } else if (head->IsInstance<AttrStmtNode>()) {
        const AttrStmtNode* attr_node = head.as<AttrStmtNode>();
        tracker.push(attr_node->body);
      } else if (head->IsInstance<IfThenElseNode>()) {
        for2if_map_[for_stmt.get()].push_back(head);
        const IfThenElseNode* if_node = head.as<IfThenElseNode>();
        tracker.push(if_node->then_case);
        if (if_node->else_case.defined()) {
          tracker.push(if_node->else_case);
        }

        // Record condition variables.
        if (!cond_var_map_.count(head.get())) {
          std::unordered_set<const Object*> new_var_set;
          cond_var_map_.insert({head.get(), new_var_set});
          PostOrderVisit(if_node->condition, [&](const ObjectRef& cond_node) {
            if (cond_node.as<VarNode>()) {
              cond_var_map_[head.get()].insert(cond_node.get());
            }
          });
        }
      } else {
        continue;
      }
    }
    ordered_for_list_.emplace_back(Downcast<Stmt, ObjectRef>(node));
  });
}

// For each IfThenElse node, find the highest For node which
// meets loop invariant condition.
void IfThenElseHoist::LocateTopFor() {
  std::unordered_map<const Object*, Stmt> if_position_map;
  std::unordered_set<const Object*> top_for_var_set;

  // Create IfThenElse -> For map.
  for (const Stmt& for_stmt : ordered_for_list_) {
    std::vector<Stmt> if_list = for2if_map_[for_stmt.get()];
    const ForNode* for_node = for_stmt.as<ForNode>();
    CHECK(for_node);
    top_for_var_map_.insert({for_node->loop_var.get(), if_list});
    for (const Stmt& if_stmt : if_list) {
      const Object* if_node = if_stmt.get();
      if2for_map_[if_node].push_back(for_stmt);
    }
  }

  // Locate the highest For node which is loop invariant.
  for (const auto& item : if2for_map_) {
    Stmt top_for;
    const Object* if_stmt = item.first;
    std::vector<Stmt> for_list = item.second;
    for (size_t i = 0; i < for_list.size(); ++i) {
      const Stmt& for_stmt = for_list.at(i);
      const ForNode* for_node = for_stmt.as<ForNode>();
      CHECK(for_node);
      std::vector<Stmt> new_for_list{for_stmt};
      for_tracking_map_.insert({for_stmt.get(), new_for_list});
      if (cond_var_map_[if_stmt]
        .count(for_node->loop_var.get())) {
        std::vector<Stmt> updated_for_list(for_list.begin(),
                                           for_list.begin() + i);
        if2for_map_[if_stmt] = updated_for_list;
        break;
      } else {
        top_for = for_stmt;
      }
    }
    if (top_for.as<ForNode>()) {
      if_position_map.insert({if_stmt, top_for});
    }
  }

  for (const auto& item : if_position_map) {
    top_for_var_set.insert(item.second.as<ForNode>()->loop_var.get());
  }

  std::vector<const Object*> removed_for_var_list;
  for (const auto& item : top_for_var_map_) {
    const Object* top_for_var = item.first;
    std::vector<Stmt> if_list = item.second;
    if (!top_for_var_set.count(top_for_var)) {
      removed_for_var_list.push_back(top_for_var);
    } else {
      std::vector<Stmt> actual_if_list;
      for (const Stmt& if_stmt : if_list) {
        if (if_position_map.count(if_stmt.get())) {
          actual_if_list.push_back(if_stmt);
        }
      }
      top_for_var_map_[top_for_var] = actual_if_list;
    }
  }
  for (const Object* top_for_var : removed_for_var_list) {
    top_for_var_map_.erase(top_for_var);
  }
}

// When we try to mutate a For node, some child For nodes can have already
// been mutated. This function is to get the updated For node and further
// hoisting can be done based on this new node.
// We keep all For nodes tracing in for_tracking_map_. When we get a
// hoisted IfThenElse, we match it with tracing For nodes to pick
// the updated one.
size_t IfThenElseHoist::GetUpdatedFor(const Stmt& for_stmt,
                                       const Stmt& if_stmt) {
  std::vector<Stmt> tracked_for_list = for_tracking_map_[for_stmt.get()];
  size_t updated_for_idx = 0;
  for (size_t i = 0; i < tracked_for_list.size(); ++i) {
    const Stmt& current_for =
      tracked_for_list.at(tracked_for_list.size() - 1 - i);
    if (is_first_if(current_for, if_stmt)) {
      updated_for_idx = tracked_for_list.size() - 1 - i;
      break;
    }
  }
  return updated_for_idx;
}

// Hoist an IfThenElse node as high as possible.
// This function iterates on all candidate For nodes. For each For node,
// it first removes IfThenElse nodes. Then it generates a new IfThenElse
// node using mutated For nodes.
Stmt IfThenElseHoist::HoistIf(const Stmt& if_stmt) {
  Stmt new_if = if_stmt;

  for (size_t i = 0; i < if2for_map_[if_stmt.get()].size(); ++i) {
    const Stmt& for_stmt = if2for_map_[if_stmt.get()].at(i);
    size_t updated_for_idx = GetUpdatedFor(for_stmt, new_if);
    const Stmt& updated_for_node =
      for_tracking_map_[for_stmt.get()].at(updated_for_idx);
    auto generated_for_pair = RemoveIf(updated_for_node, new_if);
    const Stmt& then_for = generated_for_pair.first;
    const Stmt& else_for = generated_for_pair.second;;
    for_tracking_map_[for_stmt.get()].at(updated_for_idx) = then_for;

    if (else_for.get()) {
      for_tracking_map_[for_stmt.get()].push_back(else_for);
    }

    const IfThenElseNode* new_if_node = new_if.as<IfThenElseNode>();
    CHECK(new_if_node);
    new_if = IfThenElseNode::make(new_if_node->condition, then_for, else_for);
    if (i < if2for_map_[if_stmt.get()].size() - 1) {
      const Stmt& original_next_for = if2for_map_[if_stmt.get()].at(i + 1);
      const Stmt& actual_next_for =
        for_tracking_map_[original_next_for.get()].at(updated_for_idx);
      Stmt update_for_stmt = update_for(actual_next_for, new_if);

      for_tracking_map_[original_next_for.get()].
        at(updated_for_idx) = update_for_stmt;
    }
  }
  return new_if;
}

// Mutate For nodes in post order DFS manner.
Stmt IfThenElseHoist::PostOrderMutate(const Stmt& stmt) {
  PackedFunc replace_top_for = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const ObjectRef& current_for = args[0];
      const ForNode* for_node = current_for.as<ForNode>();
      if (!for_node) return;

      if (top_for_var_map_.count(for_node->loop_var.get())) {
        std::vector<Stmt> new_if_list;
        for (const Stmt& if_stmt :
          top_for_var_map_[for_node->loop_var.get()]) {
          new_if_list.emplace_back(HoistIf(if_stmt));
        }

        const IfThenElseNode* next_if_node;
        const IfThenElseNode* current_if_node =
          new_if_list.back().as<IfThenElseNode>();
        Stmt new_for = Stmt();
        for (size_t i = new_if_list.size() - 1; i > 0; --i) {
          CHECK(current_if_node);
          const Stmt current_if_stmt =
            IfThenElseNode::make(current_if_node->condition,
                             current_if_node->then_case,
                             current_if_node->else_case);
          next_if_node = new_if_list[i - 1].as<IfThenElseNode>();
          CHECK(next_if_node);
          new_for = IfThenElseNode::make(next_if_node->condition, current_if_stmt,
                                     next_if_node->else_case);
          current_if_node = new_for.as<IfThenElseNode>();
        }

        if (!new_for.get()) {
          const IfThenElseNode* first_if_node = new_if_list[0].as<IfThenElseNode>();
          CHECK(first_if_node);
          new_for = IfThenElseNode::make(first_if_node->condition,
                                     first_if_node->then_case,
                                     first_if_node->else_case);
        }
        *ret = new_for;
      }
    });
  return IRTransform(stmt, nullptr, replace_top_for, Array<String>{"For"});
}

Stmt HoistIfThenElse(Stmt stmt) {
  return IfThenElseHoist().VisitAndMutate(stmt);
}


TVM_REGISTER_GLOBAL("testing.HoistIfThenElse")
.set_body_typed(HoistIfThenElse);

}  // namespace tir
}  // namespace tvm
