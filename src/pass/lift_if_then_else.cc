#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <tvm/api_registry.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "../arithmetic/int_set.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using LifterMap = std::unordered_map<const Node*, std::vector<Stmt>>;
using VarMap = std::unordered_map<const Node*, std::unordered_set<const Node*>>;

class IfThenElseLifter : public IRMutator {
 public:
  Stmt VisitAndMutate(const Stmt& stmt) {
    GenerateInternalData(stmt);
    return PostOrderMutate(stmt);
  }

  Stmt PostOrderMutate(const Stmt& stmt) {
    PackedFunc replace_top_for = PackedFunc(
      [&](TVMArgs args, TVMRetValue *ret){
        const NodeRef& current_for = args[0];
        if (current_for.as<For>()) {
          const For* for_node = current_for.as<For>();
          if (top_for_map_.count(for_node->loop_var.get())) {
            std::vector<Stmt> new_if_list;
            for (const Stmt& if_stmt : top_for_map_[for_node->loop_var.get()]) {
              new_if_list.emplace_back(LiftIf(if_stmt));
            }

            const IfThenElse* next_if_node;
            const IfThenElse* current_if_node = new_if_list.back().as<IfThenElse>();
            Stmt new_for = Stmt();
            for (size_t i = new_if_list.size() - 1; i > 0; --i) {
              const Stmt current_if_stmt = IfThenElse::make(current_if_node->condition,
                                                            current_if_node->then_case,
                                                            current_if_node->else_case);
              next_if_node = new_if_list[i - 1].as<IfThenElse>();
              new_for = IfThenElse::make(next_if_node->condition, current_if_stmt, next_if_node->else_case);
              current_if_node = new_for.as<IfThenElse>();
            }

            if (!new_for.get()) {
              const IfThenElse* first_if_node = new_if_list[0].as<IfThenElse>();
              new_for = IfThenElse::make(first_if_node->condition,
                                         first_if_node->then_case,
                                         first_if_node->else_case);
            }
            *ret = new_for;
          }
        }
      });
    return IRTransform(stmt, nullptr, replace_top_for, {Expr("For")});
  }

 private:
  void GenerateInternalData(const Stmt& stmt);
  size_t GetActualFor(const Stmt& for_stmt, const Stmt& if_stmt);
  Stmt LiftIf(const Stmt& if_stmt);

  LifterMap if2for_map_;
  LifterMap top_for_map_;
  LifterMap for_tracking_map_;
  LifterMap for2if_map_;
  std::vector<Stmt> ordered_for_list_;
  VarMap cond_var_map_;

};

// Check whether a given IfThenElse stmt is the first one appearing in a For stmt.
bool is_first_if(const Stmt& for_stmt, const Stmt& if_stmt) {
  std::vector<size_t> if_hash_list;

  PostOrderVisit(for_stmt.as<For>()->body, [&](const NodeRef& node) {
    if (node.as<IfThenElse>()) {
      if_hash_list.push_back(node.hash());
    }
  });
  return if_hash_list.empty() ? false : if_stmt.hash() == if_hash_list.back();
}

// Update upper level for loop when current for loop is modified.
Stmt update_for(const Stmt& parent_for_stmt, const Stmt& new_if_stmt) {
  std::vector<size_t> for_hash_list;

  PostOrderVisit(parent_for_stmt.as<For>()->body, [&](const NodeRef& node) {
    if (node.as<For>()) {
      for_hash_list.push_back(node.hash());
    }
  });

  PackedFunc replace_target_for = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const NodeRef& current_for = args[0];
      if (current_for.hash() == for_hash_list.back()) {
        *ret = new_if_stmt;
      }
    });

  return IRTransform(parent_for_stmt, nullptr, replace_target_for, {Expr("For")});
}

// Remove If statement from a for statement
std::pair<Stmt, Stmt> RemoveIf(const Stmt& for_stmt, const Stmt& if_stmt) {
  const For* for_node = for_stmt.as<For>();
  const Stmt make_for = For::make(for_node->loop_var, for_node->min, for_node->extent,
                                  for_node->for_type, for_node->device_api, for_node->body);

  Stmt then_for;
  Stmt else_for;

  PackedFunc replace_then_case = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const NodeRef& node  = args[0];
      if (node == if_stmt) {
        *ret = node.as<IfThenElse>()->then_case;
      }
    });

  PackedFunc replace_else_case = PackedFunc(
    [&](TVMArgs args, TVMRetValue *ret){
      const NodeRef& node  = args[0];
      if (node == if_stmt) {
        *ret = node.as<IfThenElse>()->else_case;
      }
    });

  then_for = IRTransform(make_for, nullptr, replace_then_case, {Expr("IfThenElse")});
  if (if_stmt.as<IfThenElse>()->else_case) {
    else_for = IRTransform(make_for, nullptr, replace_else_case, {Expr("IfThenElse")});
  }

  return std::make_pair(then_for, else_for);
}

void IfThenElseLifter::GenerateInternalData(const Stmt& stmt) {
  std::unordered_map<const Node*, Stmt> if_position_map;
  std::unordered_set<const Node*> top_for_var_set;

  PostOrderVisit(stmt, [&](const NodeRef& node){
    const For* for_node = node.as<For>();
    if (for_node) {
      std::queue<Stmt> tracker;
      tracker.push(for_node->body);
      Stmt for_stmt = Downcast<Stmt, NodeRef>(node);
      for2if_map_.insert({for_stmt.get(), std::vector<Stmt>()});
      while(!tracker.empty()) {
        Stmt head = tracker.front();
        tracker.pop();
        if (head->is_type<For>()) {
          for (const auto& if_stmt : for2if_map_.at(head.get())) {
            for2if_map_[for_stmt.get()].push_back(if_stmt);
          }
        } else if (head->is_type<AttrStmt>()) {
          const AttrStmt* attr_node = head.as<AttrStmt>();
          tracker.push(attr_node->body);
        } else if (head->is_type<IfThenElse>()) {
          for2if_map_[for_stmt.get()].push_back(head);
          const IfThenElse* if_node = head.as<IfThenElse>();
          tracker.push(if_node->then_case);
          if (if_node->else_case) {
            tracker.push(if_node->else_case);
          }

          if (!cond_var_map_.count(head.get())) {
            std::unordered_set<const Node*> new_var_set;
            cond_var_map_.insert({head.get(), new_var_set});
            PostOrderVisit(if_node->condition, [&](const NodeRef& var) {
              if (var.as<Variable>()) {
                cond_var_map_[head.get()].insert(var.get());
              }
            });
          }
        } else {
          continue;
        }
      }
      ordered_for_list_.emplace_back(Downcast<Stmt, NodeRef>(node));
    }
  });


  for (const Stmt& for_stmt : ordered_for_list_) {
    std::vector<Stmt> if_list = for2if_map_[for_stmt.get()];
    top_for_map_.insert({for_stmt.as<For>()->loop_var.get(), if_list});
    for (const Stmt& if_stmt : if_list) {
      if (!if2for_map_.count(if_stmt.get())) {
        std::vector<Stmt> new_for_list;
        if2for_map_.insert({if_stmt.get(), new_for_list});
      }
      if2for_map_[if_stmt.get()].push_back(for_stmt);
    }
  }

  for (const auto& item : if2for_map_) {
    Stmt top_for;
    const Node* if_stmt = item.first;
    std::vector<Stmt> for_list = item.second;
    for (size_t i = 0; i < for_list.size(); ++i) {
      const Stmt& for_stmt = for_list.at(i);
      std::vector<Stmt> new_for_list{for_stmt};
      for_tracking_map_.insert({for_stmt.get(), new_for_list});
      if (cond_var_map_[if_stmt]
        .count(for_stmt.as<For>()->loop_var.get())) {
        std::vector<Stmt> updated_for_list(for_list.begin(), for_list.begin() + i);
        if2for_map_[if_stmt] = updated_for_list;
        break;
      } else {
        top_for = for_stmt;
      }
    }
    if (top_for.as<For>()) {
        if_position_map.insert({if_stmt, top_for});
    }
  }

  for ( const auto& item : if_position_map) {
    top_for_var_set.insert(item.second.as<For>()->loop_var.get());
  }

  std::vector<const Node*> removed_for_var_list;
  for (const auto& item : top_for_map_) {
    const Node* top_for_var = item.first;
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
      top_for_map_[top_for_var] = actual_if_list;
    }
  }
  for (const Node* top_for_var : removed_for_var_list) {
    top_for_map_.erase(top_for_var);
  }
}

size_t IfThenElseLifter::GetActualFor(const Stmt& for_stmt, const Stmt& if_stmt) {
  std::vector<Stmt> tracked_for_list = for_tracking_map_[for_stmt.get()];
  size_t actual_idx = 0;
  for (size_t i = 0; i < tracked_for_list.size(); ++i) {
    const Stmt& current_for = tracked_for_list.at(tracked_for_list.size() - 1 - i);
    if (is_first_if(current_for, if_stmt)) {
      actual_idx = tracked_for_list.size() - 1 - i;
      break;
    }
  }
  return actual_idx;
}

Stmt IfThenElseLifter::LiftIf(const Stmt& if_stmt) {
  Stmt new_if = if_stmt;

  for (size_t i = 0; i < if2for_map_[if_stmt.get()].size(); ++i) {
    const Stmt& for_stmt = if2for_map_[if_stmt.get()].at(i);
    size_t actual_for_idx = GetActualFor(for_stmt, new_if);
    const Stmt& actual_for_node = for_tracking_map_[for_stmt.get()].at(actual_for_idx);
    auto generated_for_pair = RemoveIf(actual_for_node, new_if);
    const Stmt& then_for = generated_for_pair.first;
    const Stmt& else_for = generated_for_pair.second;;
    for_tracking_map_[for_stmt.get()].at(actual_for_idx) = then_for;

    if (else_for.get()) {
      for_tracking_map_[for_stmt.get()].push_back(else_for);
    }
    new_if = IfThenElse::make(new_if.as<IfThenElse>()->condition, then_for, else_for);
    if (i < if2for_map_[if_stmt.get()].size() - 1) {
      const Stmt& original_next_for = if2for_map_[if_stmt.get()].at(i + 1);
      const Stmt& actual_next_for = for_tracking_map_[original_next_for.get()].at(actual_for_idx);
      Stmt update_for_stmt = update_for(actual_next_for, new_if);

      for_tracking_map_[original_next_for.get()].at(actual_for_idx) = update_for_stmt;
    }
  }
  return new_if;
}


Stmt LiftIfThenElse(Stmt stmt) {
  return IfThenElseLifter().VisitAndMutate(stmt);
}

}  // namespace ir
}  // namespace tvm