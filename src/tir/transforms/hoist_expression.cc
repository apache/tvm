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
 * \file hoist_expression.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../arith/interval_set.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

enum class HoistedConditionals : int {
  kNone = 0,
  kIfElseStmt = (1 << 0),
  kIfElseExpr = (1 << 1),
  kBooleanExpression = (1 << 2),
  kUsingBlockVar = (1 << 3),
};

enum class HoistedLetBindings : int {
  kNone = 0,
  kRequiredByCondition = (1 << 0),
  kLetStmt = (1 << 1),
  kLetExpr = (1 << 2),
};

struct HoistExpressionConfigNode : public tvm::AttrsNode<HoistExpressionConfigNode> {
  int hoisted_conditionals;
  int hoisted_let_bindings;

  TVM_DECLARE_ATTRS(HoistExpressionConfigNode, "tir.transform.HoistExpressionConfig") {
    TVM_ATTR_FIELD(hoisted_conditionals)
        .describe("Bitflags for the types of boolean expressions to hoist")
        .set_default(static_cast<int>(HoistedConditionals::kIfElseStmt) |
                     static_cast<int>(HoistedConditionals::kIfElseExpr) |
                     static_cast<int>(HoistedConditionals::kBooleanExpression));
    TVM_ATTR_FIELD(hoisted_let_bindings)
        .describe("Bitflags for the types of let bindings to hoist")
        .set_default(static_cast<int>(HoistedLetBindings::kRequiredByCondition) |
                     static_cast<int>(HoistedLetBindings::kLetStmt) |
                     static_cast<int>(HoistedLetBindings::kLetExpr));
  }

  bool FlagSet(HoistedConditionals flag) const {
    return static_cast<int>(flag) & hoisted_conditionals;
  }
  bool FlagSet(HoistedLetBindings flag) const {
    return static_cast<int>(flag) & hoisted_let_bindings;
  }
};

class HoistExpressionConfig : public Attrs {
 public:
  HoistExpressionConfig(int hoisted_conditionals, int hoisted_let_bindings) {
    auto node = make_object<HoistExpressionConfigNode>();
    node->hoisted_conditionals = hoisted_conditionals;
    node->hoisted_let_bindings = hoisted_let_bindings;
    data_ = std::move(node);
  }
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(HoistExpressionConfig, Attrs,
                                            HoistExpressionConfigNode);
};

TVM_REGISTER_NODE_TYPE(HoistExpressionConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.HoistExpression", HoistExpressionConfig);

struct HoistIfThenElseConfigNode : public tvm::AttrsNode<HoistIfThenElseConfigNode> {
  // Would like to replace the typo here from "hosting" to "hoisting",
  // but that may impact user configurations.
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

class HoistInfoCollector : public StmtExprVisitor {
 public:
  struct ConditionInfo {
    ConditionInfo(PrimExpr condition, HoistedConditionals hoist_from, bool uses_block_var,
                  std::unordered_set<const VarNode*> required_let_bindings, bool generate_else_case)
        : condition(condition),
          hoist_from(hoist_from),
          uses_block_var(uses_block_var),
          required_let_bindings(required_let_bindings),
          generate_else_case(generate_else_case) {}
    PrimExpr condition;
    HoistedConditionals hoist_from;
    bool uses_block_var;
    std::unordered_set<const VarNode*> required_let_bindings;
    bool generate_else_case;

    bool IsEnabled(const HoistExpressionConfig& config) const {
      bool valid_source = config->FlagSet(hoist_from);

      bool all_required_bindings_are_hoisted =
          required_let_bindings.empty() ||
          config->FlagSet(HoistedLetBindings::kRequiredByCondition) ||
          config->FlagSet(HoistedLetBindings::kLetStmt);

      bool valid_block_var_usage =
          config->FlagSet(HoistedConditionals::kUsingBlockVar) || !uses_block_var;
      return valid_source && all_required_bindings_are_hoisted && valid_block_var_usage;
    }
  };

  struct LetBindingInfo {
    LetBindingInfo(Var var, PrimExpr value, HoistedLetBindings hoist_from)
        : var(var), value(value), hoist_from(hoist_from) {}
    Var var;
    PrimExpr value;
    HoistedLetBindings hoist_from;

    bool IsEnabled(const HoistExpressionConfig& config) const {
      return config->FlagSet(hoist_from);
    }
  };

  struct HoistInfo {
    // The loop variable
    Var loop_var;

    // The For or AttrStmt that defines the loop var.
    Stmt loop_def;

    // Bindings defined in LetStmt inside the for-loop whose value
    // does not depend on the loop variable.  These can be hoisted
    // outside this for-loop.
    std::vector<LetBindingInfo> let_bindings;

    // Conditions evaluated inside the for-loop whose value does not
    // depend on the loop variable.  These can be hoisted outside this
    // for loop.  These may depend on the let_bindings.
    std::vector<ConditionInfo> conditions;

    // Only conditions that impact the entire body of the loop
    // hoisted.  Conditionals may not be hoisted from inside a
    // sequential node to outside.
    bool reached_sequential_node{false};

    // True if the loop variable representing a block variable
    // (e.g. blockIdx.x, threadIdx.x), false otherwise.
    bool IsBlockVariable() const { return !loop_def.as<ForNode>(); }
  };

  static std::vector<HoistInfo> Collect(Stmt stmt, HoistExpressionConfig config) {
    HoistInfoCollector collector(config);
    collector(stmt);
    return collector.completed_loops;
  }

 private:
  using Parent = StmtExprVisitor;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  explicit HoistInfoCollector(HoistExpressionConfig config) : config(config) {}

  void AttemptHoistConditional(PrimExpr cond, HoistedConditionals hoist_from,
                               bool generate_else_block = true) {
    if (SideEffect(cond) > CallEffectKind::kPure) {
      return;
    }
    if (auto info = FindHoistDestination(cond)) {
      if (!info->reached_sequential_node) {
        // Record whether this conditional uses any block variables.
        bool uses_block_var = active_block_vars.size() && UsesVar(cond, [&](const VarNode* var) {
                                return active_block_vars.count(var);
                              });

        std::unordered_set<const VarNode*> let_bindings_used;

        for (Var var : UndefinedVars(cond)) {
          auto it = let_var_to_let_vars.find(var.get());
          if (it != let_var_to_let_vars.end()) {
            let_bindings_used.insert(it->first);
            for (auto used : it->second) {
              let_bindings_used.insert(used);
            }
          }
        }
        info->conditions.push_back(ConditionInfo(cond, hoist_from, uses_block_var,
                                                 let_bindings_used, generate_else_block));
      }
    }
  }

  void VisitExpr_(const AndNode* op) final {
    AttemptHoistConditional(op->a, HoistedConditionals::kBooleanExpression);
    AttemptHoistConditional(op->b, HoistedConditionals::kBooleanExpression);
    Parent::VisitExpr_(op);
  }

  void VisitExpr_(const OrNode* op) final {
    AttemptHoistConditional(op->a, HoistedConditionals::kBooleanExpression);
    AttemptHoistConditional(op->b, HoistedConditionals::kBooleanExpression);
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    active_loops.push_back({op->loop_var, GetRef<Stmt>(op)});
    active_loop_vars.insert(op->loop_var.get());

    Parent::VisitStmt_(op);
    completed_loops.push_back(active_loops.back());

    active_loop_vars.erase(op->loop_var.get());
    active_loops.pop_back();
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    Var var;
    if (const auto* node_iter_var = op->node.as<IterVarNode>()) {
      var = node_iter_var->var;
    } else if (auto opt = op->node.as<Var>()) {
      var = opt.value();
    } else {
      return Parent::VisitStmt_(op);
    }

    active_block_vars.insert(var.get());
    active_loop_vars.insert(var.get());
    active_loops.push_back({var, GetRef<Stmt>(op)});

    Parent::VisitStmt_(op);

    completed_loops.push_back(active_loops.back());
    active_loops.pop_back();

    active_loop_vars.erase(var.get());
    active_block_vars.erase(var.get());
  }

  void VisitBinding(Var var, PrimExpr value, HoistedLetBindings hoist_from) {
    ICHECK_EQ(let_var_to_loop_vars.count(var.get()), 0)
        << "Multiple nested definitions of variable " << var;
    ICHECK_EQ(let_var_to_let_vars.count(var.get()), 0)
        << "Multiple nested definitions of variable " << var;

    if (auto info = FindHoistDestination(value)) {
      if (!info->reached_sequential_node) {
        info->let_bindings.push_back(LetBindingInfo(var, value, hoist_from));
      }
    }

    // Walk through the loop binding
    std::unordered_set<const VarNode*> loop_vars_used;
    std::unordered_set<const VarNode*> let_bindings_used;
    for (Var var : UndefinedVars(value)) {
      if (active_loop_vars.count(var.get())) {
        loop_vars_used.insert(var.get());
      } else {
        auto it = let_var_to_loop_vars.find(var.get());
        if (it != let_var_to_loop_vars.end()) {
          for (const VarNode* used : it->second) {
            loop_vars_used.insert(used);
          }
        }
      }

      auto it = let_var_to_let_vars.find(var.get());
      if (it != let_var_to_let_vars.end()) {
        let_bindings_used.insert(it->first);
        for (const VarNode* used : it->second) {
          let_bindings_used.insert(used);
        }
      }
    }

    let_var_to_loop_vars[var.get()] = std::move(loop_vars_used);
    let_var_to_let_vars[var.get()] = std::move(let_bindings_used);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    VisitBinding(op->var, op->value, HoistedLetBindings::kLetStmt);

    Parent::VisitStmt_(op);

    let_var_to_loop_vars.erase(op->var.get());
    let_var_to_let_vars.erase(op->var.get());
  }

  void VisitExpr_(const LetNode* op) final {
    VisitBinding(op->var, op->value, HoistedLetBindings::kLetExpr);

    Parent::VisitExpr_(op);

    let_var_to_loop_vars.erase(op->var.get());
    let_var_to_let_vars.erase(op->var.get());
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    AttemptHoistConditional(op->condition, HoistedConditionals::kIfElseStmt,
                            op->else_case.defined());
    Parent::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr cond = op->args[0];
      AttemptHoistConditional(cond, HoistedConditionals::kIfElseExpr);
    }
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    if (active_loops.size()) {
      active_loops.back().reached_sequential_node = true;
    }
    Parent::VisitStmt_(op);
  }

  // Find the loop above which this expression could be hoisted.  If
  // nullptr, the expression cannot be hoisted.
  HoistInfo* FindHoistDestination(PrimExpr expr) {
    // Cannot hoist above a loop if we aren't already in a loop.
    if (active_loops.empty()) {
      return nullptr;
    }

    for (auto it = active_loops.rbegin(); it != active_loops.rend(); it++) {
      Var loop_var = it->loop_var;
      bool uses_loop_var = UsesVar(expr, [&](const VarNode* var) -> bool {
        if (var == loop_var.get()) {
          return true;
        }

        auto it = let_var_to_loop_vars.find(var);
        if (it == let_var_to_loop_vars.end()) {
          return false;
        }

        return it->second.count(loop_var.get());
      });

      bool is_disabled_hoist_across_block_var =
          !config->FlagSet(HoistedConditionals::kUsingBlockVar) && it->IsBlockVariable();

      if (it->reached_sequential_node || uses_loop_var || is_disabled_hoist_across_block_var) {
        if (it == active_loops.rbegin()) {
          // Cannot hoist beyond the innermost loop iterator.
          return nullptr;
        } else {
          // Hoist to just below the loop iterator that is required.
          it--;
          return &(*it);
        }
      }
    }

    // If no loop variables are used, can hoist above the outermost
    // loop.
    return &active_loops.front();
  }

  // The user-provided config describing which expressions should be
  // hoisted.
  HoistExpressionConfig config;

  // Current thread_extent bindings of block variables.
  std::unordered_set<const VarNode*> active_block_vars;

  // An ordered list of loops that are currently being visited.
  std::vector<HoistInfo> active_loops;

  // Loops that have already been visited
  std::vector<HoistInfo> completed_loops;

  // Map from a bound variable to the loop variables it depends on.
  // Includes indirect usage.
  std::unordered_map<const VarNode*, std::unordered_set<const VarNode*>> let_var_to_loop_vars;

  // Map from a bound variable to the other let bindings it depends on.
  // Includes indirect usage.
  std::unordered_map<const VarNode*, std::unordered_set<const VarNode*>> let_var_to_let_vars;

  // Lookup table for the currently active loops.
  std::unordered_set<const VarNode*> active_loop_vars;
};

class ExpressionHoister : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Hoist(Stmt stmt, HoistExpressionConfig config) {
    auto loop_info = HoistInfoCollector::Collect(stmt, config);

    arith::Analyzer analyzer;
    ExpressionHoister hoister(std::move(loop_info), config, &analyzer);
    stmt = hoister(std::move(stmt));
    stmt = ConvertSSA(std::move(stmt));
    return stmt;
  }

 private:
  using Parent = arith::IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  explicit ExpressionHoister(std::vector<HoistInfoCollector::HoistInfo> loop_info,
                             HoistExpressionConfig config, arith::Analyzer* analyzer)
      : Parent(analyzer), config_(config) {
    for (auto& info : loop_info) {
      // Mark let bindings to use if they are enabled on their own.
      for (const auto& binding : info.let_bindings) {
        if (binding.IsEnabled(config)) {
          hoisted_let_bindings.insert(binding.var.get());
        }
      }

      // Or if they are required by a conditional
      if (config->FlagSet(HoistedLetBindings::kRequiredByCondition)) {
        for (const auto& conditional : info.conditions) {
          if (conditional.IsEnabled(config)) {
            for (const auto& var : conditional.required_let_bindings) {
              hoisted_let_bindings.insert(var);
            }
          }
        }
      }

      loop_info_lookup[info.loop_def.get()] = std::move(info);
    }
  }

  Stmt WrapHoistedStatements(Stmt stmt, const HoistInfoCollector::HoistInfo& info) {
    for (auto cond_it = info.conditions.rbegin(); cond_it != info.conditions.rend(); cond_it++) {
      if (cond_it->IsEnabled(config_)) {
        if (cond_it->generate_else_case) {
          stmt = IfThenElse(cond_it->condition, stmt, stmt);
        } else {
          stmt = IfThenElse(cond_it->condition, stmt);
        }
      }
    }
    for (auto let_it = info.let_bindings.rbegin(); let_it != info.let_bindings.rend(); let_it++) {
      if (hoisted_let_bindings.count(let_it->var.get())) {
        stmt = LetStmt(let_it->var, let_it->value, stmt);
      }
    }

    return stmt;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);

    auto it = loop_info_lookup.find(op);
    ICHECK(it != loop_info_lookup.end())
        << "Could not find pre-pass information for loop over " << op->loop_var;
    return WrapHoistedStatements(stmt, it->second);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);

    auto it = loop_info_lookup.find(op);
    if (it == loop_info_lookup.end()) {
      return stmt;
    } else {
      return WrapHoistedStatements(stmt, it->second);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (hoisted_let_bindings.count(op->var.get())) {
      return this->VisitStmt(op->body);
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    if (hoisted_let_bindings.count(op->var.get())) {
      return this->VisitExpr(op->body);
    } else {
      return Parent::VisitExpr_(op);
    }
  }

  HoistExpressionConfig config_;

  std::unordered_map<const StmtNode*, HoistInfoCollector::HoistInfo> loop_info_lookup;
  std::unordered_set<const VarNode*> hoisted_let_bindings;
};

Stmt HoistExpression(Stmt stmt, HoistExpressionConfig config) {
  return ExpressionHoister::Hoist(stmt, config);
}

namespace transform {

Pass HoistExpression() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<HoistExpressionConfig>("tir.HoistExpression");

    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<HoistExpressionConfig>();
    }
    n->body = ExpressionHoister::Hoist(std::move(n->body), cfg.value());
    return f;
  };
  auto insertion_pass = CreatePrimFuncPass(pass_func, 0, "tir.InsertHoistedExpression", {});

  return Sequential(
      {
          insertion_pass,
          Simplify(),
          RemoveNoOp(),
      },
      "tir.HoistExpression");
}

TVM_REGISTER_GLOBAL("tir.transform.HoistExpression").set_body_typed(HoistExpression);

Pass HoistIfThenElse() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<HoistIfThenElseConfig>("tir.HoistIfThenElse");

    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<HoistIfThenElseConfig>();
    }
    int block_var = static_cast<int>(cfg.value()->support_block_scope_hosting
                                         ? HoistedConditionals::kUsingBlockVar
                                         : HoistedConditionals::kNone);
    HoistExpressionConfig config(block_var | static_cast<int>(HoistedConditionals::kIfElseStmt),
                                 static_cast<int>(HoistedLetBindings::kNone));
    n->body = ExpressionHoister::Hoist(std::move(n->body), config);
    return f;
  };
  auto insertion_pass = CreatePrimFuncPass(pass_func, 0, "tir.InsertHoistIfThenElse", {});
  return Sequential(
      {
          insertion_pass,
          Simplify(),
          RemoveNoOp(),
      },
      "tir.HoistIfThenElse");
}

TVM_REGISTER_GLOBAL("tir.transform.HoistIfThenElse").set_body_typed(HoistIfThenElse);

Pass HoistIfThenElseBasic() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    HoistExpressionConfig config(static_cast<int>(HoistedConditionals::kIfElseStmt),
                                 static_cast<int>(HoistedLetBindings::kNone));
    n->body = ExpressionHoister::Hoist(std::move(n->body), config);
    return f;
  };
  auto insertion_pass = CreatePrimFuncPass(pass_func, 0, "tir.InsertHoistIfThenElseBasic", {});
  return Sequential(
      {
          insertion_pass,
          Simplify(),
          RemoveNoOp(),
      },
      "tir.HoistIfThenElseBasic");
}

TVM_REGISTER_GLOBAL("tir.transform.HoistIfThenElseBasic").set_body_typed(HoistIfThenElseBasic);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
