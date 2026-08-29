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
 * \file src/arith/z3_prover.cc
 * \brief Optional Z3 SMT solver backend for arith::Analyzer.
 *
 * The real implementation is compiled only when TVM_USE_Z3 is defined (set by
 * the USE_Z3 CMake option). Otherwise a conservative stub is compiled so the
 * C++ and Python APIs stay available without a Z3 dependency.
 */
#ifdef TVM_USE_Z3

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <algorithm>
#include <climits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/dtype.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "z3++.h"

namespace tvm::arith {

using namespace tirx;
using namespace ffi;

namespace {

struct Namespace {
  std::unordered_set<std::string> used_names;
  /// @brief Get a new name that is not used before
  /// This function is used to generate z3 variable names
  ///
  /// Z3 may deduplicate variables with the same name, which
  /// causes issues when different TVM variables are mapped to
  /// the same z3 variable.
  ///
  /// This function generates unique names by appending
  /// suffixes to the original expression string representation.
  ///
  /// such as : "x", "x$1", "x$2", ...
  std::string GetNewName(const PrimExpr& expr) {
    std::stringstream ss;
    ss << expr;
    auto name = ss.str();
    if (used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while (used_names.count(check_name)) {
      idx++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
};

}  // namespace

// Deprecated no-ops, kept only for callers of
// arith.EnterZ3ContextScope / arith.ExitZ3ContextScope (Z3ContextScope in
// Python). Shared per-compile Z3 contexts are gone: every materialized solver
// owns a private context (see Z3Prover::Impl::Materialize), which subsumes the
// per-compilation isolation these scopes provided. Remove together with the
// remaining call sites.
void EnterZ3ContextScope() {}

void ExitZ3ContextScope() {}

class Z3Prover::Impl : ExprFunctor<z3::expr(const Expr&)> {
 public:
  using Base = ExprFunctor<z3::expr(const Expr&)>;
  using Self = Z3Prover::Impl;

  AnalyzerObj* analyzer;
  /// @brief Z3 context owning every handle of this prover.
  // Null until the solver materializes; Materialize creates a context private
  // to this prover, which clones of a materialized prover then share so their
  // copied handles stay valid.
  std::shared_ptr<z3::context> ctx;

  /// @brief Z3 solver instance
  std::optional<z3::solver> solver;

  /// @brief Memoized PrimExpr -> slot in z3_pool_. Holds no Z3 handles, so
  /// its pointer-hashed bucket order cannot affect Z3 object lifetime.
  std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual> memo_;

  /// @brief Slots owning the memoized Z3 handles, plus a free-slot stack.
  /// Handles are created and released only at fixed points of the execution
  /// path (never in hash-bucket order), so Z3's AST-ID recycling -- and thus
  /// solver behavior under rlimit -- stays deterministic across processes.
  std::vector<std::optional<z3::expr>> z3_pool_;
  std::vector<size_t> free_slots_;

  void MemoPut(const PrimExpr& expr, const z3::expr& z3_expr) {
    auto [it, inserted] = memo_.emplace(expr, 0);
    if (!inserted) return;
    if (free_slots_.empty()) {
      it->second = z3_pool_.size();
      z3_pool_.emplace_back(z3_expr);
    } else {
      it->second = free_slots_.back();
      free_slots_.pop_back();
      z3_pool_[it->second] = z3_expr;
    }
  }

  void MemoErase(const PrimExpr& expr) {
    auto it = memo_.find(expr);
    if (it == memo_.end()) return;
    z3_pool_[it->second].reset();
    free_slots_.push_back(it->second);
    memo_.erase(it);
  }

  const z3::expr* MemoGet(const PrimExpr& expr) const {
    auto it = memo_.find(expr);
    return it == memo_.end() ? nullptr : &*z3_pool_[it->second];
  }

  /// @brief Namespace for variable naming
  Namespace ns;

  /// @brief Timeout in milliseconds
  unsigned timeout_ms{UINT_MAX};

  /// @brief Max steps
  unsigned rlimit{UINT_MAX};

  /// @brief Create a z3 solver with custom options
  static z3::solver CreateSolver(z3::context& ctx) {
    z3::solver result(ctx);
    // here we disable model generation to speed up the solving process
    result.set("model", false);
    // ensure determinstic behavior
    result.set("random_seed", (unsigned)42);
    return result;
  }

  Impl(AnalyzerObj* parent) : analyzer(parent) {
    // The solver (and its context) is created lazily (see Materialize).
    // Analyzers are constructed in large numbers as cheap scratch objects,
    // and only a tiny fraction ever reaches the Z3 fallback of CanProve;
    // creating the solver (and translating every Bind into Z3 constraints)
    // eagerly made every Analyzer pay Z3's initialization cost up front.
    // Until the first operation that actually needs the solver,
    // Bind/EnterConstraint only journal their arguments into scope_stack_.
    scope_stack_.push_back({});
    scope_side_effects_.push_back({});
    // use rlimit, not timeout to ensure deterministic behavior
    SetRLimit(10000U);
  }

  /// @brief Create the solver on first use and replay the recorded journal.
  ///
  /// The live scope_stack_ (which also feeds the SMTLIB2 debug output) holds
  /// exactly the binds and constraints an eagerly-built solver would carry:
  /// binds append to the innermost scope and scopes pop LIFO, so walking the
  /// scopes front-to-back replays the surviving entries in the order they
  /// were recorded. Entries from scopes that already exited were popped
  /// together with their scope and are correctly absent. The replay performs
  /// the same MemoPut/solver->add sequence the eager path would have done
  /// for these entries, so CanProve answers are unchanged.
  void Materialize() {
    if (solver) return;
    // The solver gets its own context: verdicts under the deterministic
    // rlimit budget depend on the context's accumulated AST state (ids feed
    // the search heuristics), so sharing a context couples every solver's
    // borderline answers to whichever analyzers happened to materialize
    // before it. A private context makes each analyzer's answers a pure
    // function of its own journal and query history. Unmaterialized provers
    // hold no context (and no Z3 state) at all.
    ctx = std::make_shared<z3::context>();
    solver.emplace(CreateSolver(*ctx));
    if (timeout_ms != UINT_MAX) {
      solver->set("timeout", timeout_ms);
    }
    if (rlimit != UINT_MAX) {
      solver->set("rlimit", rlimit);
    }
    TVM_FFI_ICHECK_EQ(scope_side_effects_.size(), scope_stack_.size());
    for (size_t scope_index = 0; scope_index < scope_stack_.size(); ++scope_index) {
      for (const Scope& entry : scope_stack_[scope_index]) {
        switch (entry.kind) {
          case Scope::BindValue:
            ApplyBindValue(entry.var, entry.value);
            break;
          case Scope::BindRange:
            ApplyBindRange(entry.var, entry.min, entry.extent);
            break;
          case Scope::Constraint:
            ApplyConstraint(entry.constraint, entry.is_assume, scope_index);
            break;
        }
      }
    }
  }

  /// @brief Create a Free z3 expression from a primitive-valued ExprNode.
  z3::expr Create(const ExprNode* op) {
    // Expression translation only happens with a live solver: every public
    // entry point that can reach a visitor materializes first.
    TVM_FFI_ICHECK(solver.has_value());
    auto ref = ffi::GetRef<Expr>(op).as_or_throw<PrimExpr>();
    PrimType dtype = ref.ty();
    std::string name = ns.GetNewName(ref);
    /// TVM max_val can't handle uint64 max correctly, so we special case it here
    if (dtype.MatchesCode(DLDataTypeCode::kDLBool)) {
      return ctx->bool_const(name.c_str());
    } else {
      z3::expr e = ctx->int_const(name.c_str());
      if (dtype.MatchesCode(DLDataTypeCode::kDLUInt) && dtype.bits() == 64) {
        solver->add(ctx->int_val(0) <= e && e <= ctx->int_val((uint64_t)UINT64_MAX));
      } else {
        auto min_val = min_value(dtype).as_or_throw<IntImm>()->value;
        auto max_val = max_value(dtype).as_or_throw<IntImm>()->value;
        solver->add(ctx->int_val(min_val) <= e && e <= ctx->int_val(max_val));
      }
      return e;
    }
  }

  struct Scope {
    enum Kind {
      BindValue,
      BindRange,
      Constraint,
    } kind;
    Var var;
    PrimExpr value;
    PrimExpr min;
    PrimExpr extent;
    PrimExpr constraint;
    /// Assume-mode constraints keep side-effectful expressions memoized for
    /// the lifetime of their scope. This flag preserves that behavior during
    /// deferred replay.
    bool is_assume = false;
  };

  /// @brief scope_stack memorizes existing constraint and bindings
  ///        to generate SMTLIB2 representation with comments
  std::vector<std::vector<Scope>> scope_stack_;

  /// @brief Per-scope memoized side-effect expressions, parallel to scope_stack_.
  std::vector<std::vector<PrimExpr>> scope_side_effects_;

  /// @brief Enter a constraint scope
  std::function<void()> EnterConstraint(const PrimExpr& constraint, bool is_assume) {
    scope_stack_.push_back({});
    scope_stack_.back().push_back(
        Scope{Scope::Constraint, Var(), PrimExpr(), PrimExpr(), PrimExpr(), constraint, is_assume});
    scope_side_effects_.push_back({});
    if (solver) {
      ApplyConstraint(constraint, is_assume, scope_side_effects_.size() - 1);
    }
    // Exit callback: scopes leave LIFO. If the solver materialized while this
    // scope was live, replay pushed its frame and retained its assume memos.
    return [this]() {
      if (solver) solver->pop();
      for (const PrimExpr& expr : scope_side_effects_.back()) {
        MemoErase(expr);
      }
      scope_side_effects_.pop_back();
      scope_stack_.pop_back();
    };
  }

  /// @brief Translate a constraint into the solver (push + assert).
  void ApplyConstraint(const PrimExpr& constraint, bool is_assume, size_t scope_index) {
    solver->push();
    this->is_assume = is_assume;
    solver->add(VisitBool(constraint));
    this->is_assume = false;
    auto side_effect_exprs = std::move(side_effect_exprs_);
    side_effect_exprs_.clear();
    if (is_assume) {
      auto& keep = scope_side_effects_[scope_index];
      keep.insert(keep.end(), side_effect_exprs.begin(), side_effect_exprs.end());
    } else {
      for (const PrimExpr& expr : side_effect_exprs) {
        MemoErase(expr);
      }
    }
  }

  /// @brief Check trivil bad cases, return true if the expr is a bad case
  /// Z3 prover may take a long time to initialize (at least 200us),
  /// This optimization can speedup 30% of the test cases in our unit tests
  bool CheckTrivilBadCases(const PrimExpr& expr) {
    if (IsFreeNode(expr)) {
      return true;
    }
    auto checkTrivilCmp = [this](const PrimExpr& lhs, const PrimExpr& rhs) {
      if (IsFreeNode(lhs) && rhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if (IsFreeNode(rhs) && lhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if (IsFreeNode(lhs) && IsFreeNode(rhs)) {
        return true;
      }
      // cast('xxx', free_var) == constant
      if (auto cast = lhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && rhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      // constant == cast('xxx', free_var)
      if (auto cast = rhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && lhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      return false;
    };
    if (auto eq = expr.as<EQNode>()) {
      auto lhs = eq->a;
      auto rhs = eq->b;
      return checkTrivilCmp(lhs, rhs);
    } else if (auto ne = expr.as<NENode>()) {
      auto lhs = ne->a;
      auto rhs = ne->b;
      return checkTrivilCmp(lhs, rhs);
    }
    return false;
  }

  /// @brief Check if the expression can be proved
  bool CanProve(const PrimExpr& expr) {
    // Z3 is only a fallback. Any failure (including z3::exception thrown by the
    // solver) must degrade to "cannot prove" instead of escaping to the caller.
    try {
      if (!IsZ3SupportedExpr(expr.get())) return false;
      // The trivial-bad-case check below consults the memo (IsFreeNode), so
      // the journal must be replayed first for it to see the bound vars.
      Materialize();
      if (CheckTrivilBadCases(expr)) return false;
      z3::expr_vector constr(*ctx);
      constr.push_back(!ConvertBool(expr));
      auto result = solver->check(constr);
      constr.pop_back();
      return result == z3::unsat;
    } catch (const z3::exception&) {
      return false;
    }
  }

  /// @brief Binded
  /// @brief Bind a variable to a value or a range
  /// A bind journaled inside a live constraint scope is dropped with that
  /// scope. The eager path differed only cosmetically: it kept the var's memo
  /// translation after the scope's solver frame (and with it the var's range
  /// assertions) was popped, leaving an unconstrained placeholder --
  /// observably the same as translating the var as a fresh free variable on
  /// demand, which is what a later query on the journal does.
  void Bind(const Var& var, const PrimExpr& value, bool allow_override = false) {
    if (!IsZ3SupportedExpr(var.get())) return;
    scope_stack_.back().push_back(Scope{Scope::BindValue, var, value});
    if (!solver) return;  // journaled; translated when the solver materializes
    ApplyBindValue(var, value);
  }

  void ApplyBindValue(const Var& var, const PrimExpr& value) {
    // we add the binding whenever the value is pure,
    // because non-pure parts are handling by creating free variables in VisitExpr
    MemoPut(var.as_or_throw<PrimExpr>(), ConvertInt(value));
  }

  /// @brief Bind a variable to a range
  void Bind(const Var& var, const Range& range, bool allow_override = false) {
    if (!IsZ3SupportedExpr(var.get())) return;
    scope_stack_.back().push_back(
        Scope{Scope::BindRange, var, PrimExpr(), range->min, range->extent});
    if (!solver) return;  // journaled; translated when the solver materializes
    ApplyBindRange(var, range->min, range->extent);
  }

  void ApplyBindRange(const Var& var, const PrimExpr& min, const PrimExpr& extent) {
    // 1. Create a placeholder for the var, and save it in the memo
    //    if the var is overrided later, we can just update the memo, and the old placeholder will
    //    be ignored
    auto var_expr = Create(var.get());
    MemoPut(var.as_or_throw<PrimExpr>(), var_expr);

    // 2. Add constraint on the placeholder
    //    when min_expr >= max_expr, the range is empty, which is under undefined behavior
    //    instead of adding an unsat constraint, we just skip the range constraint to leave it a
    //    free var
    //
    //    NOTE: min + extent builds a fresh AddNode that is not folded, so we must
    //    test is_const_int on min and extent individually and add the two constants
    //    in C++. Otherwise this fast path is never taken and we always emit the more expensive
    //    symbolic constraint below.
    if (tirx::is_const_int(min) && tirx::is_const_int(extent)) {
      int64_t min_value = *tirx::as_const_int(min);
      int64_t extent_value = *tirx::as_const_int(extent);
      int64_t max_value = min_value + extent_value;
      if (min_value < max_value) {
        solver->add(ctx->int_val(min_value) <= var_expr);
        solver->add(var_expr < ctx->int_val(max_value));
      }
    } else {
      PrimExpr prim_var = var.as_or_throw<PrimExpr>();
      solver->add(ConvertBool(extent <= 0 || (min <= prim_var && prim_var < min + extent)));
    }
  }

  void CopyFrom(const Self& other_) {
    if (!other_.solver) {
      // The source has not materialized: its entire Z3 state is the journal.
      // Copy it and stay lazy; a later Materialize on this clone rebuilds
      // the same solver state the source would have built.
      solver.reset();
      memo_.clear();
      z3_pool_.clear();
      free_slots_.clear();
      ns = Namespace();
      side_effect_exprs_.clear();
      ctx = other_.ctx;
      scope_stack_ = other_.scope_stack_;
      scope_side_effects_.assign(scope_stack_.size(), {});
      timeout_ms = other_.timeout_ms;
      rlimit = other_.rlimit;
      return;
    }
    // Z3 handles cannot move between contexts. Destroy every handle owned by
    // this fresh clone before adopting the source Analyzer's context.
    solver.reset();
    memo_.clear();
    z3_pool_.clear();
    free_slots_.clear();
    ctx = other_.ctx;
    solver.emplace(CreateSolver(*ctx));

    // Copy other objects. The source and destination now share one context, so
    // copying Z3 handles is valid and both provers retain its ownership.
    ns = other_.ns;
    std::vector<std::pair<size_t, const PrimExpr*>> live;
    live.reserve(other_.memo_.size());
    for (auto& item : other_.memo_) {
      live.emplace_back(item.second, &item.first);
    }
    std::sort(live.begin(), live.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    for (auto [index, expr] : live) {
      MemoPut(*expr, *other_.z3_pool_[index]);
    }
    for (auto a : other_.solver->assertions()) {
      solver->add(a);
    }
    // Copy timeout options, but not other solver options.
    SetTimeoutMs(other_.timeout_ms);
    SetRLimit(other_.rlimit);
    // Copy the scope stack, which contains comments for SMTLIB2 generation.
    scope_stack_ = other_.scope_stack_;
    scope_side_effects_ = other_.scope_side_effects_;
  }

  /// @brief Set timeout in milliseconds
  void SetTimeoutMs(unsigned timeout_ms) {
    this->timeout_ms = timeout_ms;
    if (!solver) return;  // stored; applied when the solver materializes
    solver->set("timeout", timeout_ms);
  }

  /// @brief Set max steps
  void SetRLimit(unsigned rlimit) {
    this->rlimit = rlimit;
    if (!solver) return;  // stored; applied when the solver materializes
    solver->set("rlimit", rlimit);
  }

  /// @brief Get the SMTLIB2 representation of the current solver state
  ffi::String GetSMTLIB2() {
    Materialize();
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << solver->to_smt2();
    return ss.str();
  }

  void AddScopeDebugMsg(std::ostream& ss) {
    for (const auto& scope : scope_stack_) {
      ss << "; Entering Scope\n";
      for (const auto& s : scope) {
        switch (s.kind) {
          case Scope::Constraint:
            ss << "; constraint: " << s.constraint << "\n";
            break;
          case Scope::BindValue:
            ss << "; bind value: " << s.var << " = " << s.value << "\n";
            break;
          case Scope::BindRange:
            ss << "; bind range: " << s.var << " in [" << s.min << ", " << s.min + s.extent
               << ")\n";
            break;
        }
      }
    }
  }

  /// @brief Get the SMTLIB2 representation of the current solver state with additional expr trying
  /// to prove
  ffi::String GetSMTLIB2(const PrimExpr& expr) {
    Materialize();
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << "; Trying to prove: " << expr << "\n";
    solver->push();
    solver->add(!ConvertBool(expr));
    ss << solver->to_smt2();
    solver->pop();
    return ss.str();
  }

  /// @brief Get the statistics of the solver
  ffi::String GetStats() {
    Materialize();
    std::stringstream ss;
    ss << solver->statistics();
    return ss.str();
  }

  ffi::String GetModel(const PrimExpr& expr) {
    Materialize();
    solver->set("model", true);
    solver->push();
    solver->add(!ConvertBool(expr));
    auto result = solver->check();
    ffi::String model_str;
    if (result == z3::sat) {
      z3::model m = solver->get_model();
      std::map<std::string, z3::expr> model_map;
      for (unsigned i = 0; i < m.size(); i++) {
        z3::func_decl d = m[i];
        model_map.emplace(d.name().str(), m.get_const_interp(d));
      }
      std::stringstream ss;
      for (const auto& [k, v] : model_map) {
        ss << "  " << k << " = " << v << "\n";
      }
      model_str = ss.str();
    }
    solver->pop();
    solver->set("model", false);
    return model_str;
  }

  /*!
   * \brief Count the number of distinct integer values satisfying current constraints.
   *
   * Uses Z3's model enumeration (AllSAT pattern) to count solutions:
   * 1. Find a satisfying assignment
   * 2. Add a blocking clause to exclude it
   * 3. Repeat until UNSAT
   *
   * \param var The variable to count values for
   * \param max_count Safety limit on enumeration
   * \param min_consecutive Minimum consecutive count requirement (0 to disable)
   * \return Number of satisfying values, -1 on error, -2 if min_consecutive constraint not met
   */
  int64_t CountSatisfyingValues(const Var& var, int64_t max_count, int64_t min_consecutive = 1) {
    if (!IsZ3SupportedExpr(var.get())) {
      return -1;
    }

    Materialize();
    solver->set("model", true);
    solver->push();

    // Convert the TVM variable to Z3 expression
    z3::expr z3_var = VisitInt(var.as_or_throw<PrimExpr>());

    int64_t count = 0;
    std::vector<int64_t> found_values;

    while (count < max_count) {
      auto result = solver->check();
      if (result != z3::sat) {
        break;  // No more solutions
      }

      z3::model m = solver->get_model();
      z3::expr val_expr = m.eval(z3_var, true);

      // Extract the integer value from Z3 expression
      int64_t val;
      if (val_expr.is_numeral()) {
        val = val_expr.get_numeral_int64();
      } else {
        // If we can't get a concrete value, stop enumeration
        break;
      }

      found_values.push_back(val);
      count++;

      // Add blocking clause: var != val (exclude this solution)
      solver->add(z3_var != ctx->int_val(val));
    }

    solver->pop();
    solver->set("model", false);

    // Clear any side effects from visiting the variable
    for (const auto& expr : side_effect_exprs_) {
      MemoErase(expr);
    }
    side_effect_exprs_.clear();

    // Check minimum consecutive constraint if enabled
    if (min_consecutive > 0 && count > 0) {
      // Sort the values to check consecutive groups
      std::sort(found_values.begin(), found_values.end());

      // Check that all values form groups of at least min_consecutive consecutive numbers
      int64_t consecutive_count = 1;
      for (size_t i = 1; i < found_values.size(); i++) {
        if (found_values[i] == found_values[i - 1] + 1) {
          // Consecutive value
          consecutive_count++;
        } else {
          // Gap found, check if the previous group meets the minimum
          if (consecutive_count < min_consecutive) {
            return -2;  // Previous group too small
          }
          consecutive_count = 1;  // Start new group
        }
      }
      // Check the last group
      if (consecutive_count < min_consecutive) {
        return -2;  // Last group too small
      }
    }

    return count;
  }

 private:
  using Z3BinOp = z3::expr (*)(const z3::expr&, const z3::expr&);

  std::vector<PrimExpr> side_effect_exprs_;
  bool is_assume = false;

  z3::expr ConvertBool(const PrimExpr& e, bool is_assume = false) {
    this->is_assume = is_assume;
    auto res = VisitBool(e);
    for (auto& expr : side_effect_exprs_) {
      MemoErase(expr);
    }
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  z3::expr ConvertInt(const PrimExpr& e, bool is_assume = false) {
    this->is_assume = is_assume;
    auto res = VisitInt(e);
    for (auto& expr : side_effect_exprs_) {
      MemoErase(expr);
    }
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  /// @brief Visit expression with memoization
  z3::expr VisitExpr(const Expr& expr) override {
    PrimExpr e = expr.as_or_throw<PrimExpr>();
    if (const z3::expr* hit = MemoGet(e)) {
      return *hit;
    }
    auto res = Base::VisitExpr(e);
    auto side_effect = SideEffect(e);
    if (side_effect <= CallEffectKind::kPure) {
      MemoPut(e, res);
    } else if (side_effect <= CallEffectKind::kReadState) {
      MemoPut(e, res);
      side_effect_exprs_.emplace_back(e);
    } else {
      if (is_assume) {
        MemoPut(e, res);
      }
      side_effect_exprs_.emplace_back(e);
    }
    return res;
  }

  /// @brief Check if the expression is a free node having no constraints
  bool IsFreeNode(const PrimExpr& e) {
    if (memo_.count(e)) {
      return false;
    }
    return e->IsInstance<CallNode>() || e->IsInstance<BufferLoadNode>() ||
           e->IsInstance<ReduceNode>() ||
           (e->IsInstance<CastNode>() && !IsZ3SupportedExpr(e.as_or_throw<Cast>()->value.get()));
  }

  /// @brief Check if the expression type is supported by z3 integer operations.
  static bool IsZ3SupportedExpr(const ExprNode* expr) {
    TVM_FFI_DCHECK(expr != nullptr);
    TVM_FFI_DCHECK(!expr->ExprNode::ty.IsMissing());
    PrimType prim_ty = expr->ExprNode::ty.as_or_throw<PrimType>();
    return (prim_ty->dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLInt) ||
            prim_ty->dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLUInt) ||
            prim_ty->dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLBool)) &&
           prim_ty->dtype.lanes == 1;
  }

  /// @brief Visit the expression and convert it into z3 integer expression
  z3::expr VisitInt(const PrimExpr& expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return z3::ite(e, ctx->int_val(1), ctx->int_val(0));
    } else {
      return e;
    }
  }

  /// @brief Visit the expression and convert it into z3 boolean expression
  z3::expr VisitBool(const PrimExpr& e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      return expr != ctx->int_val(0);
    }
  }

  /// @brief Helper function to visit binary arithmetic operations
  z3::expr VisitArith(Z3BinOp signed_op, const ExprNode* op, const PrimExpr& a, const PrimExpr& b) {
    if (IsZ3SupportedExpr(a.get()) && IsZ3SupportedExpr(b.get())) {
      return signed_op(VisitInt(a), VisitInt(b));
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExpr_(const LetNode* op) override {
    if (IsZ3SupportedExpr(op->var.get())) {
      MemoPut(op->var.as_or_throw<PrimExpr>(), VisitInt(op->value));
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CastNode* op) override {
    // if the inner dtype is valid, we just visit it
    if (IsZ3SupportedExpr(op->value.get()) && IsZ3SupportedExpr(op)) {
      return VisitInt(op->value);
    } else {
      // otherwise, we create a new free z3 variable
      return Create(op);
    }
  }
  z3::expr VisitExpr_(const VarNode* op) override { return Create(op); }
  z3::expr VisitExpr_(const BufferLoadNode* op) override { return Create(op); }
  z3::expr VisitExpr_(const ReduceNode* op) override { return Create(op); }
  z3::expr VisitExpr_(const MinNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a < b, a, b);
  }
  z3::expr VisitExpr_(const MaxNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a > b, a, b);
  }
  // TVM Div/Mod are truncated (round toward zero), while Z3's native operator/
  // and operator% are Euclidean. Using the raw operators is unsound once the
  // dividend can be negative, so we implement truncating helpers explicitly.
  static z3::expr truncdiv(const z3::expr& a, const z3::expr& b) {
    z3::expr abs_a = z3::ite(a >= 0, a, -a);
    z3::expr abs_b = z3::ite(b >= 0, b, -b);
    // |a| / |b| is exact (Euclidean == truncated for non-negative operands).
    z3::expr q = abs_a / abs_b;
    return z3::ite((a >= 0) == (b >= 0), q, -q);
  }
  static z3::expr truncmod(const z3::expr& a, const z3::expr& b) {
    // TVM Mod follows the sign of the dividend: a - b * truncdiv(a, b).
    return a - b * truncdiv(a, b);
  }
  static z3::expr floordiv(const z3::expr& a, const z3::expr& b) {
    return z3::ite(b > 0, a / b, -((-a) / b));
  }
  static z3::expr floormod(const z3::expr& a, const z3::expr& b) {
    return z3::ite(b > 0, a % b, -((-a) % b));
  }
  z3::expr VisitExpr_(const AddNode* op) override {
    return VisitArith(z3::operator+, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const SubNode* op) override {
    return VisitArith(z3::operator-, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const MulNode* op) override {
    return VisitArith(z3::operator*, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const DivNode* op) override { return VisitArith(truncdiv, op, op->a, op->b); }
  z3::expr VisitExpr_(const ModNode* op) override { return VisitArith(truncmod, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorDivNode* op) override {
    return VisitArith(floordiv, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const FloorModNode* op) override {
    return VisitArith(floormod, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const EQNode* op) override {
    return VisitArith(z3::operator==, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const NENode* op) override {
    return VisitArith(z3::operator!=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LTNode* op) override {
    return VisitArith(z3::operator<, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LENode* op) override {
    return VisitArith(z3::operator<=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GTNode* op) override {
    return VisitArith(z3::operator>, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GENode* op) override {
    return VisitArith(z3::operator>=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const AndNode* op) override { return VisitBool(op->a) && VisitBool(op->b); }
  z3::expr VisitExpr_(const OrNode* op) override { return VisitBool(op->a) || VisitBool(op->b); }
  z3::expr VisitExpr_(const NotNode* op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode* op) override {
    return z3::ite(VisitBool(op->condition), VisitInt(op->true_value), VisitInt(op->false_value));
  }
  z3::expr VisitExpr_(const IntImmNode* op) override { return ctx->int_val(op->value); }

  // Bitwise operations
  z3::expr VisitExpr_(const CallNode* op) override {
    // Check if this is a bitwise operation
    if (op->op.same_as(tirx::builtin::bitwise_and())) {
      return VisitBitwiseOp(z3::operator&, op);
    } else if (op->op.same_as(tirx::builtin::bitwise_or())) {
      return VisitBitwiseOp(z3::operator|, op);
    } else if (op->op.same_as(tirx::builtin::bitwise_xor())) {
      return VisitBitwiseOp(z3::operator^, op);
    } else if (op->op.same_as(tirx::builtin::bitwise_not())) {
      return VisitBitwiseNotOp(op);
    } else if (op->op.same_as(tirx::builtin::shift_left())) {
      return VisitShiftOp(z3::shl, op);
    } else if (op->op.same_as(tirx::builtin::shift_right())) {
      return VisitShiftOp(z3::ashr, op);
    } else if (op->op.same_as(tirx::builtin::if_then_else()) && op->args.size() == 3 &&
               IsZ3SupportedExpr(op->args[1].get()) && IsZ3SupportedExpr(op->args[2].get())) {
      // tir.if_then_else(cond, a, b) is a select-like ternary.
      return z3::ite(VisitBool(op->args[0].as_or_throw<PrimExpr>()),
                     VisitInt(op->args[1].as_or_throw<PrimExpr>()),
                     VisitInt(op->args[2].as_or_throw<PrimExpr>()));
    } else {
      // For other call nodes, create a free variable
      return Create(op);
    }
  }

  /// @brief Helper function to visit binary bitwise operations
  z3::expr VisitBitwiseOp(z3::expr (*op_func)(const z3::expr&, const z3::expr&),
                          const CallNode* op) {
    if (op->args.size() != 2) {
      LOG(FATAL) << "Binary bitwise operation expects 2 arguments, got " << op->args.size();
      TVM_FFI_UNREACHABLE();
    }

    PrimExpr a = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr b = op->args[1].as_or_throw<PrimExpr>();
    unsigned bit_width = std::max(a.ty().bits(), b.ty().bits());

    if (IsZ3SupportedExpr(a.get()) && IsZ3SupportedExpr(b.get())) {
      return z3::bv2int(
          op_func(z3::int2bv(bit_width, VisitInt(a)), z3::int2bv(bit_width, VisitInt(b))), true);
    } else {
      return Create(op);
    }
  }

  /// @brief Helper function to visit unary bitwise not operation
  z3::expr VisitBitwiseNotOp(const CallNode* op) {
    if (op->args.size() != 1) {
      LOG(FATAL) << "Bitwise not operation expects 1 argument, got " << op->args.size();
      TVM_FFI_UNREACHABLE();
    }

    PrimExpr a = op->args[0].as_or_throw<PrimExpr>();

    if (IsZ3SupportedExpr(a.get())) {
      // Cast integer to bit-vector, apply bitwise not, then cast back.
      unsigned bit_width = a.ty().bits();
      z3::expr a_int = VisitInt(a);
      z3::expr a_bv = z3::int2bv(bit_width, a_int);
      return z3::bv2int(~a_bv, true);
    } else {
      return Create(op);
    }
  }

  /// @brief Helper function to visit shift operations
  z3::expr VisitShiftOp(z3::expr (*op_func)(const z3::expr&, const z3::expr&), const CallNode* op) {
    if (op->args.size() != 2) {
      LOG(FATAL) << "Shift operation expects 2 arguments, got " << op->args.size();
      TVM_FFI_UNREACHABLE();
    }

    PrimExpr a = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr b = op->args[1].as_or_throw<PrimExpr>();

    // Shift operations require integer types for both operands
    if (IsZ3SupportedExpr(a.get()) && IsZ3SupportedExpr(b.get())) {
      z3::expr a_expr = VisitInt(a);
      z3::expr b_expr = VisitInt(b);

      // Rely on Z3's native bit-vector shift behavior. We must NOT add hard
      // assertions such as `b_expr >= 0` to the solver here: solver.add() has no
      // matching push/pop in this path, so the assertion would permanently
      // poison the shared solver and make all subsequent unrelated proofs about
      // `b` unsound.
      unsigned bit_width = std::max(a.ty().bits(), b.ty().bits());
      z3::expr a_bv = z3::int2bv(bit_width, a_expr);
      z3::expr b_bv = z3::int2bv(bit_width, b_expr);

      // Perform the shift in bit-vector domain, then cast back to int.
      z3::expr result_bv = op_func(a_bv, b_bv);
      return z3::bv2int(result_bv, true);
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExprDefault_(const Object* op) override {
    // Z3 is a best-effort fallback that runs only after the native analyzers
    // have already failed. An unsupported node must not crash the build, so we
    // model it as a fresh unconstrained free variable, which keeps the proof
    // sound (it can only make CanProve more conservative).
    return Create(static_cast<const ExprNode*>(op));
  }
};

TVM_DLL bool Z3Prover::IsEnabled() const { return true; }
TVM_DLL bool Z3Prover::CanProve(const PrimExpr& expr) { return impl_->CanProve(expr); }
TVM_DLL void Z3Prover::Bind(const Var& var, const Range& new_range, bool allow_override) {
  return impl_->Bind(var, new_range, allow_override);
}
TVM_DLL void Z3Prover::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {
  return impl_->Bind(var, expr, allow_override);
}
std::function<void()> Z3Prover::EnterConstraint(const PrimExpr& constraint, bool is_assume) {
  return impl_->EnterConstraint(constraint, is_assume);
}
ffi::String Z3Prover::GetSMTLIB2(const ffi::Optional<PrimExpr> expr) {
  if (expr.has_value()) {
    return impl_->GetSMTLIB2(expr.value());
  } else {
    return impl_->GetSMTLIB2();
  }
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) { impl_->SetTimeoutMs(timeout_ms); }
void Z3Prover::SetRLimit(unsigned max_step) { impl_->SetRLimit(max_step); }
void Z3Prover::CopyFrom(const Z3Prover& other) { impl_->CopyFrom(*other.impl_); }
ffi::String Z3Prover::GetStats() { return impl_->GetStats(); }
ffi::String Z3Prover::GetModel(const PrimExpr& expr) { return impl_->GetModel(expr); }
TVM_DLL int64_t Z3Prover::CountSatisfyingValues(const Var& var, int64_t max_count,
                                                int64_t min_consecutive) {
  return impl_->CountSatisfyingValues(var, max_count, min_consecutive);
}
Z3Prover::Z3Prover(AnalyzerObj* parent) : impl_(new Impl{parent}) {}
TVM_DLL Z3Prover::~Z3Prover() { delete impl_; }

}  // namespace tvm::arith

#else  // TVM_USE_Z3

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"

namespace tvm::arith {

using namespace tirx;
using namespace ffi;

void EnterZ3ContextScope() {}
void ExitZ3ContextScope() {}

// Stub implementation used when Z3 support is not built. All proving queries
// conservatively report "cannot prove" while keeping the public API available.
class Z3Prover::Impl {};

TVM_DLL bool Z3Prover::IsEnabled() const { return false; }
TVM_DLL bool Z3Prover::CanProve(const PrimExpr& expr) { return false; }
TVM_DLL void Z3Prover::Bind(const Var& var, const Range& new_range, bool allow_override) {}
TVM_DLL void Z3Prover::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {}
std::function<void()> Z3Prover::EnterConstraint(const PrimExpr& constraint, bool is_assume) {
  return []() {};
}
ffi::String Z3Prover::GetSMTLIB2(const ffi::Optional<PrimExpr> expr) {
  return "; Z3 Prover is disabled.";
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) {}
void Z3Prover::SetRLimit(unsigned rlimit) {}
ffi::String Z3Prover::GetModel(const PrimExpr& expr) { return "; Z3 Prover is disabled."; }
TVM_DLL int64_t Z3Prover::CountSatisfyingValues(const Var& var, int64_t max_count,
                                                int64_t min_consecutive) {
  return -1;  // Z3 disabled, return error
}

void Z3Prover::CopyFrom(const Z3Prover& other) {}
ffi::String Z3Prover::GetStats() { return "; Z3 Prover is disabled."; }
Z3Prover::Z3Prover(AnalyzerObj*) : impl_(nullptr) {}
TVM_DLL Z3Prover::~Z3Prover() {}

}  // namespace tvm::arith

#endif  // TVM_USE_Z3
