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

#include "tvm/arith/analyzer.h"
#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"
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

class Z3Prover::Impl : ExprFunctor<z3::expr(const PrimExpr&)> {
 public:
  using Base = ExprFunctor<z3::expr(const PrimExpr&)>;
  using Self = Z3Prover::Impl;

  AnalyzerObj* analyzer;
  /// @brief Z3 context, a shared ptr, because tilelang want to copy the Analyzer
  // We use a thread_local static Z3 context so all analyzers within the same thread
  // can share a common context, because Z3 initialization is slow on some CPUs
  // (e.g., AMD EPYC 7502 32-Core). Using thread_local ensures thread safety.
  inline static thread_local std::shared_ptr<z3::context> ctx{new z3::context()};

  /// @brief Z3 solver instance
  z3::solver solver{*ctx};

  /// @brief Memorize pure expressions
  std::unordered_map<PrimExpr, z3::expr, StructuralHash, ExprDeepEqual> memo_;

  bool is_assume = false;

  /// @brief Namespace for variable naming
  Namespace ns;

  /// @brief Timeout in milliseconds
  unsigned timeout_ms{UINT_MAX};

  /// @brief Max steps
  unsigned rlimit{UINT_MAX};

  /// @brief Create a z3 solver with custom options
  static z3::solver CreateSolver(z3::context& ctx) {
    z3::solver solver(ctx);
    // here we disable model generation to speed up the solving process
    solver.set("model", false);
    // ensure determinstic behavior
    solver.set("random_seed", (unsigned)42);
    return solver;
  }

  Impl(AnalyzerObj* parent) : analyzer(parent) {
    scope_stack_.push_back({});
    solver = CreateSolver(*ctx);
    // default timeout 5ms
    // Z3's implementation of timeout, when setting timeout T ms, it will stop at T - 1 ms
    // SetTimeoutMs(5);
    // use rlimit, not timeout to ensure determinstic behavior
    SetRLimit(1e4);
  }

  /// @brief Create a Free z3 expression from PrimExprNode
  z3::expr Create(const PrimExprNode* op) {
    auto ref = ffi::GetRef<PrimExpr>(op);
    auto dtype = op->dtype;
    std::string name = ns.GetNewName(ref);
    /// TVM max_val can't handle uint64 max correctly, so we special case it here
    if (dtype.is_bool()) {
      return ctx->bool_const(name.c_str());
    } else {
      z3::expr e = ctx->int_const(name.c_str());
      if (dtype.is_uint() && dtype.bits() == 64) {
        solver.add(ctx->int_val(0) <= e && e <= ctx->int_val((uint64_t)UINT64_MAX));
      } else {
        auto min_val = Downcast<IntImm>(min_value(dtype))->value;
        auto max_val = Downcast<IntImm>(max_value(dtype))->value;
        solver.add(ctx->int_val(min_val) <= e && e <= ctx->int_val(max_val));
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
  };

  /// @brief scope_stack memorizes existing constraint and bindings
  ///        to generate SMTLIB2 representation with comments
  std::vector<std::vector<Scope>> scope_stack_;

  /// @brief Enter a constraint scope
  std::function<void()> EnterConstraint(const PrimExpr& constraint, bool is_assume = false) {
    scope_stack_.push_back({});
    scope_stack_.back().push_back(
        Scope{Scope::Constraint, Var(), PrimExpr(), PrimExpr(), PrimExpr(), constraint});
    solver.push();
    this->is_assume = is_assume;
    solver.add(VisitBool(constraint));
    this->is_assume = false;
    auto side_effect_exprs = std::move(side_effect_exprs_);
    side_effect_exprs_.clear();
    if (is_assume) {
      return [this, side_effect_exprs]() {
        solver.pop();
        for (const auto& expr : side_effect_exprs) {
          memo_.erase(expr);
        }
        scope_stack_.pop_back();
      };
    } else {
      for (const auto& expr : side_effect_exprs) {
        memo_.erase(expr);
      }
      return [this]() {
        solver.pop();
        scope_stack_.pop_back();
      };
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
    if (CheckTrivilBadCases(expr)) return false;
    if (!IsValidDType(expr->dtype)) return false;
    z3::expr_vector constr(*ctx);
    constr.push_back(!ConvertBool(expr));
    auto result = solver.check(constr);
    constr.pop_back();
    return result == z3::unsat;
  }

  /// @brief Binded
  /// @brief Bind a variable to a value or a range
  void Bind(const Var& var, const PrimExpr& value, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    scope_stack_.back().push_back(Scope{Scope::BindValue, var, value});
    // we add the binding whenever the value is pure,
    // because non-pure parts are handling by creating free variables in VisitExpr
    memo_.emplace(var, ConvertInt(value));
  }

  /// @brief Bind a variable to a range
  void Bind(const Var& var, const Range& range, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    scope_stack_.back().push_back(
        Scope{Scope::BindRange, var, PrimExpr(), range->min, range->extent});
    // 1. Create a placeholder for the var, and save it in the memo
    //    if the var is overrided later, we can just update the memo, and the old placeholder will
    //    be ignored
    auto var_expr = Create(var.as<PrimExprNode>());
    memo_.emplace(var, var_expr);

    // 2. Add constraint on the placeholder
    //    when min_expr >= max_expr, the range is empty, which is under undefined behavior
    //    instead of adding an unsat constraint, we just skip the range constraint to leave it a
    //    free var
    if (tirx::is_const_int(range->min) && tirx::is_const_int(range->min + range->extent)) {
      int64_t min_value = *tirx::as_const_int(range->min);
      int64_t max_value = *tirx::as_const_int(range->min + range->extent);
      if (min_value < max_value) {
        solver.add(ctx->int_val(min_value) <= var_expr);
        solver.add(var_expr < ctx->int_val(max_value));
      }
    } else {
      solver.add(ConvertBool(range->extent <= 0 ||
                             (range->min <= var && var < range->min + range->extent)));
    }
  }

  void CopyFrom(const Self& other_) {
    // 1. create a new solver
    //    because this->solver depends on this->ctx
    //    we need to deconstruct the old solver, and create a new one depending on other_.ctx
    solver = CreateSolver(*other_.ctx);
    // 2. copy the context
    //    the context is a shared_ptr, we can just copy the pointer
    ctx = other_.ctx;
    // 3. copy other objects
    ns = other_.ns;
    for (auto& item : other_.memo_) {
      memo_.emplace(item.first, item.second);
    }
    for (auto a : other_.solver.assertions()) {
      solver.add(a);
    }
    // 4. copy timeout options
    //    but other solver options are not copied
    SetTimeoutMs(other_.timeout_ms);
    SetRLimit(other_.rlimit);
    // 5. copy the scope stack, which containing comments for SMTLIB2 generation
    scope_stack_ = other_.scope_stack_;
  }

  /// @brief Set timeout in milliseconds
  void SetTimeoutMs(unsigned timeout_ms) {
    this->timeout_ms = timeout_ms;
    solver.set("timeout", timeout_ms);
  }

  /// @brief Set max steps
  void SetRLimit(unsigned rlimit) {
    this->rlimit = rlimit;
    solver.set("rlimit", rlimit);
  }

  /// @brief Get the SMTLIB2 representation of the current solver state
  ffi::String GetSMTLIB2() {
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << solver.to_smt2();
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
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << "; Trying to prove: " << expr << "\n";
    solver.push();
    solver.add(!ConvertBool(expr));
    ss << solver.to_smt2();
    solver.pop();
    return ss.str();
  }

  /// @brief Get the statistics of the solver
  ffi::String GetStats() {
    std::stringstream ss;
    ss << solver.statistics();
    return ss.str();
  }

  ffi::String GetModel(const PrimExpr& expr) {
    solver.set("model", true);
    solver.push();
    solver.add(!ConvertBool(expr));
    auto result = solver.check();
    ffi::String model_str;
    if (result == z3::sat) {
      z3::model m = solver.get_model();
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
    solver.pop();
    solver.set("model", false);
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
    if (!IsValidDType(var->dtype)) {
      return -1;
    }

    solver.set("model", true);
    solver.push();

    // Convert the TVM variable to Z3 expression
    z3::expr z3_var = VisitInt(var);

    int64_t count = 0;
    std::vector<int64_t> found_values;

    while (count < max_count) {
      auto result = solver.check();
      if (result != z3::sat) {
        break;  // No more solutions
      }

      z3::model m = solver.get_model();
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
      solver.add(z3_var != ctx->int_val(val));
    }

    solver.pop();
    solver.set("model", false);

    // Clear any side effects from visiting the variable
    for (const auto& expr : side_effect_exprs_) {
      memo_.erase(expr);
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

  z3::expr ConvertBool(const PrimExpr& e, bool is_assume = false) {
    this->is_assume = is_assume;
    auto res = VisitBool(e);
    for (auto& expr : side_effect_exprs_) {
      memo_.erase(expr);
    }
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  z3::expr ConvertInt(const PrimExpr& e, bool is_assume = false) {
    this->is_assume = is_assume;
    auto res = VisitInt(e);
    for (auto& expr : side_effect_exprs_) {
      memo_.erase(expr);
    }
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  /// @brief Visit expression with memoization
  z3::expr VisitExpr(const PrimExpr& e) override {
    if (memo_.count(e)) {
      return memo_.at(e);
    }
    auto res = Base::VisitExpr(e);
    auto side_effect = SideEffect(e);
    if (side_effect <= CallEffectKind::kPure) {
      memo_.emplace(e, res);
    } else if (side_effect <= CallEffectKind::kReadState) {
      memo_.emplace(e, res);
      side_effect_exprs_.emplace_back(e);
    } else {
      if (is_assume) {
        memo_.emplace(e, res);
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
           e->IsInstance<ProducerLoadNode>() || e->IsInstance<ReduceNode>() ||
           (e->IsInstance<CastNode>() && !IsValidDType(Downcast<Cast>(e)->value->dtype));
  }

  /// @brief Check if the dtype is valid for z3 integer operations
  static bool IsValidDType(const DataType& dtype) {
    return (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) && dtype.lanes() == 1;
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
  z3::expr VisitArith(Z3BinOp signed_op, const PrimExprNode* op, const PrimExpr& a,
                      const PrimExpr& b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      return signed_op(VisitInt(a), VisitInt(b));
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExpr_(const LetNode* op) override {
    if (IsValidDType(op->var->dtype)) {
      memo_.emplace(op->var, VisitInt(op->value));
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CastNode* op) override {
    // if the inner dtype is valid, we just visit it
    if (IsValidDType(op->value->dtype) && IsValidDType(op->dtype)) {
      return VisitInt(op->value);
    } else {
      // otherwise, we create a new free z3 variable
      return Create(op);
    }
  }
  z3::expr VisitExpr_(const VarNode* op) override { return Create(op); }
  z3::expr VisitExpr_(const BufferLoadNode* op) override { return Create(op); }
  z3::expr VisitExpr_(const ProducerLoadNode* op) override { return Create(op); }
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
  z3::expr VisitExpr_(const DivNode* op) override {
    return VisitArith(z3::operator/, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const ModNode* op) override {
    return VisitArith(z3::operator%, op, op->a, op->b);
  }
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

    const PrimExpr& a = op->args[0];
    const PrimExpr& b = op->args[1];
    unsigned bit_width = std::max(op->args[0].dtype().bits(), op->args[1].dtype().bits());

    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
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

    const PrimExpr& a = op->args[0];

    if (IsValidDType(a->dtype)) {
      // Cast integer to bit-vector, apply bitwise not, then cast back.
      unsigned bit_width = a.dtype().bits();
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

    const PrimExpr& a = op->args[0];
    const PrimExpr& b = op->args[1];

    // Shift operations require integer types for both operands
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      // For shift operations, we need to ensure the shift amount is non-negative
      // and within reasonable bounds
      z3::expr a_expr = VisitInt(a);
      z3::expr b_expr = VisitInt(b);

      // Add constraint that shift amount should be non-negative
      // This is a common assumption in many programming languages
      solver.add(b_expr >= 0);

      // Also limit shift amount to avoid unrealistic large shifts
      // We'll limit to 64 bits (reasonable for most use cases)
      solver.add(b_expr < 64);

      unsigned bit_width = std::max(a.dtype().bits(), b.dtype().bits());
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
    LOG(FATAL) << "Z3Prover only support integers, but got " << op->GetTypeKey() << ".";
    TVM_FFI_UNREACHABLE();
  }
};

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
