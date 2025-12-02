#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <z3++.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "tvm/node/structural_equal.h"
#include "tvm/node/structural_hash.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/expr_functor.h"
#include "tvm/arith/analyzer.h"
#include "tvm/tir/op_attr_types.h"

namespace tvm::arith {

using namespace tir;
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
  std::string GetNewName(const PrimExpr & expr) {
    std::stringstream ss;
    ss << expr;
    auto name = ss.str();
    if(used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while(used_names.count(check_name)) {
      idx ++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
};

} // namespace

class Z3Prover::Impl : ExprFunctor<z3::expr(const PrimExpr &)> {
public:
  using Base = ExprFunctor<z3::expr(const PrimExpr &)>;
  using Self = Z3Prover::Impl;

  /// @brief Z3 context, a shared ptr, because tilelang want to copy the Analyzer
  std::shared_ptr<z3::context> ctx { new z3::context() };

  /// @brief Z3 solver instance
  z3::solver solver {*ctx};

  /// @brief Memorize pure expressions
  std::unordered_map<PrimExpr, z3::expr, StructuralHash, StructuralEqual> memo_;

  /// @brief Assume overrides
  std::vector<PrimExpr> assume_overrides_;
  bool is_assume = false;

  /// @brief Namespace for variable naming
  Namespace ns;

  /// @brief Timeout in milliseconds
  unsigned timeout_ms {UINT_MAX};

  /// @brief Max steps
  unsigned max_step {UINT_MAX};

  /// @brief Create a z3 solver with custom options
  static z3::solver CreateSolver(z3::context & ctx) {
    z3::solver solver(ctx);
    // here we disable model generation to speed up the solving process
    solver.set("model", false);
    return solver;
  }

  Impl() {
    solver = CreateSolver(*ctx);
    // default timeout 5ms
    // Z3's implementation of timeout, when setting timeout T ms, it will stop at T - 1 ms
    SetTimeoutMs(5);
  }

  /// @brief Create a Free z3 expression from PrimExprNode
  z3::expr Create(const PrimExprNode *op) {
    auto ref = ffi::GetRef<PrimExpr>(op);
    auto dtype = op->dtype;
    std::string name = ns.GetNewName(ref);
    z3::expr e = ctx->int_const(name.c_str());
    /// TVM max_val can't handle uint64 max correctly, so we special case it here
    if(dtype.is_uint() && dtype.bits() == 64) {
      solver.add(e >= ctx->int_val(0));
      solver.add(e <= ctx->int_val((uint64_t)UINT64_MAX));
    } else {
      auto max_val = Downcast<IntImm>(max_value(dtype))->value;
      auto min_val = Downcast<IntImm>(min_value(dtype))->value;
      solver.add(e <= ctx->int_val(max_val));
      solver.add(e >= ctx->int_val(min_val));
    }
    return e;
  }

  /// @brief Enter a constraint scope
  std::function<void()> EnterConstraint(const PrimExpr& constraint, bool is_assume=false) {
    solver.push();
    is_assume = true;
    auto e = VisitBool(constraint);
    is_assume = false;
    solver.add(e);
    auto overrides = std::move(assume_overrides_);
    assume_overrides_.clear();
    return [this, overrides]() {
      solver.pop();
      for (const auto& expr : assume_overrides_) {
        memo_.erase(expr);
      }
    };
  }

  /// @brief Check trivil bad cases, return true if the expr is a bad case
  /// Z3 prover may take a long time to initialize (at least 200us),
  /// This optimization can speedup 30% of the test cases in our unit tests
  bool CheckTrivilBadCases(const PrimExpr & expr) {
    if(IsFreeNode(expr)) {
      return true;
    }
    auto checkTrivilCmp = [this](const PrimExpr & lhs, const PrimExpr & rhs) {
      if(IsFreeNode(lhs) && rhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if(IsFreeNode(rhs) && lhs->IsInstance<IntImmNode>()) {
        return true;
      }
      if(IsFreeNode(lhs) && IsFreeNode(rhs)) {
        return true;
      }
      // cast('xxx', free_var) == constant
      if(auto cast = lhs.as<CastNode>()) {
        if(IsFreeNode(cast->value) && rhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      // constant == cast('xxx', free_var)
      if(auto cast = rhs.as<CastNode>()) {
        if(IsFreeNode(cast->value) && lhs->IsInstance<IntImmNode>()) {
          return true;
        }
      }
      return false;
    };
    if(auto eq = expr.as<EQNode>()) {
      auto lhs = eq->a;
      auto rhs = eq->b;
      return checkTrivilCmp(lhs, rhs);
    } else if(auto ne = expr.as<NENode>()) {
      auto lhs = ne->a;
      auto rhs = ne->b;
      return checkTrivilCmp(lhs, rhs);
    }
    return false;
  }

  /// @brief Check if the expression can be proved
  bool CanProve(const PrimExpr &expr) {
    if (CheckTrivilBadCases(expr)) return false;
    if (!IsValidDType(expr->dtype)) return false;
    z3::check_result result = z3::unknown;
    z3::expr_vector constr(*ctx);
    constr.push_back(!VisitBool(expr));
    try {
      result = solver.check(constr);
    } catch(std::exception & e) {
      std::string reason = e.what();
      if(reason != "max. steps exceeded") {
        LOG(FATAL) << "Z3 encountered an error: " << e.what();
      }
    }
    constr.pop_back();
    return result == z3::unsat;
  }

  /// @brief Bind a variable to a value or a range
  void Bind(const Var & var, const PrimExpr & value, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    // ICHECK(!allow_override) << "Z3Prover does not support override binding.";
    if(SideEffect(value) <= CallEffectKind::kPure) {
      memo_.emplace(var, VisitInt(value));
    } else {
      solver.add(VisitBool(var == value));
    }
  }

  /// @brief Bind a variable to a range
  void Bind(const Var & var, const Range & range, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    // ICHECK(!allow_override) << "Z3Prover does not support override binding.";
    auto name = ns.GetNewName(var);
    auto var_expr = VisitExpr(var);
    // auto var_expr = ctx->int_const(name.c_str());
    auto min_expr = VisitInt(range->min);
    auto extent_expr = VisitInt(range->extent);
    solver.add(var_expr >= min_expr);
    solver.add(var_expr < (min_expr + extent_expr));
  }

  void CopyFrom(const Self & other_) {
    // 1. must copy solver first, because the old solver holds the context, if we drop the old context, the solver will be invalid
    solver = CreateSolver(*other_.ctx);
    // 2. then copy context
    ctx = other_.ctx;
    // copy other objects
    ns = other_.ns;
    for(auto & item: other_.memo_) {
      memo_.emplace(item.first, item.second);
    }
    for(auto a: other_.solver.assertions()) {
      solver.add(a);
    }
    SetTimeoutMs(other_.timeout_ms);
    SetMaxStep(other_.max_step);
  }

  /// @brief Set timeout in milliseconds
  void SetTimeoutMs(unsigned timeout_ms) {
    this->timeout_ms = timeout_ms;
    solver.set("timeout", timeout_ms);
  }

  /// @brief Set max steps
  void SetMaxStep(unsigned max_step) {
    this->max_step = max_step;
    solver.set("max_steps", max_step);
  }

  /// @brief Get the SMTLIB2 representation of the current solver state
  ffi::String GetSMTLIB2() {
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    ss <<  solver.to_smt2();
    return ss.str();
  }

  /// @brief Get the SMTLIB2 representation of the current solver state with additional expr trying to prove
  ffi::String GetSMTLIB2(const PrimExpr & expr) {
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    solver.push();
    solver.add(!VisitBool(expr));
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

private:

  using Z3BinOp = z3::expr(*)(const z3::expr &, const z3::expr &);

  /// @brief Visit expression with memoization
  z3::expr VisitExpr(const PrimExpr & e) override {
    if(memo_.count(e)) {
      return memo_.at(e);
    }
    auto res =  Base::VisitExpr(e);
    if(is_assume || SideEffect(e) <= CallEffectKind::kPure) {
      memo_.emplace(e, res);
      assume_overrides_.emplace_back(e);
    }
    return res;
  }

  bool IsFreeNode(const PrimExpr & e) {
    if(memo_.count(e)) {
      return false;
    }
    return e->IsInstance<CallNode>() 
      || e->IsInstance<BufferLoadNode>()
      || e->IsInstance<ProducerLoadNode>()
      || e->IsInstance<ReduceNode>()
      || (e->IsInstance<CastNode>() && !IsValidDType(Downcast<Cast>(e)->value->dtype));
  }
  /// @brief Check if the dtype is valid for z3 integer operations
  static bool IsValidDType(const DataType & dtype) {
    return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1;
  }

  /// @brief Visit the expression and convert it into z3 integer expression
  z3::expr VisitInt(const PrimExpr &expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return z3::ite(e, ctx->int_val(1), ctx->int_val(0));
    } else {
      return e;
    }
  }

  /// @brief Visit the expression and convert it into z3 boolean expression
  z3::expr VisitBool(const PrimExpr &e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      return expr != ctx->int_val(0);
    }
  }

  /// @brief Helper function to visit binary arithmetic operations
  z3::expr VisitArith(Z3BinOp signed_op, const PrimExprNode *op, const PrimExpr &a, const PrimExpr &b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      return signed_op(VisitInt(a), VisitInt(b));
    } else {
      return Create(op);
    }
  }

  z3::expr VisitExpr_(const LetNode *op) override { 
    if (IsValidDType(op->var->dtype)) {
      // if the expression is pure, we just bind it to the var
      if(SideEffect(op->value) <= CallEffectKind::kPure) {
        memo_.emplace(op->var, VisitInt(op->value));
      } else {
        // if the expression is not pure, we create a new z3 variable and add equality constraint
        solver.add(VisitBool(op->var == op->value));
      }
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CastNode * op) override {
    // if the inner dtype is valid, we just visit it
    if (IsValidDType(op->value->dtype) && IsValidDType(op->dtype)) {
      return VisitInt(op->value);
    } else {
      // otherwise, we create a new free z3 variable
      return Create(op);
    }
  }
  z3::expr VisitExpr_(const CallNode *op) override {
    // We don't know what the call does, so we create a new free z3 variable
    return Create(op);
  }
  z3::expr VisitExpr_(const VarNode *op) override {
    // We create a new free z3 variable for the variable node, it should be memorized in parent VisitExpr call
    return Create(op);
  }
  z3::expr VisitExpr_(const BufferLoadNode *op) override {
    // The buffer load may have side effects, we create a new free z3 variable
    return Create(op);
  }
  z3::expr VisitExpr_(const ProducerLoadNode *op) override {
    // The producer load may have side effects, we create a new free z3 variable
    return Create(op);
  }
  z3::expr VisitExpr_(const ReduceNode *op) override {
    // The reduce node may have side effects, we create a new free z3 variable
    return Create(op);
  }
  z3::expr VisitExpr_(const MinNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a < b, a, b);
  }
  z3::expr VisitExpr_(const MaxNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a > b, a, b);
  }
  z3::expr VisitExpr_(const AddNode *op) override { return VisitArith(z3::operator +, op, op->a, op->b); }
  z3::expr VisitExpr_(const SubNode *op) override { return VisitArith(z3::operator -, op, op->a, op->b); }
  z3::expr VisitExpr_(const MulNode *op) override { return VisitArith(z3::operator *, op, op->a, op->b); }
  z3::expr VisitExpr_(const DivNode *op) override { return VisitArith(z3::operator /, op, op->a, op->b); }
  z3::expr VisitExpr_(const ModNode *op) override { return VisitArith(z3::operator %, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorDivNode *op) override { return VisitArith(z3::operator /, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorModNode *op) override { return VisitArith(z3::operator %, op, op->a, op->b); }
  z3::expr VisitExpr_(const EQNode *op) override { return VisitArith(z3::operator==, op, op->a, op->b); }
  z3::expr VisitExpr_(const NENode *op) override { return VisitArith(z3::operator!=, op, op->a, op->b); }
  z3::expr VisitExpr_(const LTNode *op) override { return VisitArith(z3::operator<, op, op->a, op->b); }
  z3::expr VisitExpr_(const LENode *op) override { return VisitArith(z3::operator<=, op, op->a, op->b); }
  z3::expr VisitExpr_(const GTNode *op) override { return VisitArith(z3::operator>, op, op->a, op->b); }
  z3::expr VisitExpr_(const GENode *op) override { return VisitArith(z3::operator>=, op, op->a, op->b); }
  z3::expr VisitExpr_(const AndNode *op) override { return VisitBool(op->a) && VisitBool(op->b); }
  z3::expr VisitExpr_(const OrNode *op) override { return VisitBool(op->a) || VisitBool(op->b); }
  z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode *op) override { return z3::ite(VisitBool(op->condition), VisitInt(op->true_value), VisitInt(op->false_value)); }
  z3::expr VisitExpr_(const IntImmNode *op) override { return ctx->int_val(op->value); }
  z3::expr VisitExprDefault_(const Object* op) override {
    LOG(FATAL) << "Z3Prover only support integers, but got " << op->GetTypeKey() << ".";
    TVM_FFI_UNREACHABLE();
  }
};

TVM_DLL bool Z3Prover::CanProve(const PrimExpr & expr) {
  return impl_->CanProve(expr);
}
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
  if(expr.has_value()) {
    return impl_->GetSMTLIB2(expr.value());
  } else {
    return impl_->GetSMTLIB2();
  }
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) {
  impl_->SetTimeoutMs(timeout_ms);
}
void Z3Prover::SetMaxStep(unsigned max_step) {
  impl_->SetMaxStep(max_step);
}
void Z3Prover::CopyFrom(const Z3Prover & other) {
  impl_->CopyFrom(*other.impl_);
}
ffi::String Z3Prover::GetStats() {
  return impl_->GetStats();
}
Z3Prover::Z3Prover(Analyzer* parent): impl_(new Impl) {}
TVM_DLL Z3Prover::~Z3Prover() {
  delete impl_;
}

} // namespace tvm::arith