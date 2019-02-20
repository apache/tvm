/*!
 *  Copyright (c) 2018 by Contributors
 * \file zero_elimination.cc
 * \brief Transform tensors in such a way as to eliminate summation over zeros.
 */
#include "zero_elimination.h"

#include <tvm/api_registry.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "arithmetic/ModulusRemainder.h"
#include "../op/op_util.h"

namespace tvm {
namespace ir {

using HalideIR::Internal::gcd;
using HalideIR::Internal::lcm;

struct ExprLess {
    bool operator()(const Expr& l, const Expr& r) const {
      return Compare(l, r) < 0;
    }
};

struct ExprEq {
    bool operator()(const Expr& l, const Expr& r) const {
      return Compare(l, r) == 0;
    }
};

// Merge two maps, prefer the right one on conflict
template <class K, class V>
Map<K, V> Merge(Map<K, V> original, const Map<K, V>& update) {
  for (const auto& p : update) {
    original.Set(p.first, p.second);
  }
  return std::move(original);
}

// Concatenate two arrays
template <class T>
Array<T> Concat(Array<T> a, const Array<T>& b) {
  for (const auto& x : b) {
    a.push_back(x);
  }
  return std::move(a);
}

// Combine all expressions from the container using &&.
template <class container>
Expr All(const container& c) {
  Expr res;
  for (const auto& e : c) {
    if (res.get()) {
      res = res && e;
    } else {
      res = e;
    }
  }
  if (res.get()) {
    return res;
  } else {
    return const_true();
  }
}

// Create a select statement of the form cond ? on_true : 0
Expr SelectElseZero(const Expr& cond, const Expr& on_true) {
  return Select::make(cond, on_true, make_zero(on_true.type()));
}

// Simplify the expression as thoroughly as possible by using all available simplifiers.
Expr SuperSimplify(Expr e, const Map<Var, Range>& vranges = Map<Var, Range>()) {
  // For some reason no simplifier can detect that there is only one value of the variable
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& var_range : vranges) {
    if (is_const_int(var_range.second->extent, 1)) {
      vmap[var_range.first.get()] = var_range.second->min;
    }
  }
  if (!vmap.empty()) {
    e = Substitute(e, vmap);
  }

  return CanonicalSimplify(Simplify(CanonicalSimplify(e, vranges), vranges), vranges);
}

// Provability check that uses SuperSimplify
bool CanProve(Expr e, const Map<Var, Range>& vranges = Map<Var, Range>()) {
  return is_one(SuperSimplify(e, vranges));
}

class ExprFreeVarsVisitor : public IRVisitor {
 public:
  std::vector<Var> free_array;
  std::unordered_set<const Variable*> bound;
  std::unordered_set<const Variable*> free;

  virtual void Visit(const NodeRef& node) {
    if (const Variable* v = node.as<Variable>()) {
      if (!bound.count(v) && !free.count(v)) {
        free.insert(v);
        free_array.push_back(Var(node.node_));
      }
    } else {
      IRVisitor::Visit(node);
    }
  }

  void Visit_(const Variable* op) {
    CHECK(false) << "This case shouldn't happen";
  }

  void Visit_(const LetStmt* op) {
    bound.insert(op->var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const For* op) {
    bound.insert(op->loop_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Let* op) {
    bound.insert(op->var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Reduce* op) {
    for (const auto& iv : op->axis) {
      bound.insert(iv->var.get());
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate* op) {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Free* op) {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load* op) {
    Visit(op->buffer_var);
    IRVisitor::Visit_(op);
  }
};

// Get free variables of an expression
Array<Var> ExprFreeVars(const Expr& expr) {
  ExprFreeVarsVisitor visitor;
  visitor.Visit(expr);
  return visitor.free_array;
}

// Clone iter vars and return both the new vars and the substitution from old to new.
std::pair<Array<IterVar>, std::unordered_map<const Variable*, Expr>> CloneIterVars(
    const Array<IterVar>& vars) {
  Array<IterVar> new_vars;
  std::unordered_map<const Variable*, Expr> vmap;
  for (const IterVar& iv : vars) {
    IterVar new_v =
      IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""),
          iv->iter_type, iv->thread_tag);
    new_vars.push_back(new_v);
    vmap[iv->var.get()] = new_v;
  }
  return std::make_pair(std::move(new_vars), std::move(vmap));
}

// Clone reduction by cloning the axis variables.
Expr CloneReduction(const Expr& expr) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(red->axis);

    Array<Expr> src_with_newaxis;
    for (const auto& src : red->source) {
      src_with_newaxis.push_back(Substitute(src, vmap));
    }

    return Reduce::make(red->combiner, src_with_newaxis,
        new_axis, Substitute(red->condition, vmap), red->value_index);
  } else {
    return expr;
  }
}

// Convert an array of itervars to an array of inequalities
Array<Expr> IterVarsToInequalities(const Array<IterVar>& itervars) {
  Array<Expr> res;
  for (const IterVar& v : itervars) {
    res.push_back(GE::make(v->var, v->dom->min));
    res.push_back(LT::make(v->var, v->dom->min + v->dom->extent));
  }
  return res;
}

// Convert an array of itervars to a map from vars to ranges
Map<Var, Range> IterVarsToMap(const Array<IterVar>& itervars) {
  Map<Var, Range> res;
  for (const IterVar& v : itervars) {
    res.Set(v->var, v->dom);
  }
  return res;
}

// Convert an array of itervars to an array of vars
Array<Var> IterVarsToVars(const Array<IterVar>& itervars) {
  Array<Var> res;
  for (const IterVar& v : itervars) {
    res.push_back(v->var);
  }
  return res;
}

// Given a map from vars to ranges create an array of itervars
Array<IterVar> IterVarsFromMap(const Array<Var>& vars, const Map<Var, Range>& vranges,
                               IterVarType iter_type = kDataPar, std::string thread_tag = "") {
  Array<IterVar> res;
  for (const Var& v : vars) {
    CHECK(vranges.count(v)) << "A range for the variable " << v
      << " was not provided in map " << vranges;
    res.push_back(IterVarNode::make(vranges[v], v, iter_type, thread_tag));
  }
  return res;
}

// Return true if this combiner is just a sum.
bool IsSumCombiner(const CommReducer& combiner) {
  if (combiner->result.size() != 1) {
    return false;
  }

  if (!is_const_value(SuperSimplify(combiner->identity_element[0]), 0)) {
    return false;
  }

  return is_const_value(SuperSimplify(combiner->result[0] -
                                      (combiner->lhs[0] + combiner->rhs[0])),
                        0);
}

// Return true if zero may be factored out of a reduction with this combiner.
bool CanFactorZeroFromCombiner(const CommReducer& combiner, int value_index) {
  if (!is_const_value(combiner->identity_element[value_index], 0)) {
    return false;
  }

  Expr zero = make_zero(combiner->result[value_index].type());
  Expr in = Substitute(combiner->result[value_index],
                       {{combiner->lhs[value_index], zero},
                        {combiner->rhs[value_index], zero}});
  in = SuperSimplify(in);

  return is_const_value(in, 0);
}

Expr InlineThisCall(const Expr& expr) {
  if (const Call* op = expr.as<Call>()) {
    if (op->call_type == Call::CallType::Halide) {
      if (const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>()) {
        Array<Var> tensor_axes;
        for (const auto& var : op_comp->axis) {
          tensor_axes.push_back(var->var);
        }

        Stmt inlined = Inline(Evaluate::make(expr), op->func, tensor_axes,
                              op_comp->body[op->value_index]);
        if (const ir::Evaluate* ev = inlined.as<ir::Evaluate>()) {
          // If it is a reduction, clone it
          return CloneReduction(ev->value);
        }
      }
    }
  }

  return expr;
}

Tensor InlineTailCall(const Tensor& tensor) {
  return op::TransformBody(tensor, InlineThisCall);
}

class InlineTensorsMutator : public IRMutator {
 public:
  explicit InlineTensorsMutator(const Array<Tensor>& inlineable, bool inline_reductions = false)
      : inline_reductions_(inline_reductions) {
    for (const Tensor& tensor : inlineable) {
      inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }
  }

  Expr Mutate_(const Call* op, const Expr& e) {
    if (op->call_type == Call::CallType::Halide) {
      const ComputeOpNode* op_comp = op->func.as<ComputeOpNode>();
      if (inlineable_.empty() || inlineable_.count({op_comp, op->value_index})) {
        if (op_comp && (inline_reductions_ || !op_comp->body[0].as<Reduce>())) {
          Array<Var> tensor_axes;
          for (const auto& var : op_comp->axis) {
            tensor_axes.push_back(var->var);
          }

          Stmt inlined = Inline(Evaluate::make(e), op->func, tensor_axes,
                                op_comp->body[op->value_index]);
          if (const ir::Evaluate* ev = inlined.as<ir::Evaluate>()) {
            // If it is a reduction, clone it
            return Mutate(ev->value);
          }
        }
      }
    }

    return e;
  }

 private:
  std::set<std::pair<const OperationNode*, int>> inlineable_;
  bool inline_reductions_;
};

Expr InlineTensors(const Expr& expr, const Array<Tensor>& inlineable,
                   bool inline_reductions) {
  return InlineTensorsMutator(inlineable, inline_reductions).Mutate(expr);
}

Tensor InlineTensors(const Tensor& tensor, const Array<Tensor>& inlineable,
                     bool inline_reductions) {
  auto transformation =
    [inlineable, inline_reductions](const Expr& e) {
      return InlineTensorsMutator(inlineable, inline_reductions).Mutate(e); };
  return op::TransformBody(tensor, transformation);
}


struct NonzeronessConditionResult {
  Expr cond;
  Expr value;

  Expr to_expr() const {
    return SelectElseZero(cond, value);
  }
};

class NonzeronessConditionFunctor
  : public ExprFunctor<NonzeronessConditionResult(const Expr&, const Expr&)> {
 public:
  NonzeronessConditionResult NonzeronessCondition(const Expr& e) {
    return VisitExpr(e, e);
  }

  result_type VisitExpr_(const Variable*, const Expr& e) final { return Default_(e); }
  result_type VisitExpr_(const IntImm* op, const Expr& e) final { return Const_(op, e); }
  result_type VisitExpr_(const UIntImm* op, const Expr& e) final { return Const_(op, e); }
  result_type VisitExpr_(const FloatImm* op, const Expr& e) final { return Const_(op, e); }
  result_type VisitExpr_(const StringImm*, const Expr& e) final { return Default_(e); }
  result_type VisitExpr_(const Add* op, const Expr& e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Sub* op, const Expr& e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Mul* op, const Expr& e) final { return BinOpMulLike_(op, e); }
  result_type VisitExpr_(const Div* op, const Expr& e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const Mod* op, const Expr& e) final { return BinOpDivLike_(op, e); }
  result_type VisitExpr_(const Min* op, const Expr& e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const Max* op, const Expr& e) final { return BinOpAddLike_(op, e); }
  result_type VisitExpr_(const EQ* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const NE* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const LE* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const LT* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const GE* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const GT* op, const Expr& e) final { return Bool_(op, e); }
  result_type VisitExpr_(const Not* op, const Expr& e) final { return Bool_(op, e); }

  result_type VisitExpr_(const Cast* op, const Expr& e) final {
    if (op->value.type().is_bool()) {
      return {op->value, make_const(e.type(), 1)};
    } else {
      auto nz_a = NonzeronessCondition(op->value);

      if (nz_a.value.same_as(op->value)) {
        return {nz_a.cond, e};
      } else {
        return {nz_a.cond, Cast::make(op->type, nz_a.value)};
      }
    }
  }

  result_type VisitExpr_(const Select* op, const Expr& e) final {
    return SelectLike_(e, op->condition, op->true_value, op->false_value, Select::make);
  }

  result_type VisitExpr_(const Call* op, const Expr& e) final {
    if (op->name == intrinsic::tvm_if_then_else) {
      return SelectLike_(e, op->args[0], op->args[1], op->args[2], if_then_else);
    } else {
      return Default_(e);
    }
  }

  NonzeronessConditionResult Default_(const Expr& e) {
    return {const_true(), e};
  }

  template <class TNode>
  NonzeronessConditionResult Const_(const TNode* op, const Expr& e) {
    if (op->value == 0) {
      return {const_false(), e};
    } else {
      return {const_true(), e};
    }
  }

  template <class make_select_type>
  NonzeronessConditionResult SelectLike_(const Expr& e, const Expr& cond, const Expr& true_val,
                                         const Expr& false_val, make_select_type make_select) {
    auto nz_a = NonzeronessCondition(true_val);
    auto nz_b = NonzeronessCondition(false_val);

    if (is_const_value(nz_b.value, 0)) {
      Expr new_cond = SuperSimplify(nz_a.cond && cond);
      return {new_cond, nz_a.value};
    }

    if (is_const_value(nz_a.value, 0)) {
      Expr new_cond = SuperSimplify(nz_b.cond && !cond);
      return {new_cond, nz_b.value};
    }

    Expr new_cond =
      SuperSimplify(Or::make(cond && nz_a.cond,
                             !cond &&  nz_b.cond));
    if (nz_a.value.same_as(true_val) && nz_b.value.same_as(false_val)) {
      return {new_cond, e};
    } else {
      return {new_cond, make_select(cond, nz_a.value, nz_b.value)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpAddLike_(const TNode* op, const Expr& e) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);

    if (Equal(nz_a.cond, nz_b.cond)) {
      if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
        return {nz_a.cond, e};
      } else {
        return {nz_a.cond, TNode::make(nz_a.value, nz_b.value)};
      }
    } else {
      Expr new_cond = SuperSimplify(Or::make(nz_a.cond, nz_b.cond));
      Expr new_a = Equal(nz_a.cond, new_cond) ? nz_a.value : nz_a.to_expr();
      Expr new_b = Equal(nz_b.cond, new_cond) ? nz_b.value : nz_b.to_expr();
      Expr new_expr = TNode::make(new_a, new_b);
      return {new_cond, new_expr};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpMulLike_(const TNode* op, const Expr& e) {
    auto nz_a = NonzeronessCondition(op->a);
    auto nz_b = NonzeronessCondition(op->b);

    Expr new_cond = SuperSimplify(nz_a.cond && nz_b.cond);

    if (nz_a.value.same_as(op->a) && nz_b.value.same_as(op->b)) {
      return {new_cond, e};
    } else {
      return {new_cond, TNode::make(nz_a.value, nz_b.value)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult BinOpDivLike_(const TNode* op, const Expr& e) {
    auto nz_a = NonzeronessCondition(op->a);

    if (nz_a.value.same_as(op->a)) {
      return {nz_a.cond, e};
    } else {
      return {nz_a.cond, TNode::make(nz_a.value, op->b)};
    }
  }

  template <class TNode>
  NonzeronessConditionResult Bool_(const TNode* op, const Expr& e) {
    return {e, make_const(e.type(), 1)};
  }
};

NonzeronessConditionResult NonzeronessCondition(const Expr& expr) {
  return NonzeronessConditionFunctor().NonzeronessCondition(expr);
}

Expr LiftNonzeronessCondition(const Expr& expr) {
  return NonzeronessCondition(expr).to_expr();
}


class NormalizeComparisonsMutator : public IRMutator {
 public:
  virtual Expr Mutate_(const EQ* op, const Expr& e) { return Make<EQ>(op->a, op->b); }
  virtual Expr Mutate_(const NE* op, const Expr& e) { return Make<NE>(op->a, op->b); }
  virtual Expr Mutate_(const LT* op, const Expr& e) { return Make<LT>(op->a, op->b); }
  virtual Expr Mutate_(const LE* op, const Expr& e) { return Make<LE>(op->a, op->b); }
  virtual Expr Mutate_(const GT* op, const Expr& e) { return Make<LT>(op->b, op->a); }
  virtual Expr Mutate_(const GE* op, const Expr& e) { return Make<LE>(op->b, op->a); }

 private:
  template <class TNode>
  Expr Make(const Expr& a, const Expr& b) {
    // rewrite LT to LE for ints
    if (std::is_same<TNode, LT>::value && (a.type().is_int() || a.type().is_uint())) {
      return LE::make(SuperSimplify(a - b + 1), make_zero(a.type()));
    }
    return TNode::make(SuperSimplify(a - b), make_zero(a.type()));
  }
};

// Rewrite every comparison into the form a == 0, a != 0, a <= 0, and sometimes for floats a < 0
Expr NormalizeComparisons(const Expr& expr) {
  return NormalizeComparisonsMutator().Mutate(expr);
}


struct FactorOutAtomicFormulasResult {
  std::vector<Expr> atomic_formulas;
  Expr rest;

  Expr to_expr() const {
    Expr res = rest;
    for (const Expr& e : atomic_formulas) {
      res = And::make(e, res);
    }
    return res;
  }
};

class FactorOutAtomicFormulasFunctor
  : public ExprFunctor<FactorOutAtomicFormulasResult(const Expr&, const Expr&)> {
 public:
  result_type Atomic_(const Expr& e) {
    return {{e}, make_const(e.type(), 1)};
  }

  result_type VisitExpr_(const Variable*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const Call*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const IntImm*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const UIntImm*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const EQ*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const NE*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const LE*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const LT*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const GE*, const Expr& e) final { return Atomic_(e); }
  result_type VisitExpr_(const GT*, const Expr& e) final { return Atomic_(e); }

  result_type VisitExpr_(const And* op, const Expr& e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    std::vector<Expr> res;
    res.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::set_union(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                   res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                   std::back_inserter(res),
                   ExprLess());

    return {res, res_a.rest && res_b.rest};
  }

  result_type VisitExpr_(const Mul* op, const Expr& e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    std::vector<Expr> res;
    res.reserve(res_a.atomic_formulas.size() + res_b.atomic_formulas.size());
    std::set_union(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                   res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                   std::back_inserter(res),
                   ExprLess());

    return {res, res_a.rest * res_b.rest};
  }

  result_type VisitExpr_(const Or* op, const Expr& e) final {
    auto res_a = VisitExpr(op->a, op->a);
    auto res_b = VisitExpr(op->b, op->b);

    std::vector<Expr> res;
    res.reserve(std::min(res_a.atomic_formulas.size(), res_b.atomic_formulas.size()));
    std::set_intersection(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                          res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                          std::back_inserter(res),
                          ExprLess());

    std::vector<Expr> new_cond_a;
    new_cond_a.reserve(res_a.atomic_formulas.size() - res.size());
    std::set_difference(res_a.atomic_formulas.begin(), res_a.atomic_formulas.end(),
                        res.begin(), res.end(),
                        std::back_inserter(new_cond_a),
                        ExprLess());

    std::vector<Expr> new_cond_b;
    new_cond_b.reserve(res_b.atomic_formulas.size() - res.size());
    std::set_difference(res_b.atomic_formulas.begin(), res_b.atomic_formulas.end(),
                        res.begin(), res.end(),
                        std::back_inserter(new_cond_b),
                        ExprLess());

    res_a.atomic_formulas = std::move(new_cond_a);
    res_b.atomic_formulas = std::move(new_cond_b);

    Expr new_rest = Or::make(res_a.to_expr(), res_b.to_expr());

    return {res, new_rest};
  }
};

// Transform the given formula into an array of atomic formulas and a non-atomic residual.
FactorOutAtomicFormulasResult FactorOutAtomicFormulas(const Expr& e) {
  return FactorOutAtomicFormulasFunctor().VisitExpr(e, e);
}


struct EliminateDivModResult {
  Expr expr;
  Map<Var, Expr> substitution;
  Array<Var> new_variables;
  Array<Expr> conditions;
  Map<Var, Range> ranges;
};

class EliminateDivModMutator : public IRMutator {
 public:
  Map<Var, Expr> substitution;
  Array<Var> new_variables;
  Array<Expr> conditions;
  Map<Var, Range> ranges;

  explicit EliminateDivModMutator(Map<Var, Range> ranges)
    : ranges(ranges) {}

  virtual Expr Mutate_(const Div* op, const Expr& e) {
    const IntImm* imm = op->b.as<IntImm>();
    if (imm && imm->value > 0) {
      auto it = expr_to_vars_.find({op->a.get(), imm->value});
      if (it != expr_to_vars_.end()) {
        return it->second.first;
      }

      Expr mutated_a = Mutate(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value)) {
        return var_pair_opt.value().first;
      } else {
        return Div::make(mutated_a, Mutate(op->b));
      }
    }

    return Div::make(Mutate(op->a), Mutate(op->b));
  }

  virtual Expr Mutate_(const Mod* op, const Expr& e) {
    const IntImm* imm = op->b.as<IntImm>();
    if (imm && imm->value > 0) {
      auto it = expr_to_vars_.find({op->a.get(), imm->value});
      if (it != expr_to_vars_.end()) {
        return it->second.second;
      }

      Expr mutated_a = Mutate(op->a);
      if (auto var_pair_opt = AddNewVarPair(op->a, mutated_a, imm->value)) {
        return var_pair_opt.value().second;
      } else {
        return Mod::make(mutated_a, Mutate(op->b));
      }
    }

    return Mod::make(Mutate(op->a), Mutate(op->b));
  }

 private:
  dmlc::optional<std::pair<Var, Var>> AddNewVarPair(const Expr& e, const Expr& mut, int64_t val) {
    using tresult = dmlc::optional<std::pair<Var, Var>>;

    Expr val_e = make_const(e.type(), val);
    idx_ += 1;

    std::unordered_map<const Variable*, IntSet> var_intsets;
    for (const auto& p : ranges) {
      var_intsets[p.first.get()] = IntSet::range(p.second);
    }

    Range div_range = EvalSet(mut / val_e, var_intsets).cover_range(Range());
    Range mod_range = EvalSet(mut % val_e, var_intsets).cover_range(Range());

    if (!div_range.get() || !mod_range.get()) {
      LOG(WARNING) << "EliminateDivMod: won't eliminate div or mod of expr " << e
                   << "  because its bounds cannot be inferred";
      return tresult();
    }

    auto div = Var("div" + std::to_string(idx_), e.type());
    auto mod = Var("mod" + std::to_string(idx_), e.type());

    new_variables.push_back(div);
    new_variables.push_back(mod);

    substitution.Set(div, mut / val_e);
    substitution.Set(mod, mut % val_e);

    ranges.Set(div, div_range);
    ranges.Set(mod, mod_range);

    conditions.push_back(mut == div*val_e + mod);

    if (!CanProve(mod_range->extent <= val_e)) {
      LOG(WARNING) << "EliminateDivMod: cannot fully eliminate div or mod of expr " << e
                   << "  (probably it may change its sign)";
      // We cannot prove that mod is unique, so add additional condition
      conditions.push_back(Select::make(e >= 0, mod >= 0, mod <= 0));
    }

    auto p = std::make_pair(div, mod);
    expr_to_vars_[{e.get(), val}] = p;
    return tresult(p);
  }

  int idx_{0};
  std::map<std::pair<const HalideIR::Internal::IRNode*, int64_t>, std::pair<Var, Var>>
    expr_to_vars_;
};

// replace every subexpr of the form e/const and e % const with a new variable
EliminateDivModResult EliminateDivMod(const Expr& expr, Map<Var, Range> ranges) {
  EliminateDivModResult res;
  EliminateDivModMutator mutator(ranges);
  res.expr = mutator.Mutate(expr);
  res.conditions = std::move(mutator.conditions);
  res.new_variables = std::move(mutator.new_variables);
  res.substitution = std::move(mutator.substitution);
  res.ranges = std::move(mutator.ranges);
  return res;
}

// run EliminateDivMod from the condition of a reduction
Expr EliminateDivModFromReductionCondition(const Expr& expr,
                                           Map<Var, Range> vranges = Map<Var, Range>()) {
  if (const Reduce* red = expr.as<Reduce>()) {
    for (const IterVar& iv : red->axis) {
      vranges.Set(iv->var, iv->dom);
    }

    auto elim_res = EliminateDivMod(red->condition, vranges);

    vranges = elim_res.ranges;

    Array<IterVar> new_axis =
        Concat(red->axis, IterVarsFromMap(elim_res.new_variables, vranges, kCommReduce));

    Expr new_cond = elim_res.expr && All(elim_res.conditions);

    return Reduce::make(red->combiner, red->source, new_axis, new_cond, red->value_index);
  } else {
    return expr;
  }
}


VarBounds VarBounds::substitute(const Map<Var, Expr>& subst) const {
  auto apply_fun = [&subst](const Expr& e) { return Substitute(e, subst); };
  return {Substitute(coef, subst),
          UpdateArray(lower, apply_fun),
          UpdateArray(equal, apply_fun),
          UpdateArray(upper, apply_fun)};
}

Array<Expr> SolveSystemOfInequalitiesResult::as_conditions() const {
  Array<Expr> res;
  for (const Var& v : variables) {
    auto it = bounds.find(v.get());
    CHECK(it != bounds.end());
    const VarBounds& bnds = it->second;
    Expr lhs = bnds.coef * v;
    for (const Expr& rhs : bnds.equal) {
      res.push_back(EQ::make(lhs, rhs));
    }
    for (const Expr& rhs : bnds.lower) {
      res.push_back(GE::make(lhs, rhs));
    }
    for (const Expr& rhs : bnds.upper) {
      res.push_back(LE::make(lhs, rhs));
    }
  }
  for (const Expr& e : other_conditions) {
    res.push_back(e);
  }
  return res;
}

// Rewrite the system of inequalities using Fourier-Motzkin elimination
// Note that variable ranges help a lot, so this parameter is even non-optional
SolveSystemOfInequalitiesResult SolveSystemOfInequalities(const Array<Expr>& inequalities,
                                                          const Array<Var>& variables,
                                                          const Map<Var, Range>& vranges) {
  SolveSystemOfInequalitiesResult res;
  res.variables = variables;

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current` and classify them according to polarity wrt v
  // - Combine each formula of positive polarity (wrt v) with each formula of negative polarity
  // - Put the resulting combinations into `new_current` along with unclassifiable formulas
  // - Replace `current` with `new_current` and move to the next variable

  // current and new_current are sorted to enable some heuristics
  std::set<Expr, ExprLess> current;
  std::set<Expr, ExprLess> new_current;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, Expr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, Expr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<Expr> rest;

  // A helper that adds an inequality to new_current if it's not obviously redundant
  auto add_to_new_current = [&new_current, &vranges] (const Expr& new_ineq) {
    if (CanProve(new_ineq, vranges)) {
      // redundant: follows from the vranges
      return;
    }
    if (const LE* new_le = new_ineq.as<LE>()) {
        // A heuristic: check if the new inequality is a consequence of one
        // of its future neighbors (in this case don't add it) or if a future neighbor is
        // a consequence of the new ineq (in which case remove the neighbor)
        auto it_neighbor = new_current.lower_bound(new_ineq);
        if (it_neighbor != new_current.begin()) {
          const LE* le = std::prev(it_neighbor)->as<LE>();
          if (le && CanProve(new_le->a - le->a <= 0, vranges)) {
            return;
          } else if (le && CanProve(le->a - new_le->a <= 0, vranges)) {
            new_current.erase(std::prev(it_neighbor));
          }
        }
        // Check the other neighbor
        if (it_neighbor != new_current.end()) {
          const LE* le = it_neighbor->as<LE>();
          if (le && CanProve(new_le->a - le->a <= 0, vranges)) {
            return;
          } else if (le && CanProve(le->a - new_le->a <= 0, vranges)) {
            it_neighbor = new_current.erase(it_neighbor);
          }
        }

        new_current.insert(it_neighbor, new_ineq);
    } else {
      new_current.insert(new_ineq);
    }
  };

  // Simplify each inequality into the form `expr <= 0` and add to new_current formulas
  for (const Expr& ineq : inequalities) {
    add_to_new_current(NormalizeComparisons(SuperSimplify(ineq, vranges)));
  }

  std::swap(current, new_current);

  for (const Var& v : variables) {
    CHECK(!res.bounds.count(v.get())) <<
      "Variable " << v << " appears several times in the `variables` which might be a bug";

    new_current.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (vranges.count(v)) {
      const Range& range = vranges[v];
      Expr range_lbound = SuperSimplify(range->min, vranges);
      Expr range_ubound = SuperSimplify(range->min + range->extent - 1, vranges);
      coef_neg.push_back({-1, range_lbound});
      coef_pos.push_back({1, -range_ubound});
    }

    // Take formulas from `current` and classify them according to polarity wrt v
    for (const Expr& ineq : current) {
      if (const LE* le = ineq.as<LE>()) {
        Array<Expr> coef = arith::DetectLinearEquation(le->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            coef_pos.push_back({coef0, coef[1]});
          } else if (coef0 < 0) {
            coef_neg.push_back({coef0, coef[1]});
          }
          continue;
        }
      } else if (const EQ* eq = ineq.as<EQ>()) {
        Array<Expr> coef = arith::DetectLinearEquation(eq->a, {v});
        if (!coef.empty() && is_const(coef[0])) {
          int64_t coef0 = *as_const_int(coef[0]);
          if (coef0 == 0) {
            // zero polarity, straight to new_current
            add_to_new_current(ineq);
          } else if (coef0 > 0) {
            // Equalities may be considered as pairs of two inequalities
            coef_pos.push_back({coef0, coef[1]});
            coef_neg.push_back({-coef0, -coef[1]});
          } else if (coef0 < 0) {
            coef_pos.push_back({-coef0, -coef[1]});
            coef_neg.push_back({coef0, coef[1]});
          }
          continue;
        }
      }

      // if nothing worked, put it in rest
      rest.push_back(ineq);
    }

    // Combine each positive inequality with each negative one (by adding them together)
    for (const auto& pos : coef_pos) {
      for (const auto& neg : coef_neg) {
        auto first_gcd = gcd(pos.first, -neg.first);
        Expr c_pos = make_const(v.type(), neg.first/first_gcd);
        Expr c_neg = make_const(v.type(), pos.first/first_gcd);
        Expr new_lhs = c_neg*neg.second - c_pos*pos.second;
        Expr new_ineq = LE::make(new_lhs, make_zero(pos.second.type()));
        new_ineq = NormalizeComparisons(SuperSimplify(new_ineq, vranges));
        add_to_new_current(new_ineq);
      }
    }

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos) {
      coef_lcm = lcm(coef_lcm, pos.first);
    }
    for (const auto& neg : coef_neg) {
      coef_lcm = lcm(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds stored in sorted vectors
    std::vector<Expr> upper_bounds;
    std::vector<Expr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      Expr bound = make_const(v.type(), -coef_lcm/pos.first)*pos.second;
      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &vranges](const Expr& o) { return CanProve(o - bound <= 0,
                                                                          vranges); })) {
        continue;
      }
      // Erase all worse bounds
      upper_bounds.erase(
        std::remove_if(upper_bounds.begin(), upper_bounds.end(),
                       [&bound, &vranges](const Expr& o) { return CanProve(o - bound >= 0,
                                                                           vranges); }),
        upper_bounds.end());
      // Add
      upper_bounds.push_back(bound);
    }
    for (const auto& neg : coef_neg) {
      Expr bound = make_const(v.type(), -coef_lcm/neg.first)*neg.second;
      bound = SuperSimplify(bound, vranges);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &vranges](const Expr& o) { return CanProve(o - bound >= 0,
                                                                          vranges); })) {
        continue;
      }
      // Erase all worse bounds
      lower_bounds.erase(
        std::remove_if(lower_bounds.begin(), lower_bounds.end(),
                       [&bound, &vranges](const Expr& o) { return CanProve(o - bound <= 0,
                                                                           vranges); }),
        lower_bounds.end());
      // Add
      lower_bounds.push_back(bound);
    }

    // Sort the vectors and remove duplicates
    for (std::vector<Expr>* bounds : {&upper_bounds, &lower_bounds}) {
      std::sort(bounds->begin(), bounds->end(), ExprLess());
      bounds->erase(std::unique(bounds->begin(), bounds->end(), ExprEq()), bounds->end());
    }

    // Bounds which are both lower and upper should go to equal...
    std::vector<Expr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    std::set_intersection(upper_bounds.begin(), upper_bounds.end(),
                          lower_bounds.begin(), lower_bounds.end(),
                          std::back_inserter(equal), ExprLess());

    // ...and be removed from upper bounds...
    std::vector<Expr> new_upper;
    new_upper.reserve(upper_bounds.size() - equal.size());
    std::set_difference(upper_bounds.begin(), upper_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_upper), ExprLess());

    // ...and from lower bounds.
    std::vector<Expr> new_lower;
    new_lower.reserve(lower_bounds.size() - equal.size());
    std::set_difference(lower_bounds.begin(), lower_bounds.end(),
                        equal.begin(), equal.end(),
                        std::back_inserter(new_lower), ExprLess());

    // Write it to the result.
    auto& bnds = res.bounds[v.get()];
    bnds.coef = make_const(v.type(), coef_lcm);
    bnds.equal = equal;
    bnds.lower = new_lower;
    bnds.upper = new_upper;

    std::swap(current, new_current);
  }

  // Everything that is left goes to res.other_conditions
  for (const Expr& e : current) {
    Expr e_simp = SuperSimplify(e, vranges);
    if (is_const_int(e_simp, 0)) {
      // contradiction detected
      res.other_conditions = {const_false()};
      return res;
    } else if (is_const_int(e_simp, 1)) {
      continue;
    } else {
      res.other_conditions.push_back(e_simp);
    }
  }

  for (const Expr& e : rest)
    res.other_conditions.push_back(e);

  return res;
}


// Simplify an iteration domain.
DomainSimplificationResult SimplifyDomain(const Expr& cond,
                                          const Array<Var>& axis,
                                          Map<Var, Range> vranges,
                                          bool eliminate_div_mod) {
  if (eliminate_div_mod) {
    auto elim_res = EliminateDivMod(cond, vranges);

    Map<Var, Range> new_vranges = elim_res.ranges;
    Array<Var> new_axis = Concat(axis, elim_res.new_variables);
    Expr new_cond = elim_res.expr && All(elim_res.conditions);

    auto res = SimplifyDomain(new_cond, new_axis, new_vranges, false);

    Map<Var, Expr> new_old_to_new;
    for (const Var& v : axis) {
      new_old_to_new.Set(v, res.old_to_new[v]);
    }

    Map<Var, Expr> new_new_to_old;
    for (const auto& pair : res.new_to_old) {
      new_new_to_old.Set(pair.first, Substitute(pair.second, elim_res.substitution));
    }

    res.old_to_new = std::move(new_old_to_new);
    res.new_to_old = std::move(new_new_to_old);

    return res;
  }

  auto factoratomic_res = FactorOutAtomicFormulas(cond);
  std::vector<Expr>& atomic_formulas = factoratomic_res.atomic_formulas;
  Expr rest_of_cond = factoratomic_res.rest;

  // Put rest_of_cond into the vector of atomic formulas so that we don't forget about it.
  // Although rest_of_cond is not atomic, the subsequent functions won't complain about it.
  atomic_formulas.push_back(rest_of_cond);

  // vars are variables from axis followed by all the other variables from vranges
  Array<Var> vars = axis;
  for (const auto& pair : vranges) {
    bool already = false;
    for (const Var& v : vars) {
      already = already || v.same_as(pair.first);
    }
    if (!already) {
      vars.push_back(pair.first);
    }
  }

  auto solved_system = SolveSystemOfInequalities(atomic_formulas, vars, vranges);

  DomainSimplificationResult res;
  std::unordered_map<const Variable*, IntSet> new_var_intsets;

  // Initialize new_var_intsets with the old var intsets
  for (const auto& pair : vranges) {
    new_var_intsets[pair.first.get()] = IntSet::range(pair.second);
  }

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = axis.rbegin(); it != axis.rend(); ++it) {
    const Var& var = *it;
    auto& bnd = solved_system.bounds[var.get()];
    // Note that we replace old vars with new ones
    bnd = bnd.substitute(res.old_to_new);
    if (is_one(bnd.coef) && !bnd.equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity, so it must be
      // the simplest one.
      res.old_to_new.Set(var, bnd.equal[0]);
    } else {
      Array<Expr> lowers = Concat(bnd.equal, bnd.lower);
      Array<Expr> uppers = Concat(bnd.equal, bnd.upper);

      // Here we will try all pairs of lower and upper bounds and find the best pair, that is, the
      // pair with the minimal difference between the upper and the lower.
      // Note that the bounds are for v*coef, not for v (because we don't want complex expressions
      // involving division).

      // The lower bound of the best pair so far
      Expr best_lower = vranges[var]->min * bnd.coef;
      // The difference between the upper and the lower of the best pair so far
      Expr best_diff = (vranges[var]->extent - 1) * bnd.coef;
      // The overapproximation of the best difference
      Expr best_diff_over = best_diff;

      for (const Expr& low : lowers) {
        for (const Expr& upp : uppers) {
          Expr diff = SuperSimplify(upp - low, vranges);
          // Since diff may depend on some other variables, we compute its overapproximation
          Expr diff_over = EvalSet(diff, new_var_intsets).max();

          if (diff_over.same_as(HalideIR::Internal::Interval::pos_inf)) {
            continue;
          }

          // If it is provable that the new one is strictly better than the current best one,
          // then replace it. Note that we are biased towards earlier pairs which should be simpler.
          if (CanProve(diff_over - best_diff_over < 0, vranges)) {
            best_lower = low;
            best_diff = diff;
            best_diff_over = diff_over;
          }
        }
      }

      if (is_const_int(best_diff, 0)) {
        // In this case coef*iv = best_lower
        // Don't create an itervar, just replace it everywhere with its min
        res.old_to_new.Set(var, SuperSimplify(best_lower / bnd.coef, vranges));
        // To assure correctness, we have to add a condition that best_lower can be divided by coef
        res.conditions.push_back(SuperSimplify(best_lower % bnd.coef == 0, vranges));
      } else {
        std::string suffix = Equal(best_lower, vranges[var]->min * bnd.coef) ? "" : ".shifted";
        Var new_var = var.copy_with_suffix(suffix);

        // We will replace our iv with new_var + shift.
        // We use rounding-up division to compute shift. Since we want to use a single formula
        // without selects in as many cases as possible, we try to prove conditions manually.
        Expr shift;
        if (CanProve(best_lower <= 0, vranges)) {
          shift = best_lower / bnd.coef;
        } else if (CanProve(best_lower > -bnd.coef, vranges)) {
          shift = (best_lower + bnd.coef - 1)/bnd.coef;
        } else {
          shift = Select::make(best_lower <= -bnd.coef,
                               best_lower / bnd.coef,
                               (best_lower + bnd.coef - 1)/bnd.coef);
        }
        shift = SuperSimplify(shift, vranges);

        Expr diff = SuperSimplify(best_diff_over / bnd.coef, vranges);

        if (is_const_int(diff, 0)) {
          // Don't create an itervar, just replace it everywhere with its min
          res.old_to_new.Set(var, shift);
        } else {
          res.old_to_new.Set(var, new_var + shift);
          // Note that we are substituting old with new, so best_lower contains new var,
          // that is we have to substitute new with old in best_lower here
          res.new_to_old.Set(new_var,
                             SuperSimplify(var - Substitute(shift, res.new_to_old), vranges));

          new_var_intsets[new_var.get()] = IntSet::interval(make_zero(new_var.type()), diff);

          // Add the new var to the resulting axis
          auto range = Range(make_zero(new_var.type()), SuperSimplify(diff + 1, vranges));
          res.axis.push_back(new_var);
          res.ranges.Set(new_var, range);
          vranges.Set(new_var, range);
        }
      }
    }
  }

  // Add the original conditions (with variables substituted) to the resulting conditions
  for (const Expr& old_cond : solved_system.as_conditions()) {
    res.conditions.push_back(SuperSimplify(Substitute(old_cond, res.old_to_new), vranges));
  }

  return res;
}

// Use the condition of a reduction op to simplify its domain (axis)
Expr SimplifyReductionDomain(const Expr& expr, const Map<Var, Range>& outer_vranges) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Map<Var, Range> vranges = Merge(outer_vranges, IterVarsToMap(red->axis));
    auto res = SimplifyDomain(red->condition, IterVarsToVars(red->axis),
                              Merge(outer_vranges, IterVarsToMap(red->axis)));

    Array<Expr> new_source;
    for (const Expr& src : red->source) {
      new_source.push_back(Substitute(src, res.old_to_new));
    }

    Array<IterVar> new_axis = IterVarsFromMap(res.axis, res.ranges, kCommReduce);

    // Perform simplification mainly to remove a possibly empty reduction.
    return Simplify(Reduce::make(red->combiner, new_source, new_axis,
                                 All(res.conditions), red->value_index));
  } else {
    return expr;
  }
}

// Extract the given expr under the given condition as a separate tensor if the volume of the
// extracted tensor will be less than the volume of the outer_axis
Expr ExtractAsTensorMaybe(const Expr& e, const Expr& cond,
                          const Array<Var>& outer_axis,
                          const Map<Var, Range>& vranges) {
  // TODO(sgrechanik-h): We don't use divmod elimination here because of some performance problems
  auto res = SimplifyDomain(cond, outer_axis, vranges, false);

  Expr new_expr = SuperSimplify(Substitute(e, res.old_to_new), vranges);

  // Keep only those variables of the new axis which are used in the new_expr
  {
    Array<Var> used_res_axis;
    for (const Var& var : res.axis) {
      if (ExprUseVar(new_expr, var)) {
        used_res_axis.push_back(var);
      }
    }

    res.axis = std::move(used_res_axis);
  }

  // Use the new axis to simplify the new expr, removing redundant inequalities
  new_expr = SuperSimplify(new_expr, res.ranges);

  // If the expression does not use vars then it is probably better to keep it inlined
  if (res.axis.empty()) {
    return new_expr;
  }

  // Compute volumes before and after
  Expr old_volume = make_const(Int(64), 1);
  for (const Var& var : outer_axis) {
    old_volume = old_volume * vranges[var]->extent;
  }

  Expr new_volume = make_const(Int(64), 1);
  for (const Var& var : res.axis) {
    new_volume = new_volume * res.ranges[var]->extent;
  }

  // if we can prove that the old volume is not greater than the new volume then
  // prefer the old expression.
  if (CanProve(old_volume <= new_volume, vranges)) {
    return e;
  }

  Tensor tensor = op::TensorFromExpr(new_expr, IterVarsFromMap(res.axis, res.ranges),
                                     "extracted_tensor");

  Array<Expr> args;
  for (const Var& var : res.axis) {
    args.push_back(res.new_to_old[var]);
  }

  return Call::make(e.type(), tensor->op->name, args,
                    Call::CallType::Halide, tensor->op, tensor->value_index);
}


class RemoveRedundantInequalitiesMutator : public IRMutator {
 public:
  explicit RemoveRedundantInequalitiesMutator(Array<Expr> known) {
    for (const Expr& cond : known) {
      known_.push_back(SuperSimplify(cond));
    }
  }

  virtual Expr Mutate_(const Select* op, const Expr& e) {
    bool has_side_effect = HasSideEffect(e);
    Expr new_cond = SuperSimplify(Mutate(op->condition));
    if (is_one(new_cond) && !has_side_effect) {
      return Mutate(op->true_value);
    } else if (is_zero(new_cond) && !has_side_effect) {
      return Mutate(op->false_value);
    } else {
      Array<Expr> new_known = known_;
      for (const Expr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
        new_known.push_back(atomic);
      }
      RemoveRedundantInequalitiesMutator new_mutator(new_known);
      // Note that we mutate only the true value with the new mutator
      // TODO(sgrechanik-h): Update known conditions for the false value as well
      return Select::make(new_cond, new_mutator.Mutate(op->true_value), Mutate(op->false_value));
    }
  }

  virtual Expr Mutate_(const Call* op, const Expr& e) {
    if (op->name == intrinsic::tvm_if_then_else) {
      Expr new_cond = SuperSimplify(Mutate(op->args[0]));
      if (is_one(new_cond)) {
        return Mutate(op->args[1]);
      } else if (is_zero(new_cond)) {
        return Mutate(op->args[2]);
      } else {
        Array<Expr> new_known = known_;
        for (const Expr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
          new_known.push_back(atomic);
        }
        RemoveRedundantInequalitiesMutator new_mutator(new_known);
        // Note that we mutate only the true value with the new mutator
        // TODO(sgrechanik-h): Update known conditions for the false value as well
        return if_then_else(new_cond, new_mutator.Mutate(op->args[1]), Mutate(op->args[2]));
      }
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  virtual Expr Mutate_(const Reduce* op, const Expr& e) {
    Array<Expr> known_with_axes = known_;
    for (const Expr& axis_cond : IterVarsToInequalities(op->axis)) {
      known_with_axes.push_back(axis_cond);
    }
    RemoveRedundantInequalitiesMutator mutator_with_axes(known_with_axes);

    Expr new_cond = mutator_with_axes.Mutate(op->condition);

    Array<Expr> new_known = known_with_axes;
    for (const Expr& atomic : FactorOutAtomicFormulas(new_cond).atomic_formulas) {
      new_known.push_back(atomic);
    }
    RemoveRedundantInequalitiesMutator new_mutator(new_known);

    Array<Expr> new_source;
    for (const Expr& src : op->source) {
      new_source.push_back(new_mutator.Mutate(src));
    }

    return Reduce::make(op->combiner, new_source, op->axis, new_cond, op->value_index);
  }

  virtual Expr Mutate_(const EQ* op, const Expr& e) { return MutateAtomic_(e); }
  virtual Expr Mutate_(const NE* op, const Expr& e) { return MutateAtomic_(e); }
  virtual Expr Mutate_(const LT* op, const Expr& e) { return MutateAtomic_(e); }
  virtual Expr Mutate_(const LE* op, const Expr& e) { return MutateAtomic_(e); }
  virtual Expr Mutate_(const GT* op, const Expr& e) { return MutateAtomic_(e); }
  virtual Expr Mutate_(const GE* op, const Expr& e) { return MutateAtomic_(e); }

  virtual Expr Mutate_(const And* op, const Expr& e) {
    return Mutate(op->a) && Mutate(op->b);
  }

 private:
  Expr MutateAtomic_(const Expr& e) {
    Expr simplified = SuperSimplify(e);
    for (const Expr& other : known_) {
      if (Equal(simplified, other)) {
        return const_true();
      }
    }
    return simplified;
  }

  Array<Expr> known_;
};

// Propagate information from conditions and remove redundant inequalities
// TODO(sgrechanik-h): This should be merged into standard simplifiers
Expr RemoveRedundantInequalities(const Expr& expr, const Array<Expr>& known) {
  return RemoveRedundantInequalitiesMutator(known).Mutate(expr);
}

// Extract from cond an implication of cond not containing vars
std::pair<Expr, Expr> ImplicationNotContainingVars(
    const Expr& cond, const std::unordered_set<const Variable*>& vars) {
  CHECK(cond.type().is_bool()) << "The type of cond must be bool";
  // TODO(sgrechanik-h): not
  if (const And* op = cond.as<And>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return {pair_a.first && pair_b.first,
            pair_a.second && pair_b.second};
  } else if (const Or* op = cond.as<Or>()) {
    auto pair_a = ImplicationNotContainingVars(op->a, vars);
    auto pair_b = ImplicationNotContainingVars(op->b, vars);
    return {Or::make(pair_a.first, pair_b.first), cond};
  } else if (!ExprUseVar(cond, vars)) {
    return {cond, const_true()};
  } else {
    return {const_true(), cond};
  }
}

// Factor conditions out of a reduction by applying Fourier-Motzkin elimination and moving out
// (in)equalities which do not depend on the reduction variables.
std::pair<Expr, Expr> LiftConditionsThroughReduction(const Expr& cond,
                                                     const Array<IterVar>& red_axis,
                                                     const Array<IterVar>& outer_axis) {
  // Factor out atomics so that we can consider this as a system of inequalities
  auto factoratomic_res = FactorOutAtomicFormulas(cond);
  Array<Expr> atomics = factoratomic_res.atomic_formulas;
  const Expr& rest = factoratomic_res.rest;

  Array<Var> allvars;
  for (const IterVar& v : red_axis) {
    allvars.push_back(v->var);
  }
  for (const IterVar& v : outer_axis) {
    allvars.push_back(v->var);
  }

  auto vranges = Merge(IterVarsToMap(red_axis), IterVarsToMap(outer_axis));
  // start from reduction vars, so that input vars don't depend on them
  atomics = SolveSystemOfInequalities(atomics, allvars, vranges).as_conditions();

  // Append the rest part
  Expr rewritten_cond = All(atomics) && rest;

  std::unordered_set<const Variable*> vset;
  for (const IterVar& v : red_axis) {
    vset.insert(v->var.get());
  }

  // The outer (first) condition does not contain reduction vars,
  // the inner (second) condition is everything else
  return ImplicationNotContainingVars(rewritten_cond, vset);
}

class ExtractReductionsMutator : public IRMutator {
 public:
  explicit ExtractReductionsMutator(const Array<Var>& outer_axis,
                                    Map<Var, Range> vranges,
                                    std::string name = "extracted_reduction")
    : outer_axis_(outer_axis), vranges_(std::move(vranges)), name_(std::move(name)) {}

  Expr Mutate_(const Reduce* op, const Expr& e) {
    ExtractReductionsMutator new_mutator(Concat(IterVarsToVars(op->axis), outer_axis_),
                                         Merge(vranges_, IterVarsToMap(op->axis)),
                                         name_);

    Array<Expr> new_source;
    for (const Expr& src : op->source) {
      new_source.push_back(new_mutator.Mutate(src));
    }

    Expr new_reduce =
      Reduce::make(op->combiner, new_source, op->axis, op->condition, op->value_index);

    ExprFreeVarsVisitor fv_visitor;
    fv_visitor.Visit(new_reduce);

    // Vars of the tensor we are going to create for this reduction
    Array<Var> vars;
    for (const Var& v : outer_axis_) {
      // We take variables from the outer_axis_ which are also present in the new reduction
      if (fv_visitor.free.count(v.get())) {
        vars.push_back(v);
      }
    }

    auto newaxis_vmap_pair = CloneIterVars(IterVarsFromMap(vars, vranges_));
    Array<IterVar> new_axis = newaxis_vmap_pair.first;
    new_reduce = SuperSimplify(Substitute(new_reduce, newaxis_vmap_pair.second),
                               IterVarsToMap(new_axis));

    Tensor tensor = op::TensorFromExpr(new_reduce, new_axis, name_, tag_, attrs_);

    Array<Expr> args;
    for (const Var& v : vars) {
      args.push_back(v);
    }

    return Call::make(e.type(), tensor->op->name, args,
                      Call::CallType::Halide, tensor->op, tensor->value_index);
  }

 private:
  Array<Var> outer_axis_;
  Map<Var, Range> vranges_;
  std::string name_;
  std::string tag_;
  Map<std::string, NodeRef> attrs_;
};

// Extract reductions as separate tensors.
Expr ExtractReductions(const Expr& expr,
                       const Array<Var>& outer_axis,
                       const Map<Var, Range>& vranges) {
  return ExtractReductionsMutator(outer_axis, vranges).Mutate(expr);
}

Expr ExtractNonTopReductions(const Expr& expr,
                             const Array<Var>& outer_axis,
                             const Map<Var, Range>& vranges) {
  if (const Reduce* red = expr.as<Reduce>()) {
    Array<Var> new_outer_axis = Concat(IterVarsToVars(red->axis), outer_axis);
    Map<Var, Range> new_vranges = Merge(vranges, IterVarsToMap(red->axis));
    Array<Expr> new_source;
    for (const Expr& src : red->source) {
      new_source.push_back(ExtractReductions(src, new_outer_axis, new_vranges));
    }
    Expr new_condition = ExtractReductions(red->condition, new_outer_axis, new_vranges);

    return Reduce::make(red->combiner, new_source, red->axis,
                        new_condition, red->value_index);
  } else {
    return ExtractReductions(expr, outer_axis, vranges);
  }
}

Expr OptimizeAndLiftNonzeronessConditionsImpl(const Expr& expr, const Array<IterVar>& axis) {
  Expr result;

  if (const Reduce* red = expr.as<Reduce>()) {
    // TODO(sgrechanik-h): There are some other operations which behave like sum
    bool is_sum = IsSumCombiner(red->combiner);
    if (is_sum || CanFactorZeroFromCombiner(red->combiner, red->value_index)) {
      Expr new_red = expr;

      // Here we simplify the reduction
      {
        Expr cond = red->condition;
        Array<Expr> source = red->source;

        // If it is a summation then we can lift nonzeroness conditions from the source
        // and add them to the reduction conditions
        if (is_sum) {
          auto nz = NonzeronessCondition(red->source[red->value_index]);
          cond = nz.cond && cond;
          source.Set(0, nz.value);
        }

        new_red = Reduce::make(red->combiner, source, red->axis, cond, red->value_index);
        new_red = SimplifyReductionDomain(new_red, IterVarsToMap(axis));
        red = new_red.as<Reduce>();

        // If the reduction disappears completely then transform the result as a non-reduction
        if (!red) {
          return OptimizeAndLiftNonzeronessConditionsImpl(new_red, axis);
        }
      }

      Expr new_outer_cond, new_reduce_cond;
      Array<Expr> new_source = red->source;

      // Partially lift conditions from the reduce condition
      std::tie(new_outer_cond, new_reduce_cond) =
        LiftConditionsThroughReduction(red->condition, red->axis, axis);

      // If it's not sum then we haven't yet lifted nonzeroness cond from the source
      if (!is_sum) {
        Expr outer_nz_cond, nz_cond, nz_source;
        auto nz = NonzeronessCondition(red->source[red->value_index]);
        // Append conditions from the reduction
        nz_cond = new_reduce_cond && nz.cond;
        nz_source = nz.value;
        std::tie(outer_nz_cond, nz_cond) =
          LiftConditionsThroughReduction(nz_cond, red->axis, axis);
        new_outer_cond = new_outer_cond && outer_nz_cond;
        new_source.Set(red->value_index, SelectElseZero(nz_cond, nz_source));
      }

      Expr new_reduce = Reduce::make(red->combiner, new_source, red->axis,
                                     new_reduce_cond, red->value_index);
      new_reduce = ExtractAsTensorMaybe(new_reduce, new_outer_cond,
                                        IterVarsToVars(axis), IterVarsToMap(axis));
      result = SelectElseZero(new_outer_cond, new_reduce);
    } else {
      return SimplifyReductionDomain(expr, IterVarsToMap(axis));
    }
  } else {
    auto nz = NonzeronessCondition(expr);
    Expr new_expr = ExtractAsTensorMaybe(nz.value, nz.cond,
                                         IterVarsToVars(axis), IterVarsToMap(axis));
    result = SelectElseZero(nz.cond, new_expr);
  }

  // Note that RemoveRedundantInequalities can sometimes propagate equalities which
  // other simplifiers cannot, like (i % 3) == 0.
  Array<Expr> axis_conds = IterVarsToInequalities(axis);
  result = RemoveRedundantInequalities(result, axis_conds);

  // Sometimes ExtractAsTensorMaybe doesn't perform extraction, so there may be some non-top
  // reductions left, take care of them
  Map<Var, Range> vrange = IterVarsToMap(axis);
  return SuperSimplify(ExtractReductions(result, IterVarsToVars(axis), vrange),
                       vrange);
}

Tensor OptimizeAndLiftNonzeronessConditions(const Tensor& tensor) {
  return op::TransformBody(tensor, OptimizeAndLiftNonzeronessConditionsImpl);
}

TVM_REGISTER_API("ir_pass.IsSumCombiner")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IsSumCombiner(args[0]);
  });

TVM_REGISTER_API("ir_pass.CanFactorZeroFromCombiner")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = CanFactorZeroFromCombiner(args[0], args[1]);
  });

TVM_REGISTER_API("ir_pass.LiftNonzeronessCondition")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = LiftNonzeronessCondition(args[0]);
  });

TVM_REGISTER_API("ir_pass.InlineTailCall")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = InlineTailCall(args[0]);
  });

TVM_REGISTER_API("ir_pass.InlineTensors")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Expr>()) {
      Expr e = args[0];
      if (args.size() == 1) {
        *ret = InlineTensors(e);
      } else if (args.size() == 2) {
        *ret = InlineTensors(e, args[1]);
      } else if (args.size() >= 3) {
        *ret = InlineTensors(e, args[1], args[2]);
      }
    } else if (args[0].IsNodeType<Tensor>()) {
      Tensor t = args[0];
      if (args.size() == 1) {
        *ret = InlineTensors(t);
      } else if (args.size() == 2) {
        *ret = InlineTensors(t, args[1]);
      } else if (args.size() >= 3) {
        *ret = InlineTensors(t, args[1], args[2]);
      }
    }
  });

TVM_REGISTER_API("ir_pass.SolveSystemOfInequalities")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SolveSystemOfInequalities(args[0], args[1], args[2]).as_conditions();
  });

TVM_REGISTER_API("ir_pass.SimplifyDomain")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    auto res = SimplifyDomain(args[0], args[1], args[2]);
    Array<IterVar> axis = IterVarsFromMap(res.axis, res.ranges);
    *ret = Array<NodeRef>({All(res.conditions), axis, res.old_to_new, res.new_to_old});
  });

TVM_REGISTER_API("ir_pass.SimplifyReductionDomain")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SimplifyReductionDomain(args[0], args[1]);
  });

TVM_REGISTER_API("ir_pass.ExtractAsTensorMaybe")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExtractAsTensorMaybe(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_API("ir_pass.ExtractReductions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExtractReductions(args[0], args[1], args[2]);
  });

TVM_REGISTER_API("ir_pass.ExtractNonTopReductions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExtractNonTopReductions(args[0], args[1], args[2]);
  });

TVM_REGISTER_API("ir_pass.OptimizeAndLiftNonzeronessConditions")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = OptimizeAndLiftNonzeronessConditions(args[0]);
  });

}  // namespace ir
}  // namespace tvm
