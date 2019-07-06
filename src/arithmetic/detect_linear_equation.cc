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
 *  Copyright (c) 2017 by Contributors
 * \file detect_linear_equation.cc
 * \brief Utility to detect patterns in the expression.
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/arithmetic.h>

namespace tvm {
namespace arith {

using namespace ir;

// Linear equation, the components can be undefined.
struct LinearEqEntry {
  Expr base;
  Expr coeff;
};

struct IntervalEntry {
  Expr min_value;
  Expr max_value;
};

class LinearEqDetector
    : public ExprFunctor<LinearEqEntry(const Expr&, const Expr &)> {
 public:
  explicit LinearEqDetector(Var var)
      : var_(var) {}

  bool Detect(const Expr& e, LinearEqEntry* ret) {
    *ret = VisitExpr(e, e);
    if (fail_) return false;
    if (!ret->base.defined()) {
      ret->base = make_zero(var_.type());
    }
    if (!ret->coeff.defined()) {
      ret->coeff = make_zero(var_.type());
    }
    return true;
  }

  LinearEqEntry VisitExpr_(const Add* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    LinearEqEntry ret;
    ret.base = AddCombine(a.base, b.base);
    ret.coeff = AddCombine(a.coeff, b.coeff);
    return ret;
  }

  LinearEqEntry VisitExpr_(const Sub* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    LinearEqEntry ret;
    ret.base = SubCombine(a.base, b.base);
    ret.coeff = SubCombine(a.coeff, b.coeff);
    return ret;
  }

  LinearEqEntry VisitExpr_(const Mul* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    if (a.coeff.defined()) {
      std::swap(a, b);
    }
    if (a.coeff.defined()) {
      fail_ = true;
      return LinearEqEntry();
    }
    LinearEqEntry ret;
    ret.base = MulCombine(a.base, b.base);
    ret.coeff = MulCombine(a.base, b.coeff);
    return ret;
  }
  LinearEqEntry VisitExpr_(const Variable* op, const Expr& e) final {
    LinearEqEntry ret;
    if (op == var_.get()) {
      ret.coeff = make_const(op->type, 1);
    } else {
      ret.base = e;
    }
    return ret;
  }
  LinearEqEntry VisitExprDefault_(const Node* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    if (ExprUseVar(e, var_)) {
      fail_ = true;
      return LinearEqEntry();
    } else {
      LinearEqEntry ret;
      ret.base = e;
      return ret;
    }
  }

 private:
  Var var_;
  bool fail_{false};
  // Combine by add
  Expr AddCombine(Expr a, Expr b) {
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    return a + b;
  }
  Expr SubCombine(Expr a, Expr b) {
    // Check b first in case they are both undefined
    if (!b.defined()) return a;
    if (!a.defined()) return -b;
    return a - b;
  }
  Expr MulCombine(Expr a, Expr b) {
    if (!a.defined()) return a;
    if (!b.defined()) return b;
    return a * b;
  }
};

Array<Expr> DetectLinearEquation(const Expr& e, const Array<Var>& vars) {
  Expr base = e;
  Array<Expr> coeff;

  for (Var v : vars) {
    LinearEqEntry ret;
    if (!LinearEqDetector(v).Detect(base, &ret)) {
      return Array<Expr>();
    }
    coeff.push_back(ret.coeff);
    base = std::move(ret.base);
  }

  std::unordered_set<const Variable*> vset;
  for (size_t i = vars.size(); i > 1; --i) {
    vset.insert(vars[i - 1].get());
    // The previous coeff contains the variable
    if (ExprUseVar(coeff[i - 2], vset)) {
      return Array<Expr>();
    }
  }
  coeff.push_back(base);
  return coeff;
}

// Detect clip condition as min max value
bool DetectClipBound(
    const Expr& cond,
    std::unordered_map<const Variable*, IntervalEntry>* bmap) {
  int flag = 0;
  Var var;
  auto fvisit = [&bmap, &flag, &var](const NodeRef& n) {
    if (const Variable* v = n.as<Variable>()) {
      if (bmap->count(v)) {
        if (flag == 0) {
          var = Var(n.node_);
          flag = 1;
        } else if (flag == 1) {
          if (!var.same_as(n)) {
            flag = -1;
          }
        }
      }
    }
  };
  PostOrderVisit(cond, fvisit);
  if (flag != 1) return false;
  // canonical form: exp >= 0
  Expr canonical;
  if (const LT* op = cond.as<LT>()) {
    if (!op->a.type().is_int()) return false;
    canonical = op->b - op->a - make_const(op->a.type(), 1);
  } else if (const LE* op = cond.as<LE>()) {
    if (!op->a.type().is_int()) return false;
    canonical = op->b - op->a;
  } else if (const GT* op = cond.as<GT>()) {
    if (!op->a.type().is_int()) return false;
    canonical = op->a - op->b - make_const(op->a.type(), 1);
  } else if (const GE* op = cond.as<GE>()) {
    if (!op->a.type().is_int()) return false;
    canonical = op->a - op->b;
  } else {
    return false;
  }
  LinearEqEntry ret;
  if (!LinearEqDetector(var).Detect(canonical, &ret)) return false;
  ret.coeff = Simplify(ret.coeff);
  IntervalEntry& p = (*bmap)[var.get()];
  if (is_const_int(ret.coeff, 1)) {
    // var + shift >=0 -> var >= -shift
    if (p.min_value.defined()) {
      p.min_value = ir::Max::make(p.min_value, -ret.base);
    } else {
      p.min_value = -ret.base;
    }
    return true;
  }
  if (is_const_int(ret.coeff, -1)) {
    // -var + shift >=0 -> var <= shift
    if (p.max_value.defined()) {
      p.max_value = ir::Min::make(p.max_value, ret.base);
    } else {
      p.max_value = ret.base;
    }
    return true;
  }
  return false;
}


template<typename OP>
void SplitCommExpr(const Expr& e, std::vector<Expr>* ret) {
  if (const OP* op = e.as<OP>()) {
    SplitCommExpr<OP>(op->a, ret);
    SplitCommExpr<OP>(op->b, ret);
  } else {
    ret->push_back(e);
  }
}

// Detect the lower and upper bound from the expression.
// e must be connected by and.
Array<Expr> DetectClipBound(const Expr& e, const Array<Var>& vars) {
  std::vector<Expr> splits;
  SplitCommExpr<ir::And>(e, &splits);
  std::unordered_map<const Variable*, IntervalEntry> rmap;
  for (Var v : vars) {
    rmap[v.get()] = IntervalEntry();
  }
  for (Expr cond : splits) {
    if (!DetectClipBound(cond, &rmap)) return Array<Expr>();
  }
  Array<Expr> ret;
  for (Var v : vars) {
    IntervalEntry e = rmap[v.get()];
    if (e.min_value.defined()) {
      e.min_value = Simplify(e.min_value);
    }
    if (e.max_value.defined()) {
      e.max_value = Simplify(e.max_value);
    }
    ret.push_back(e.min_value);
    ret.push_back(e.max_value);
  }
  return ret;
}


}  // namespace arith
}  // namespace tvm
