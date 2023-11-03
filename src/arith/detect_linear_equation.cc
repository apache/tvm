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
 * \file detect_linear_equation.cc
 * \brief Utility to detect patterns in the expression.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

using namespace tir;

// Linear equation, the components can be undefined.
struct LinearEqEntry {
  PrimExpr base;
  PrimExpr coeff;
};

struct IntervalEntry {
  PrimExpr min_value;
  PrimExpr max_value;
};

class LinearEqDetector : public ExprFunctor<LinearEqEntry(const PrimExpr&, const PrimExpr&)> {
 public:
  explicit LinearEqDetector(Var var) : var_(var) {}

  bool Detect(const PrimExpr& e, LinearEqEntry* ret) {
    *ret = VisitExpr(e, e);
    if (fail_) return false;
    if (!ret->base.defined()) {
      ret->base = make_zero(var_.dtype());
    }
    if (!ret->coeff.defined()) {
      ret->coeff = make_zero(var_.dtype());
    }
    return true;
  }

  LinearEqEntry VisitExpr_(const AddNode* op, const PrimExpr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    LinearEqEntry ret;
    ret.base = AddCombine(a.base, b.base);
    ret.coeff = AddCombine(a.coeff, b.coeff);
    return ret;
  }

  LinearEqEntry VisitExpr_(const SubNode* op, const PrimExpr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    LinearEqEntry ret;
    ret.base = SubCombine(a.base, b.base);
    ret.coeff = SubCombine(a.coeff, b.coeff);
    return ret;
  }

  LinearEqEntry VisitExpr_(const MulNode* op, const PrimExpr& e) final {
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
  LinearEqEntry VisitExpr_(const VarNode* op, const PrimExpr& e) final {
    LinearEqEntry ret;
    if (op == var_.get()) {
      auto dtype = op->dtype;
      ret.coeff = make_const(DataType::Int(dtype.bits(), dtype.lanes()), 1);
    } else {
      ret.base = e;
    }
    return ret;
  }
  LinearEqEntry VisitExprDefault_(const Object* op, const PrimExpr& e) final {
    if (fail_) return LinearEqEntry();
    if (UsesVar(e, [this](const VarNode* var) { return var == var_.get(); })) {
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
  PrimExpr AddCombine(PrimExpr a, PrimExpr b) {
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    return a + b;
  }
  PrimExpr SubCombine(PrimExpr a, PrimExpr b) {
    // Check b first in case they are both undefined
    if (!b.defined()) return a;
    if (!a.defined()) return -b;
    return a - b;
  }
  PrimExpr MulCombine(PrimExpr a, PrimExpr b) {
    if (!a.defined()) return a;
    if (!b.defined()) return b;
    return a * b;
  }
};

Array<PrimExpr> DetectLinearEquation(const PrimExpr& e, const Array<Var>& vars) {
  PrimExpr base = e;
  Array<PrimExpr> coeff;

  for (Var v : vars) {
    LinearEqEntry ret;
    if (!LinearEqDetector(v).Detect(base, &ret)) {
      return Array<PrimExpr>();
    }
    coeff.push_back(ret.coeff);
    base = std::move(ret.base);
  }

  std::unordered_set<const VarNode*> vset;
  auto vset_contains = [&](const VarNode* node) { return vset.count(node) != 0; };

  for (size_t i = vars.size(); i > 1; --i) {
    vset.insert(vars[i - 1].get());
    // The previous coeff contains the variable
    if (UsesVar(coeff[i - 2], vset_contains)) {
      return Array<PrimExpr>();
    }
  }
  coeff.push_back(base);
  return coeff;
}

// Detect clip condition as min max value
bool DetectClipBound(const PrimExpr& cond,
                     std::unordered_map<const VarNode*, IntervalEntry>* bmap) {
  int flag = 0;
  Var var;
  auto fvisit = [&bmap, &flag, &var](const ObjectRef& n) {
    if (const VarNode* v = n.as<VarNode>()) {
      if (bmap->count(v)) {
        if (flag == 0) {
          var = Downcast<Var>(n);
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
  bool is_eq = false;
  PrimExpr canonical;
  if (const LTNode* op = cond.as<LTNode>()) {
    if (!op->a.dtype().is_int()) return false;
    canonical = op->b - op->a - make_const(op->a.dtype(), 1);
  } else if (const LENode* op = cond.as<LENode>()) {
    if (!op->a.dtype().is_int()) return false;
    canonical = op->b - op->a;
  } else if (const GTNode* op = cond.as<GTNode>()) {
    if (!op->a.dtype().is_int()) return false;
    canonical = op->a - op->b - make_const(op->a.dtype(), 1);
  } else if (const GENode* op = cond.as<GENode>()) {
    if (!op->a.dtype().is_int()) return false;
    canonical = op->a - op->b;
  } else if (const EQNode* op = cond.as<EQNode>()) {
    if (!op->a.dtype().is_int()) return false;
    canonical = op->a - op->b;
    is_eq = true;
  } else {
    return false;
  }
  LinearEqEntry ret;
  Analyzer analyzer;
  if (!LinearEqDetector(var).Detect(canonical, &ret)) return false;
  ret.coeff = analyzer.Simplify(ret.coeff);
  IntervalEntry& p = (*bmap)[var.get()];

  Optional<PrimExpr> min_value;
  Optional<PrimExpr> max_value;
  if (is_const_int(ret.coeff, 1)) {
    // var + shift >=0 -> var >= -shift
    min_value = -ret.base;
    if (is_eq) {
      max_value = min_value;
    }
  } else if (is_const_int(ret.coeff, -1)) {
    // -var + shift >=0 -> var <= shift
    max_value = ret.base;
    if (is_eq) {
      min_value = max_value;
    }
  }
  if (!min_value.defined() && !max_value.defined()) {
    return false;
  }
  if (min_value.defined()) {
    if (p.min_value.defined()) {
      p.min_value = max(p.min_value, min_value.value());
    } else {
      p.min_value = min_value.value();
    }
  }
  if (max_value.defined()) {
    if (p.max_value.defined()) {
      p.max_value = min(p.max_value, max_value.value());
    } else {
      p.max_value = max_value.value();
    }
  }
  return true;
}

template <typename OP>
void SplitCommExpr(const PrimExpr& e, std::vector<PrimExpr>* ret) {
  if (const OP* op = e.as<OP>()) {
    SplitCommExpr<OP>(op->a, ret);
    SplitCommExpr<OP>(op->b, ret);
  } else {
    ret->push_back(e);
  }
}

// Detect the lower and upper bound from the expression.
// e must be connected by and.
Array<PrimExpr> DetectClipBound(const PrimExpr& e, const Array<Var>& vars) {
  std::vector<PrimExpr> splits;
  Analyzer analyzer;
  SplitCommExpr<tir::AndNode>(analyzer.Simplify(e), &splits);
  std::unordered_map<const VarNode*, IntervalEntry> rmap;
  for (Var v : vars) {
    rmap[v.get()] = IntervalEntry();
  }
  for (PrimExpr cond : splits) {
    if (!DetectClipBound(cond, &rmap)) return Array<PrimExpr>();
  }
  Array<PrimExpr> ret;
  for (Var v : vars) {
    IntervalEntry e = rmap[v.get()];
    if (e.min_value.defined()) {
      e.min_value = analyzer.Simplify(e.min_value);
    }
    if (e.max_value.defined()) {
      e.max_value = analyzer.Simplify(e.max_value);
    }
    ret.push_back(e.min_value);
    ret.push_back(e.max_value);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("arith.DetectLinearEquation").set_body_typed(DetectLinearEquation);

TVM_REGISTER_GLOBAL("arith.DetectClipBound")
    .set_body_typed([](const PrimExpr& e, const Array<Var>& vars) {
      return DetectClipBound(e, vars);
    });
}  // namespace arith
}  // namespace tvm
