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
 * \file presburger_set.cc
 * \brief The presburger set functions
 */
#include "presburger_set.h"

#include <tvm/arith/int_set.h>
#include <tvm/arith/int_solver.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "constraint_extract.h"
#include "interval_set.h"

namespace tvm {
namespace arith {

#ifdef TVM_MLIR_VERSION
#if TVM_MLIR_VERSION >= 150
using namespace tir;

static void Update(const PrimExpr& constraint, PresburgerSetNode* intset) {
  auto& space = intset->space;
  auto constraints_union = ExtractComponents(constraint);
  for (const PrimExpr& subconstraint : constraints_union) {
    auto entries = ExtractConstraints(subconstraint, false);
    auto vars = intset->GetVars();
    IntegerRelation disjunct(entries.size(), 0, vars.size() + 1, space);
    for (const PrimExpr& entry : entries) {
      // The expression is expect to be simplified to only contain ==, <= or <
      if (entry.as<LENode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<LENode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<LENode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_b[i]) - *as_const_int(coeffs_a[i]));
        }
        disjunct.addInequality(int_coeffs);
      } else if (entry.as<LTNode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<LTNode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<LTNode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_b[i]) - *as_const_int(coeffs_a[i]));
        }
        int_coeffs[int_coeffs.size() - 1] -= 1;
        disjunct.addInequality(int_coeffs);
      } else if (entry.as<EQNode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<EQNode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<EQNode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_a[i]) - *as_const_int(coeffs_b[i]));
        }
        disjunct.addEquality(int_coeffs);
      } else {
        LOG(FATAL) << "Unsupported constraint expression: " << entry->GetTypeKey();
      }
    }
    intset->unionInPlace(disjunct);
  }
}

PresburgerSet::PresburgerSet(const PrimExpr& constraint) {
  Array<Var> vars;
  PostOrderVisit(constraint, [&vars](const ObjectRef& obj) {
    if (const VarNode* new_var = obj.as<VarNode>()) {
      auto var = GetRef<Var>(new_var);
      if (!std::any_of(vars.begin(), vars.end(), [&var](const Var& v) { return v.same_as(var); })) {
        vars.push_back(var);
      }
    }
  });
  auto constraints_union = ExtractComponents(constraint);
  Analyzer analyzer;
  PrimExpr simplified_constraint = analyzer.Simplify(constraint, kSimplifyRewriteCanonicalRewrite);
  auto space = PresburgerSpace::getRelationSpace(vars.size(), 0, 0, 0);
  auto node = make_object<PresburgerSetNode>(std::move(space), vars);
  node->SetVars(vars);
  Update(simplified_constraint, node.get());
  data_ = std::move(node);
}

PresburgerSet::PresburgerSet(const std::vector<IntegerRelation>& disjuncts,
                             const Array<Var>& vars) {
  auto node = make_object<PresburgerSetNode>(disjuncts, disjuncts[0].getSpace(), vars);
  data_ = std::move(node);
}

void PresburgerSetNode::UpdateConstraint(const PrimExpr& constraint, const Array<Var>& vars) {
  Analyzer analyzer;
  PrimExpr simplified_constraint = analyzer.Simplify(constraint, kSimplifyRewriteCanonicalRewrite);
  Update(simplified_constraint, this);
  SetVars(vars);
}

PrimExpr PresburgerSetNode::GenerateConstraint() const {
  PrimExpr constraint = Bool(0);
  for (const IntegerRelation& disjunct : disjuncts) {
    PrimExpr union_entry = Bool(1);
    for (unsigned i = 0, e = disjunct.getNumEqualities(); i < e; ++i) {
      PrimExpr linear_eq = IntImm(DataType::Int(64), 0);
      if (disjunct.getNumCols() > 1) {
        for (unsigned j = 0, f = disjunct.getNumCols() - 1; j < f; ++j) {
#if TVM_MLIR_VERSION >= 160
          auto coeff = int64_t(disjunct.atEq(i, j));
#else
          auto coeff = disjunct.atEq(i, j);
#endif
          if (coeff >= 0 || is_zero(linear_eq)) {
            linear_eq = linear_eq + IntImm(DataType::Int(64), coeff) * vars[j];
          } else {
            linear_eq = linear_eq - IntImm(DataType::Int(64), -coeff) * vars[j];
          }
        }
      }
#if TVM_MLIR_VERSION >= 160
      auto c0 = int64_t(disjunct.atEq(i, disjunct.getNumCols() - 1));
#else
      auto c0 = disjunct.atEq(i, disjunct.getNumCols() - 1);
#endif
      linear_eq = linear_eq + IntImm(DataType::Int(64), c0);
      union_entry = (union_entry && (linear_eq == 0));
    }
    for (unsigned i = 0, e = disjunct.getNumInequalities(); i < e; ++i) {
      PrimExpr linear_eq = IntImm(DataType::Int(64), 0);
      if (disjunct.getNumCols() > 1) {
        for (unsigned j = 0, f = disjunct.getNumCols() - 1; j < f; ++j) {
#if TVM_MLIR_VERSION >= 160
          auto coeff = int64_t(disjunct.atIneq(i, j));
#else
          auto coeff = disjunct.atIneq(i, j);
#endif
          if (coeff >= 0 || is_zero(linear_eq)) {
            linear_eq = linear_eq + IntImm(DataType::Int(64), coeff) * vars[j];
          } else {
            linear_eq = linear_eq - IntImm(DataType::Int(64), -coeff) * vars[j];
          }
        }
      }
#if TVM_MLIR_VERSION >= 160
      auto c0 = int64_t(disjunct.atIneq(i, disjunct.getNumCols() - 1));
#else
      auto c0 = disjunct.atIneq(i, disjunct.getNumCols() - 1);
#endif
      if (c0 >= 0) {
        linear_eq = linear_eq + IntImm(DataType::Int(64), c0);
      } else {
        linear_eq = linear_eq - IntImm(DataType::Int(64), -c0);
      }
      union_entry = (union_entry && (linear_eq >= 0));
    }
    constraint = constraint || union_entry;
  }

  return constraint;
}

PresburgerSet Union(const Array<PresburgerSet>& sets) {
  CHECK_GT(sets.size(), 0);
  if (sets.size() == 1) return sets[0];
  auto relations = sets[0]->disjuncts;
  for (size_t i = 1; i < sets.size(); ++i) {
    for (const IntegerRelation& rel : sets[i]->disjuncts) {
      relations.push_back(rel);
    }
  }
  return PresburgerSet(std::move(relations), sets[0]->GetVars());
}

PresburgerSet Intersect(const Array<PresburgerSet>& sets) {
  CHECK_GT(sets.size(), 0);
  if (sets.size() == 1) return sets[0];
  auto relations = sets[0]->disjuncts;
  const auto& space = sets[0]->space;

  for (size_t i = 1; i < sets.size(); ++i) {
    ICHECK(space.isCompatible(sets[i]->space)) << "Spaces should match";
    for (const IntegerRelation& relA : sets[i]->disjuncts) {
      for (const IntegerRelation& relB : relations) {
        IntegerRelation intersection = relA.intersect(relB);
        if (!intersection.isEmpty()) relations.push_back(intersection);
      }
    }
  }
  return PresburgerSet(std::move(relations), sets[0]->GetVars());
}

IntSet EvalSet(const PrimExpr& e, const PresburgerSet& set) {
  Array<PrimExpr> tvm_coeffs = DetectLinearEquation(e, set->GetVars());
#if TVM_MLIR_VERSION >= 160
  SmallVector<mlir::presburger::MPInt> coeffs;
#else
  SmallVector<int64_t> coeffs;
#endif

  coeffs.reserve(tvm_coeffs.size());
  for (const PrimExpr& it : tvm_coeffs) {
#if TVM_MLIR_VERSION >= 160
    coeffs.push_back(mlir::presburger::MPInt(*as_const_int(it)));
#else
    coeffs.push_back(*as_const_int(it));
#endif
  }

  IntSet result = IntSet().Nothing();
  for (const IntegerRelation& it : set->disjuncts) {
    Simplex simplex(it);
    auto range = simplex.computeIntegerBounds(coeffs);
    auto maxRoundedDown(simplex.computeOptimum(Simplex::Direction::Up, coeffs));
    auto opt = range.first.getOptimumIfBounded();
#if TVM_MLIR_VERSION >= 160
    auto min = opt.has_value() ? IntImm(DataType::Int(64), int64_t(opt.value())) : neg_inf();
#else
    auto min = opt.hasValue() ? IntImm(DataType::Int(64), opt.getValue()) : neg_inf();
#endif
    opt = range.second.getOptimumIfBounded();
#if TVM_MLIR_VERSION >= 160
    auto max = opt.has_value() ? IntImm(DataType::Int(64), int64_t(opt.value())) : pos_inf();
#else
    auto max = opt.hasValue() ? IntImm(DataType::Int(64), opt.getValue()) : pos_inf();
#endif
    auto interval = IntervalSet(min, max);
    result = Union({result, interval});
  }
  return result;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PresburgerSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto set = node.as<PresburgerSetNode>();
      ICHECK(ret) << "Unknown type:" << node->GetTypeKey();
      p->stream << "{";
      p->stream << set->GetVars() << ": ";
      p->stream << node.as<PresburgerSetNode>()->GenerateConstraint();
      p->stream << "}";
    });

#endif  // TVM_MLIR_VERSION >= 150
#endif  // TVM_MLIR_VERSION

PresburgerSet MakePresburgerSet(const PrimExpr& constraint) { return PresburgerSet(constraint); }

TVM_REGISTER_GLOBAL("arith.PresburgerSet").set_body_typed(MakePresburgerSet);

TVM_REGISTER_NODE_TYPE(PresburgerSetNode);

}  // namespace arith
}  // namespace tvm
