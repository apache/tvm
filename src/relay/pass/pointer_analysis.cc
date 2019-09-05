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
 * Copyright (c) 2019 by Contributors
 *
 * \file partial_eval.cc
 *
 * \brief Pointer analysis via Andersen's Algorithm.
 * Analyze may-alias, escape, pointer content.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include "pass_util.h"

namespace tvm {
namespace relay {

struct AllocAbstractLocation : ExprVisitor {
  Map<Expr, AbstractLocation> spawn;
  Map<AbstractLocation, Expr> origin;
  void VisitExpr_(const RefCreateNode* op) final {
    Expr refc = GetRef<RefCreate>(op);
    auto loc = AbstractLocationNode::make(spawn.size() + 1);  // 0 is reserved for external ref
    spawn.Set(refc, loc);
    origin.Set(loc, refc);
    ExprVisitor::VisitExpr_(op);
  }
};

AbstractLocation external_ref = AbstractLocationNode::make(0);

struct Analyze : ExprFunctor<void(const Expr&)> {
  Map<Expr, AbstractLocation> spawn;
  Map<AbstractLocation, Expr> origin;
  Map<Expr, Set<AbstractLocation>> contain;
  Map<AbstractLocation, Set<Expr>> store;
  bool progress = true;
  Analyze(Map<Expr, AbstractLocation> spawn,
          Map<AbstractLocation, Expr> origin) : spawn(spawn), origin(origin) {
    for (const auto& p : origin) {
      store.Set(p.first, Set<Expr>());
    }
  }

  void analyze(const Expr& e) {
    progress = false;
    VisitExpr(e);
  }

  void VisitExpr(const Expr& e) final {
    if (contain.count(e) == 0) {
      contain.Set(e, Set<AbstractLocation>());
    }
    ExprFunctor::VisitExpr(e);
  }

  void VisitExpr_(const ConstantNode* op) final { }
  void VisitExpr_(const OpNode* op) final { }
  void VisitExpr_(const ConstructorNode* op) final { }

  void AddContain(const Expr& e, AbstractLocation loc) {
    Set<AbstractLocation> locs = contain.at(e);
    if (locs.count(loc) == 0) {
      progress = true;
      locs.Insert(loc);
      contain.Set(e, locs);
    }
  }

  void AddStore(AbstractLocation loc, const Expr& expr) {
    Set<Expr> exprs = store.at(loc);
    if (exprs.count(expr) == 0) {
      progress = true;
      exprs.Insert(expr);
      store.Set(loc, exprs);
    }
  }

  void UnionContain(const Expr& parent, const Expr& child) {
    for (auto loc : contain.at(child)) {
      AddContain(parent, loc);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    VisitExpr(op->op);
    for (Expr e : op->args) {
      VisitExpr(e);
    }
    Expr expr = GetRef<Expr>(op);
    CHECK(op->op.as<OpNode>()) << "Only support call to operator right now.";
    UnionContain(expr, op->op);
    for (Expr e : op->args) {
      UnionContain(expr, e);
    }
  }

  void VisitExpr_(const RefCreateNode* op) final {
    VisitExpr(op->value);
    Expr expr = GetRef<Expr>(op);
    AddContain(expr, spawn.at(expr));
    AddStore(spawn.at(expr), op->value);
  }

  void VisitExpr_(const RefReadNode* op) final {
    VisitExpr(op->ref);
    Expr lhs = GetRef<Expr>(op);
    CHECK_GT(contain.count(op->ref), 0);
    std::unordered_set<Expr, NodeHash, NodeEqual> exprs;
    for (const auto& abs_loc : contain.at(op->ref)) {
      if (abs_loc == external_ref) {
        AddContain(lhs, external_ref);
      } else {
        CHECK_GT(store.count(abs_loc), 0);
        for (const auto& expr : store.at(abs_loc)) {
          exprs.insert(expr);
        }
      }
    }
    for (const auto& rhs : exprs) {
      UnionContain(lhs, rhs);
    }
  }

  void VisitExpr_(const RefWriteNode* op) final {
    VisitExpr(op->ref);
    VisitExpr(op->value);
    CHECK_GT(contain.count(op->ref), 0);
    for (const auto& loc : contain.at(op->ref)) {
      AddStore(loc, op->value);
    }
  }
};

PointerAnalysisResult PointerAnalysis(const Expr& e) {
  AllocAbstractLocation aal;
  aal(e);
  Map<Expr, AbstractLocation> spawn = aal.spawn;
  Map<AbstractLocation, Expr> origin = aal.origin;
  Analyze ana(spawn, origin);
  while (ana.progress) {
    ana.analyze(e);
  }
  return PointerAnalysisResultNode::make(spawn, origin, ana.contain, ana.store);
}

TVM_REGISTER_API("relay._analysis.PointerAnalysis")
.set_body_typed(PointerAnalysis);

}  // namespace relay
}  // namespace tvm
