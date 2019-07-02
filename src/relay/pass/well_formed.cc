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
 *  Copyright (c) 2018 by Contributors
 * \file well_formed.cc
 * \brief check that expression is well formed.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <unordered_set>

namespace tvm {
namespace relay {


//! brief make sure each Var is bind at most once.
class WellFormedChecker : private ExprVisitor, PatternVisitor {
  bool well_formed = true;

  std::unordered_set<Var, NodeHash, NodeEqual> s;

  void Check(const Var& v) {
    if (s.count(v) != 0) {
      well_formed = false;
    }
    s.insert(v);
  }

  void VisitExpr_(const LetNode* l) final {
    // we do letrec only for FunctionNode,
    // but shadowing let in let binding is likely programming error, and we should forbidden it.
    Check(l->var);
    CheckWellFormed(l->value);
    CheckWellFormed(l->body);
  }

  void VisitExpr_(const FunctionNode* f) final {
    for (const Var& param : f->params) {
      Check(param);
    }
    CheckWellFormed(f->body);
  }

  void VisitPattern(const Pattern& p) final {
    PatternVisitor::VisitPattern(p);
  }

  void VisitVar(const Var& v) final {
    Check(v);
  }

 public:
  bool CheckWellFormed(const Expr& e) {
    this->VisitExpr(e);
    return well_formed;
  }
};

bool WellFormed(const Expr& e) {
  return WellFormedChecker().CheckWellFormed(e);
}

TVM_REGISTER_API("relay._analysis.well_formed")
.set_body_typed(WellFormed);

}  // namespace relay
}  // namespace tvm
