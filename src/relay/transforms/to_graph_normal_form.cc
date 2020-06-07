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
 *
 * \file to_gnf.cc
 *
 * \brief Turn A normal form into graph normal form.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "let_list.h"

namespace tvm {
namespace relay {

class UseVarVisitor : public ExprVisitor {
 public:
  explicit UseVarVisitor(const Var& v) : v(v) {}

  static bool UseVar(const Var& v, const Expr& e) {
    UseVarVisitor uv(v);
    uv(e);
    return uv.use_var;
  }

 private:
  bool use_var = false;
  Var v;

  void VisitExpr_(const VarNode* vn) override { use_var = use_var || (v == GetRef<Var>(vn)); }
};

class GNF : public ExprMutator {
 private:
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> var_map_;
  Expr VisitExpr_(const VarNode* vn) override {
    Var v = GetRef<Var>(vn);
    return var_map_.count(v) == 0 ? v : var_map_.at(v);
  }

  static bool UseVar(const Var& v, const Expr& e) { return UseVarVisitor::UseVar(v, e); }

  static Expr WrapRec(const Var& var, const Expr& val) {
    return UseVar(var, val) ? Let(var, val, var) : val;
  }

  Expr VisitExpr_(const LetNode* ln) override {
    var_map_.insert(std::pair<Var, Expr>(ln->var, WrapRec(ln->var, VisitExpr(ln->value))));
    return VisitExpr(ln->body);
  }
};

Expr ToGraphNormalForm(const Expr& e) { return GNF()(e); }

namespace transform {

Pass ToGraphNormalForm() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ToGraphNormalForm(f));
      };
  return CreateFunctionPass(pass_func, 1, "ToGraphNormalForm", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToGraphNormalForm").set_body_typed(ToGraphNormalForm);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
