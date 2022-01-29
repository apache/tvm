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
 * \file de_duplicate.cc
 * \brief Use a fresh Id for every Var to make the result well-formed.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

#include <stack>

namespace tvm {
namespace relay {

Expr DeDup(const Expr& e) {
  class DeDupMutator : public TypeMutator, public MixedModeMutator, public PatternMutator {
   public:
    TypeVar Fresh(const TypeVar& tv) {
      TypeVar ret = TypeVar(tv->name_hint, tv->kind);
      type_rename_[tv] = ret;
      return ret;
    }

    Var Fresh(const Var& v) {
      ICHECK_EQ(rename_.count(v), 0);
      ICHECK_EQ(memo_.count(v), 0) << v.as<VarNode>();
      Var ret = Var(v->name_hint(), VisitType(v->type_annotation));
      rename_[v] = ret;
      return ret;
    }

    Expr DispatchVisitExpr(const Expr& e) final {
      auto ret = ExprMutator::VisitExpr(e);
      ret->checked_type_ = e->checked_type_;
      ret->virtual_device_ = e->virtual_device_;
      return ret;
    }

    using MixedModeMutator::VisitExpr_;

    Expr VisitExpr_(const VarNode* op) final {
      Var v = GetRef<Var>(op);
      return rename_.count(v) != 0 ? rename_.at(v) : v;
    }

    Expr VisitExpr_(const LetNode* op) final {
      std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> new_vars;
      auto pre_visit = [this, &new_vars](const LetNode* op) {
        Expr expr = GetRef<Expr>(op);
        new_vars[expr] = this->Fresh(op->var);
        // Rely on the Memoizer to cache pre-visit values
        this->VisitExpr(op->value);
      };
      auto post_visit = [this, &new_vars](const LetNode* op) {
        Expr expr = GetRef<Expr>(op);
        this->memo_[expr] =
            Let(new_vars[expr], this->VisitExpr(op->value), this->VisitExpr(op->body));
      };
      ExpandANormalForm(op, pre_visit, post_visit);
      return memo_[GetRef<Expr>(op)];
    }

    Type VisitType(const Type& t) final { return t.defined() ? TypeMutator::VisitType(t) : t; }

    Expr VisitExpr_(const FunctionNode* func_node) final {
      tvm::Array<TypeVar> type_params;
      for (const TypeVar& type_param : func_node->type_params) {
        type_params.push_back(Fresh(type_param));
      }
      tvm::Array<Var> params;
      for (const Var& param : func_node->params) {
        params.push_back(Fresh(param));
      }
      return WithFields(GetRef<Function>(func_node), params, VisitExpr(func_node->body),
                        VisitType(func_node->ret_type), type_params);
    }

    Pattern VisitPattern(const Pattern& p) final { return PatternFunctor::VisitPattern(p); }

    Pattern VisitPattern_(const PatternVarNode* op) final { return PatternVar(Fresh(op->var)); }

    Type VisitType_(const TypeVarNode* op) final {
      TypeVar v = GetRef<TypeVar>(op);
      return type_rename_.count(v) != 0 ? type_rename_.at(v) : v;
    }

    Var VisitVar(const Var& v) final { return Fresh(v); }

   private:
    std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> rename_;
    std::unordered_map<TypeVar, TypeVar, ObjectPtrHash, ObjectPtrEqual> type_rename_;
  };
  ICHECK(WellFormed(e)) << AsText(e, false);
  Expr ret = DeDupMutator().VisitExpr(e);
  ICHECK(WellFormed(ret));
  ICHECK_EQ(FreeVars(e).size(), FreeVars(ret).size());
  return ret;
}  // namespace relay

TVM_REGISTER_GLOBAL("relay._transform.dedup").set_body_typed(DeDup);

}  // namespace relay
}  // namespace tvm
