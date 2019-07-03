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
 * \file de_duplicate.cc
 * \brief Use a fresh Id for every Var to make the result well-formed.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/pattern_functor.h>
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

Expr DeDup(const Expr& e) {
  class DeDupMutator : public TypeMutator,
                       public ExprMutator,
                       public PatternMutator {
   public:
    TypeVar Fresh(const TypeVar& tv) {
      TypeVar ret = TypeVarNode::make(tv->var->name_hint, tv->kind);
      type_rename_[tv] = ret;
      return ret;
    }

    Var Fresh(const Var& v) {
      Var ret = VarNode::make(v->name_hint(), VisitType(v->type_annotation));
      rename_[v] = ret;
      return ret;
    }

    Expr VisitExpr(const Expr& e) final {
      return ExprMutator::VisitExpr(e);
    }

    Expr VisitExpr_(const VarNode* op) final {
      Var v = GetRef<Var>(op);
      return rename_.count(v) != 0 ? rename_.at(v) : v;
    }

    Expr VisitExpr_(const LetNode* op) final {
      Var v = Fresh(op->var);
      return LetNode::make(v, VisitExpr(op->value), VisitExpr(op->body));
    }

    Type VisitType(const Type& t) final {
      return t.defined() ? TypeMutator::VisitType(t) : t;
    }

    Expr VisitExpr_(const FunctionNode* op) final {
      tvm::Array<TypeVar> type_params;
      for (const TypeVar& type_param : op->type_params) {
        type_params.push_back(Fresh(type_param));
      }
      tvm::Array<Var> params;
      for (const Var& param : op->params) {
        params.push_back(Fresh(param));
      }
      return FunctionNode::make(params,
                                VisitExpr(op->body),
                                VisitType(op->ret_type),
                                type_params,
                                op->attrs);
    }

    Pattern VisitPattern(const Pattern& p) final {
      return PatternMutator::VisitPattern(p);
    }

    Pattern VisitPattern_(const PatternVarNode* op) final {
      return PatternVarNode::make(Fresh(op->var));
    }

    Clause VisitClause(const Clause& c) final {
      Pattern pat = VisitPattern(c->lhs);
      return ClauseNode::make(pat, VisitExpr(c->rhs));
    }

    Type VisitType_(const TypeVarNode* op) final {
      TypeVar v = GetRef<TypeVar>(op);
      return type_rename_.count(v) != 0 ? type_rename_.at(v) : v;
    }

    Var VisitVar(const Var& v) final {
      return Fresh(v);
    }

   private:
    std::unordered_map<Var, Var, NodeHash, NodeEqual> rename_;
    std::unordered_map<TypeVar, TypeVar, NodeHash, NodeEqual> type_rename_;
  };

  Expr ret = DeDupMutator().VisitExpr(e);
  CHECK_EQ(FreeVars(ret).size(), FreeVars(e).size());
  return ret;
}

TVM_REGISTER_API("relay._transform.dedup")
.set_body_typed(DeDup);

}  // namespace relay
}  // namespace tvm
