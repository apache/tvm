/*!
 *  Copyright (c) 2018 by Contributors
 * \file copy.cc
 * \brief Copy expressions and types.
 */

#include "type_functor.h"
#include <tvm/relay/expr_functor.h>
#include <unordered_map>

namespace tvm {
namespace relay {

// not basing on typemutator as it might one day memorize
struct CopyType : TypeFunctor<Type(const Type &)> {
  Type Copy(const Type & t) {
    return this->VisitType(t);
  }
};

// not basing on exprmutator as it memorize
class CopyExpr : ExprFunctor<Expr(const Expr &)> {
  CopyType ct;
  std::unordered_map<Var, Var, NodeHash, NodeEqual> rename;

  virtual Expr VisitExpr_(const ConstantNode * c) final {
    return GetRef<Constant>(c);
  }

  virtual Expr VisitExpr_(const TupleNode * c) final {
    std::vector<Expr> v;
    for (const auto & f : c->fields) {
      v.push_back((*this)(f));
    }
    return TupleNode::make(v);
  }

  virtual Expr VisitExpr_(const VarNode * v) final {
    auto var = GetRef<Var>(v);
    return rename.count(var) == 0 ? var : rename.at(var);
  }

  virtual Expr VisitExpr_(const GlobalVarNode * g) final {
    return GetRef<GlobalVar>(g);
  }

  Var fresh(const Var & v) {
    if (rename.count(v) == 0) {
      rename.insert(std::pair<Var, Var>(v, VarNode::make(v->name_hint)));
    }
    return rename.at(v);
  }

  virtual Expr VisitExpr_(const ParamNode * p) final {
    return ParamNode::make(fresh(p->var), ct.Copy(p->type));
  }

  virtual Expr VisitExpr_(const FunctionNode * f) final {
    std::vector<Param> params;
    for (const auto & param : f->params) {
      // params.push_back(Copy(param));
    }
    std::vector<TypeParam> type_params;
    for (const auto & type_param : f->type_params) {
      
    }
    return FunctionNode::make(params, ct.Copy(f->ret_type), Copy(f->body), type_params);
  }

 public:
  Expr Copy(const Expr & e) {
    return this->VisitExpr(e);
  }
};

Expr Copy(const Expr & e) {
  return CopyExpr().Copy(e);
}

Type Copy(const Type & t) {
  return CopyType()(t);
}

}  // namespace relay
}  // namespace tvm
