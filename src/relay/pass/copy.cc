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

template<typename A, typename B>
tvm::Array<B> map(const tvm::Array<A> & arr, const std::function<B(A)> & f) {
  std::vector<B> v;
  for (const A & a : arr) {
    v.push_back(f(a));
  }
  return v;
}

// not basing on typemutator as it might one day memorize
class CopyType : TypeFunctor<Type(const Type &)> {

 public:
  Type Copy(const Type & t) {
    return this->VisitType(t);
  }

  tvm::Array<TypeParam> Copy(const tvm::Array<TypeParam> & tp) {
    return map<TypeParam, TypeParam>(tp, [this](const TypeParam & tp) {
        return Downcast<TypeParam>(Copy(tp));
      });
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
    return TupleNode::make(Copy(c->fields));
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
    return FunctionNode::make(Copy(f->params),
                              ct.Copy(f->ret_type),
                              Copy(f->body),
                              ct.Copy(f->type_params));
  }

  virtual Expr VisitExpr_(const CallNode * c) final {
    return CallNode::make(Copy(c->op), Copy(c->args), c->attrs);
  }

  virtual Expr VisitExpr_(const LetNode * l) final {
    return LetNode::make(fresh(l->var),
                          Copy(l->value),
                          Copy(l->body),
                          ct.Copy(l->value_type));
  }

  virtual Expr VisitExpr_(const IfNode * i) final {
    return IfNode::make(Copy(i->cond),
                        Copy(i->true_branch),
                        Copy(i->false_branch));
  }

 public:
  Expr Copy(const Expr & e) {
    return this->VisitExpr(e);
  }

  tvm::Array<Expr> Copy(const tvm::Array<Expr> & a) {
    return map<Expr, Expr>(a, [this](const Expr & e) { return Copy(e); });
  }

  tvm::Array<Param> Copy(const tvm::Array<Param> & a) {
    return map<Param, Param>(a, [this](const Param & p) {
        return Downcast<Param>(Copy(p));
      });
  }

};

Expr Copy(const Expr & e) {
  return CopyExpr().Copy(e);
}

Type Copy(const Type & t) {
  return CopyType().Copy(t);
}

}  // namespace relay
}  // namespace tvm
