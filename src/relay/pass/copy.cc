/*!
 *  Copyright (c) 2018 by Contributors
 * \file copy.cc
 * \brief Copy expressions and types.
 */

#include "type_functor.h"
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

// not basing on exprmutator as it memorize
struct CopyExpr : ExprFunctor<Expr(const Expr &)> {
  virtual Expr VisitExpr_(const ConstantNode * c) override {
    return GetRef<Constant>(c);
  }
  virtual Expr VisitExpr_(const TupleNode * c) override {
    std::vector<Expr> v;
    for (const auto & f : c->fields) {
      v.push_back((*this)(f));
    }
    return TupleNode::make(v);
  }
};

// not basing on typemutator as it might one day memorize
struct CopyType : TypeFunctor<Type(const Type &)> {
  
};

Expr Copy(const Expr & e) {
  return CopyExpr()(e);
}

Type Copy(const Type & t) {
  return CopyType()(t);
}

}  // namespace relay
}  // namespace tvm
