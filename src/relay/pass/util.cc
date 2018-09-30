/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file util.cc
 *
 * \brief simple util for relay.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "./type_visitor.h"

namespace tvm {
namespace relay {

class FreeVar;
class FreeTypeVar : private TypeVisitor<> {
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * free;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound;
  FreeTypeVar(std::unordered_set<TypeParam, NodeHash, NodeEqual> * free,
              std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound) :
    free(free), bound(bound) { }

  void VisitType_(const TypeParamNode* tp) final {
    auto var = GetRef<TypeParam>(tp);
    if (bound->count(var) == 0) {
      free->insert(var);
    }
  }

  void VisitType_(const FuncTypeNode* f) final {
    for (auto type_param : f->type_params) {
      bound->insert(type_param);
    }

    for (auto type_cs : f->type_constraints) {
      this->VisitType(type_cs);
    }

    for (auto arg_type : f->arg_types) {
      this->VisitType(arg_type);
    }
    this->VisitType(f->ret_type);
  }
  friend FreeVar;
};

class FreeVar : public ExprVisitor {
  void VisitExpr_(const VarNode *v) final {
    auto var = GetRef<Var>(v);
    if (bound.count(var) == 0) {
      free.insert(var);
    }
  }

  void VisitExpr_(const FunctionNode *f) final {
    for (const auto & tp : f->type_params) {
      bound_type.insert(tp);
    }
    for (const auto & p : f->params) {
      bound.insert(p->var);
    }
    VisitExpr(f->body);
    VisitType(f->ret_type);
  }

  void VisitExpr_(const LetNode *l) final {
    bound.insert(l->var);
    VisitExpr(l->value);
    VisitExpr(l->body);
    VisitType(l->value_type);
  }

 public:
  std::unordered_set<Var, NodeHash, NodeEqual> free;
  std::unordered_set<Var, NodeHash, NodeEqual> bound;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> free_type;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> bound_type;

  void VisitType(const Type& t) final {
    FreeTypeVar(&free_type, &bound_type)(t);
  }
};

tvm::Array<Var> FreeVariables(const Expr & e) {
  FreeVar fv;
  fv.VisitExpr(e);
  return tvm::Array<Var>(fv.free.begin(), fv.free.end());
}

tvm::Array<TypeParam> FreeTypeVariables(const Expr & e) {
  FreeVar fv;
  fv.VisitExpr(e);
  return tvm::Array<TypeParam>(fv.free_type.begin(), fv.free_type.end());
}

tvm::Array<TypeParam> FreeTypeVariables(const Type & t) {
  FreeVar fv;
  fv.VisitType(t);
  return tvm::Array<TypeParam>(fv.free_type.begin(), fv.free_type.end());
}

// not exposed to python for now, as tvm ffi does not support unordered_set

}  // namespace relay
}  // namespace tvm
