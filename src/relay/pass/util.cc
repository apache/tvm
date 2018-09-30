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
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * free_vars;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound_vars;
  FreeTypeVar(std::unordered_set<TypeParam, NodeHash, NodeEqual> * free_vars,
              std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound_vars) :
    free_vars(free_vars), bound_vars(bound_vars) { }

  void VisitType_(const TypeParamNode* tp) final {
    auto var = GetRef<TypeParam>(tp);
    if (bound_vars->count(var) == 0) {
      free_vars->insert(var);
    }
  }

  void VisitType_(const FuncTypeNode* f) final {
    for (auto type_param : f->type_params) {
      bound_vars->insert(type_param);
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
    if (bound_vars.count(var) == 0) {
      free_vars.insert(var);
    }
  }

  void VisitExpr_(const FunctionNode *f) final {
    for (const auto& tp : f->type_params) {
      bound_types.insert(tp);
    }
    for (const auto& p : f->params) {
      bound_vars.insert(p->var);
    }
    VisitExpr(f->body);
    VisitType(f->ret_type);
  }

  void VisitExpr_(const LetNode *l) final {
    bound_vars.insert(l->var);
    VisitExpr(l->value);
    VisitExpr(l->body);
    VisitType(l->value_type);
  }

 public:
  std::unordered_set<Var, NodeHash, NodeEqual> free_vars;
  std::unordered_set<Var, NodeHash, NodeEqual> bound_vars;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> free_types;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> bound_types;

  void VisitType(const Type& t) final {
    FreeTypeVar(&free_types, &bound_types)(t);
  }
};

tvm::Array<Var> FreeVariables(const Expr& e) {
  FreeVar fv;
  fv.VisitExpr(e);
  return tvm::Array<Var>(fv.free_vars.begin(), fv.free_vars.end());
}

tvm::Array<TypeParam> FreeTypeVariables(const Expr& e) {
  FreeVar fv;
  fv.VisitExpr(e);
  return tvm::Array<TypeParam>(fv.free_types.begin(), fv.free_types.end());
}

tvm::Array<TypeParam> FreeTypeVariables(const Type& t) {
  FreeVar fv;
  fv.VisitType(t);
  return tvm::Array<TypeParam>(fv.free_types.begin(), fv.free_types.end());
}

TVM_REGISTER_API("relay._ir_pass.free_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = FreeVariables(args[0]);
  });

TVM_REGISTER_API("relay._ir_pass.free_type_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef x = args[0];
    if (x.as<TypeNode>()) {
      *ret = FreeTypeVariables(Downcast<Type>(x));
    } else {
      *ret = FreeTypeVariables(Downcast<Expr>(x));
    }
  });

}  // namespace relay
}  // namespace tvm
