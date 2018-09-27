/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file lambda_lift.cc
 *
 * \brief remove all inner lambda by turning them into global definition.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "./type_visitor.h"

namespace tvm {
namespace relay {

struct FreeTypeVar : TypeVisitor<> {
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * free;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound;
  FreeTypeVar(std::unordered_set<TypeParam, NodeHash, NodeEqual> * free,
                  std::unordered_set<TypeParam, NodeHash, NodeEqual> * bound) : free(free), bound(bound) { }
};

struct FreeVar : ExprVisitor {
  std::unordered_set<Var, NodeHash, NodeEqual> free;
  std::unordered_set<Var, NodeHash, NodeEqual> bound;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> freeType;
  std::unordered_set<TypeParam, NodeHash, NodeEqual> boundType;

  void VisitExpr_(const VarNode *v) final {
    auto var = GetRef<Var>(v);
    if (bound.count(var) == 0) {
      free.insert(var);
    }
  }

  void VisitExpr_(const FunctionNode *f) final {
    for (const auto & tp : f->type_params) {
      boundType.insert(tp);
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

  void VisitType(const Type& t) final {
    FreeTypeVar(&freeType, &boundType)(t);
  }
};
Expr LambdaLift(std::unordered_map<Var, Expr, NodeHash, NodeEqual> * rename, const Environment& env, const Expr& e);

struct LiftAllLambda : ExprMutator {
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> * rename;
  Environment env;
  explicit LiftAllLambda(std::unordered_map<Var, Expr, NodeHash, NodeEqual> * rename, const Environment& env) :
    rename(rename), env(env) { }

  Expr copy(const Expr& e) {
    return e;
  }

  Expr VisitExpr_(const VarNode *v) final {
    auto var = GetRef<Var>(v);
    return rename->count(var) == 0 ? var : rename->at(var);
  }

  // jared we should sit down and review this, I am scared
  Expr LiftFunction(const Var & self, const GlobalVar& gv, const Function& func) {
    FreeVar fv;
    fv.bound.insert(self);
    fv(func);
    std::vector<Param> p;
    std::vector<Expr> e;
    std::vector<TypeParam> tp;
    for (const auto & v : fv.free) {
      p.push_back(ParamNode::make(v, IncompleteTypeNode::make(TypeParamNode::kType)));
      e.push_back(v);
    }
    for (const auto & t : fv.freeType) {
      tp.push_back(t);
    }
    auto ret = CallNode::make(gv, e);
    rename->insert(std::pair<Var, Expr>(self, ret));
    auto wrapped = FunctionNode::make(p, IncompleteTypeNode::make(TypeParamNode::kType), func, tp);
    env->Add(gv, Downcast<Function, Expr>(copy(LambdaLift(rename, env, wrapped)))); // copy is called before wrapped, or rename will stop working
    return ret;
  }

  Expr VisitExpr_(const FunctionNode *f) final {
    return LiftFunction(VarNode::make("unused"), GlobalVarNode::make("lifted"), GetRef<Function>(f));
  }

  // what about mutual recursion? a clean version probably involve tying the knot.
  // since we are in C++ going two pass seems to be the way to go.
  // mutual recursion is not supported yet, so no worry for now.
  Expr VisitExpr_(const LetNode *l) final {
    if (auto f = Downcast<Function, Expr>(l->value)) {
      return LetNode::make(l->var, LiftFunction(l->var, GlobalVarNode::make("lifted"), f), l->body, l->value_type);
      throw Error("still thinking about the recursion case");
    } else {
      return LetNode::make(l->var, VisitExpr(l->value), VisitExpr(l->body), l->value_type);
    }
  }
};

Expr LambdaLift(std::unordered_map<Var, Expr, NodeHash, NodeEqual> * rename, const Environment& env, const Expr& e) {
  if (auto f = e.as<FunctionNode>()) {
    return FunctionNode::make(f->params, f->ret_type, LambdaLift(rename, env, f->body), f->type_params);
  } else {
    return LiftAllLambda(rename, env)(e);
  }
}

Expr LambdaLift(const Environment& env, const Expr& e) {
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> rename;
  return LambdaLift(&rename, env, e);
}

TVM_REGISTER_API("relay._ir_pass.lambda_lift")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = LambdaLift(args[0], args[1]);
  });

}  // namespace relay
}  // namespace tvm
