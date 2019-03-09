/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file to_gnf.cc
 *
 * \brief Turn A normal form into graph normal form.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "let_list.h"

namespace tvm {
namespace relay {

class UseVarVisitor : public ExprVisitor {
 public:
  explicit UseVarVisitor(const Var& v) : v(v) { }

  static bool UseVar(const Var& v, const Expr& e) {
    UseVarVisitor uv(v);
    uv(e);
    return uv.use_var;
  }

 private:
  bool use_var = false;
  Var v;

  void VisitExpr_(const VarNode* vn) override {
    use_var = use_var || (v == GetRef<Var>(vn));
  }
};

class GNF : public ExprMutator {
 private:
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> var_map_;
  Expr VisitExpr_(const VarNode* vn) override {
    Var v = GetRef<Var>(vn);
    return var_map_.count(v) == 0 ? v : var_map_.at(v);
  }

  static bool UseVar(const Var& v, const Expr& e) {
    return UseVarVisitor::UseVar(v, e);
  }

  static Expr WrapRec(const Var& var, const Expr& val) {
    return UseVar(var, val) ? LetNode::make(var, val, var) : val;
  }

  Expr VisitExpr_(const LetNode* ln) override {
    var_map_.insert(std::pair<Var, Expr>(ln->var, VisitExpr(WrapRec(ln->var, ln->value))));
    return VisitExpr(ln->body);
  }
};

Expr ToGraphNormalForm(const Expr& e) {
  return GNF()(e);
}

TVM_REGISTER_API("relay._ir_pass.to_graph_normal_form")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = ToGraphNormalForm(args[0]);
});

}  // namespace relay
}  // namespace tvm
