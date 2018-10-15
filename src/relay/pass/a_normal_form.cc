/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/pass/a_normal_form.cc
 * \brief Transform the program into A Normal Form.
 */
#include <tvm/relay/expr_functor.h>
#include "./type_visitor.h"
#include "tvm/relay/pass.h"
#include "./let_list.h"
namespace tvm {
namespace relay {

class ANFMutator : private ExprMutator {
  LetList let_list_;
  template<typename K, typename V>
  using Map = std::unordered_map<K, V, NodeHash, NodeEqual> *;

  // map let bounded var to the transformed non-compound expr.
  Map<Var, Expr> var_to_expr_;

  // map let bound var to transformed var. this is for letrec.
  Map<Var, Var> var_to_var_;

  // map an expr to a var that it should bind to.
  Map<Expr, Var> expr_to_var_;

  ANFMutator(Map<Var, Expr> var_to_expr,
             Map<Var, Var> var_to_var,
             Map<Expr, Var> expr_to_var) :
    var_to_expr_(var_to_expr), var_to_var_(var_to_var), expr_to_var_(expr_to_var) { }

  /* while it seems like a good idea to override VisitExpr to push into letlist once for all,
   * at quite a few case we dont want that (let, param, localvar, globalvar).
   */

  /*! \brief Transform a compound Expr into a Var.
   *
   *  \param original the original Expr with possibly compound child.
   *
   *  \param e the Expr with no compound child.
   *
   *  \return the transformed Expr.
   */
  Var Compound(const Expr& original, const Expr& e) {
    // use the specified mapping if possible.
    Var v = expr_to_var_->count(original) != 0 ?
      expr_to_var_->at(original) :
      VarNode::make("x", Type());
    return let_list_.Push(v, e);
  }

  /*! \brief transform an Expr that has it's own scope.
   *
   *  \param e the Expr to transform.
   *
   *  \return the transformed Expr.
   */
  Expr Scoped(const Expr& e) {
    return ANormalForm(var_to_expr_, var_to_var_, expr_to_var_, e);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return Compound(GetRef<Constant>(op), GetRef<Constant>(op));
  }

  Expr VisitExpr_(const TupleNode* op) final {
    std::vector<Expr> f;
    for (const Expr& e : op->fields) {
      f.push_back(VisitExpr(e));
    }
    return Compound(GetRef<Tuple>(op), TupleNode::make(f));
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    return Compound(GetRef<TupleGetItem>(op),
                    TupleGetItemNode::make(VisitExpr(op->tuple), op->index));
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    // if it has been transformed to non-compound, use it.
    if (var_to_expr_->count(var) != 0) {
      return var_to_expr_->at(var);
    }
    // if it has been transformed into another variable (letrec), use it.
    if (var_to_var_->count(var) != 0) {
      return var_to_var_->at(var);
    }
    return var;
  }

  Expr VisitExpr_(const GlobalVarNode* op) final {
    return GetRef<GlobalVar>(op);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    return Compound(GetRef<Function>(op),
                    FunctionNode::make(op->params,
                                       op->ret_type,
                                       Scoped(op->body),
                                       op->type_params));
  }

  Expr VisitExpr_(const CallNode* op) final {
    std::vector<Expr> args;
    for (const Expr& e : op->args) {
      args.push_back(VisitExpr(e));
    }
    return Compound(GetRef<Call>(op),
                    CallNode::make(VisitExpr(op->op), args, op->attrs, op->type_args));
  }

  Expr VisitExpr_(const LetNode* op) final {
    Var x = VarNode::make("x", Type());
    // make itself in letrec get bound to x.
    var_to_var_->insert(std::pair<Var, Var>(op->var, x));
    // make the op->value get bounded to x.
    expr_to_var_->insert(std::pair<Expr, Var>(op->value, x));
    Expr val = VisitExpr(op->value);
    // make the var get bound to val, and overshadow the binding to x.
    var_to_expr_->insert(std::pair<Var, Expr>(op->var, val));
    return VisitExpr(op->body);
  }

  Expr VisitExpr_(const IfNode* op) final {
    return Compound(GetRef<If>(op),
                    IfNode::make(VisitExpr(op->cond),
                                 Scoped(op->true_branch),
                                 Scoped(op->false_branch)));
  }

  Expr VisitExpr_(const OpNode* op) final {
    return GetRef<Op>(op);
  }

 public:
  static Expr ANormalForm(Map<Var, Expr> var_to_expr,
                          Map<Var, Var> var_to_var,
                          Map<Expr, Var> expr_to_var,
                          const Expr& e) {
    ANFMutator anf(var_to_expr, var_to_var, expr_to_var);
    return anf.let_list_.Get(anf.VisitExpr(e));
  }
};

Expr ANormalForm(const Expr& e) {
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> var_to_expr;
  std::unordered_map<Var, Var, NodeHash, NodeEqual> var_to_var;
  std::unordered_map<Expr, Var, NodeHash, NodeEqual> expr_to_var;
  return ANFMutator::ANormalForm(
    &var_to_expr,
    &var_to_var,
    &expr_to_var,
    e);
}

TVM_REGISTER_API("relay._ir_pass.a_normal_form")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = ANormalForm(args[0]);
    });

}  // namespace relay
}  // namespace tvm
