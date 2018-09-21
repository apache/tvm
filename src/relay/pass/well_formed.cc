/*!
 *  Copyright (c) 2018 by Contributors
 * \file well_formed.cc
 * \brief check that expression is well formed.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <unordered_set>

namespace tvm {
namespace relay {

struct NotWellFormed { };

//! brief make sure each Var is bind at most once.
class WellFormedChecker : private ExprVisitor {
  bool well_formed = true;

  std::unordered_set<Var, NodeHash, NodeEqual> s;

  void Check(const Var & v) {
    if (s.count(v) != 0) {
      well_formed = false;
    }
    s.insert(v);
  }

  void VisitExpr_(const LetNode * l) final {
    // we do letrec only for FunctionNode,
    // but shadowing let in let binding is likely programming error, and we should forbidden it.
    Check(l->var);
    CheckWellFormed(l->value);
    CheckWellFormed(l->body);
  }

  void VisitExpr_(const FunctionNode * f) final {
    for (const Param & p : f->params) {
      Check(p->var);
    }
    CheckWellFormed(f->body);
  }

 public:
  bool CheckWellFormed(const Expr & e) {
    this->VisitExpr(e);
    return well_formed;
  }
};

bool WellFormed(const Expr & e) {
  return WellFormedChecker().CheckWellFormed(e);
}

TVM_REGISTER_API("relay._ir_pass.well_formed")
  .set_body([](TVMArgs args, TVMRetValue *ret) {
      Expr e = args[0];
      *ret = WellFormed(e);
    });

}  // namespace relay
}  // namespace tvm
