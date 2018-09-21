/*!
 *  Copyright (c) 2018 by Contributors
 * \file var_well_formed.cc
 * \brief Function for substituting a concrete type in place of a type ID
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <unordered_set>

namespace tvm {
namespace relay {

struct ShadowDetected { };

struct DetectShadow : ExprVisitor {
  struct Insert {
    DetectShadow * ds;
    Var lv;
    Insert(DetectShadow * ds, const Var & lv) : ds(ds), lv(lv) {
      if (ds->s.count(lv) != 0) {
        throw ShadowDetected();
      }
      ds->s.insert(lv);
    }
    Insert(const Insert &) = delete;
    Insert(Insert &&) = default;
    ~Insert() {
      ds->s.erase(lv);
    }
  };
  std::unordered_set<Var> s;
  void VisitExpr_(const LetNode & l) {
    // we do letrec only for FunctionNode,
    // but shadowing let in let binding is likely programming error, and we should forbidden it.
    Insert ins(this, l.var);
    (*this)(l.value);
    (*this)(l.body);
  }
  void VisitExpr_(const FunctionNode & f) {
    std::vector<Insert> ins;
    for (const Param & p : f.params) {
      ins.push_back(Insert(this, p->var));
    }
    (*this)(f.body);
  }
};

bool VarWellFormed(const Expr & e) {
  try {
    DetectShadow()(e);
    return true;
  } catch (const ShadowDetected &) {
    return false;
  }
}

}  // namespace relay
}  // namespace tvm
