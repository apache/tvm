/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file dead_code.cc
 *
 * \brief Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "let_list.h"

namespace tvm {
namespace relay {

bool IsBoolLit(const Expr& e, bool b) {
  if (const ConstantNode* c = e.as<ConstantNode>()) {
    if (c->is_scalar()) {
      auto dt = c->tensor_type()->dtype;
      if (dt == UInt(8)) {
        return *reinterpret_cast<const uint8_t*>(c->data->data) == b;
      } else if (dt == UInt(16)) {
        return *reinterpret_cast<const uint16_t*>(c->data->data) == b;
      } else if (dt == UInt(32)) {
        return *reinterpret_cast<const uint32_t*>(c->data->data) == b;
      } else if (dt == UInt(64)) {
        return *reinterpret_cast<const uint64_t*>(c->data->data) == b;
      } else if (dt == Int(8)) {
        return *reinterpret_cast<const int8_t*>(c->data->data) == b;
      } else if (dt == Int(16)) {
        return *reinterpret_cast<const int16_t*>(c->data->data) == b;
      } else if (dt == Int(32)) {
        return *reinterpret_cast<const int32_t*>(c->data->data) == b;
      } else if (dt == Int(64)) {
        return *reinterpret_cast<const int64_t*>(c->data->data) == b;
      }
    }
  }
  return false;
}

// calculate the dependency graph from expression
class CalcDep : private ExprMutator {
 public:
  static Expr Eliminate(const Expr& e) {
    CalcDep cd;
    auto res = cd(e);
    GenLet gl(cd.var_map_);
    gl(res);
    return gl.lets_.Get(res);
  }

 private:
  struct Binder {
    Type t;
    Expr e;
    Binder(const Type& t, const Expr& e) : t(t), e(e) { }
  };
  using VarMap = std::unordered_map<Var, Binder, NodeHash, NodeEqual>;
  VarMap var_map_;

  Expr VisitExpr_(const IfNode* i) final {
    auto cond = VisitExpr(i->cond);
    if (IsBoolLit(cond, true)) {
      return Eliminate(i->true_branch);
    } else if (IsBoolLit(cond, false)) {
      return Eliminate(i->false_branch);
    } else {
      return IfNode::make(cond, Eliminate(i->true_branch), Eliminate(i->false_branch));
    }
  }

  Expr VisitExpr_(const LetNode* l) final {
    var_map_.insert(std::pair<Var, Binder>(l->var,
                                           Binder(l->value_type,
                                                  Eliminate(l->value))));
    return VisitExpr(l->body);
  }

  Expr VisitExpr_(const FunctionNode* f) final {
    return FunctionNode::make(f->params, f->ret_type, Eliminate(f->body), f->type_params);
  }

  // generate the let list from dependency graph
  class GenLet : private ExprVisitor {
   private:
    LetList lets_;
    VarMap var_map_;
    explicit GenLet(const VarMap& var_map) : var_map_(var_map) { }
    friend CalcDep;

    void VisitExpr_(const VarNode* vn) final {
      Var v = GetRef<Var>(vn);
      if (var_map_.count(v) != 0) {
        auto val = var_map_.at(v);
        var_map_.erase(v);
        // erase before visit to handle letrec
        VisitExpr(val.e);
        // visit before push back so the dependency of dependency is before the dependency
        lets_.Push(v, val.t, val.e);
      }
    }
  };
};

Expr DeadCodeElimination(const Expr& e) {
  return CalcDep::Eliminate(e);
}

TVM_REGISTER_API("relay._ir_pass.dead_code_elimination")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = DeadCodeElimination(args[0]);
  });

}  // namespace relay
}  // namespace tvm
