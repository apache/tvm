/*!
 *  Copyright (c) 2016 by Contributors
 *  SSA related checks and pass.
 * \file ssa.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace ir {
namespace {

// global functor to get var definition from
struct FGetVarDef {
  using FType = IRFunctor<VarExpr (const IRNodeRef&)>;
  static FType& vtable() {  // NOLINT(*)
    static FType inst; return inst;
  }
};
TVM_STATIC_IR_FUNCTOR(FGetVarDef, vtable)
.set_dispatch<Let>([](const Let* op) {
    return op->var;
  })
.set_dispatch<LetStmt>([](const LetStmt* op) {
    return op->var;
  })
.set_dispatch<For>([](const For* op) {
    return op->loop_var;
  })
.set_dispatch<Allocate>([](const Allocate* op) {
    return op->buffer_var;
  });

struct FSetVarDef {
  using FTypeExpr = IRFunctor<Expr (const IRNodeRef&, VarExpr)>;
  using FTypeStmt = IRFunctor<Stmt (const IRNodeRef&, VarExpr)>;
  static FTypeExpr& vtable_expr() {  // NOLINT(*)
    static FTypeExpr inst; return inst;
  }
  static FTypeStmt& vtable_stmt() {  // NOLINT(*)
    static FTypeStmt inst; return inst;
  }
};
TVM_STATIC_IR_FUNCTOR(FSetVarDef, vtable_expr)
.set_dispatch<Let>([](const Let* op, VarExpr var) {
    std::shared_ptr<Let> x = std::make_shared<Let>(*op);
    x->var = var;
    return Expr(x);
  });

TVM_STATIC_IR_FUNCTOR(FSetVarDef, vtable_stmt)
.set_dispatch<LetStmt>([](const LetStmt* op, VarExpr var) {
    std::shared_ptr<LetStmt> x = std::make_shared<LetStmt>(*op);
    x->var = var;
    return Stmt(x);
  })
.set_dispatch<For>([](const For* op, VarExpr var) {
    std::shared_ptr<For> x = std::make_shared<For>(*op);
    x->loop_var = var;
    return Stmt(x);
  });

class IRVerifySSA : public IRVisitor {
 public:
  bool is_ssa{true};

  void Visit(const IRNodeRef& n) final {
    if (!is_ssa) return;
    static auto& fget_var_def = FGetVarDef::vtable();
    if (fget_var_def.can_dispatch(n)) {
      VarExpr v = fget_var_def(n);
      if (defined_.count(v.get()) != 0) {
        is_ssa = false; return;
      } else {
        defined_[v.get()] = 1;
      }
    }
    IRVisitor::Visit(n);
  }

 private:
  std::unordered_map<const Variable*, int> defined_;
};

class IRConvertSSA : public IRMutator {
 public:
  Expr Mutate(Expr expr) final {
    static auto& fget_var_def = FGetVarDef::vtable();
    static auto& fset_var_def = FSetVarDef::vtable_expr();
    if (fget_var_def.can_dispatch(expr)) {
      VarExpr v = fget_var_def(expr);
      VarExpr new_var = v;
      if (defined_.count(v.get()) != 0) {
        CHECK(expr.as<Allocate>() == nullptr)
            << "One allocation in two places, cannot rename buffer in allocate";
        new_var = Variable::make(v->type, v->name_hint);
      } else {
        defined_.insert(v.get());
      }
      scope_[v.get()].push_back(new_var);
      Expr new_expr = IRMutator::Mutate(expr);
      scope_[v.get()].pop_back();

      if (!new_var.same_as(v)) {
        return fset_var_def(new_expr, new_var);
      } else {
        return new_expr;
      }
    } else if (expr.as<Variable>()) {
      const Variable* v = expr.as<Variable>();
      if (scope_.count(v) != 0) {
        return scope_[v].back();
      } else {
        return expr;
      }
    } else {
      Expr e = IRMutator::Mutate(expr);
      return e;
    }
  }

  Stmt Mutate(Stmt stmt) final {
    static auto& fget_var_def = FGetVarDef::vtable();
    static auto& fset_var_def = FSetVarDef::vtable_stmt();
    if (fget_var_def.can_dispatch(stmt)) {
      VarExpr v = fget_var_def(stmt);
      VarExpr new_var = v;
      if (defined_.count(v.get()) != 0) {
        new_var = Variable::make(v->type, v->name_hint);
      } else {
        defined_.insert(v.get());
      }
      scope_[v.get()].push_back(new_var);
      Stmt new_stmt = IRMutator::Mutate(stmt);
      scope_[v.get()].pop_back();

      if (!new_var.same_as(v)) {
        return fset_var_def(new_stmt, new_var);
      } else {
        return new_stmt;
      }
    } else {
      return IRMutator::Mutate(stmt);
    }
  }

 private:
  std::unordered_map<const Variable*, std::vector<VarExpr> > scope_;
  std::unordered_set<const Variable*> defined_;
};

}  // namespace

bool VerifySSA(const Stmt& ir) {
  IRVerifySSA v;
  v.Visit(ir);
  return v.is_ssa;
}

Stmt ConvertSSA(Stmt stmt) {
  return IRConvertSSA().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
