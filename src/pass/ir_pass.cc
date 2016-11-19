/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_pass.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>

namespace tvm {
namespace ir {
namespace {

struct SetVarDef {
  // get var definition from node
  using FType = IRFunctor<const Variable*(const IRNodeRef&)>;
  static FGetVarDef& vtable_get_var_def() {  // NOLINT(*)
    static FGetVarDef inst; return inst;
  }
  static FSetVarExpr& vtable_set_var_expr() {  // NOLINT(*)
    static FSetVarExpr inst; return inst;
  }
  static FSetVarStmt& vtable_set_var_expr() {  // NOLINT(*)
    static FSetVarStmt inst; return inst;
  }
};

  // return a new node to
  using FSetVarExpr = IRFunctor<Expr (const IRNodeRef&, VarExpr)>;
  // return a new node to
  using FSetVarStmt = IRFunctor<Expr (const IRNodeRef&, VarExpr)>;

inline const Variable* GetVarDef(const IRNodeRef& n) {
  if (n.as<Let>()) {
    return n.as<Let>()->var.get();
  } else if (n.as<LetStmt>()) {
    return n.as<LetStmt>()->var.get();
  } else if (n.as<For>()) {
    return n.as<For>()->loop_var.get();
  } else if (n.as<Allocate>()) {
    return n.as<Allocate>()->buffer_var.get();
  } else {
    return nullptr;
  }
}

inline Expr ResetVar(const Expr& n, VarExpr var) {
  if (n.as<Let>()) {
    std::shared_ptr<Let> x = std::make_shared<Let>(*n.as<Let>());
    x->var = var;
    return Expr(x);
  } else if (n.as<Allocate>()) {
  }
}

inline Stmt ResetVarDef(const Stmt& n, VarExpr var) {
  if (n.as<LetStmt>()) {
    std::shared_ptr<LetStmt> x = std::make_shared<LetStmt>(*n.as<Let>());
    x->var = var;
    return Expr(x);
  } else if (n.as<For>()) {
    std::shared_ptr<For> x = std::make_shared<For>(*n.as<Let>());
    x->loop_var = var;
    return Expr(x);
  } else {
    LOG(FATAL) << "not reached";
  }
}

class IRVerifySSA : public IRVisitor {
 public:
  bool is_ssa{true};
  std::unordered_set<const Variable*> defined;

  void Visit(const IRNodeRef& n) final {
    if (!is_ssa) return;
    const Variable* v = GetVarDef(n);
    if (v != nullptr) {
      if (defined.count(v) != 0) {
        is_ssa = false; return;
      } else {
        defined.insert(v);
      }
    }
    IRVisitor::Visit(n);
  }
};

class IRConvertSSA : public IRMutator {
 public:
  Expr Mutate(Expr expr) final {
    static const auto& f = IRConvertSSA::vtable_expr();
    return (f.can_dispatch(expr) ?
            f(expr, expr, this) : IRMutator::Mutate(expr));
  }
  Stmt Mutate(Stmt stmt) final {
    static const auto& f = IRMutatorExample::vtable_stmt();
    return (f.can_dispatch(stmt) ?
            f(stmt, stmt, this) : IRMutator::Mutate(stmt));
  }
  using FConvertExpr = IRFunctor<Expr(const IRNodeRef&, const Expr&, IRConvertSSA *)>;
  using FConvertStmt = IRFunctor<Stmt(const IRNodeRef&, const Expr&, IRConvertSSA *)>;
  std::unordered_map<const Variable*, std::vector<VarExpr> > scope;
  std::unordered_set<const Variable*> defined;
};

temple<>

TVM_STATIC_IR_FUNCTOR(IRConvertSSA, vtable_expr)
.set_dispatch<Let>([](const Let* op, const Expr& e, IRConvertSSA* m) {
    VarExpr var = op->var;
    if (m->defined.count(var.get()) != 0) {
      var = Variable::make(var->type, var->name_hint);
    }
    // insert scope before recursion.
    m->scope[var.get()].push_back(var);
    Expr new_expr = Mutate(e);
    m->scope[var.get()].pop_back();

    if (!var.same_as(op->var)) {
      std::shared_ptr<Let> x = std::make_shared<Let>(*new_expr.as<Let>());
      x->var = var;
      return Expr(x);
    } else {
      return new_expr;
    }
  });

}  // namespace

bool VerifySSA(const IRNodeRef& ir) {
  IRVerifySSA v;
  v.Visit(ir);
  return v.is_ssa;
}

}  // namespace ir
}  // namespace tvm
