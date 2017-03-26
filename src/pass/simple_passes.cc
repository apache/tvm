/*!
 *  Copyright (c) 2016 by Contributors
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {

class IRSideEffect : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (has_side_effect_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    if (!op->is_pure()) {
      has_side_effect_ = true; return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const Expr& e) {
  IRSideEffect v;
  v.Visit(e);
  return v.has_side_effect_;
}

class IRSubstitue : public IRMutator {
 public:
  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = smap.find(op);
    if (it != smap.end()) {
      return it->second;
    } else {
      return e;
    }
  }
  std::unordered_map<const Variable*, Expr> smap;
};

Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map) {
  if (value_map.size() == 0) return stmt;
  IRSubstitue m;
  for (auto kv : value_map) {
    m.smap[kv.first.get()] = kv.second;
  }
  return m.Mutate(stmt);
}

class ExprUseVarVisitor : public IRVisitor {
 public:
  explicit ExprUseVarVisitor(const Variable* var)
      : var_(var) {}

  void Visit(const NodeRef& e) final {
    if (use_var_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Variable* op) final {
    if (op == var_) {
      use_var_ = true;
    }
  }

  void Visit_(const Load* op) final {
    if (op->buffer_var.get() == var_) {
      use_var_ = true;
    }
    IRVisitor::Visit_(op);
  }

  const Variable* var_;
  bool use_var_{false};
};

bool ExprUseVar(const Expr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor.Visit(e);
  return visitor.use_var_;
}

}  // namespace ir
}  // namespace tvm
