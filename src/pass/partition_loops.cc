#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>


namespace tvm {
namespace ir {

class VariableFinder: public IRVisitor {
 public:
  VariableFinder(Var target) : target_(target) {}

  void Visit(const NodeRef& node) final {
    if (finded) return;
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());

    path_.push_back(node.get());
    if (node.same_as(target_)) finded = true;
    IRVisitor::Visit(node);
    if (!finded) path_.pop_back();
  }

  std::vector<const Node*> path_;

 private:
  bool finded{false};
  Var target_;
  std::unordered_set<const Node*> visited_;
};


// Get the path to the variable
std::vector<const Node*> GetPath(Var target, Expr expr) {
  VariableFinder v(target);
  v.Visit(expr);
  return v.path_;
}



// IRVisitor version
class Deducer: public IRVisitor {
 public:
  Expr Deduce(Var target, Expr expr) {
    path_ = GetPath(target, expr);
    target_ = target;
    iter = 0;

    LOG(INFO) << "Path";
    for (const Node* n : path_) {
      LOG(INFO) << n->type_key();
    }
    Visit(expr);
    return result;
  }

  void Visit(const NodeRef& e) final {
    if (e.get() == path_[iter++]) {
      LOG(INFO) << "Deduce " << e->type_key();
      IRVisitor::Visit(e);
    } else {
      LOG(INFO) << "ERROR " << e->type_key();
    }
  }

  void Visit_(const LT* op) final {
    result = op->b;
    Visit(op->a);
  }

  void Visit_(const Add* op) final {
    bool left = op->a.get() == path_[iter];
    result -= left ? op->b : op->a;
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Mul* op) final {
    bool left = op->a.get() == path_[iter];
    result /= left ? op->b : op->a;
    Visit(left ? op->a : op->b);
  }

  Expr result;
 private:
  Var  target_;
  std::vector<const Node*> path_;
  size_t iter;
};


// IRMutator version
class DeduceMutator {
 public:
  Expr Deduce(Var target, Expr expr) {
    this->path_ = GetPath(target, expr);
    this->target = target;
    this->iter = 0;

    LOG(INFO) << "Path";
    for (const Node* n : path_) {
      LOG(INFO) << n->type_key();
    }
    return Mutate(expr, expr);
  }

  Expr Mutate(const NodeRef& node, Expr result) {
    if (node.get() == path_[iter++]) {
      LOG(INFO) << "Deduce " << node->type_key();
      static const FMutateExpr& f = vtable_expr();
      return f(node, result, this);
    } else {
      LOG(INFO) << "Error " << node->type_key();
      return result;
    }
  }

  const Node* GetCurrentNode() {
    return path_[iter];
  }

  using FMutateExpr = IRFunctor<Expr(const NodeRef&, Expr&, DeduceMutator*)>;
  static FMutateExpr& vtable_expr();

  Var target;
 private:
  std::vector<const Node*> path_;
  size_t iter;
};

DeduceMutator::FMutateExpr& DeduceMutator::vtable_expr() {  // NOLINT(*)
  static FMutateExpr inst; return inst;
}

TVM_STATIC_IR_FUNCTOR(DeduceMutator, vtable_expr)
.set_dispatch<LT>([](const LT* op, Expr& res, DeduceMutator* m) {
    return m->Mutate(op->a, op->b);
})
.set_dispatch<Mul>([](const Mul* op, Expr& res, DeduceMutator* m) {
    bool left = op->a.get() == m->GetCurrentNode();
    res /= left ? op->b : op->a;
    return m->Mutate(left ? op->a : op->b, res);
})
.set_dispatch<Add>([](const Add* op, Expr& res, DeduceMutator* m) {
    bool left = op->a.get() == m->GetCurrentNode();
    res -= left ? op->b : op->a;
    return m->Mutate(left ? op->a : op->b, res);
})
.set_dispatch<Variable>([](const Variable* op, Expr& res, DeduceMutator* m) {
    return res;
});




Expr Deduce(Var v, Expr e) {
    // x*y+z < a
    LOG(INFO) << "Deduce";
    // Deducer deducer;
    // deducer.Deduce(v, e);
    // return deducer.result;
    DeduceMutator deducer;
    return deducer.Deduce(v, e);
}

}
}
