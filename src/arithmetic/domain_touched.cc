/*!
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/arithmetic.h>
#include <tvm/api_registry.h>

#include <unordered_set>
#include <unordered_map>

namespace tvm {
namespace arith {

using namespace ir;

// Find Read region of the tensor in the stmt.
class FuncTouchedDomain final : public IRVisitor {
public:
  FuncTouchedDomain(FunctionRef func, bool consider_calls, bool consider_provides)
    : func_(func), consider_calls_(consider_calls), consider_provides_(consider_provides)  {}

  Domain Find(const Stmt& stmt) {
    this->Visit(stmt);
    Domain ret;
    Range none;
    for (size_t i = 0; i < bounds_.size(); ++i) {
      ret.push_back(arith::Union(bounds_[i]).cover_range(none));
    }
    return ret;
  }

  void Visit_(const For *op) final {
    const Variable* var = op->loop_var.get();
    dom_map_[var] = IntSet::range(
        Range::make_with_min_extent(op->min, op->extent));
    IRVisitor::Visit_(op);
    dom_map_.erase(var);
  }

  void Visit_(const LetStmt* op) final {
    dom_map_[op->var.get()] =
        arith::EvalSet(op->value, dom_map_);
    IRVisitor::Visit_(op);
    dom_map_.erase(op->var.get());
  }

  /* TODO: Thread extent unitest not generated.*/
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      CHECK(thread_axis);
      const Variable* var = thread_axis->var.get();
      dom_map_[var] = IntSet::range(Range(make_zero(op->value.type()), op->value));
      IRVisitor::Visit_(op);
      dom_map_.erase(var);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Call* op) final {
    if (consider_calls_ && op->func.same_as(func_)) {
      Touch(op->args);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide* op) final {
    if (consider_provides_ && op->func.same_as(func_)) {
      Touch(op->args);
    }
    IRVisitor::Visit_(op);
  }

private:
  void Touch(const Array<Expr>& args) {
    if (args.size() > bounds_.size()) {
      bounds_.resize(args.size());
    }
    for (size_t i = 0; i < args.size(); ++i) {
      bounds_[i].emplace_back(EvalSet(args[i], dom_map_));
    }
  }

  FunctionRef func_;
  bool consider_calls_, consider_provides_;
  std::vector<std::vector<IntSet> > bounds_;
  std::unordered_map<const Variable*, IntSet> dom_map_;
};

Domain RegionTouched(Stmt stmt, const FunctionRef &func, bool consider_calls, bool consider_provides) {
  return FuncTouchedDomain(func, consider_calls, consider_provides).Find(stmt);
}

}  // namespace arith
}  // namespace tvm
