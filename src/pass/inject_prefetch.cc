/*!
 *  Copyright (c) 2017 by Contributors
 * \file inject_prefetch.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <unordered_set>

namespace tvm {
namespace ir {

using arith::IntSet;

// Find Read region of the tensor in the stmt.
class FuncTouchRegion final : public IRVisitor {
 public:
  FuncTouchRegion(
      FunctionRef func,
      int value_index,
      std::unordered_map<const Variable*, IntSet> dom_map)
      : func_(func), value_index_(value_index), dom_map_(dom_map) {
  }

  std::vector<Range> Find(const Stmt& stmt) {
    this->Visit(stmt);
    std::vector<Range> ret;
    Range none;
    for (size_t i = 0; i < bounds_.size(); ++i) {
      ret.emplace_back(arith::Union(bounds_[i]).cover_range(none));
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
    if (op->func.same_as(func_) && op->value_index == value_index_) {
      Touch(op->args);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide* op) final {
    if (op->func.same_as(func_) && op->value_index == value_index_) {
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
  int value_index_;
  std::vector<std::vector<IntSet> > bounds_;
  std::unordered_map<const Variable*, IntSet> dom_map_;
};

class PrefetchInjector : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    /*Stmt ret = IRMutator::Mutate_(op, s);
    op = ret.as<AttrStmt>();
    if (op->attr_key == attr::prefetch_scope) {
      Tensor ts(op->node.node_);
      CHECK_NE(loop_nest_.size(), 0U);
      const VarExpr& v = loop_nest_.back();
      std::unordered_map<const Variable*, IntSet> dmap;
      // remap loop_var to loop_var + offset;
      dmap[v.get()] = IntSet::single_point(v + op->value);
      FuncTouchRegion ftouch(ts->op, ts->value_index, dmap);
      std::vector<Range> bounds = ftouch.Find(op->body);
      Halide::Internal::Region region;
      for (Range r : bounds) {
        if (!r.defined()) {
          LOG(WARNING) << "Cannot decide prefetch region for " << ts;
          return op->body;
        }
        region.push_back(Halide::IR::Range::make_by_min_extent(
            r->min, r->extent));
      }
      Stmt prefetch = Prefetch::make(
          ts->op, ts->value_index, ts->dtype, region);
      return Block::make(prefetch, op->body);
    } else {
      return ret;
    }*/
    Stmt ret = IRMutator::Mutate_(op, s);
    if (op->attr_key == attr::prefetch_scope) {
      //std::cout << "prefetch over " << loop_nest_.back() << " offset is " << op->value << "\n";
    }
    return ret;
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    loop_nest_.push_back(op->loop_var);
    Stmt ret = IRMutator::Mutate_(op, s);
    loop_nest_.pop_back();
    return ret;
  }

 private:
  std::vector<VarExpr> loop_nest_;
};

Stmt InjectPrefetch(Stmt stmt) {
  return PrefetchInjector().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
