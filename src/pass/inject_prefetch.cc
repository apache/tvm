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
