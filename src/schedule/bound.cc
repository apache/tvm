/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include "./int_set.h"
#include "./bound.h"

namespace tvm {
namespace schedule {

// result = ceil((a / b)), both a and b are positive integer
inline Expr DivCeil(Expr a, Expr b) {
  return (a + b - 1) / b;
}

// Downward message passing algorithm on schedule s,
// pass the range state down from the root to the leaves
// after this pass, every IterVar in the schedule hyper graph will have a range(domain)
void PassDown(const Schedule& s,
              std::unordered_map<IterVar, Range>* p_state) {
  auto& state = *p_state;
  // forwar iteration on relations
  for (IterVarRelation rel : s->relations) {
    if (rel.as<SplitNode>()) {
      const SplitNode* r = rel.as<SplitNode>();
      CHECK(state.count(r->parent));
      CHECK(!state.count(r->inner));
      const Range& range_parent = state.at(r->parent);
      if (r->factor.defined()) {
        state[r->inner] = Range::make_with_min_extent(0, r->factor);
        if (r->outer->dom.defined()) {
          state[r->outer] = r->outer->dom;
        } else {
          CHECK(!state.count(r->outer));
          state[r->outer] = Range::make_with_min_extent(
              0, DivCeil(range_parent->extent, r->factor));
        }
      } else {
        CHECK(r->outer->dom.defined());
        state[r->outer] = r->outer->dom;
        state[r->inner] = Range::make_with_min_extent(
            0, DivCeil(range_parent->extent, r->outer->dom->extent));
      }
    } else if (rel.as<FuseNode>()) {
      const FuseNode* r = rel.as<FuseNode>();
      CHECK(state.count(r->outer));
      CHECK(state.count(r->inner));
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner = state.at(r->inner);
      state[r->fused] = Range::make_with_min_extent(
          0, range_outer->extent * range_inner->extent);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// upward message passing algorithm
// pass the integer set on each leave loop up to the root
// dom_map is the result of PassDown, it records the domain of each IterVar.
// dom_map can be used to get cached result in reverse construction.
void PassUp(const Schedule& s,
            const std::unordered_map<IterVar, Range>& dom_map,
            std::unordered_map<IterVar, IntSet>* p_state) {
  auto& state = *p_state;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      IntSet parent;
      const SplitNode* r = rel.as<SplitNode>();
      PassUp(r, dom_map,
             state.at(r->outer), state.at(r->inner),
             &parent);
      state[r->parent] = parent;
    } else if (rel.as<FuseNode>()) {
      IntSet outer, inner;
      const FuseNode* r = rel.as<FuseNode>();
      PassUp(r, dom_map,
             state.at(r->fused),
             &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

void PassBound(
    const Tensor& tensor,
    const std::vector<IntSet>& arg_bounds,
    std::unordered_map<IterVar, std::vector<IntSet> >* result) {
  if (tensor->op.as<ComputeOpNode>()) {
    auto root_iter_vars = tensor->op->root_iter_vars();
    CHECK_EQ(tensor.ndim(), root_iter_vars.size());
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      (*result)[root_iter_vars[i]].push_back(arg_bounds[i]);
    }
  } else {
    LOG(FATAL) << "unknown operation mode";
  }
}

void PassBound(
    Operation op,
    std::unordered_map<IterVar, IntSet>* ebound) {
  if (op.as<ComputeOpNode>()) {
    auto fvisit = [ebound](const NodeRef& n) {
      auto *call = n.as<ir::Call>();
      if (call != nullptr && call->func.defined()) {
        Tensor t(call->func.node_);
        std::vector<IntSet> arg_bounds;
        for (size_t i = 0; i < t.ndim(); ++i) {
          arg_bounds.push_back(Eval(call->args[i], *ebound));
        }
      }
    };
    ir::PostOrderVisit(op.as<ComputeOpNode>()->body, fvisit);
  } else {
    LOG(FATAL) << "unknown operation mode";
  }
}

void InferBound(const Schedule& sch,
                std::unordered_map<IterVar, Range>* rmap) {
  CHECK_NE(sch->attach_type, kNone);
  if (sch->attach_type == kInline) return;
  if (sch->attach_type == kRoot) {
    auto root_iter_vars = sch->op->root_iter_vars();
    for (size_t i = 0; i < root_iter_vars.size(); ++i) {
      auto v = root_iter_vars[i];
      CHECK(v->dom.defined());
      CHECK(!rmap->count(v));
      (*rmap)[v] = v->dom;
    }
  }
  // get range of all child iter vars.
  PassDown(sch, rmap);
  // pass iteration variable to children
}


Map<IterVar, Range> InferBound(Schedule sch) {
  return {};
}

}  // namespace schedule
}  // namespace tvm
