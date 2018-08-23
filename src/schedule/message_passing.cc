/*!
 *  Copyright (c) 2017 by Contributors
 * \file message_passing.cc
 * \brief The message passing domain.
 */
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include "message_passing.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace schedule {

using namespace ir;
using namespace arith;

// result = ceil((a / b)), both a and b are positive integer
inline Expr DivCeil(Expr a, Expr b) {
  return ir::Simplify((a + b - 1) / b);
}

inline bool prove_equal(Expr lhs, Expr rhs) {
  return is_zero(ir::Simplify(lhs - rhs));
}

void Update(std::unordered_map<IterVar, Range>* p_state,
            const IterVar& iv,
            Range r) {
  auto it = p_state->find(iv);
  if (it == p_state->end()) {
    (*p_state)[iv] = r;
  } else {
    bool match = is_zero(it->second->min);
    if (!prove_equal(r->extent, it->second->extent)) match = false;
    CHECK(match)
        << iv
        << " domain already inferred,"
        << " cannot prove their extents are the same "
        << it->second->extent << " vs " << r->extent;
  }
}

void PassDownDomain(const Stage& stage,
                    std::unordered_map<IterVar, Range>* p_state,
                    bool allow_missing) {
  auto& state = *p_state;
  // forwar iteration on relations
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (!state.count(r->parent)) {
        CHECK(allow_missing);
        continue;
      }
      CHECK(!state.count(r->inner));
      const Range& range_parent = state.at(r->parent);
      if (r->factor.defined()) {
        Update(p_state, r->inner, Range::make_by_min_extent(0, r->factor));
        Update(p_state, r->outer,
               Range::make_by_min_extent(
                   0, DivCeil(range_parent->extent, r->factor)));
      } else {
        Update(p_state, r->outer, Range::make_by_min_extent(0, r->nparts));
        Update(p_state, r->inner,
               Range::make_by_min_extent(
                   0, DivCeil(range_parent->extent, r->nparts)));
      }
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (!state.count(r->outer) || !state.count(r->inner)) {
        CHECK(allow_missing);
        continue;
      }
      const Range& range_outer = state.at(r->outer);
      const Range& range_inner = state.at(r->inner);
      state[r->fused] = Range::make_by_min_extent(
          0, range_outer->extent * range_inner->extent);
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (!state.count(r->parent)) {
        CHECK(allow_missing);
        continue;
      }
      Update(p_state, r->rebased,
             Range::make_by_min_extent(
                 0, state.at(r->parent)->extent));
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      Update(p_state, s->iter, Range::make_by_min_extent(0, 1));
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
  // update the extents of binded threads.
  for (auto kv : stage->iter_var_attrs) {
    if (kv.second->bind_thread.defined()) {
      CHECK(state.count(kv.first));
      Update(p_state, kv.second->bind_thread, state.at(kv.first));
    }
  }
}

void PassUpIndex(const Stage& stage,
                 const Map<IterVar, Range>& dom_map,
                 std::unordered_map<IterVar, Expr>* p_state,
                 bool allow_missing) {
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->outer) || !state.count(s->inner)) {
        CHECK(allow_missing);
        continue;
      }
      Expr outer = state.at(s->outer);
      Expr inner = state.at(s->inner);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr parent_min = dom_map.at(s->parent)->min;
      state[s->parent] = inner + outer * factor;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = state[s->parent] + parent_min;
      }
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      Expr value = state.at(s->fused);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr outer_min = dom_map.at(s->outer)->min;
      Expr inner_min = dom_map.at(s->inner)->min;
      state[s->outer] = ComputeExpr<Div>(value, factor);
      state[s->inner] = ComputeExpr<Mod>(value, factor);
      // add min if they exist
      if (!is_zero(outer_min)) {
        state[s->outer] = state[s->outer] + outer_min;
      }
      if (!is_zero(inner_min)) {
        state[s->inner] = state[s->inner] + inner_min;
      }
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      Expr value = state.at(s->rebased);
      Expr parent_min = dom_map.at(s->parent)->min;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = value + parent_min;
      } else {
        state[s->parent] = value;
      }
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

void PassDownIndex(const Stage& stage,
                   const Map<IterVar, Range>& dom_map,
                   std::unordered_map<IterVar, Expr>* p_state,
                   bool allow_missing) {
  auto& state = *p_state;
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      Range r = dom_map.at(s->inner);
      CHECK(is_zero(r->min));
      Expr parent = state.at(s->parent);
      Expr factor = r->extent;
      state[s->outer] = ComputeExpr<Div>(parent, factor);
      state[s->inner] = ComputeExpr<Mod>(parent, factor);
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->inner) && !state.count(s->outer)) {
        CHECK(allow_missing);
        continue;
      }
      Expr factor = dom_map.at(s->inner)->extent;
      Expr outer_min = dom_map.at(s->outer)->min;
      Expr inner_min = dom_map.at(s->inner)->min;
      Expr inner = state.at(s->inner);
      Expr outer = state.at(s->outer);
      CHECK(is_zero(outer_min));
      CHECK(is_zero(inner_min));
      state[s->fused] = outer * factor + inner;
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      Expr value = state.at(s->parent);
      Expr parent_min = dom_map.at(s->parent)->min;
      CHECK(is_zero(parent_min));
      state[s->rebased] = value;
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      state[s->iter] = make_zero(s->iter->var.type());
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// Domain message passing.
void PassUpDomain(const SplitNode* s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& outer,
                  const IntSet& inner,
                  IntSet* parent) {
  if (dom_map.count(s->outer) &&
      dom_map.count(s->inner) &&
      dom_map.count(s->parent) &&
      outer.match_range(dom_map.at(s->outer)) &&
      inner.match_range(dom_map.at(s->inner))) {
    *parent = IntSet::range(dom_map.at(s->parent));
    return;
  }
  Expr factor = dom_map.at(s->inner)->extent;
  Expr parent_min = dom_map.at(s->parent)->min;
  CHECK(outer.defined());
  CHECK(inner.defined());
  CHECK(factor.defined());
  *parent = EvalSet(
      s->outer->var * factor + s->inner->var + parent_min,
      {{s->outer, outer}, {s->inner, inner}});
}

void PassUpDomain(const FuseNode* s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& fused,
                  IntSet* outer,
                  IntSet* inner) {
  CHECK(dom_map.count(s->outer));
  CHECK(dom_map.count(s->inner));
  CHECK(dom_map.count(s->fused));

  if (fused.match_range(dom_map.at(s->fused))) {
    *outer = IntSet::range(dom_map.at(s->outer));
    *inner = IntSet::range(dom_map.at(s->inner));
    return;
  }
  Expr outer_min = dom_map.at(s->outer)->min;
  Expr inner_min = dom_map.at(s->inner)->min;

  if (fused.is_single_point()) {
    Expr value = fused.point_value();
    Expr factor = dom_map.at(s->inner)->extent;
    Expr v_outer  = value / factor;
    Expr v_inner  = value % factor;
    if (!is_zero(outer_min)) v_outer = v_outer + outer_min;
    if (!is_zero(inner_min)) v_inner = v_inner + inner_min;
    *outer = IntSet::single_point(v_outer);
    *inner = IntSet::single_point(v_inner);
  } else {
    LOG(WARNING) << "use fallback inference rule in fuse";
    // simply use the entire set, this rule can be enhanced.
    *outer = IntSet::range(dom_map.at(s->outer));
    *inner = IntSet::range(dom_map.at(s->inner));
    return;
  }
}

void PassUpDomain(const RebaseNode* s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  const IntSet& rebased,
                  IntSet* parent) {
  CHECK(dom_map.count(s->parent));
  if (rebased.match_range(dom_map.at(s->rebased))) {
    *parent = IntSet::range(dom_map.at(s->parent));
    return;
  }
  Expr parent_min = dom_map.at(s->parent)->min;
  *parent = EvalSet(s->rebased->var + parent_min,
                    {{s->rebased, rebased}});
}

void PassUpDomain(const Stage& stage,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, IntSet>* p_state) {
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* r = rel.as<SplitNode>()) {
      IntSet parent;
      PassUpDomain(r, dom_map,
                   state.at(r->outer), state.at(r->inner),
                   &parent);
      state[r->parent] = parent;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      IntSet outer, inner;
      PassUpDomain(r, dom_map,
                   state.at(r->fused),
                   &outer, &inner);
      state[r->outer] = outer;
      state[r->inner] = inner;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      IntSet parent;
      PassUpDomain(r, dom_map,
                   state.at(r->rebased),
                   &parent);
      state[r->parent] = parent;
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// Pass up bit mask with or relation.
void PassUpBitMaskOr(const Stage& stage,
                     std::unordered_map<IterVar, int>* p_state,
                     bool allow_missing) {
  auto& state = *p_state;
  for (size_t i = stage->relations.size(); i != 0; --i) {
    IterVarRelation rel = stage->relations[i - 1];
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->inner) && !state.count(s->outer)) {
        CHECK(allow_missing);
        continue;
      }
      int res = 0;
      if (!state.count(s->parent)) res |= state[s->parent];
      if (!state.count(s->inner)) res |= state[s->inner];
      if (!state.count(s->outer)) res |= state[s->outer];
      state[s->parent] = res;
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->fused)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer)) {
        state[s->outer] = state[s->fused];
      } else {
        state[s->outer] |= state[s->fused];
      }
      if (!state.count(s->inner)) {
        state[s->inner] = state[s->fused];
      } else {
        state[s->inner] |= state[s->fused];
      }
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->rebased)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->parent)) {
        state[s->parent] = state[s->rebased];
      } else {
        state[s->parent] |= state[s->rebased];
      }
    } else if (rel.as<SingletonNode>()) {
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

void PassDownBitMaskOr(const Stage& stage,
                       std::unordered_map<IterVar, int>* p_state,
                       bool allow_missing) {
  auto& state = *p_state;
  for (IterVarRelation rel : stage->relations) {
    if (const SplitNode* s = rel.as<SplitNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->outer)) {
        state[s->outer] = state.at(s->parent);
      } else {
        state[s->outer] |= state.at(s->parent);
      }
      if (!state.count(s->inner)) {
        state[s->inner] = state.at(s->parent);
      } else {
        state[s->inner] |= state.at(s->parent);
      }
    } else if (const FuseNode* s = rel.as<FuseNode>()) {
      if (!state.count(s->outer) && !state.count(s->inner)) {
        CHECK(allow_missing);
        continue;
      }
      int res = 0;
      if (state.count(s->outer)) res |= state.at(s->outer);
      if (state.count(s->inner)) res |= state.at(s->inner);
      if (state.count(s->fused)) res |= state.at(s->fused);
      state[s->fused] = res;
    } else if (const RebaseNode* s = rel.as<RebaseNode>()) {
      if (!state.count(s->parent)) {
        CHECK(allow_missing);
        continue;
      }
      if (!state.count(s->rebased)) {
        state[s->rebased] = state.at(s->parent);
      } else {
        state[s->rebased] |= state.at(s->parent);
      }
    } else if (const SingletonNode* s = rel.as<SingletonNode>()) {
      state[s->iter] = 0;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}


/*!
 * \brief message passing to find if boundary checking on IterVar is needed.
 * \param s The stage to be used.
 * \param p_state The message passing state
 *     IterVar->flag
 */
void PassUpBoundCheck(const Stage& s,
                      const Map<IterVar, Range>& dom_map,
                      std::unordered_map<IterVar, bool>* p_state) {
  auto& state = *p_state;
  using HalideIR::Internal::can_prove;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      bool outer = state.at(s->outer);
      bool inner = state.at(s->inner);

      if (dom_map.count(s->inner) && dom_map.count(s->outer)) {
        Expr factor = dom_map.at(s->inner)->extent;
        Expr step = dom_map.at(s->outer)->extent;
        if (outer || inner) {
          state[s->parent] = true;
        } else {
          if (can_prove(dom_map.at(s->parent)->extent == factor * step)) {
            state[s->parent] = false;
          } else {
            state[s->parent] = true;
          }
        }
      } else {
        state[s->parent] = true;
      }
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      bool fused = state.at(s->fused);
      state[s->outer] = fused;
      state[s->inner] = fused;
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
      state[s->parent] = state.at(s->rebased);
    } else if (rel.as<SingletonNode>()) {
      // nop
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

std::vector<Expr> MakeBoundCheck(
    const Stage& stage,
    const Map<IterVar, Range>& dom_map,
    const std::unordered_map<IterVar, Expr>& value_map,
    bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter) {
  std::unordered_map<IterVar, bool> bound_state;
  for (IterVar iv : stage->leaf_iter_vars) {
    bound_state[iv] = false;
  }
  PassUpBoundCheck(stage, dom_map, &bound_state);

  std::vector<Expr> preds;
  std::unordered_map<const Variable*, IntSet> iset_dmap;

  // setup domain map for set analysis
  for (const auto& kv : dom_map) {
    iset_dmap[kv.first->var.get()] = IntSet::range(kv.second);
  }

  for (IterVar iv : stage->op->root_iter_vars()) {
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) continue;
    Range dom = dom_map.at(iv);
    if (bound_state.at(iv)) {
      Expr value = ComputeExpr<Sub>(value_map.at(iv), dom->min);
      Expr vmax = EvalSet(value, iset_dmap).max();
      if (vmax.type() != value.type() || !can_prove(vmax < dom->extent)) {
        preds.emplace_back(value < dom->extent);
      }
    }
    CHECK(iv->dom.defined());
    if (!skip_ivar_domain && !iv->dom.same_as(dom)) {
      Expr value = ComputeExpr<Sub>(value_map.at(iv), iv->dom->min);
      IntSet s = EvalSet(value, iset_dmap);
      Expr vmin = s.min();
      Expr vmax = s.max();
      if (vmin.type() != value.type() || !can_prove(vmin >= iv->dom->min)) {
        preds.emplace_back(value >= 0);
      }
      if (vmax.type() != value.type() || !can_prove(vmax < iv->dom->extent)) {
        preds.emplace_back(value < (iv->dom->extent - iv->dom->min));
      }
    }
  }
  return preds;
}
}  // namespace schedule
}  // namespace tvm
