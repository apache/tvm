/*!
 *  Copyright (c) 2017 by Contributors
 * \file message_passing.cc
 * \brief The message passing domain.
 */
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include "./message_passing.h"

namespace tvm {
namespace schedule {

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
      state[s->outer] = value / factor;
      state[s->inner] = value % factor;
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
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

}  // namespace schedule
}  // namespace tvm
