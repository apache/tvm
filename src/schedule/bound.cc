/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/schedule_pass.h>
#include <tvm/arithmetic.h>
#include <unordered_map>
#include <unordered_set>
#include "./graph.h"
#include "../runtime/thread_storage_scope.h"

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

// Downward message passing algorithm on stage schedule s,
// pass the range state down from the root to the leaves
// after this pass, every IterVar in the stage hyper graph will have a range(domain)
void PassDown(const Stage& s,
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
          if (!state.count(r->outer)) {
            state[r->outer] = Range::make_with_min_extent(
                0, DivCeil(range_parent->extent, r->factor));
          } else {
            Expr outer_ext = DivCeil(range_parent->extent, r->factor);
            Range outer_rng = state.at(r->outer);
            bool match = is_zero(outer_rng->min);
            if (!prove_equal(outer_ext, outer_rng->extent)) match = false;
            CHECK(match)
                << "IterVar is used in two places as outer scope,"
                << " cannot prove their extents are the same";
          }
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
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* r = rel.as<RebaseNode>();
      CHECK(state.count(r->parent));
      state[r->rebased] = Range::make_with_min_extent(
          0, state.at(r->parent)->extent);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// upward message passing algorithm
// pass the integer set on each leave loop up to the root
// dom_map is the result of PassDown, it records the domain of each IterVar.
// dom_map can be used to get cached result in reverse construction.
// Implementation of Evaluations and passing.
void PassUp(const SplitNode* s,
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

void PassUp(const FuseNode* s,
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

void PassUp(const RebaseNode* s,
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

void PassUp(const Stage& s,
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
    } else if (rel.as<RebaseNode>()) {
      IntSet parent;
      const RebaseNode* r = rel.as<RebaseNode>();
      PassUp(r, dom_map,
             state.at(r->rebased),
             &parent);
      state[r->parent] = parent;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

// All the itervars that are needed to output bound of op.
// For most op, it is root_iter_vars
// For Scan, it also contains the additional spatial axis.
Array<IterVar> OutputRelatedIterVars(const Operation& op) {
  if (op.as<ScanOpNode>()) {
    const ScanOpNode* scan = op.as<ScanOpNode>();
    Array<IterVar> ret{scan->scan_axis};
    for (IterVar iv : scan->spatial_axis_) {
      ret.push_back(iv);
    }
    return ret;
  } else {
    return op->root_iter_vars();
  }
}

/*! \brief temporary data structure to store Tensor domain */
struct TensorDom {
  // constructor
  explicit TensorDom(int ndim)
      : data(ndim) {}
  /*! \brief The domain data*/
  std::vector<std::vector<IntSet> > data;
};

/*!
 * \brief Propagate bound to target
 * \param dom_map The domain map to be propagated
 * \param out The tensor set to be passed
 * \return The result bound
 */
void BoundProp(const Operation& op,
               const std::unordered_map<const Variable*, IntSet>& dom_map,
               std::unordered_map<Tensor, TensorDom> *out) {
  if (op.as<ComputeOpNode>()) {
    auto fvisit = [&dom_map, out](const NodeRef& n) {
      auto *call = n.as<ir::Call>();
      if (call != nullptr && call->func.defined()) {
        Tensor t = Operation(call->func.node_).output(call->value_index);
        if (t->op.defined() && out->count(t)) {
          TensorDom& dom = out->at(t);
          for (size_t i = 0; i < t.ndim(); ++i) {
            dom.data[i].push_back(EvalSet(call->args[i], dom_map));
          }
        }
      }
    };
    ir::PostOrderVisit(op.as<ComputeOpNode>()->body, fvisit);
  } else if (op.as<ScanOpNode>()) {
    const ScanOpNode* scan = op.as<ScanOpNode>();
    size_t sp_idx = 0;
    for (size_t i = 0; i < scan->init.size(); ++i) {
      TensorDom* init_dom = nullptr;
      TensorDom* update_dom = nullptr;
      if (out->count(scan->init[i])) {
        init_dom = &out->at(scan->init[i]);
      }
      if (out->count(scan->update[i])) {
        update_dom = &out->at(scan->update[i]);
      }
      // first dimension, always needed.
      if (init_dom) {
        init_dom->data[0].push_back(IntSet::range(
            Range::make_with_min_extent(0, scan->init[i]->shape[0])));
      }
      if (update_dom) {
        update_dom->data[0].push_back(dom_map.at(scan->scan_axis->var.get()));
      }
      // The update dimensions
      for (size_t k = 1; k < scan->update[i]->shape.size(); ++k, ++sp_idx) {
        IterVar sp_ax = scan->spatial_axis_[sp_idx];
        if (init_dom) {
          init_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
        }
        if (update_dom) {
          update_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
        }
      }
    }
  } else if (op.as<PlaceholderOpNode>()) {
    // do nothing
  } else {
    LOG(FATAL) << "unknown operation mode " << op->type_key();
  }
}


// Given the bound of output of op
// Pass the bound to the related axis in op.
void GatherOpBound(const ScanOpNode* scan,
                   const Operation& op,
                   const FeedGraph& fg,
                   const std::unordered_map<Tensor, TensorDom>& tmap,
                   std::unordered_map<IterVar, Range>* rmap) {
  CHECK(!rmap->count(scan->scan_axis));
  std::vector<Tensor> output(op->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = op.output(i);
  }
  // Update for time axis.
  std::vector<IntSet> time_dom;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tmap.at(output[i]);
    time_dom.insert(time_dom.end(), d.data[0].begin(), d.data[0].end());
  }
  CHECK(!rmap->count(scan->scan_axis));
  Range sdom = scan->scan_axis->dom;
  Range r = arith::Union(time_dom).cover_range(sdom);
  (*rmap)[scan->scan_axis] = Range::make_with_min_extent(
      sdom->min, ir::Simplify(r->extent + r->min - sdom->min));
  Array<Operation> body = ScanGetBody_(scan, fg);
  Map<IterVar, Expr> fix_pt = ScanFixPointAnalysis(op, body);
  // Update for spatial axis.
  size_t sp_idx = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tmap.at(output[i]);
    for (size_t k = 1; k < scan->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = scan->spatial_axis_[sp_idx];
      CHECK(!rmap->count(sp_ax));
      CHECK(fix_pt.count(sp_ax));
      if (fix_pt[sp_ax].as<ir::IntImm>()->value) {
        // fix point, we can slice it.
        (*rmap)[sp_ax] = arith::Union(d.data[k + 1]).cover_range(sp_ax->dom);
      } else {
        // not a fix point, need to include everything.
        (*rmap)[sp_ax] = sp_ax->dom;
      }
    }
  }
}

void GatherOpBound(const Operation& op,
                   const FeedGraph& fg,
                   const std::unordered_map<Tensor, TensorDom>& tmap,
                   std::unordered_map<IterVar, Range>* rmap) {
  if (op.as<ComputeOpNode>()) {
    const ComputeOpNode* compute = op.as<ComputeOpNode>();
    const TensorDom& tdom = tmap.at(op.output(0));
    for (size_t i = 0; i < compute->axis.size(); ++i) {
      Range r = arith::Union(tdom.data.at(i)).cover_range(compute->axis[i]->dom);
      CHECK(!rmap->count(compute->axis[i]));
      (*rmap)[compute->axis[i]] = r;
    }
    for (size_t i = 0; i < compute->reduce_axis.size(); ++i) {
      CHECK(!rmap->count(compute->reduce_axis[i]));
      (*rmap)[compute->reduce_axis[i]] = compute->reduce_axis[i]->dom;
    }
  } else if (op.as<ScanOpNode>()) {
    GatherOpBound(op.as<ScanOpNode>(), op, fg, tmap, rmap);
  } else if (op.as<PlaceholderOpNode>()) {
    // dp nothing
  } else {
    LOG(FATAL) << "unknown operation mode " << op->type_key();
  }
}

// check if scope
inline bool ScopeRelax(const IterVar& iv, const std::string& scope) {
  using runtime::ThreadScope;
  using runtime::StorageScope;
  if (iv->thread_tag.length() == 0) return false;
  if (scope.length() == 0) return false;

  return StorageScope::make(scope).rank <= ThreadScope::make(iv->thread_tag).rank;
}

void InferRootBound(const Stage& stage,
                    const FeedGraph& feed_graph,
                    const AttachPath& attach_path,
                    std::unordered_map<IterVar, Range>* rmap) {
  CHECK_NE(stage->attach_type, kInline)
      << "call schedule.normalize before scheduleops";
  if (stage->attach_type == kInlinedAlready) return;
  if (stage->is_output || stage->op.as<PlaceholderOpNode>()) {
    for (auto iv :  OutputRelatedIterVars(stage->op)) {
      CHECK(iv->dom.defined());
      CHECK(!rmap->count(iv));
      (*rmap)[iv] = iv->dom;
    }
    return;
  }
  // parent stage, if any
  Stage parent;
  if (stage->attach_type == kScope || stage->attach_type == kScanUpdate) {
    parent = stage->attach_stage;
  }
  // The tensor domain.
  std::unordered_map<Tensor, TensorDom> tmap;
  // consumers other than parent
  std::unordered_set<Operation> consumers;
  // initialize the result
  bool direct_consume_by_parent = false;
  for (int i = 0; i < stage->op->num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    tmap.emplace(t, TensorDom(static_cast<int>(t.ndim())));
    auto it = feed_graph.find(t);
    if (it != feed_graph.end()) {
      for (const Operation& op : it->second) {
        if (!parent.defined() || op != parent->op) {
          consumers.insert(op);
        } else {
          direct_consume_by_parent = true;
        }
      }
    } else {
      LOG(INFO) << "not in feed graph consumer = " << stage->op;
    }
  }
  // The relax set
  // Thie specifieds the iteration variables that need to be relaxed
  // from the already inferred bounds.
  std::unordered_map<const Variable*, IntSet> relax_set;
  for (IterVar iv : attach_path.at(stage->op)) {
    if (ScopeRelax(iv, stage->scope)) {
      relax_set[iv->var.get()] = IntSet::range(rmap->at(iv));
    }
  }
  if (direct_consume_by_parent) {
    // parent stage if exist
    Stage parent = stage->attach_stage;
    // Bound inference logics in parent.
    std::unordered_map<IterVar, IntSet> up_state;
    bool fix_value = true;
    for (auto iv : parent->leaf_iter_vars) {
      auto it = rmap->find(iv);
      CHECK(it != rmap->end());
      Range vrange = it->second;
      CHECK(is_zero(vrange->min))
          << "InferBound requires every leaf iter var's min equals 0, "
          << " call schedule.normalize to achieve this. "
          << " stage=" << parent;
      // special optimization to remove trivial loop
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::single_point(vrange->min);
      }
      if (fix_value && !ScopeRelax(iv, stage->scope)) {
        up_state[iv] = IntSet::single_point(iv->var);
      } else {
        up_state[iv] = IntSet::range(vrange);
      }
      if (stage->attach_ivar == iv) {
        fix_value = false;
      }
    }
    // get the bound of the root IterVars given current location.
    PassUp(parent, *rmap, &up_state);

    std::unordered_map<const Variable*, IntSet> dom_map;
    for (auto iv : OutputRelatedIterVars(parent->op)) {
      Range r;
      if (up_state.count(iv)) {
        r = up_state.at(iv).cover_range(iv->dom);
      } else {
        r = iv->dom;
      }
      if (relax_set.size() != 0) {
        dom_map[iv->var.get()] = EvalSet(r, relax_set);
      } else {
        dom_map[iv->var.get()] = IntSet::range(r);
      }
    }
    // prop from parent.
    BoundProp(parent->op, dom_map, &tmap);
  }
  // Bound prop by other consumers.
  // To explain the the general logic, consider the example:
  //
  // for (i_outer, 0, 10) {
  //   producer
  //
  //   for (i_inner, 0, 4) {
  //     consumer op
  //   }
  // }
  // - Get domain of each of consumer op, say [i_inner + i_outer*8, extent=4)
  // - We need to relax it since the producer is attached at i_outer
  // - Consumer's path is [i_inner, i_outer], then [i_inner] need to be relaxed
  // - Traverse attach_path, relax until reaching the producer's attachment point.
  for (const Operation& op : consumers) {
    std::unordered_map<const Variable*, IntSet> dom_map;
    bool found = false;
    Array<IterVar> attach = attach_path.at(stage->op);
    for (IterVar iv : attach_path.at(op)) {
      if (attach.size() != 0 && iv == attach[0]) {
        found = true; break;
      }
      Range vrange = rmap->at(iv);
      CHECK(is_zero(vrange->min))
          << "InferBound requires every leaf iter var's min equals 0, "
          << "call schedule.normalize to achieve this.";
      relax_set[iv->var.get()] = IntSet::range(vrange);
    }
    CHECK(found || attach.size() == 0)
        << "Invalid Schedule, cannot find the producer " << stage->op
        << " along the loop nest specified by compute_at of consumer " << op;
    for (auto iv : OutputRelatedIterVars(op)) {
      Range r = rmap->at(iv);
      dom_map[iv->var.get()] = EvalSet(r, relax_set);
    }
    BoundProp(op, dom_map, &tmap);
  }
  GatherOpBound(stage->op, feed_graph, tmap, rmap);
}

Map<IterVar, Range> InferBound(const Schedule& sch) {
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  FeedGraph feed_graph = CreateFeedGraph(CreateReadGraph(roots));
  AttachPath attach_path = CreateAttachPath(sch);

  std::unordered_map<IterVar, Range> ret;
  for (size_t i = sch->stages.size(); i != 0; --i) {
    const Stage& stage = sch->stages[i - 1];
    InferRootBound(stage, feed_graph, attach_path, &ret);
    // pass down to get bound of all iter vars.
    PassDown(stage, &ret);
    // setup outer most threads.
    for (IterVar iv : stage->outermost_threads) {
      CHECK(iv->dom.defined());
      ret[iv] = iv->dom;
    }
  }
  return Map<IterVar, Range>(ret.begin(), ret.end());
}

}  // namespace schedule
}  // namespace tvm
