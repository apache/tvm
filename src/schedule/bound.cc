/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>
#include "./int_set.h"
#include "./graph.h"

namespace tvm {
namespace schedule {

// result = ceil((a / b)), both a and b are positive integer
inline Expr DivCeil(Expr a, Expr b) {
  return (a + b - 1) / b;
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
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

/*!
 * \brief Pass the bound of tensor read
 *  to the corresponding bound of the IterVar of operation
 * \param tensor The tensor to be passed.
 * \param dim_bounds The read index set on each dimension.
 * \param The result IterVar bound .
 */
void PassToOperation(
    const Tensor& tensor,
    const std::vector<IntSet>& dim_bounds,
    std::unordered_map<IterVar, std::vector<IntSet> >* result) {
  // This is a push style operation, given output bound, push to the op IterVar bound.
  // It cannot handle complicated cases where op bound is coupled with bounds of
  // all of its outputs, without having a simple communicative union relation.
  //
  // Eventually, we need to change the inference to be a Pull style inference
  if (tensor->op.as<ComputeOpNode>()) {
    auto root_iter_vars = tensor->op->root_iter_vars();
    CHECK_EQ(tensor.ndim(), root_iter_vars.size());
    for (size_t i = 0; i < tensor.ndim(); ++i) {
      (*result)[root_iter_vars[i]].push_back(dim_bounds[i]);
    }
  } else {
    LOG(FATAL) << "unknown operation mode " << tensor->op->type_key();
  }
}

/*!
 * \brief Recursively propagate bound
 * \param post_order The propagation order.
 * \param dom_map The domain map to be propagated
 * \return The result bound
 */
std::unordered_map<IterVar, IntSet>
BoundProp(const Array<Operation>& post_order,
          std::unordered_map<IterVar, std::vector<IntSet> > *p_state) {
  std::unordered_map<IterVar, IntSet> result;

  for (size_t i = post_order.size(); i != 0; --i) {
    Operation op = post_order[i - 1];
    if (op.as<ComputeOpNode>()) {
      for (auto iv : op->root_iter_vars()) {
        CHECK(p_state->count(iv))
            << "Bound of root operator must exists";
        CHECK(!result.count(iv));
        result[iv] = Union(p_state->at(iv));
      }
      auto fvisit = [p_state, &result](const NodeRef& n) {
        auto *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Tensor t = Operation(call->func.node_).output(call->value_index);
          if (t->op.defined() && !t->op.as<PlaceholderOpNode>()) {
            std::vector<IntSet> arg_bounds;
            for (size_t i = 0; i < t.ndim(); ++i) {
              arg_bounds.push_back(EvalSet(call->args[i], result));
            }
            PassToOperation(t, arg_bounds, p_state);
          }
        }
      };
      ir::PostOrderVisit(op.as<ComputeOpNode>()->body, fvisit);
    } else if (op.as<PlaceholderOpNode>()) {
      // do nothing
    } else {
      LOG(FATAL) << "unknown operation mode " << op->type_key();
    }
  }
  return result;
}


// check if scope
bool ScopeRelax(const IterVar& iv, const std::string& scope) {
  if (iv->thread_tag.length() == 0) return false;
  if (scope.length() == 0) return false;

  static std::unordered_map<std::string, int> scope_rank{
    {"global", 0},
    {"shared", 1},
    {"local", 2}
  };
  static std::unordered_map<std::string, int> thread_tag_rank{
    {"gridIdx.x", 0},
    {"gridIdx.y", 0},
    {"gridIdx.z", 0},
    {"threadIdx.x", 1},
    {"threadIdx.y", 1},
    {"threadIdx.z", 1}
  };
  return scope_rank.at(scope) <= thread_tag_rank.at(iv->thread_tag);
}

void InferBound(const Stage& stage,
                std::unordered_map<IterVar, Range>* rmap) {
  if (stage->attach_type == kInline) return;
  if (stage->attach_type == kRoot || stage->attach_type == kNone) {
    auto root_iter_vars = stage->op->root_iter_vars();
    for (auto iv :  root_iter_vars) {
      CHECK(iv->dom.defined());
      CHECK(!rmap->count(iv));
      (*rmap)[iv] = iv->dom;
    }
  }
  // get range of all child iter vars.
  PassDown(stage, rmap);

  if (stage->attach_type == kScope) {
    Stage parent = stage->attach_stage;
    CHECK(parent.defined());
    auto g = CreateReadGraph({parent->op});
    auto post_order = PostDFSOrder({parent->op}, g);
    std::unordered_map<IterVar, IntSet> up_state;

    bool fix_value = true;
    for (auto iv : parent->leaf_iter_vars) {
      if (fix_value && !ScopeRelax(iv, stage->scope)) {
        up_state[iv] = IntSet::make_point(iv->var);
      } else {
        up_state[iv] = IntSet::make_range(rmap->at(iv));
      }
      if (stage->attach_ivar == iv) {
        fix_value = false;
      }
    }
    // get the bound of the root IterVars given the current condition
    PassUp(parent, *rmap, &up_state);
    std::unordered_map<IterVar, std::vector<IntSet> > bp_state;
    for (auto iv : parent->op->root_iter_vars()) {
      CHECK(up_state.count(iv));
      bp_state[iv] = {up_state.at(iv)};
    }
    auto result = BoundProp(post_order, &bp_state);
    for (auto iv : stage->op->root_iter_vars()) {
      CHECK(result.count(iv));
      CHECK(!rmap->count(iv));
      (*rmap)[iv] = result.at(iv).GetCoverRange();
    }
  }
}


Map<IterVar, Range> InferBound(Schedule sch) {
  std::unordered_map<IterVar, Range> ret;
  // reverse post DFS order, from out most stage to the innermost
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    InferBound(stage, &ret);
  }
  return Map<IterVar, Range>(ret.begin(), ret.end());
}

}  // namespace schedule
}  // namespace tvm
