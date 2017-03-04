/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Utility to make loop nest.
 * \file make_loop.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include "./make_loop.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace op {

using namespace arith;
using namespace ir;

/*!
 * \brief use message passing to calculate the assignment of each Var inside the loop body.
 * \param s The schedule to be used.
 * \param dom_map The domain map of each iteration variable's domain
 * \param p_state The message passing state
 *     IterVar->The assignment.
 */
void PassUpOffset(const Stage& s,
                  const Map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, Expr>* p_state) {
  auto& state = *p_state;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      Expr outer = state.at(s->outer);
      Expr inner = state.at(s->inner);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr parent_min = dom_map.at(s->parent)->min;
      state[s->parent] = inner + outer * factor;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = state[s->parent] + parent_min;
      }
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
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
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
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

std::vector<std::vector<Stmt> >
MakeLoopNest(const Stage& stage,
             const std::unordered_map<IterVar, Range>& dom_map,
             size_t begin_iter_pos,
             bool new_loop_var,
             const std::unordered_set<IterVar>& skip_iter,
             std::unordered_map<IterVar, Expr>* p_value_map) {
  auto leaf_iter_vars = stage->leaf_iter_vars;
  Stmt no_op = Evaluate::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, Expr>& value_map = *p_value_map;

  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    Range dom = dom_map.at(iv);
    // initialize the offset and loop_level
    Var var = iv->var;
    if (new_loop_var) {
      var = Var(iv->var->name_hint + ".init", iv->var.type());
    }
    // Mark the iter var in the IR, to remember the point
    if (iv->thread_tag.length() == 0) {
      ForType for_type = ForType::Serial;
      if (stage->iter_var_attrs.count(iv)) {
        switch (stage->iter_var_attrs[iv]->iter_type) {
          case kUnrolled: for_type = ForType::Unrolled; break;
          case kVectorized: for_type = ForType::Vectorized; break;
          case kParallelized: for_type = ForType::Parallel; break;
          default: LOG(FATAL) << "Unknown iter type"
                              << stage->iter_var_attrs[iv]->iter_type
                              << " in the iter_var_attrs";
        }
      }
      if (is_one(dom->extent)) {
        nest[i + 1].emplace_back(
            LetStmt::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            For::make(var, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
      } else {
        Var idx(iv->var->name_hint + ".idx", iv->var.type());
        nest[i + 1].emplace_back(
            For::make(idx, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        Expr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(
            LetStmt::make(var, new_value, no_op));
      }
    } else if (iv->thread_tag == "vthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, ir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, ir::attr::thread_extent, dom->extent, no_op));
      value_map[iv] = var;
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  PassUpOffset(stage, dom_map, &value_map);
  return nest;
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
  using Halide::Internal::can_prove;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      bool outer = state.at(s->outer);
      bool inner = state.at(s->inner);
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
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      bool fused = state.at(s->fused);
      state[s->outer] = fused;
      state[s->inner] = fused;
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
      state[s->parent] = state.at(s->rebased);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

std::vector<Stmt> MakeBoundCheck(
    const Stage& stage,
    const Map<IterVar, Range>& dom_map,
    bool skip_ivar_domain,
    const std::unordered_set<IterVar>& skip_iter,
    const std::unordered_map<IterVar, Expr>& value_map) {
  Stmt no_op = Evaluate::make(0);
  std::unordered_map<IterVar, bool> bound_state;
  for (IterVar iv : stage->leaf_iter_vars) {
    bound_state[iv] = false;
  }
  PassUpBoundCheck(stage, dom_map, &bound_state);
  // insert conditions
  std::vector<Stmt> nest;
  for (IterVar iv : stage->op->root_iter_vars()) {
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) continue;
    Range dom = dom_map.at(iv);
    if (bound_state.at(iv)) {
      Expr condition = ComputeExpr<Sub>(value_map.at(iv), dom->min) < dom->extent;
      nest.emplace_back(IfThenElse::make(condition, no_op));
    }
    CHECK(iv->dom.defined());
    if (!skip_ivar_domain && !iv->dom.same_as(dom)) {
      Expr condition = ComputeExpr<Sub>(value_map.at(iv), iv->dom->min) < iv->dom->extent;
      nest.emplace_back(IfThenElse::make(condition, no_op));
    }
  }
  return nest;
}

}  // namespace op
}  // namespace tvm
