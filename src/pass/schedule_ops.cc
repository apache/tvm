/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>

#include "./scope.h"
#include "../schedule/graph.h"

namespace tvm {
namespace ir {
namespace {

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
      Expr offset = inner + outer * factor;
      Expr outer_min = dom_map.at(s->parent)->min;
      if (!is_zero(outer_min)) {
        offset = outer_min + offset;
      }
      state[s->parent] = offset;
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      Expr value = state.at(s->fused);
      Expr factor = dom_map.at(s->inner)->extent;
      state[s->outer] = value / factor;
      state[s->inner] = value % factor;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

/*!
 * \brief split the expr by addition.
 * \param expr The expression to be splitted.
 * \param loop_level The loop level of each Variable
 * \param result vector of (level, expr)
 *   The level gives the mimimum loop level this expression need to be computed.
 *   The Expr gives the expression content.
 */
void SplitByAdd(Expr expr,
                const std::unordered_map<const Variable*, size_t>& loop_level,
                std::vector<std::pair<size_t, Expr> > *result) {
  const Add* op = expr.as<Add>();
  if (op != nullptr) {
    SplitByAdd(op->a, loop_level, result);
    SplitByAdd(op->b, loop_level, result);
  } else {
    size_t max_level = 0;
    auto fvisit = [&max_level, &loop_level](const NodeRef& n) {
      const Variable* op = n.as<Variable>();
      if (op != nullptr) {
        auto it = loop_level.find(op);
        if (it != loop_level.end()) {
          max_level = std::max(max_level, it->second);
        }
      }
    };
    PostOrderVisit(expr, fvisit);
    result->push_back(std::make_pair(max_level, expr));
  }
}

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 */
Stmt MergeNest(std::vector<std::vector<Stmt> > nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    for (auto rj = ri->rbegin(); rj != ri->rend(); ++rj) {
      Stmt s = *rj;
      if (s.as<For>()) {
        auto n = std::make_shared<For>(*s.as<For>());
        CHECK(is_no_op(n->body));
        n->body = body;
        body = Stmt(n);
      } else if (s.as<LetStmt>()) {
        auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
        CHECK(is_no_op(n->body));
        n->body = body;
        body = Stmt(n);
      } else if (s.as<AttrStmt>()) {
        auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
        CHECK(is_no_op(n->body));
        n->body = body;
        body = Stmt(n);
      } else if (s.as<IfThenElse>()) {
        auto n = std::make_shared<IfThenElse>(*s.as<IfThenElse>());
        CHECK(is_no_op(n->then_case));
        CHECK(!n->else_case.defined());
        n->then_case = body;
        body = Stmt(n);
      } else {
        LOG(FATAL) << "not supported nest type";
      }
    }
  }
  return body;
}

/*!
 * \brief Make the loop nest of the correspondings schedule.
 * \param sch The schedule.
 * \param dom_map The domain map.
 *
 * \return a nested representation of loop statements.
 *  The flattened Stmt are ordered from outmost to inner most order.
 */
std::vector<std::vector<Stmt> > MakeLoopNest(
    const Stage& sch,
    const Map<IterVar, Range>& dom_map) {
  // optional, use let to define some CSE in dom_map.
  auto leaf_iter_vars = sch->leaf_iter_vars;
  std::unordered_map<IterVar, Expr> offset;
  std::unordered_map<const Variable*, size_t> loop_level;
  Stmt no_op = Evaluate::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);

  for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    // initialize the offset and loop_level
    offset[iv] = iv->var;
    loop_level[iv->var.as<Variable>()] = i + 1;
    // Mark the iter var in the IR, to remember the point
    if (iv->thread_tag.length() == 0) {
      Range dom = dom_map.at(iv);
      nest[i + 1].emplace_back(
          For::make(iv->var, dom->min, dom->extent,
                    ForType::Serial, DeviceAPI::None, no_op));
    }
    nest[i + 1].emplace_back(
        AttrStmt::make(iv, "scope", iv->var, no_op));
  }
  // message passing to get offset of root iter vars.
  PassUpOffset(sch, dom_map, &offset);

  for (IterVar iv : sch->op->root_iter_vars()) {
    Expr value = offset.at(iv);
    if (!value.same_as(iv->var)) {
      using Entry = std::pair<size_t, Expr>;
      std::vector<Entry> splits;
      SplitByAdd(value, loop_level, &splits);

      Expr offset = 0;
      size_t nsplit_left = splits.size() - 1;
      for (size_t i = 0; i <= leaf_iter_vars.size(); ++i) {
        size_t hit = 0;
        for (const auto& kv : splits) {
          if (kv.first == i) {
            if (is_zero(offset)) {
              offset = kv.second;
            } else {
              offset = offset + kv.second;
              ++hit;
            }
          }
        }
        nsplit_left -= hit;
        if (hit != 0) {
          std::ostringstream os;
          os << iv->var->name_hint << ".at.l" << i;
          Var base_offset(os.str());
          if (nsplit_left == 0) {
            base_offset = iv->var;
          }
          nest[i].emplace_back(
              LetStmt::make(base_offset, offset, no_op));
          offset = base_offset;
        }
      }
      Range dom = dom_map.at(iv);
      if (!offset.same_as(iv->var)) {
        // define the iv->var
        nest.back().emplace_back(
            LetStmt::make(iv->var, offset, no_op));
      }
      Expr condition = (iv->var - dom->min) < dom->extent;
      // Boundary condition checking
      // Need better boundary condition here.
      nest.back().emplace_back(IfThenElse::make(condition, no_op));
    }
  }
  return nest;
}


/*!
 * \brief Make pipeline specifically for compute op node.
 * \param op The compute node
 * \param tensors The tensors generated by provide.
 */
Stmt MakeProvide(const ComputeOpNode* op,
                 const std::vector<Tensor>& tensors) {
  Tensor t = tensors[0];
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body, args);
}

/*!
 * \brief Make pipeline specifically for compute op node.
 * \param op The compute node
 * \param dom_map The domain map
 * \param tensors The tensors generated by provide.
 * \param body The content of the pipeline.
 */
Stmt MakeRealize(const ComputeOpNode* op,
                 const Map<IterVar, Range>& dom_map,
                 const std::vector<Tensor>& tensors,
                 Stmt body) {
  Tensor t = tensors[0];
  Halide::Internal::Region bounds;
  for (IterVar iv : op->axis) {
    bounds.push_back(dom_map.at(iv));
  }
  return Realize::make(t->op, t->value_index, t->dtype,
                       bounds, make_const(Bool(1), true), body);
}

Stmt MakePipeline(const Stage& sch,
                  const Map<IterVar, Range>& dom_map,
                  Stmt consumer) {
  std::vector<Tensor> tensors;
  for (int i = 0; i < sch->op->num_outputs(); ++i) {
    tensors.emplace_back(sch->op.output(i));
  }

  Stmt provide;
  if (sch->op.as<ComputeOpNode>()) {
    provide = MakeProvide(sch->op.as<ComputeOpNode>(), tensors);
  } else {
    LOG(FATAL) << "not supported op " << sch->op->type_key();
  }
  std::vector<std::vector<Stmt> > nest = MakeLoopNest(sch, dom_map);
  Stmt producer = MergeNest(nest, provide);
  producer = ProducerConsumer::make(sch->op, true, producer);

  Stmt pipeline = producer;
  if (consumer.defined()) {
    consumer = ProducerConsumer::make(sch->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }

  if (sch->op.as<ComputeOpNode>()) {
    return MakeRealize(sch->op.as<ComputeOpNode>(),
                       dom_map, tensors, pipeline);
  } else {
    LOG(FATAL) << "not supported op";
    return Stmt();
  }
}

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  InjectRealize(Stage schedule, Map<IterVar, Range> dom_map)
      : schedule(schedule), dom_map(dom_map) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        op->type_key == "scope") {
      if (op->node == schedule->attach_ivar) {
        CHECK(!found_attach);
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->type_key, op->value,
            MakePipeline(schedule, dom_map,
                         IRMutator::Mutate(op->body)));
      }
    }
    return stmt;
  }
  // the operations to be carried
  Stage schedule;
  // domain map
  Map<IterVar, Range> dom_map;
  // whether attach point is found
  bool found_attach{false};
};

Stmt InjectInline(const Operation op, Stmt body) {
  CHECK(body.defined());
  const ComputeOpNode* compute = op.as<ComputeOpNode>();
  CHECK(compute != nullptr)
      << "can only inline compute op";
  Array<Var> args;
  for (auto iv : compute->axis) {
    args.push_back(iv->var);
  }
  return Inline(op, args, compute->body, body);
}

}  // namespace

Stmt ScheduleOps(
    Schedule sch, Map<IterVar, Range> dom_map) {
  Stmt body = Stmt();
  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    if (s->attach_type == kInline) {
      body = InjectInline(s->op, body);
    } else if (s->attach_type == kRoot || s-> attach_type == kNone) {
      body = MakePipeline(s, dom_map, body);
    } else if (s->attach_type == kScope) {
      CHECK(body.defined());
      InjectRealize mutator(s, dom_map);
      body = mutator.Mutate(body);
      CHECK(mutator.found_attach)
          << "did not find attachment point";
    }
  }
  return body;
}

}  // namespace ir
}  // namespace tvm
