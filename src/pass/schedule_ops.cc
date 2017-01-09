/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include "./scope.h"

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
void PassUpOffset(const Schedule& s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, Expr>* p_state) {
  auto& state = *p_state;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      Expr outer = state.at(s->outer);
      Expr inner = state.at(s->outer);
      Expr factor = dom_map.at(s->outer)->extent;
      Expr offset = inner + outer * factor;
      Expr outer_min = dom_map.at(s->parent)->min;
      if (!is_zero(outer_min)) {
        offset = outer_min + offset;
      }
      state[s->parent] = offset;
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      Expr value = state.at(s->fused);
      Expr factor = dom_map.at(s->outer)->extent;
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
Stmt CombineNest(std::vector<Stmt>&& nest, Stmt body) {
  while (!nest.empty()) {
    Stmt s = std::move(nest.back());
    nest.pop_back();
    if (s.as<For>()) {
      auto n = std::make_shared<For>(*s.as<For>());
      n->body = body;
      body = Stmt(n);
    } else if (s.as<LetStmt>()) {
      auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
      n->body = body;
      body = Stmt(n);
    } else if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

/*!
 * \brief Make the loop nest of the correspondings schedule.
 * \param sch The schedule.
 * \param dom_map The domain map.
 */
std::vector<Stmt> MakeLoopNest(
    const Schedule& sch,
    const std::unordered_map<IterVar, Range>& dom_map) {
  // optional, use let to define some CSE in dom_map.
  auto leaf_iter_vars = sch->leaf_iter_vars;
  std::unordered_map<IterVar, Expr> offset;
  std::unordered_map<const Variable*, size_t> loop_level;

  // create the loop nest
  std::vector<Stmt> nest;
  nest.resize(leaf_iter_vars.size() + 1, Stmt());

  for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    // initialize the offset and loop_level
    offset[iv] = iv->var;
    loop_level[iv->var.as<Variable>()] = i + 1;

    nest[i] = AttrStmt::make(iv->var, "scope", iv, Stmt());
    if (iv->thread_tag.length() == 0) {
      Range dom = dom_map.at(iv);
      nest[i] = For::make(iv->var, dom->min, dom->extent,
                          ForType::Serial, DeviceAPI::None, nest[i]);
    }
  }
  // message passing to get offset of root iter vars.
  PassUpOffset(sch, dom_map, &offset);
  for (IterVar iv : sch->op->root_iter_vars()) {
    Expr value = offset.at(iv);
    if (value.same_as(iv->var)) continue;
    using Entry = std::pair<size_t, Expr>;
    std::vector<Entry> splits;
    SplitByAdd(value, loop_level, &splits);

    Expr offset = 0;
    for (size_t i = 0; i <= leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      for (const auto& kv : splits) {
        if (kv.first == i) {
          offset = offset + splits[i].second;
        }
      }
      std::ostringstream os;
      os << iv->var->name_hint << ".at.l" << i;
      Var base_offset(os.str());
      nest[i] = LetStmt::make(base_offset, offset, nest[i]);
      offset = base_offset;
    }
    nest.back() = LetStmt::make(iv->var, offset, nest.back());
  }
  return nest;
}

/*!
 * \brief Make the loop nest of the correspondings schedule.
 * \param op The operation.
 */
Stmt MakeBody(const Operation& op) {
  Stmt body;
  if (op.as<ComputeOpNode>()) {
    const ComputeOpNode* compute = op.as<ComputeOpNode>();
    // Note: Tensor's address cannot uniquely
    Tensor t = op.output(0);
    Array<Expr> args;
    for (IterVar iv : compute->axis) {
      args.push_back(iv->var);
    }
    body = Provide::make(t, {compute->body}, args);
  } else {
    LOG(FATAL) << "not supported op";
  }
  return body;
}

Stmt MakePipeline(const Schedule& sch, Stmt body) {
  return body;
}

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  explicit InjectRealize(Schedule sch)
      : sch_(sch) {}

  Stmt Mutate(Stmt stmt) final {
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr) {
      attr_scope_.Push({op->node, op->type_key}, op->value);
      stmt = IRMutator::Mutate(stmt);
      attr_scope_.Pop({op->node, op->type_key});
    } else {
      stmt = IRMutator::Mutate(stmt);
    }

    if (op != nullptr &&
        op->type_key == "scope" &&
        op->node == sch_->attach_parent) {
      return AttrStmt::make(
          op->node, op->type_key, op->value,
          MakePipeline(sch_, op->body));
    } else {
      return stmt;
    }
  }

 private:
  // the operations to be carried
  Schedule sch_;
  Scope<AttrKey, Expr> attr_scope_;
};


}  // namespace
}  // namespace ir
}  // namespace tvm
