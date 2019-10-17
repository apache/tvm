/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Hybrid computation rule.
 * \file hybrid_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/expr_operator.h>
#include <unordered_set>
#include <string>
#include <utility>
#include "op_util.h"
#include "hybrid_op.h"

namespace tvm {
using namespace ir;
// HybridOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<HybridOpNode>([](const HybridOpNode *op, IRPrinter *p) {
    p->stream << "hybrid(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(HybridOpNode);

int HybridOpNode::num_outputs() const {
  return static_cast<int>(outputs.size());
}

Array<IterVar> HybridOpNode::root_iter_vars() const {
  return this->axis;
}

Type HybridOpNode::output_dtype(size_t i) const {
  return outputs[i]->dtype;
}

Array<Expr> HybridOpNode::output_shape(size_t i) const {
  return outputs[i]->shape;
}


Operation HybridOpNode::make(std::string name,
                             std::string tag,
                             Map<std::string, NodeRef> attrs,
                             Array<Tensor> inputs,
                             Array<Tensor> outputs,
                             Stmt body) {
  if (!attrs.defined()) {
    attrs = Map<std::string, NodeRef>();
  }
  auto n = make_node<HybridOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->axis = op::GatherLoopVars(body);
  n->body = std::move(body);
  Operation res = Operation(n);
  return res;
}

Array<Tensor> HybridOpNode::InputTensors() const {
  // Because input tensors could be potentially inlined into hybrid scripts,
  // we need to check if all input tensors are used in the body.
  std::unordered_set<Tensor> orig_inputs;
  for (auto t : inputs) {
    orig_inputs.insert(t);
  }
  std::unordered_set<Tensor> visited;
  Array<Tensor> curr_inputs;
  ir::PostOrderVisit(body, [&curr_inputs, &orig_inputs, &visited](const NodeRef& n) {
      const ir::Call *call = n.as<ir::Call>();
      if (call != nullptr && call->func.defined()) {
        Tensor t = Operation(call->func.node_).output(call->value_index);
        if (orig_inputs.count(t) && !visited.count(t)) {
          curr_inputs.push_back(t);
          visited.insert(t);
        }
      }
  });
  return curr_inputs;
}

Operation HybridOpNode::ReplaceInputs(
    const Operation &self,
    const std::unordered_map<Tensor, Tensor> &rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_node<HybridOpNode>(*this);
  n->body = op::ReplaceTensor(this->body, rmap);
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (body.same_as(n->body) &&
      inputs.same_as(n->inputs)) {
    return self;
  } else {
    return Operation(n);
  }
}

void HybridOpNode::PropBoundToInputs(
    const Operation &self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const Variable*, IntSet> &dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  auto curr_inputs = InputTensors();
  for (Tensor t : curr_inputs) {
    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom &dom = it->second;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      dom.data[i].emplace_back(IntSet::range(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i])));
    }
  }
}

void HybridOpNode::GatherBound(
    const Operation &self,
    const std::unordered_map<Tensor, TensorDom> &tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  for (auto iter_var : axis) {
    CHECK(!out_dom_map->count(iter_var));
    out_dom_map->operator[](iter_var) = iter_var->dom;
  }
}

Stmt HybridOpNode::BuildRealize(
    const Stage &stage,
    const std::unordered_map<IterVar, Range> &realize_map,
    const Stmt &body) const {
  // TODO(@were): Add attribute inject here and remove it from hybrid parser.
  CHECK_EQ(stage->op.get(), this);
  Stmt realize_body = body;
  for (int k = 0; k < num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      bounds.push_back(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i]));
    }
    realize_body = ir::Realize::make(
        t->op, t->value_index, t->dtype,
        bounds, const_true(), realize_body);
  }
  return realize_body;
}

Stmt HybridOpNode::BuildProvide(
    const Stage &stage,
    const std::unordered_map<IterVar, Range> &dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt ret = AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, this->body);
  std::unordered_map<Tensor, Tensor> rmap;
  for (int i = 0; i < this->num_outputs(); ++i) {
    rmap[outputs[i]] = stage->op.output(i);
  }
  auto n = make_node<HybridOpNode>(*this);
  /* This is a story little bit complicated.
   * The following two lines of codes replace output tensors' usage.
   * This is the simplest way I (@were) can come up with to glue
   * hybrid operation node to TVM op system.
   * In hybrid script all the tensors, especially the output tensors,
   * have their own names defined by the users. However, In TVM
   * conventional ops:
   *   1. Output tensors refer the corresponding op node so that the output
   *      tensors have the same names as the operation produces them.
   *   2. Once OpNode is wrapped up by an Operation node, it is finalized.
   *      Later access will be from a const OpNode*.
   * This is a chicken-egg paradox. It is impossible to put the output
   * tensors into the function body without forming the op node. The
   * function body is immutable after the node is formed.
   *
   * Finally, I decided to resolve this issue "lazily". During the
   * pipeline of compilation, this stage is a very preliminary stage.
   * Technically, it is before Phase 0. The actual tensors will be replaced
   * here.
   * Thus, the operation body is slightly different from the Phase 0 body.
   * This is a major difference that HybridOpNode is NOT the same as
   * ExternOpNode.
   * */
  ret = op::ReplaceTensor(ret, rmap);
  ret = op::ReplaceProvideTensor(ret, rmap);

  ret = op::ApplySchedule(stage, dom_map, ret);
  return ret;
}

namespace op {


Stmt ApplyLoopShapes(const Stage &stage,
                 const std::unordered_map<IterVar, Range> &dom_map, Stmt stmt) {
  class LoopSpliter : public IRMutator {
    Expr factor;
    const Variable *parent;
    IterVar inner, outer;

   public:
    bool splitted;
    LoopSpliter(const SplitNode *split,
                const std::unordered_map<IterVar, Range> &dom_map) :
      factor(split->factor), splitted(false) {
      parent = split->parent->var.get();

      auto &inner_ = split->inner;
      CHECK(dom_map.count(inner_));
      auto &inner_dom = dom_map.find(inner_)->second;
      CHECK(is_const_int(inner_dom->min, 0));

      auto &outer_ = split->outer;
      CHECK(dom_map.count(outer_));
      auto &outer_dom = dom_map.find(outer_)->second;
      CHECK(is_const_int(outer_dom->min, 0));

      inner = IterVarNode::make(inner_dom, inner_->var, inner_->iter_type);
      outer = IterVarNode::make(outer_dom, outer_->var, outer_->iter_type);
    }

    Stmt Mutate_(const For *op, const Stmt &stmt) {
      if (op->loop_var.get() == parent) {
        std::unordered_map<const Variable *, Expr> rmap;
        rmap[op->loop_var.get()] = inner + outer * factor;
        Stmt ret = ir::Substitute(op->body, rmap);
        Expr cond = likely(outer * factor < (op->extent - inner));
        ret = IfThenElse::make(cond, ret);
        ret = For::make(inner->var, Expr(0), inner->dom->extent,
                        IterVarTypeToForType(inner->iter_type), op->device_api, ret);
        ret = For::make(outer->var, Expr(0), outer->dom->extent,
                        IterVarTypeToForType(outer->iter_type), op->device_api, ret);
        splitted = true;
        return ret;
      }
      return IRMutator::Mutate_(op, stmt);
    }
  };

  class LoopFuser : public IRMutator {
    const IterVar &parent;
    const Variable *inner;
    const Variable *outer;
    bool under_outer;
    Expr extent;

   public:
    bool fused;
    explicit LoopFuser(const FuseNode *fuse_)
      : parent(fuse_->fused), inner(fuse_->inner->var.get()),
        outer(fuse_->outer->var.get()), under_outer(false),
        extent(0), fused(false) {}

    // TODO(@were): Handle imperfect loops

    Stmt Mutate_(const For *op, const Stmt &stmt) {
      if (op->loop_var.get() == inner) {
        CHECK(under_outer);
        std::unordered_map<const Variable *, Expr> rmap;
        rmap[op->loop_var.get()] = indexmod(parent, op->extent);
        extent = op->extent;
        fused = true;
        return ir::Substitute(op->body, rmap);
      } else if (op->loop_var.get() == outer) {
        under_outer = true;
        Stmt body = IRMutator::Mutate(op->body);
        std::unordered_map<const Variable *, Expr> rmap;
        rmap[op->loop_var.get()] = indexdiv(parent, extent);
        body = ir::Substitute(body, rmap);
        under_outer = false;
        return For::make(parent->var, Expr(0), extent * op->extent,
                         op->for_type, op->device_api, body);
      } else if (under_outer) {
        Stmt body = IRMutator::Mutate(op->body);
        std::unordered_map<const Variable *, Expr> rmap;
        rmap[op->loop_var.get()] = indexmod(indexdiv(parent, extent), op->extent);
        body = ir::Substitute(body, rmap);
        extent = extent * op->extent;
        return body;
      }
      return IRMutator::Mutate(stmt);
    }
  };

  for (auto &rel : stage->relations) {
    if (const SplitNode *split = rel.as<SplitNode>()) {
      LoopSpliter Spliter(split, dom_map);
      stmt = Spliter.Mutate(stmt);
      CHECK(Spliter.splitted);
    } else if (const FuseNode *fuse = rel.as<FuseNode>()) {
      LoopFuser Fuser(fuse);
      stmt = Fuser.Mutate(stmt);
      CHECK(Fuser.fused);
    }
  }

  return stmt;
}

Stmt ApplyLoopAnnotations(const Stage &stage,
                          const std::unordered_map<IterVar, IterVar> &rebased, Stmt stmt) {
  class LoopAnnotator : public IRMutator {
    const Variable *var;
    const IterVarAttr &attr;

   public:
    LoopAnnotator(const Variable *var_, const IterVarAttr &attr_) : var(var_), attr(attr_) {}

    Stmt Mutate_(const For *op, const Stmt &stmt) {
      if (op->loop_var.get() == var) {
        if (attr->bind_thread.defined()) {
          const auto &iter_var = attr->bind_thread;
          if (iter_var->dom.defined()) {
            CHECK(is_const_int(iter_var->dom->min, 0));
            CHECK(Equal(iter_var->dom->extent, op->extent))
              << "Thread extent and loop extent mismatch!\n";
          }
          std::unordered_map<const Variable *, Expr> rmap;
          rmap[op->loop_var.get()] = iter_var;
          Stmt body = ir::Substitute(op->body, rmap);
          return AttrStmt::make(iter_var, "thread_extent", op->extent, body);
        } else {
          return For::make(op->loop_var, op->min, op->extent,
                           IterVarTypeToForType(attr->iter_type), op->device_api, op->body);
        }
      }
      return IRMutator::Mutate_(op, stmt);
    }
  };

  for (auto &iter_var : stage->leaf_iter_vars) {
    bool need_change = false;
    int found = 0;

    const IterVar &actual = rebased.count(iter_var) ? rebased.find(iter_var)->second : iter_var;
    const Variable *var = actual->var.get();
    ForType expected = IterVarTypeToForType(iter_var->iter_type);
    IterVarAttr attr;
    if (stage->iter_var_attrs.count(iter_var)) {
      attr = stage->iter_var_attrs[iter_var];
      expected = IterVarTypeToForType(attr->iter_type);
    }

    PostOrderVisit(stmt, [&found, &var, &attr, &expected, &need_change](const NodeRef &node) {
      if (const For *op = node.as<For>()) {
        if (op->loop_var.get() == var) {
          ++found;
          need_change = expected != op->for_type || (attr.defined() && attr->bind_thread.defined());
        }
      }
    });

    CHECK_EQ(found, 1) << " iter var should be found exactly once!";
    if (need_change) {
      stmt = LoopAnnotator(var, attr).Mutate(stmt);
    }
  }
  return stmt;
}

Stmt ApplyLoopOrder(const Stage &stage,
                    const std::unordered_map<IterVar, Range> &dom_map,
                    const std::unordered_map<IterVar, IterVar> &rebased, Stmt stmt) {
  std::vector<const Variable*> current_order;
  PostOrderVisit(stmt, [&current_order](const NodeRef &node) {
    if (const For *op = node.as<For>())
      current_order.push_back(op->loop_var.get());
  });
  std::reverse(current_order.begin(), current_order.end());
  auto &required_ord = stage->leaf_iter_vars;
  CHECK_EQ(current_order.size(), required_ord.size()) << "Cannot reorder the loops!";
  std::unordered_map<const Variable *, IterVar> reorder;
  bool need_reorder = false;
  for (size_t i = 0; i < current_order.size(); ++i) {
    auto &current = current_order[i];
    const IterVar &iter_var = required_ord[i];
    const IterVar &required = rebased.count(iter_var) ? rebased.find(iter_var)->second : iter_var;
    CHECK(required->dom.defined() || dom_map.count(required)) << required << "\n";
    reorder[current] = required;
    if (current != required->var.get()) {
      need_reorder = true;
    }
  }

  class LoopReorder : public IRMutator {
    const Stage &stage;
    const std::unordered_map<IterVar, Range> &dom_map;
    const std::unordered_map<const Variable *, IterVar> &reorder;

   public:
    LoopReorder(const Stage &stage,
                const std::unordered_map<IterVar, Range> &dom_map,
                const std::unordered_map<const Variable*, IterVar> &reorder)
      : stage(stage), dom_map(dom_map), reorder(reorder) {}

    Stmt Mutate_(const For *op, const Stmt &stmt) {
      // Reorder from in to out
      Stmt body_ = IRMutator::Mutate(op->body);
      CHECK(reorder.count(op->loop_var.get()));
      auto target = reorder.find(op->loop_var.get())->second;
      if (body_.same_as(op->body) && op->loop_var.get() == target->var.get())
        return stmt;
      const Stmt &body = op->body.same_as(body_) ? op->body : body_;
      ForType for_type = IterVarTypeToForType(target->iter_type);
      if (stage->iter_var_attrs.count(target)) {
        for_type = IterVarTypeToForType(stage->iter_var_attrs[target]->iter_type);
      }
      const Range &range = target->dom.defined() ? target->dom : dom_map.find(target)->second;
      return For::make(target->var, range->min, range->extent,
                       for_type, DeviceAPI::None, body);
    }
  };

  if (need_reorder)
    return LoopReorder(stage, dom_map, reorder).Mutate(stmt);

  return stmt;
}

Stmt ApplySchedule(const Stage &stage,
                   const std::unordered_map<IterVar, Range> &dom_map, Stmt stmt) {
  // TODO(@were): Eliminate loop rebase in script parser and move the burden here
  // Gather rebased variables
  std::unordered_map<IterVar, IterVar> rebased;
  for (auto rel : stage->relations) {
    if (const auto* rebase = rel.as<RebaseNode>()) {
      rebased[rebase->rebased] = rebase->parent;
      CHECK(rebase->parent->dom.defined());
      CHECK(dom_map.count(rebase->rebased));
    }
  }
  stmt = ApplyLoopShapes(stage, dom_map, stmt);
  stmt = ApplyLoopOrder(stage, dom_map, rebased, stmt);
  stmt = ApplyLoopAnnotations(stage, rebased, stmt);
  return stmt;
}

std::vector<IterVar> GatherLoopVars(Stmt stmt) {
  // TODO(@were): Write a comprehensive pass to analyze iter var types
  std::vector<IterVar> res_;
  PostOrderVisit(stmt, [&res_](const NodeRef &node) {
    if (const For *op = node.as<For>()) {
      Var loop_var(op->loop_var);
      Range dom = Range::make_by_min_extent(op->min, op->extent);
      res_.push_back(IterVarNode::make(dom, loop_var, ForTypeToIterVarType(op->for_type)));
    }
  });
  std::reverse(res_.begin(), res_.end());
  return res_;
}

// replacer to replace tensors' usage in Provide
class ProviderReplacer : public ir::IRMutator {
 public:
  explicit ProviderReplacer(const std::unordered_map<Tensor, Tensor> &vmap)
      : vmap_(vmap) {}

  Stmt Mutate_(const ir::Provide* op, const Stmt &s) {
    Tensor t = Operation(op->func.node_).output(op->value_index);
    auto it = vmap_.find(t);
    if (it != vmap_.end()) {
      Stmt ret = ir::Provide::make(
        it->second->op, it->second->value_index, op->value, op->args);
      found = true;
      return IRMutator::Mutate_(ret.as<ir::Provide>(), ret);
    }
    return IRMutator::Mutate_(op, s);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor> &vmap_;
};

Stmt ReplaceProvideTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor> &replace) {
  ProviderReplacer repl(replace);
  Stmt ret = repl.Mutate(stmt);
  return repl.found ? ret : stmt;
}
}  // namespace op
}  // namespace tvm
