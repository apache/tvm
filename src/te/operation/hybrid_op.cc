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
#include "hybrid_op.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "op_utils.h"

namespace tvm {
namespace te {
using namespace tir;
// HybridOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<HybridOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const HybridOpNode*>(node.get());
      p->stream << "hybrid(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(HybridOpNode);

int HybridOpNode::num_outputs() const { return static_cast<int>(outputs.size()); }

Array<IterVar> HybridOpNode::root_iter_vars() const { return this->axis; }

DataType HybridOpNode::output_dtype(size_t i) const { return outputs[i]->dtype; }

Array<PrimExpr> HybridOpNode::output_shape(size_t i) const { return outputs[i]->shape; }

HybridOp::HybridOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
                   Array<Tensor> inputs, Array<Tensor> outputs, Stmt body) {
  if (!attrs.defined()) {
    attrs = Map<String, ObjectRef>();
  }
  auto n = make_object<HybridOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->axis = te::GatherLoopVars(body);
  n->body = std::move(body);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("te.HybridOp")
    .set_body_typed([](std::string name, std::string tag, Map<String, ObjectRef> attrs,
                       Array<Tensor> inputs, Array<Tensor> outputs,
                       Stmt body) { return HybridOp(name, tag, attrs, inputs, outputs, body); });

Array<Tensor> HybridOpNode::InputTensors() const {
  // Because input tensors could be potentially inlined into hybrid scripts,
  // we need to check if all input tensors are used in the body.
  std::unordered_set<Tensor> orig_inputs;
  for (auto t : inputs) {
    orig_inputs.insert(t);
  }
  std::unordered_set<Tensor> visited;
  Array<Tensor> curr_inputs;
  tir::PostOrderVisit(body, [&curr_inputs, &orig_inputs, &visited](const ObjectRef& n) {
    if (auto* pload = n.as<tir::ProducerLoadNode>()) {
      Tensor t = Downcast<Tensor>(pload->producer);
      if (orig_inputs.count(t) && !visited.count(t)) {
        curr_inputs.push_back(t);
        visited.insert(t);
      }
    }
  });
  return curr_inputs;
}

Operation HybridOpNode::ReplaceInputs(const Operation& self,
                                      const std::unordered_map<Tensor, Tensor>& rmap) const {
  ICHECK_EQ(self.operator->(), this);
  auto n = make_object<HybridOpNode>(*this);
  n->body = te::ReplaceTensor(this->body, rmap);
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (body.same_as(n->body) && inputs.same_as(n->inputs)) {
    return self;
  } else {
    return Operation(n);
  }
}

void HybridOpNode::PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                                     const std::unordered_map<const VarNode*, IntSet>& dom_map,
                                     std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  auto curr_inputs = InputTensors();
  for (Tensor t : curr_inputs) {
    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      dom.data[i].emplace_back(
          IntSet::FromRange(Range::FromMinExtent(make_const(t->shape[i].dtype(), 0), t->shape[i])));
    }
  }
}

void HybridOpNode::GatherBound(const Operation& self,
                               const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                               std::unordered_map<IterVar, Range>* out_dom_map) const {
  for (auto iter_var : axis) {
    ICHECK(!out_dom_map->count(iter_var));
    out_dom_map->operator[](iter_var) = iter_var->dom;
  }
}

Stmt HybridOpNode::BuildRealize(const Stage& stage,
                                const std::unordered_map<IterVar, Range>& realize_map,
                                const Stmt& body, String storage_scope) const {
  // TODO(@were): Add attribute inject here and remove it from hybrid parser.
  ICHECK_EQ(stage->op.get(), this);
  Stmt realize_body = body;
  for (int k = 0; k < num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      bounds.push_back(Range::FromMinExtent(make_const(t->shape[i].dtype(), 0), t->shape[i]));
    }
    realize_body = tir::ProducerRealize(t, bounds, const_true(), realize_body, storage_scope);
  }
  return realize_body;
}

Stmt HybridOpNode::BuildProvide(const Stage& stage,
                                const std::unordered_map<IterVar, Range>& dom_map,
                                bool debug_keep_trivial_loop) const {
  ICHECK_EQ(stage->op.operator->(), this);
  Stmt ret = AttrStmt(make_zero(DataType::Int(32)), tir::attr::extern_scope, 0, this->body);
  std::unordered_map<Tensor, Tensor> rmap;
  for (int i = 0; i < this->num_outputs(); ++i) {
    rmap[outputs[i]] = stage->op.output(i);
  }
  auto n = make_object<HybridOpNode>(*this);
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
  ret = te::ReplaceTensor(ret, rmap);
  ret = te::ReplaceProvideTensor(ret, rmap);

  ret = te::ApplySchedule(stage, dom_map, ret);
  return ret;
}

Stmt ApplyLoopShapes(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     Stmt stmt) {
  class LoopSpliter : public StmtExprMutator {
    PrimExpr factor;
    const VarNode* parent;
    IterVar inner, outer;

   public:
    bool splitted;
    LoopSpliter(const SplitNode* split, const std::unordered_map<IterVar, Range>& dom_map)
        : factor(split->factor), splitted(false) {
      parent = split->parent->var.get();

      auto& inner_ = split->inner;
      ICHECK(dom_map.count(inner_));
      auto& inner_dom = dom_map.find(inner_)->second;
      ICHECK(is_const_int(inner_dom->min, 0));

      auto& outer_ = split->outer;
      ICHECK(dom_map.count(outer_));
      auto& outer_dom = dom_map.find(outer_)->second;
      ICHECK(is_const_int(outer_dom->min, 0));

      inner = IterVar(inner_dom, inner_->var, inner_->iter_type);
      outer = IterVar(outer_dom, outer_->var, outer_->iter_type);
    }

    Stmt VisitStmt_(const ForNode* op) final {
      if (op->loop_var.get() == parent) {
        Stmt ret = tir::Substitute(op->body, {{op->loop_var, inner + outer * factor}});
        PrimExpr cond = likely(outer * factor < (op->extent - inner));
        ret = IfThenElse(cond, ret);
        ret = For(inner->var, PrimExpr(0), inner->dom->extent,
                  IterVarTypeToForKind(inner->iter_type), ret);
        ret = For(outer->var, PrimExpr(0), outer->dom->extent,
                  IterVarTypeToForKind(outer->iter_type), ret);
        splitted = true;
        return ret;
      }
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  class LoopFuser : public StmtExprMutator {
    const IterVar& parent;
    const VarNode* inner;
    const VarNode* outer;
    bool under_outer;
    PrimExpr extent;

   public:
    bool fused;
    explicit LoopFuser(const FuseNode* fuse_)
        : parent(fuse_->fused),
          inner(fuse_->inner->var.get()),
          outer(fuse_->outer->var.get()),
          under_outer(false),
          extent(0),
          fused(false) {}

    // TODO(@were): Handle imperfect loops
    Stmt VisitStmt_(const ForNode* op) final {
      if (op->loop_var.get() == inner) {
        ICHECK(under_outer);
        std::unordered_map<const VarNode*, PrimExpr> rmap;
        rmap[op->loop_var.get()] = indexmod(parent, op->extent);
        extent = op->extent;
        fused = true;
        return tir::Substitute(op->body, rmap);
      } else if (op->loop_var.get() == outer) {
        under_outer = true;
        Stmt body = this->VisitStmt(op->body);
        std::unordered_map<const VarNode*, PrimExpr> rmap;
        rmap[op->loop_var.get()] = indexdiv(parent, extent);
        body = tir::Substitute(body, rmap);
        under_outer = false;
        return For(parent->var, PrimExpr(0), extent * op->extent, op->kind, body,
                   op->thread_binding, op->annotations);
      } else if (under_outer) {
        Stmt body = this->VisitStmt(op->body);
        std::unordered_map<const VarNode*, PrimExpr> rmap;
        rmap[op->loop_var.get()] = indexmod(indexdiv(parent, extent), op->extent);
        body = tir::Substitute(body, rmap);
        extent = extent * op->extent;
        return body;
      }
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  for (auto& rel : stage->relations) {
    if (const SplitNode* split = rel.as<SplitNode>()) {
      LoopSpliter Spliter(split, dom_map);
      stmt = Spliter(stmt);
      ICHECK(Spliter.splitted);
    } else if (const FuseNode* fuse = rel.as<FuseNode>()) {
      LoopFuser Fuser(fuse);
      stmt = Fuser(stmt);
      ICHECK(Fuser.fused);
    }
  }

  return stmt;
}

Stmt ApplyLoopAnnotations(const Stage& stage, const std::unordered_map<IterVar, IterVar>& rebased,
                          Stmt stmt) {
  class LoopAnnotator : public StmtMutator {
    const VarNode* var;
    const IterVarAttr& attr;

   public:
    LoopAnnotator(const VarNode* var_, const IterVarAttr& attr_) : var(var_), attr(attr_) {}

    Stmt VisitStmt_(const ForNode* op) final {
      tir::ExprDeepEqual expr_equal;

      if (op->loop_var.get() == var) {
        if (attr->bind_thread.defined()) {
          const auto& iter_var = attr->bind_thread;
          if (iter_var->dom.defined()) {
            ICHECK(is_const_int(iter_var->dom->min, 0));
            ICHECK(expr_equal(iter_var->dom->extent, op->extent))
                << "Thread extent and loop extent mismatch!\n";
          }
          std::unordered_map<const VarNode*, PrimExpr> rmap;
          rmap[op->loop_var.get()] = iter_var;
          Stmt body = tir::Substitute(op->body, rmap);
          return AttrStmt(iter_var, "thread_extent", op->extent, body);
        } else {
          return For(op->loop_var, op->min, op->extent, IterVarTypeToForKind(attr->iter_type),
                     op->body, op->thread_binding, op->annotations);
        }
      }
      return StmtMutator::VisitStmt_(op);
    }
  };

  for (auto& iter_var : stage->leaf_iter_vars) {
    bool need_change = false;
    int found = 0;

    const IterVar& actual = rebased.count(iter_var) ? rebased.find(iter_var)->second : iter_var;
    const VarNode* var = actual->var.get();
    ForKind expected = IterVarTypeToForKind(iter_var->iter_type);
    IterVarAttr attr;
    if (stage->iter_var_attrs.count(iter_var)) {
      attr = stage->iter_var_attrs[iter_var];
      expected = IterVarTypeToForKind(attr->iter_type);
    }

    PostOrderVisit(stmt, [&found, &var, &attr, &expected, &need_change](const ObjectRef& node) {
      if (const ForNode* op = node.as<ForNode>()) {
        if (op->loop_var.get() == var) {
          ++found;
          need_change = expected != op->kind || (attr.defined() && attr->bind_thread.defined());
        }
      }
    });

    ICHECK_EQ(found, 1) << " iter var should be found exactly once!";
    if (need_change) {
      stmt = LoopAnnotator(var, attr)(std::move(stmt));
    }
  }
  return stmt;
}

Stmt ApplyLoopOrder(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                    const std::unordered_map<IterVar, IterVar>& rebased, Stmt stmt) {
  std::vector<const VarNode*> current_order;
  PostOrderVisit(stmt, [&current_order](const ObjectRef& node) {
    if (const ForNode* op = node.as<ForNode>()) current_order.push_back(op->loop_var.get());
  });
  std::reverse(current_order.begin(), current_order.end());
  auto& required_ord = stage->leaf_iter_vars;
  ICHECK_EQ(current_order.size(), required_ord.size()) << "Cannot reorder the loops!";
  std::unordered_map<const VarNode*, IterVar> reorder;
  bool need_reorder = false;
  for (size_t i = 0; i < current_order.size(); ++i) {
    auto& current = current_order[i];
    const IterVar& iter_var = required_ord[i];
    const IterVar& required = rebased.count(iter_var) ? rebased.find(iter_var)->second : iter_var;
    ICHECK(required->dom.defined() || dom_map.count(required)) << required << "\n";
    reorder[current] = required;
    if (current != required->var.get()) {
      need_reorder = true;
    }
  }

  class LoopReorder : public StmtMutator {
    const Stage& stage;
    const std::unordered_map<IterVar, Range>& dom_map;
    const std::unordered_map<const VarNode*, IterVar>& reorder;

   public:
    LoopReorder(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                const std::unordered_map<const VarNode*, IterVar>& reorder)
        : stage(stage), dom_map(dom_map), reorder(reorder) {}

    Stmt VisitStmt_(const ForNode* op) final {
      // Reorder from in to out
      Stmt body_ = this->VisitStmt(op->body);
      ICHECK(reorder.count(op->loop_var.get()));
      auto target = reorder.find(op->loop_var.get())->second;
      if (body_.same_as(op->body) && op->loop_var.get() == target->var.get())
        return GetRef<Stmt>(op);
      const Stmt& body = op->body.same_as(body_) ? op->body : body_;
      ForKind kind = IterVarTypeToForKind(target->iter_type);
      if (stage->iter_var_attrs.count(target)) {
        kind = IterVarTypeToForKind(stage->iter_var_attrs[target]->iter_type);
      }
      const Range& range = target->dom.defined() ? target->dom : dom_map.find(target)->second;
      return For(target->var, range->min, range->extent, kind, body, op->thread_binding,
                 op->annotations);
    }
  };

  if (need_reorder) return LoopReorder(stage, dom_map, reorder)(stmt);

  return stmt;
}

Stmt ApplySchedule(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                   Stmt stmt) {
  // TODO(@were): Eliminate loop rebase in script parser and move the burden here
  // Gather rebased variables
  std::unordered_map<IterVar, IterVar> rebased;
  for (auto rel : stage->relations) {
    if (const auto* rebase = rel.as<RebaseNode>()) {
      rebased[rebase->rebased] = rebase->parent;
      ICHECK(rebase->parent->dom.defined());
      ICHECK(dom_map.count(rebase->rebased));
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
  PostOrderVisit(stmt, [&res_](const ObjectRef& node) {
    if (const ForNode* op = node.as<ForNode>()) {
      Var loop_var(op->loop_var);
      Range dom = Range::FromMinExtent(op->min, cast(loop_var.dtype(), op->extent));
      res_.push_back(IterVar(dom, loop_var, ForKindToIterVarType(op->kind)));
    }
  });
  std::reverse(res_.begin(), res_.end());
  return res_;
}

// replacer to replace tensors' usage in Provide
class ProviderReplacer : public tir::StmtMutator {
 public:
  explicit ProviderReplacer(const std::unordered_map<Tensor, Tensor>& vmap) : vmap_(vmap) {}

  Stmt VisitStmt_(const tir::ProducerStoreNode* op) final {
    Tensor t = Downcast<Tensor>(op->producer);
    auto it = vmap_.find(t);
    if (it != vmap_.end()) {
      Stmt ret = tir::ProducerStore(it->second, op->value, op->indices);
      found = true;
      return this->VisitStmt(ret);
    }
    return StmtMutator::VisitStmt_(op);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceProvideTensor(Stmt stmt, const std::unordered_map<Tensor, Tensor>& replace) {
  ProviderReplacer repl(replace);
  Stmt ret = repl(stmt);
  return repl.found ? ret : stmt;
}
}  // namespace te
}  // namespace tvm
