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
 * \file src/relay/ir/indexed_graph.cc
 * \brief Utilties for Creating Indexed Graphs.
 */
#include "indexed_graph.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_pattern_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

// IndexedGraph

IndexedGraph<Expr> CreateIndexedGraph(const Expr& expr) {
  VLOG_CONTEXT << "CreateIndexedGraph";
  VLOG(1) << "creating for:" << std::endl << PrettyPrint(expr);
  using NodePtr = std::shared_ptr<IndexedGraph<Expr>::Node>;
  /*! \brief Creator Creates an IndexedGraph and determintes Topological order */
  class Creator : public MixedModeVisitor {
   public:
    IndexedGraph<Expr> CreateGraph(const Expr& expr) {
      VisitExpr(expr);
      graph_[expr]->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    void VisitLeaf(const Expr& expr) override {
      MixedModeVisitor::VisitLeaf(expr);
      auto node = std::make_shared<IndexedGraph<Expr>::Node>(expr, index_++);
      graph_.Set(expr, node);
      graph_.topological_order_.push_back(node);
    }
    IndexedGraph<Expr> graph_;
    size_t index_ = 0;
  };
  /*! \brief Annotator takes an IndexedGraph, fills it's forward outputs, and does dominator tree
   * analysis.
   *
   *  Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public ExprFunctor<void(const Expr&, NodePtr)> {
   public:
    Annotator(const IndexedGraph<Expr>& graph) : graph_(graph) {}
    IndexedGraph<Expr> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (const auto& node : graph_.topological_order_) {
        ExprFunctor::VisitExpr(node->ref_, nullptr);
      }
      // do the dominator analysis
      graph_.PostDom();
      return std::move(graph_);
    }

    /*! Default visitation pushes the parent to the child's outputs and the child to the parent's
     * inputs*/
    void VisitExpr(const Expr& expr, NodePtr parent) override {
      auto current = graph_[expr];
      if (parent) {
        current->outputs_.push_back(parent.get());
        parent->inputs_.push_back(current.get());
      }
    }

   protected:
    IndexedGraph<Expr> graph_;
    void VisitExpr_(const VarNode* op, NodePtr parent) override {
      if (op->type_annotation.defined()) {
        this->VisitType(op->type_annotation);
      }
    }

    void VisitExpr_(const GlobalVarNode* op, NodePtr parent) override {}

    void VisitExpr_(const ConstantNode* op, NodePtr parent) override {}

    void VisitExpr_(const TupleNode* op, NodePtr parent) override {
      for (auto field : op->fields) {
        this->VisitExpr(field, graph_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const FunctionNode* op, NodePtr parent) override {
      for (auto param : op->params) {
        this->VisitExpr(param, graph_[GetRef<Expr>(op)]);
      }

      this->VisitExpr(op->body, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const CallNode* op, NodePtr parent) override {
      this->VisitExpr(op->op, graph_[GetRef<Expr>(op)]);

      for (auto ty_arg : op->type_args) {
        this->VisitType(ty_arg);
      }

      for (auto arg : op->args) {
        this->VisitExpr(arg, graph_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const LetNode* op, NodePtr parent) override {
      this->VisitExpr(op->value, graph_[GetRef<Expr>(op)]);
      this->VisitExpr(op->var, graph_[GetRef<Expr>(op)]);
      this->VisitExpr(op->body, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const IfNode* op, NodePtr parent) override {
      this->VisitExpr(op->cond, graph_[GetRef<Expr>(op)]);
      this->VisitExpr(op->true_branch, graph_[GetRef<Expr>(op)]);
      this->VisitExpr(op->false_branch, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const OpNode* op, NodePtr parent) override { return; }

    void VisitExpr_(const TupleGetItemNode* op, NodePtr parent) override {
      this->VisitExpr(op->tuple, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefCreateNode* op, NodePtr parent) override {
      this->VisitExpr(op->value, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefReadNode* op, NodePtr parent) override {
      this->VisitExpr(op->ref, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefWriteNode* op, NodePtr parent) override {
      this->VisitExpr(op->ref, graph_[GetRef<Expr>(op)]);
      this->VisitExpr(op->value, graph_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const ConstructorNode* op, NodePtr parent) override {
      for (const Type& t : op->inputs) {
        this->VisitType(t);
      }
      this->VisitType(op->belong_to);
    }

    void VisitExpr_(const MatchNode* op, NodePtr parent) override {
      this->VisitExpr(op->data, graph_[GetRef<Expr>(op)]);
      for (const Clause& c : op->clauses) {
        this->VisitClause(c, graph_[GetRef<Expr>(op)]);
      }
    }

    void VisitClause(const Clause& op, NodePtr parent) {
      this->VisitPattern(op->lhs);
      this->VisitExpr(op->rhs, parent);
    }

    void VisitPattern(const Pattern& p) { return; }

    void VisitType(const Type& t) { return; }
  };
  return Annotator(Creator().CreateGraph(expr)).Annotate();
}

IndexedGraph<DFPattern> CreateIndexedGraph(const DFPattern& pattern) {
  using NodePtr = std::shared_ptr<IndexedGraph<DFPattern>::Node>;
  /*! \brief Creator Creates an IndexedGraph and determintes Toplogical order */
  class Creator : public DFPatternVisitor {
   public:
    IndexedGraph<DFPattern> CreateGraph(const DFPattern& pattern) {
      VisitDFPattern(pattern);
      graph_[pattern]->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    void VisitDFPattern(const DFPattern& pattern) override {
      if (this->visited_.count(pattern.get()) == 0) {
        DFPatternVisitor::VisitDFPattern(pattern);
        auto node = std::make_shared<IndexedGraph<DFPattern>::Node>(pattern, index_++);
        graph_.Set(pattern, node);
        graph_.topological_order_.push_back(node);
      }
    }
    IndexedGraph<DFPattern> graph_;
    size_t index_ = 0;
  };
  /*! \brief Annotator takes an IndexedGraph, fills it's forward outputs, and does domiantor tree
   * analysis.
   *
   *  Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public DFPatternFunctor<void(const DFPattern&, NodePtr)> {
   public:
    Annotator(const IndexedGraph<DFPattern>& graph) : graph_(graph) {}
    IndexedGraph<DFPattern> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (const auto& node : graph_.topological_order_) {
        DFPatternFunctor::VisitDFPattern(node->ref_, nullptr);
      }
      graph_.PostDom();
      // do the dominator analysis
      return std::move(graph_);
    }

    /*! Default visitation pushes the parent to the child's outputs */
    void VisitDFPattern(const DFPattern& pattern, NodePtr parent) override {
      auto current = graph_[pattern];
      if (parent) {
        current->outputs_.push_back(parent.get());
        parent->inputs_.push_back(current.get());
      }
    }

   protected:
    IndexedGraph<DFPattern> graph_;
    void VisitDFPattern_(const AltPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->left, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->right, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const AttrPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->pattern, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const CallPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->op, graph_[GetRef<DFPattern>(op)]);
      if (op->args.defined()) {
        for (auto arg : op->args) {
          VisitDFPattern(arg, graph_[GetRef<DFPattern>(op)]);
        }
      }
    }

    void VisitDFPattern_(const ConstantPatternNode* op, NodePtr parent) override {}

    void VisitDFPattern_(const DataTypePatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->pattern, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const DominatorPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->parent, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->path, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->child, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const ExprPatternNode* op, NodePtr parent) override {}

    void VisitDFPattern_(const FunctionPatternNode* op, NodePtr parent) override {
      if (op->params.defined()) {
        for (auto param : op->params) {
          VisitDFPattern(param, graph_[GetRef<DFPattern>(op)]);
        }
      }
      VisitDFPattern(op->body, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const ShapePatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->pattern, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const TupleGetItemPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->tuple, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const TuplePatternNode* op, NodePtr parent) override {
      if (op->fields.defined()) {
        for (auto field : op->fields) {
          VisitDFPattern(field, graph_[GetRef<DFPattern>(op)]);
        }
      }
    }

    void VisitDFPattern_(const IfPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->cond, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->true_branch, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->false_branch, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const LetPatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->var, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->value, graph_[GetRef<DFPattern>(op)]);
      VisitDFPattern(op->body, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const TypePatternNode* op, NodePtr parent) override {
      VisitDFPattern(op->pattern, graph_[GetRef<DFPattern>(op)]);
    }

    void VisitDFPattern_(const VarPatternNode* op, NodePtr parent) override {}

    void VisitDFPattern_(const WildcardPatternNode* op, NodePtr parent) override {}
  };
  return Annotator(Creator().CreateGraph(pattern)).Annotate();
}

}  // namespace relay
}  // namespace tvm
