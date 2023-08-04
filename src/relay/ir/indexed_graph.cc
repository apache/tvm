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
 * \brief A graph representation of the dataflow in a Relay expression or Relay (dataflow)
 * pattern.
 */
#include "indexed_graph.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_pattern_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

#include <string>

namespace tvm {
namespace relay {

std::string RefToSummary(const Expr& expr) {
  class Visitor : public ExprFunctor<std::string(const Expr&)> {
    std::string VisitExpr_(const VarNode* op) final { return "%" + op->name_hint(); }
    std::string VisitExpr_(const GlobalVarNode* op) final { return "@" + op->name_hint; }
    std::string VisitExpr_(const ConstantNode* op) final { return "const"; }
    std::string VisitExpr_(const TupleNode* op) final {
      return "tuple(" + std::to_string(op->fields.size()) + ")";
    }
    std::string VisitExpr_(const FunctionNode* op) final { return "fn"; }
    std::string VisitExpr_(const CallNode* op) final {
      return VisitExpr(op->op) + "(" + std::to_string(op->args.size()) + ")";
    }
    std::string VisitExpr_(const LetNode* op) final { return "let"; }
    std::string VisitExpr_(const IfNode* op) final { return "if"; }
    std::string VisitExpr_(const OpNode* op) final { return op->name; }
    std::string VisitExpr_(const TupleGetItemNode* op) final {
      return "." + std::to_string(op->index);
    }
    std::string VisitExpr_(const RefCreateNode* op) final { return "ref_create"; }
    std::string VisitExpr_(const RefReadNode* op) final { return "ref_read"; }
    std::string VisitExpr_(const RefWriteNode* op) final { return "ref_write"; }
    std::string VisitExpr_(const ConstructorNode* op) final { return "ctor"; }
    std::string VisitExpr_(const MatchNode* op) final { return "match"; }
  };
  return Visitor().VisitExpr(expr);
}

std::string RefToSummary(const DFPattern& pattern) {
  // TODO(mbs): Implement as debugging requires.
  return "";
}

std::unique_ptr<IndexedGraph<Expr>> CreateIndexedGraph(const Expr& expr) {
  /*!
   * \brief Adds indexed graph nodes in post-dfs order, and discovers which let-bound vars are to
   * recursive functions.
   */
  class Creator : public MixedModeVisitor {
   public:
    std::pair<std::unique_ptr<IndexedGraph<Expr>>,
              std::unique_ptr<std::unordered_set<const CallNode*>>>
    CreateGraph(const Expr& expr) {
      VisitExpr(expr);
      // Last visited node is implicitly used 'externally'.
      graph_->item_to_node(expr)->is_external_ = true;
      return {std::move(graph_), std::move(rec_calls_)};
    }

   protected:
    using MixedModeVisitor::VisitExpr_;

    // By the default the MixedModeVisitor will place
    //  - callee and arguments before a call
    //  - tuple fields before a tuple
    //  - tuple before a tuple projection
    void VisitLeaf(const Expr& expr) override {
      if (const auto* var_node = expr.as<VarNode>()) {
        if (var_node == current_let_bound_var_) {
          // Don't visit occurrences of let-rec bound vars in the recursive function body.
          // Instead, wait for them to be visited at call sites outside of the function.
          VLOG(1) << "Ignore let-rec var '" << var_node->name_hint() << "'";
          return;
        }
      }

      MixedModeVisitor::VisitLeaf(expr);
      graph_->AddNode(expr);

      if (const auto* call_node = expr.as<CallNode>()) {
        if (const auto* var_node = call_node->op.as<VarNode>()) {
          if (var_node == current_let_bound_var_) {
            // Remember this is a recursive call to the let-rec bound function.
            // The Annotator functor below will not record any dependency from the let-rec bound
            // var to the expression so that the indexed graph is always a DAG.
            VLOG(1) << "Remembering recursive call to '" << var_node->name_hint() << "'";
            rec_calls_->emplace(call_node);
          }
        }
      }
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto pre_visit = [&](const LetNode* op) {
        // Let-bound values come before their let-bound variable.
        const VarNode* prev_let_bound_var = current_let_bound_var_;
        current_let_bound_var_ = op->var.get();
        VisitExpr(op->value);
        current_let_bound_var_ = prev_let_bound_var;
        VisitExpr(op->var);
      };
      auto post_visit = [&](const LetNode* op) {
        VisitExpr(op->body);
        if (let_node != op) {
          // Replicate VisitLeaf, which we are effectively bypassing.
          visit_counter_[op]++;
          graph_->AddNode(GetRef<Expr>(op));
        }
      };
      ExpandANormalForm(let_node, pre_visit, post_visit);
    }

    class PatternCreator : public PatternVisitor {
     public:
      explicit PatternCreator(Creator* creator) : creator_(creator) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        creator_->VisitLeaf(pattern_var_node->var);
      }

      Creator* creator_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      // Matched data comes before match-bound vars then match rhs, in match order.
      VisitExpr(match_node->data);
      for (const Clause& c : match_node->clauses) {
        PatternCreator pattern_creator(this);
        pattern_creator.VisitPattern(c->lhs);
        VisitExpr(c->rhs);
      }
    }

    /*! \brief Graph we are accumulated nodes into. */
    std::unique_ptr<IndexedGraph<Expr>> graph_ = std::make_unique<IndexedGraph<Expr>>();
    /*! \brief Variable the currently visited expression is to be let-bound to, if any. */
    const VarNode* current_let_bound_var_ = nullptr;
    /*! \brief Accumulated calls to recursive functions. */
    std::unique_ptr<std::unordered_set<const CallNode*>> rec_calls_ =
        std::make_unique<std::unordered_set<const CallNode*>>();
  };

  /*!
   * \brief Fills in the inputs and outputs for all nodes, then does dominator analysis.
   *
   * Thought we use the ExprFunctor to visit nodes, we never recurse and instead just inspect
   * each sub-expression's immediate sub-sub-expressions to accumulate inputs and outputs.
   */
  class Annotator : public ExprFunctor<void(const Expr&)> {
   public:
    explicit Annotator(std::pair<std::unique_ptr<IndexedGraph<Expr>>,
                                 std::unique_ptr<std::unordered_set<const CallNode*>>>
                           args)
        : graph_(std::move(args.first)), rec_calls_(std::move(args.second)) {}

    std::unique_ptr<IndexedGraph<Expr>> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (PostDfsIndex index = 0; index < graph_->size(); ++index) {
        VisitExpr(graph_->index_to_node(index)->ref());
      }
      // do the dominator analysis
      graph_->PostDom();
      return std::move(graph_);
    }

    /*!
     * \brief Add \p parent as a possible output of the node corresponding to \p expr.
     */
    void AddOutput(const Expr& expr, IndexedGraph<Expr>::Node* parent) {
      auto current = graph_->item_to_node(expr);
      current->outputs_.push_back(parent);
      parent->inputs_.push_back(current);
    }

   protected:
    void VisitExpr_(const VarNode* var_node) override {}

    void VisitExpr_(const GlobalVarNode* global_var_node) override {}

    void VisitExpr_(const ConstantNode* constant_node) override {}

    void VisitExpr_(const TupleNode* tuple_node) override {
      auto node = graph_->item_to_node(GetRef<Tuple>(tuple_node));
      for (auto field : tuple_node->fields) {
        AddOutput(field, node);
      }
    }

    void VisitExpr_(const FunctionNode* function_node) override {
      auto node = graph_->item_to_node(GetRef<Function>(function_node));
      // Nothing to do for parameters -- each use of a parameter will contribute to its outputs.
      AddOutput(function_node->body, node);
    }

    void VisitExpr_(const CallNode* call_node) override {
      auto node = graph_->item_to_node(GetRef<Call>(call_node));
      if (rec_calls_->count(call_node)) {
        // We want the indexed graph to be a DAG, so don't consider a call to a let-rec bound
        // function from inside the function to depend on the let-rec bound var.
        VLOG(1) << "Ignoring op in call " << RefToSummary(GetRef<Call>(call_node));
      } else {
        AddOutput(call_node->op, node);
      }
      for (auto arg : call_node->args) {
        AddOutput(arg, node);
      }
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto node = graph_->item_to_node(GetRef<Let>(let_node));
      auto let_var_node = graph_->item_to_node(let_node->var);
      AddOutput(let_node->value, let_var_node);
      // Nothing to do for the let-bound variable -- each use of that variable in the let-body
      // will contribute to its outputs.
      AddOutput(let_node->body, node);
    }

    void VisitExpr_(const IfNode* if_node) override {
      auto node = graph_->item_to_node(GetRef<If>(if_node));
      AddOutput(if_node->cond, node);
      AddOutput(if_node->true_branch, node);
      AddOutput(if_node->false_branch, node);
    }

    void VisitExpr_(const OpNode* op_node) override {}

    void VisitExpr_(const TupleGetItemNode* tuple_get_item_node) override {
      auto node = graph_->item_to_node(GetRef<TupleGetItem>(tuple_get_item_node));
      AddOutput(tuple_get_item_node->tuple, node);
    }

    void VisitExpr_(const RefCreateNode* ref_create_node) override {
      auto node = graph_->item_to_node(GetRef<RefCreate>(ref_create_node));
      AddOutput(ref_create_node->value, node);
    }

    void VisitExpr_(const RefReadNode* ref_read_node) override {
      auto node = graph_->item_to_node(GetRef<RefRead>(ref_read_node));
      AddOutput(ref_read_node->ref, node);
    }

    void VisitExpr_(const RefWriteNode* ref_write_node) override {
      auto node = graph_->item_to_node(GetRef<RefWrite>(ref_write_node));
      AddOutput(ref_write_node->ref, node);
      AddOutput(ref_write_node->value, node);
    }

    void VisitExpr_(const ConstructorNode* constructor_node) override {}

    class PatternAnnotator : public PatternVisitor {
     public:
      PatternAnnotator(Annotator* annotator, const ExprNode* adt_node)
          : annotator_(annotator), adt_node_(adt_node) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        auto node = annotator_->graph_->item_to_node(pattern_var_node->var);
        annotator_->AddOutput(GetRef<Expr>(adt_node_), node);
      }

      Annotator* annotator_;
      const ExprNode* adt_node_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      // Data flows from the match data to pattern vars into match arms and out into overall
      // match.
      auto node = graph_->item_to_node(GetRef<Match>(match_node));
      for (const Clause& c : match_node->clauses) {
        PatternAnnotator pattern_annotator(this, match_node->data.get());
        pattern_annotator.VisitPattern(c->lhs);
        AddOutput(c->rhs, node);
      }
    }

    std::unique_ptr<IndexedGraph<Expr>> graph_;
    /*! \brief Accumulated calls to recursive functions. */
    std::unique_ptr<std::unordered_set<const CallNode*>> rec_calls_;
  };

  /*! \brief Fills in the basic blocks for all nodes. */
  class Blocker : public MixedModeVisitor {
   public:
    explicit Blocker(std::unique_ptr<IndexedGraph<Expr>> graph) : graph_(std::move(graph)) {}

    std::unique_ptr<IndexedGraph<Expr>> Scope(const Expr& expr) {
      VisitExpr(expr);
      return std::move(graph_);
    }

   private:
    using MixedModeVisitor::VisitExpr_;

    void VisitLeaf(const Expr& expr) override {
      MixedModeVisitor::VisitLeaf(expr);
      SetScope(expr);
    }

    void VisitExpr_(const FunctionNode* function_node) override {
      auto node = graph_->item_to_node(GetRef<Function>(function_node));
      basic_block_stack_.push_back(node);
      ExprVisitor::VisitExpr_(function_node);
      basic_block_stack_.pop_back();
    }

    void VisitExpr_(const IfNode* if_node) override {
      VisitExpr(if_node->cond);
      auto node = graph_->item_to_node(GetRef<If>(if_node));
      basic_block_stack_.push_back(node);
      VisitExpr(if_node->true_branch);
      VisitExpr(if_node->false_branch);
      basic_block_stack_.pop_back();
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto pre_visit = [&](const LetNode* op) {
        VisitExpr(op->value);
        VisitExpr(op->var);
      };
      auto post_visit = [&](const LetNode* op) {
        VisitExpr(op->body);
        if (let_node != op) {
          visit_counter_[op]++;
          SetScope(GetRef<Let>(op));
        }
      };
      ExpandANormalForm(let_node, pre_visit, post_visit);
    }

    class PatternBlocker : public PatternVisitor {
     public:
      explicit PatternBlocker(Blocker* scoper) : scoper_(scoper) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        scoper_->SetScope(pattern_var_node->var);
      }

      Blocker* scoper_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      VisitExpr(match_node->data);
      auto node = graph_->item_to_node(GetRef<Match>(match_node));
      basic_block_stack_.push_back(node);
      for (const Clause& c : match_node->clauses) {
        PatternBlocker pattern_scoper(this);
        pattern_scoper.VisitPattern(c->lhs);
        VisitExpr(c->rhs);
      }
      basic_block_stack_.pop_back();
    }

    void SetScope(const Expr& expr) {
      auto node = graph_->item_to_node(expr);
      if (!basic_block_stack_.empty()) {
        node->basic_block_ = basic_block_stack_.back();
      }
    }

    std::unique_ptr<IndexedGraph<Expr>> graph_;
    std::vector<IndexedGraph<Expr>::Node*> basic_block_stack_;
  };

  VLOG(1) << "CreateIndexedGraph:" << std::endl << PrettyPrint(expr);
  std::unique_ptr<IndexedGraph<Expr>> graph =
      Blocker(Annotator(Creator().CreateGraph(expr)).Annotate()).Scope(expr);
  VLOG(1) << "graph:" << std::endl << graph->ToString();
#if TVM_LOG_DEBUG
  graph->CheckValid();
#endif
  return graph;
}

std::unique_ptr<IndexedGraph<DFPattern>> CreateIndexedGraph(const DFPattern& pattern) {
  /*! \brief Creates an IndexedGraph and determines topological order */
  class Creator : public DFPatternVisitor {
   public:
    std::unique_ptr<IndexedGraph<DFPattern>> CreateGraph(const DFPattern& pattern) {
      graph_ = std::make_unique<IndexedGraph<DFPattern>>();
      VisitDFPattern(pattern);
      graph_->item_to_node(pattern)->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    void VisitDFPattern(const DFPattern& pattern) override {
      if (this->visited_.count(pattern.get()) == 0) {
        DFPatternVisitor::VisitDFPattern(pattern);
        graph_->AddNode(pattern);
      }
    }

    std::unique_ptr<IndexedGraph<DFPattern>> graph_;
  };

  /*! \brief Annotator takes an IndexedGraph, fills it's forward outputs, and does dominator tree
   * analysis.
   *
   *  Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public DFPatternFunctor<void(const DFPattern&)> {
   public:
    Annotator(std::unique_ptr<IndexedGraph<DFPattern>> graph) : graph_(std::move(graph)) {}

    std::unique_ptr<IndexedGraph<DFPattern>> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (PostDfsIndex index = 0; index < graph_->size(); ++index) {
        VisitDFPattern(graph_->index_to_node(index)->ref());
      }
      // do the dominator analysis
      graph_->PostDom();
      return std::move(graph_);
    }

    /*! Default visitation pushes the parent to the child's outputs */
    void AddOutput(const DFPattern& pattern, IndexedGraph<DFPattern>::Node* parent) {
      auto current = graph_->item_to_node(pattern);
      if (parent) {
        current->outputs_.push_back(parent);
        parent->inputs_.push_back(current);
      }
    }

   protected:
    void VisitDFPattern_(const AltPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<AltPattern>(op));
      AddOutput(op->left, node);
      AddOutput(op->right, node);
    }

    void VisitDFPattern_(const AttrPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<AttrPattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const CallPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<CallPattern>(op));
      AddOutput(op->op, node);
      if (op->args.defined()) {
        for (auto arg : op->args) {
          AddOutput(arg, node);
        }
      }
    }

    void VisitDFPattern_(const ConstantPatternNode* op) override {}

    void VisitDFPattern_(const DataTypePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<DataTypePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const DominatorPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<DominatorPattern>(op));
      AddOutput(op->parent, node);
      AddOutput(op->path, node);
      AddOutput(op->child, node);
    }

    void VisitDFPattern_(const ExprPatternNode* op) override {}

    void VisitDFPattern_(const FunctionPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<FunctionPattern>(op));
      if (op->params.defined()) {
        for (auto param : op->params) {
          AddOutput(param, node);
        }
      }
      AddOutput(op->body, node);
    }

    void VisitDFPattern_(const ShapePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<ShapePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const TupleGetItemPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TupleGetItemPattern>(op));
      AddOutput(op->tuple, node);
    }

    void VisitDFPattern_(const TuplePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TuplePattern>(op));
      if (op->fields.defined()) {
        for (auto field : op->fields) {
          AddOutput(field, node);
        }
      }
    }

    void VisitDFPattern_(const IfPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<IfPattern>(op));
      AddOutput(op->cond, node);
      AddOutput(op->true_branch, node);
      AddOutput(op->false_branch, node);
    }

    void VisitDFPattern_(const LetPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<LetPattern>(op));
      AddOutput(op->var, node);
      AddOutput(op->value, node);
      AddOutput(op->body, node);
    }

    void VisitDFPattern_(const TypePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TypePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const VarPatternNode* op) override {}

    void VisitDFPattern_(const WildcardPatternNode* op) override {
      if (op->pattern) {
        auto node = graph_->item_to_node(GetRef<WildcardPattern>(op));
        AddOutput(op->pattern.value(), node);
      }
    }

    std::unique_ptr<IndexedGraph<DFPattern>> graph_;
  };

  return Annotator(Creator().CreateGraph(pattern)).Annotate();
}

}  // namespace relay
}  // namespace tvm
