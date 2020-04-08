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
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/dataflow_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

// DFPatternVisitor

void DFPatternVisitor::VisitDFPattern(const DFPattern& pattern) {
  if (this->visited_.count(pattern.get()) == 0) {
    visited_.insert(pattern.get());
    DFPatternFunctor::VisitDFPattern(pattern);
  }
}

void DFPatternVisitor::VisitDFPattern_(const AltPatternNode* op) {
  VisitDFPattern(op->left);
  VisitDFPattern(op->right);
}

void DFPatternVisitor::VisitDFPattern_(const AttrPatternNode* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const CallPatternNode* op) {
  VisitDFPattern(op->op);
  for (auto arg : op->args) {
    VisitDFPattern(arg);
  }
}
void DFPatternVisitor::VisitDFPattern_(const DominatorPatternNode* op) {
  VisitDFPattern(op->parent);
  VisitDFPattern(op->path);
  VisitDFPattern(op->child);
}

void DFPatternVisitor::VisitDFPattern_(const ExprPatternNode* op) {}

void DFPatternVisitor::VisitDFPattern_(const TupleGetItemPatternNode* op) {
  VisitDFPattern(op->tuple);
}

void DFPatternVisitor::VisitDFPattern_(const TuplePatternNode* op) {
  for (auto field : op->fields) {
    VisitDFPattern(field);
  }
}

void DFPatternVisitor::VisitDFPattern_(const TypePatternNode* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const VarPatternNode* op) {}

void DFPatternVisitor::VisitDFPattern_(const WildcardPatternNode* op) {}

// DFPatternMutator

DFPattern DFPatternMutator::Mutate(const DFPattern& pattern) { return VisitDFPattern(pattern); }

DFPattern DFPatternMutator::VisitDFPattern(const DFPattern& pattern) {
  auto it = this->memo_.find(pattern);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    auto new_pattern = DFPatternFunctor::VisitDFPattern(pattern);
    memo_[pattern] = new_pattern;
    return new_pattern;
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const AltPatternNode* op) {
  auto new_left = Mutate(op->left);
  auto new_right = Mutate(op->right);

  if (new_left.same_as(op->left) && new_right.same_as(op->right)) {
    return GetRef<DFPattern>(op);
  } else {
    return AltPatternNode::make(new_left, new_right);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const AttrPatternNode* op) {
  auto new_pattern = Mutate(op->pattern);
  if (new_pattern.same_as(op->pattern)) {
    return GetRef<DFPattern>(op);
  } else {
    return AttrPatternNode::make(new_pattern, op->attrs);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const CallPatternNode* op) {
  auto new_op = Mutate(op->op);
  bool unchanged = op->op.same_as(new_op);
  tvm::Array<DFPattern> call_args;
  for (auto arg : op->args) {
    auto new_arg = Mutate(arg);
    call_args.push_back(new_arg);
    unchanged &= arg.same_as(new_arg);
  }
  if (unchanged) {
    return GetRef<DFPattern>(op);
  } else {
    return CallPatternNode::make(new_op, call_args, op->attrs, op->type_args);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const DominatorPatternNode* op) {
  auto new_parent = Mutate(op->parent);
  auto new_path = Mutate(op->path);
  auto new_child = Mutate(op->child);
  if (op->parent.same_as(new_child) && op->parent.same_as(new_child) &&
      op->parent.same_as(new_child)) {
    return GetRef<DFPattern>(op);
  } else {
    return DominatorPatternNode::make(new_parent, new_path, new_child);
  }
}


DFPattern DFPatternMutator::VisitDFPattern_(const ExprPatternNode* op) {
  return GetRef<DFPattern>(op);
}

DFPattern DFPatternMutator::VisitDFPattern_(const TupleGetItemPatternNode* op) {
  auto new_tuple = Mutate(op->tuple);
  if (new_tuple.same_as(op->tuple)) {
    return GetRef<DFPattern>(op);
  } else {
    return TupleGetItemPatternNode::make(op->tuple, op->index);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const TuplePatternNode* op) {
  bool unchanged = true;
  tvm::Array<DFPattern> fields;
  for (auto field : op->fields) {
    auto new_field = Mutate(field);
    fields.push_back(new_field);
    unchanged &= field.same_as(new_field);
  }
  if (unchanged) {
    return GetRef<DFPattern>(op);
  } else {
    return TuplePatternNode::make(fields);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const TypePatternNode* op) {
  auto new_pattern = Mutate(op->pattern);
  if (new_pattern.same_as(op->pattern)) {
    return GetRef<DFPattern>(op);
  } else {
    return TypePatternNode::make(new_pattern, op->type);
  }
}

DFPattern DFPatternMutator::VisitDFPattern_(const VarPatternNode* op) {
  return GetRef<DFPattern>(op);
}

DFPattern DFPatternMutator::VisitDFPattern_(const WildcardPatternNode* op) {
  return GetRef<DFPattern>(op);
}

//IndexedGraph

template <typename T>
struct Node {
  Node(const T& ref, const size_t index) : ref_(ref), index_(index) {}
  const T ref_;
  const size_t index_;
  std::vector<std::shared_ptr<Node<T>>> outputs_;
};

template <typename T>
struct IndexedGraph {
  std::unordered_map<T, std::shared_ptr<Node<T>>, ObjectHash, ObjectEqual> node_map_;
  std::vector<std::shared_ptr<Node<T>>> topological_order_;
};

IndexedGraph<Expr>
CreateIndexedGraph(const Expr& expr) {
  using NodePtr = std::shared_ptr<Node<Expr>>;
  class Creator : public MixedModeVisitor {
   public:
    IndexedGraph<Expr> CreateGraph(const Expr& expr) {
      VisitExpr(expr);
      return std::move(graph_);
    }
    void Create(const Expr& expr) { VisitExpr(expr); }

   protected:
    void VisitLeaf(const Expr& expr) override {
      MixedModeVisitor::VisitLeaf(expr);
      auto node = std::make_shared<Node<Expr>>(expr, index_++);
      graph_.node_map_[expr] = node;
      graph_.topological_order_.push_back(node);
    }
    IndexedGraph<Expr> graph_;
    size_t index_ = 0;
  };
  class Annotator : public ExprFunctor<void(const Expr&, NodePtr)> {
   public:
    Annotator(const IndexedGraph<Expr>& graph) : graph_(graph) {}
    IndexedGraph<Expr> Annotate() {
      for (const auto& node : graph_.topological_order_) {
        ExprFunctor::VisitExpr(node->ref_, nullptr);
      }
      return std::move(graph_);
    }

    void VisitExpr(const Expr& expr, NodePtr parent) override {
      if (parent) {
        graph_.node_map_[expr]->outputs_.push_back(parent);
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
        this->VisitExpr(field, graph_.node_map_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const FunctionNode* op, NodePtr parent) override {
      for (auto param : op->params) {
        this->VisitExpr(param, graph_.node_map_[GetRef<Expr>(op)]);
      }

      this->VisitExpr(op->body, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const CallNode* op, NodePtr parent) override {
      this->VisitExpr(op->op, graph_.node_map_[GetRef<Expr>(op)]);

      for (auto ty_arg : op->type_args) {
        this->VisitType(ty_arg);
      }

      for (auto arg : op->args) {
        this->VisitExpr(arg, graph_.node_map_[GetRef<Expr>(op)]);
      }
    }

    void VisitExpr_(const LetNode* op, NodePtr parent) override {
      this->VisitExpr(op->value, graph_.node_map_[GetRef<Expr>(op)]);
      this->VisitExpr(op->var, graph_.node_map_[GetRef<Expr>(op)]);
      this->VisitExpr(op->body, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const IfNode* op, NodePtr parent) override {
      this->VisitExpr(op->cond, graph_.node_map_[GetRef<Expr>(op)]);
      this->VisitExpr(op->true_branch, graph_.node_map_[GetRef<Expr>(op)]);
      this->VisitExpr(op->false_branch, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const OpNode* op, NodePtr parent) override { return; }

    void VisitExpr_(const TupleGetItemNode* op, NodePtr parent) override {
      this->VisitExpr(op->tuple, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefCreateNode* op, NodePtr parent) override {
      this->VisitExpr(op->value, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefReadNode* op, NodePtr parent) override {
      this->VisitExpr(op->ref, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const RefWriteNode* op, NodePtr parent) override {
      this->VisitExpr(op->ref, graph_.node_map_[GetRef<Expr>(op)]);
      this->VisitExpr(op->value, graph_.node_map_[GetRef<Expr>(op)]);
    }

    void VisitExpr_(const ConstructorNode* op, NodePtr parent) override {
      for (const Type& t : op->inputs) {
        this->VisitType(t);
      }
      this->VisitType(op->belong_to);
    }

    void VisitExpr_(const MatchNode* op, NodePtr parent) override {
      this->VisitExpr(op->data, graph_.node_map_[GetRef<Expr>(op)]);
      for (const Clause& c : op->clauses) {
        this->VisitClause(c, graph_.node_map_[GetRef<Expr>(op)]);
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
// 
// IndexedGraph<DFPattern>
// CreateIndexedGraph(const Expr& expr) {
//   using NodePtr = std::shared_ptr<IndexedGraph::Node>;
//   class Creator : public MixedModeVisitor {
//    public:
//     IndexedGraph<Expr> CreateGraph(const Expr& expr) {
//       VisitExpr(expr);
//       return std::move(graph_);
//     }
//     void Create(const Expr& expr) { VisitExpr(expr); }
// 
//    protected:
//     void DispatchVisitExpr(const Expr& expr) override {
//       MixedModeVisitor::DispatchVisitExpr(expr);
//       graph_.node_map[expr] = std::make_shared<IndexedGraph::Node>(expr, index++, {});
//       l graph_.topological_order.push_back(expr);
//     }
//     IndexdecGraph<Expr> graph_;
//     size_t index_ = 0;
//   };
//   class Annotator : public ExprFunctor<void(const Expr&, NodePtr)> {
//    public:
//     Annotator(const IndexedGraph<Expr>& graph) : graph_(graph) {}
//     Annotate() {
//       for (const auto& node : graph_.topological_order) {
//         ExprFunctor::VisitExpr(node.ref, nullptr);
//       }
//       return std::move(graph_);
//     }
// 
//     VisitExpr(const Expr& expr, NodePtr parent) override {
//       if (parent) {
//         graph_.node_map[expr].outputs.push_back(parent);
//       }
//     }
// 
//    protected:
//     void VisitExpr_(const VarNode* op, NodePtr parent) override {
//       if (op->type_annotation.defined()) {
//         this->VisitType(op->type_annotation);
//       }
//     }
// 
//     void VisitExpr_(const GlobalVarNode* op, NodePtr parent) override {}
// 
//     void VisitExpr_(const ConstantNode* op, NodePtr parent) override {}
// 
//     void VisitExpr_(const TupleNode* op, NodePtr parent) override {
//       for (auto field : op->fields) {
//         this->VisitExpr(field, GetRef<Expr>(op));
//       }
//     }
// 
//     void VisitExpr_(const FunctionNode* op, NodePtr parent) override {
//       for (auto param : op->params) {
//         this->VisitExpr(param, GetRef<Expr>(op));
//       }
// 
//       this->VisitExpr(op->body, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const CallNode* op, NodePtr parent) override {
//       this->VisitExpr(op->op, GetRef<Expr>(op));
// 
//       for (auto ty_arg : op->type_args, NodePtr parent) override {
//         this->VisitType(ty_arg, GetRef<Expr>(op));
//       }
// 
//       for (auto arg : op->args) {
//         this->VisitExpr(arg, GetRef<Expr>(op));
//       }
//     }
// 
//     void VisitExpr_(const LetNode* op, NodePtr parent) override {
//       this->VisitExpr(op->value, GetRef<Expr>(op));
//       this->VisitExpr(op->var, GetRef<Expr>(op));
//       this->VisitExpr(op->body, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const IfNode* op, NodePtr parent) override {
//       this->VisitExpr(op->cond, GetRef<Expr>(op));
//       this->VisitExpr(op->true_branch, GetRef<Expr>(op));
//       this->VisitExpr(op->false_branch, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const OpNode* op, NodePtr parent) override { return; }
// 
//     void VisitExpr_(const TupleGetItemNode* op, NodePtr parent) override {
//       this->VisitExpr(op->tuple, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const RefCreateNode* op, NodePtr parent) override {
//       this->VisitExpr(op->value, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const RefReadNode* op, NodePtr parent) override {
//       this->VisitExpr(op->ref, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const RefWriteNode* op, NodePtr parent) override {
//       this->VisitExpr(op->ref, GetRef<Expr>(op));
//       this->VisitExpr(op->value, GetRef<Expr>(op));
//     }
// 
//     void VisitExpr_(const ConstructorNode* op, NodePtr parent) override {
//       for (const Type& t : op->inputs) {
//         this->VisitType(t);
//       }
//       this->VisitType(op->belong_to);
//     }
// 
//     void VisitExpr_(const MatchNode* op, NodePtr parent) override {
//       this->VisitExpr(op->data, GetRef<Expr>(op));
//       for (const Clause& c : op->clauses) {
//         this->VisitClause(c, GetRef<Expr>(op));
//       }
//     }
// 
//     void VisitClause(const Clause& op, NodePtr parent) override {
//       this->VisitPattern(op->lhs);
//       this->VisitExpr(op->rhs, parent);
//     }
// 
//     void VisitPattern(const Pattern& p) { return; }
// 
//     void VisitType(const Type& t) { return; }
//   };
//   return Annotator(Creator().CreateGraph(expr)).Annotate();
// }

//IndexedGraph<DFPattern> CreateIndexedGraph(const DFPattern&) {
//}

}  // namespace relay
}  // namespace tvm
