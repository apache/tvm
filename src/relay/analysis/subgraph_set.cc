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

#include "subgraph_set.h"

#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>

#include <algorithm>
#include <unordered_map>
#include <vector>


namespace tvm {
namespace relay {

Subgraph SubgraphSetNode::GetSubgraph(const Expr& expr) const {
  for (auto candidate : subgraphs_) {
    if (candidate->nodes.find(expr) != candidate->nodes.end()) {
      return candidate;
    }
  }
  return Subgraph(nullptr);
}

void SubgraphSetNode::MergeSubgraph(Subgraph subgraph1,
                                    Subgraph subgraph2) {
  if (subgraph1 == subgraph2) {
    return;
  }

  // Merge subgraph 2 to subgraph 1 and erase subgraph 2.
  subgraph1->nodes.insert(subgraph2->nodes.begin(), subgraph2->nodes.end());
  for (auto arg : subgraph2->args) {
    subgraph1->args.push_back(arg);
  }
  for (auto out : subgraph2->rets) {
    subgraph1->rets.push_back(out);
  }
  // if any of the outputs of 2 are inputs of 1, they become internal nodes
  // so remove them from outs/args
  std::vector<Expr> args_to_remove;
  for (const auto& arg : subgraph1->args) {
    auto call = Downcast<Call>(arg);
    auto it = std::find(subgraph2->rets.begin(), subgraph2->rets.end(), call->args[0]);
    if (it != subgraph2->rets.end()) {
      args_to_remove.push_back(arg);
      subgraph1->rets.remove(*it);
    }
  }
  for (const auto& arg : args_to_remove) {
    subgraph1->args.remove(arg);
  }
  subgraphs_.erase(subgraph2);
}

void SubgraphSetNode::AddToSubgraph(Subgraph subgraph, const Expr& expr) {
  auto subgraph2 = GetSubgraph(expr);
  if (subgraph2.defined()) {
    MergeSubgraph(subgraph, subgraph2);
  } else {
    subgraph->nodes.insert(expr);
  }
}

Subgraph SubgraphSetNode::MakeSubgraph() {
  auto ret = subgraphs_.emplace(Subgraph());
  return *ret.first;
}

class SubgraphSet::Creator : public ExprVisitor {
 public:
  Creator(const Op &subgraph_begin_op, const Op &subgraph_end_op) :
    begin_op_(subgraph_begin_op), end_op_(subgraph_end_op) {}

  SubgraphSet Create(const Expr &expr) {
    VisitExpr(expr);
    return std::move(subgraph_set_);
  }

  void VisitExpr_(const CallNode *call) {
    auto op_node = call->op.as<OpNode>();

    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      // Propagate subgraph to arguments
      auto subgraph = subgraph_set_->GetSubgraph(GetRef<Call>(call));
      if (subgraph.defined()) {
        for (auto arg : call->args) {
          subgraph_set_->AddToSubgraph(subgraph, arg);
        }
      }
    } else if (call->op == begin_op_) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      auto subgraph = subgraph_set_->GetSubgraph(GetRef<Call>(call));
      if (!subgraph.defined()) {
        throw Error(ErrorBuilder()
                      << "Cannot find the corresponding subgraph for start annotation:\n"
                      << AsText(GetRef<Call>(call), false));
      }
      subgraph->args.push_back(GetRef<Call>(call));
    } else {
      CHECK_EQ(call->op, end_op_);
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      // Check if the argument already belongs to a subgraph
      auto subgraph = subgraph_set_->GetSubgraph(call->args[0]);
      if (!subgraph.defined()) {
        subgraph = subgraph_set_->MakeSubgraph();
        subgraph->nodes.insert(call->args[0]);
        subgraph->id = subgraph_id_++;
      }
      subgraph->nodes.insert(GetRef<Call>(call));
      subgraph->rets.push_back(GetRef<Call>(call));
    }
    ExprVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const TupleNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<Tuple>(op));
    if (subgraph.defined()) {
      for (auto field : op->fields) {
        subgraph_set_->AddToSubgraph(subgraph, field);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const TupleGetItemNode *g) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<TupleGetItem>(g));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, g->tuple);
    }
    ExprVisitor::VisitExpr_(g);
  }

  void VisitExpr_(const FunctionNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<Function>(op));
    if (subgraph.defined()) {
      for (auto param : op->params) {
        subgraph_set_->AddToSubgraph(subgraph, param);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LetNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<Let>(op));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, op->var);
      subgraph_set_->AddToSubgraph(subgraph, op->value);
      subgraph_set_->AddToSubgraph(subgraph, op->body);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IfNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<If>(op));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, op->cond);
      subgraph_set_->AddToSubgraph(subgraph, op->true_branch);
      subgraph_set_->AddToSubgraph(subgraph, op->false_branch);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefCreateNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<RefCreate>(op));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, op->value);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefReadNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<RefRead>(op));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, op->ref);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefWriteNode *op) {
    auto subgraph = subgraph_set_->GetSubgraph(GetRef<RefWrite>(op));
    if (subgraph.defined()) {
      subgraph_set_->AddToSubgraph(subgraph, op->ref);
    }
    ExprVisitor::VisitExpr_(op);
  }

 private:
  /*! \brief The subgraph set being constructed.*/
  SubgraphSet subgraph_set_;
  /*! \brief The next subgraph ID to assign. */
  int subgraph_id_{0};
  /*! \brief Subgraph 'begin' annotation operator. */
  const Op begin_op_;
  /*! \brief Subgraph 'end' annotation operator. */
  const Op end_op_;
};

SubgraphSet SubgraphSet::Create(const Expr& expr, const Op& begin, const Op& end) {
  return Creator(begin, end).Create(expr);
}

TVM_REGISTER_NODE_TYPE(SubgraphNode);
TVM_REGISTER_NODE_TYPE(SubgraphSetNode);

TVM_REGISTER_GLOBAL("relay.analysis.SubgraphSet")
.set_body_typed([](Expr expr, Op begin, Op end) {
  return SubgraphSet::Create(expr, begin, end);
});

TVM_REGISTER_GLOBAL("relay.analysis.GetSubgraph")
.set_body_typed([](SubgraphSet subgraph_set, Expr expr) {
  return subgraph_set->GetSubgraph(expr);
});


}  // namespace relay
}  // namespace tvm
