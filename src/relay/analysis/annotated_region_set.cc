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

#include "annotated_region_set.h"

#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>

#include <unordered_map>
#include <vector>


namespace tvm {
namespace relay {

AnnotatedRegion AnnotatedRegionSetNode::GetRegion(const Expr& expr) const {
  for (auto candidate : regions_) {
    if (candidate->nodes.find(expr) != candidate->nodes.end()) {
      return candidate;
    }
  }
  return AnnotatedRegion(nullptr);
}

void AnnotatedRegionSetNode::MergeRegions(AnnotatedRegion src,
                                          AnnotatedRegion dest) {
  if (dest == src) {
    return;
  }

  // Merge src to dest and erase src.
  dest->nodes.insert(src->nodes.begin(), src->nodes.end());
  for (const auto& input : src->ins) {
    dest->ins.push_back(input);
  }
  for (const auto& output : src->outs) {
    dest->outs.push_back(output);
  }
  // if any of the outputs of src are inputs of dest, they become internal nodes
  // so remove them from outs
  std::vector<Expr> ins_to_remove;
  for (const auto& input : dest->ins) {
    auto call = Downcast<Call>(input);
    auto it = src->nodes.find(call->args[0]);
    if (it != src->nodes.end()) {
      dest->outs.remove(*it);
      ins_to_remove.push_back(input);
    }
  }
  for (const auto& input : ins_to_remove) {
    dest->ins.remove(input);
  }
  regions_.erase(src);
}

void AnnotatedRegionSetNode::AddToRegion(AnnotatedRegion dest, const Expr& expr) {
  auto src = GetRegion(expr);
  if (src.defined()) {
    MergeRegions(src, dest);
  } else {
    dest->nodes.insert(expr);
  }
}

AnnotatedRegion AnnotatedRegionSetNode::MakeRegion() {
  auto ret = regions_.emplace(AnnotatedRegion());
  (*ret.first)->id = region_id_++;
  return *ret.first;
}

class AnnotatedRegionSet::Creator : public ExprVisitor {
 public:
  Creator(const Op& region_begin_op, const Op& region_end_op) :
    begin_op_(region_begin_op), end_op_(region_end_op) {}

  AnnotatedRegionSet Create(const Expr& expr) {
    VisitExpr(expr);
    return std::move(region_set_);
  }

  void VisitExpr_(const CallNode* call) {
    auto op_node = call->op.as<OpNode>();

    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      // Propagate region to arguments
      auto region = region_set_->GetRegion(GetRef<Call>(call));
      if (region.defined()) {
        for (auto arg : call->args) {
          region_set_->AddToRegion(region, arg);
        }
      }
    } else if (call->op == begin_op_) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      auto region = region_set_->GetRegion(GetRef<Call>(call));
      if (!region.defined()) {
        throw Error(ErrorBuilder()
                      << "Cannot find the corresponding region for start annotation:\n"
                      << AsText(GetRef<Call>(call), false));
      }
      region->ins.push_back(GetRef<Call>(call));
    } else {
      CHECK_EQ(call->op, end_op_);
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      // Check if the argument already belongs to a region
      auto region = region_set_->GetRegion(call->args[0]);
      if (!region.defined()) {
        region = region_set_->MakeRegion();
        region->nodes.insert(call->args[0]);
      }
      region->nodes.insert(GetRef<Call>(call));
      region->outs.push_back(GetRef<Call>(call));
    }
    ExprVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const TupleNode* op) {
    auto region = region_set_->GetRegion(GetRef<Tuple>(op));
    if (region.defined()) {
      for (auto field : op->fields) {
        region_set_->AddToRegion(region, field);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const TupleGetItemNode* g) {
    auto region = region_set_->GetRegion(GetRef<TupleGetItem>(g));
    if (region.defined()) {
      region_set_->AddToRegion(region, g->tuple);
    }
    ExprVisitor::VisitExpr_(g);
  }

  void VisitExpr_(const FunctionNode* op) {
    auto region = region_set_->GetRegion(GetRef<Function>(op));
    if (region.defined()) {
      for (auto param : op->params) {
        region_set_->AddToRegion(region, param);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LetNode* op) {
    auto region = region_set_->GetRegion(GetRef<Let>(op));
    if (region.defined()) {
      region_set_->AddToRegion(region, op->var);
      region_set_->AddToRegion(region, op->value);
      region_set_->AddToRegion(region, op->body);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IfNode* op) {
    auto region = region_set_->GetRegion(GetRef<If>(op));
    if (region.defined()) {
      region_set_->AddToRegion(region, op->cond);
      region_set_->AddToRegion(region, op->true_branch);
      region_set_->AddToRegion(region, op->false_branch);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefCreateNode* op) {
    auto region = region_set_->GetRegion(GetRef<RefCreate>(op));
    if (region.defined()) {
      region_set_->AddToRegion(region, op->value);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefReadNode* op) {
    auto region = region_set_->GetRegion(GetRef<RefRead>(op));
    if (region.defined()) {
      region_set_->AddToRegion(region, op->ref);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefWriteNode* op) {
    auto region = region_set_->GetRegion(GetRef<RefWrite>(op));
    if (region.defined()) {
      region_set_->AddToRegion(region, op->ref);
    }
    ExprVisitor::VisitExpr_(op);
  }

 private:
  /*! \brief The region set being constructed.*/
  AnnotatedRegionSet region_set_;
  /*! \brief Region 'begin' annotation operator. */
  const Op begin_op_;
  /*! \brief Region 'end' annotation operator. */
  const Op end_op_;
};

AnnotatedRegionSet AnnotatedRegionSet::Create(const Expr& expr, const Op& begin, const Op& end) {
  return Creator(begin, end).Create(expr);
}

TVM_REGISTER_NODE_TYPE(AnnotatedRegionNode);
TVM_REGISTER_NODE_TYPE(AnnotatedRegionSetNode);

TVM_REGISTER_GLOBAL("relay.analysis.AnnotatedRegionSet")
.set_body_typed([](Expr expr, Op begin, Op end) {
  return AnnotatedRegionSet::Create(expr, begin, end);
});

TVM_REGISTER_GLOBAL("relay.analysis.GetRegion")
.set_body_typed([](AnnotatedRegionSet region_set, Expr expr) {
  return region_set->GetRegion(expr);
});


}  // namespace relay
}  // namespace tvm
