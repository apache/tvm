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
#include <tvm/runtime/container.h>

#include <unordered_map>
#include <vector>


namespace tvm {
namespace relay {

AnnotatedRegion AnnotatedRegionSetNode::GetRegion(const Expr& expr) const {
  for (auto candidate : regions_) {
    if (candidate->nodes_.find(expr) != candidate->nodes_.end()) {
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
  dest->nodes_.insert(src->nodes_.begin(), src->nodes_.end());
  for (const auto& input : src->ins_) {
    dest->ins_.push_back(input);
  }
  for (const auto& output : src->outs_) {
    dest->outs_.push_back(output);
  }
  // if any of the outputs of src are inputs of dest, they become internal nodes
  // so remove them from outs
  std::vector<Expr> ins_to_remove;
  for (const auto& input : dest->ins_) {
    auto call = Downcast<Call>(input);
    auto it = src->nodes_.find(call->args[0]);
    if (it != src->nodes_.end()) {
      dest->outs_.remove(*it);
      ins_to_remove.push_back(input);
    }
  }
  for (const auto& input : ins_to_remove) {
    dest->ins_.remove(input);
  }
  regions_.erase(src);
}

void AnnotatedRegionSetNode::AddToRegion(AnnotatedRegion dest, const Expr& expr) {
  auto src = GetRegion(expr);
  if (src.defined()) {
    MergeRegions(src, dest);
  } else {
    dest->nodes_.insert(expr);
  }
}

AnnotatedRegion AnnotatedRegionSetNode::MakeRegion(const std::string& target) {
  auto ret = regions_.emplace(AnnotatedRegion());
  (*ret.first)->id_ = region_id_++;
  (*ret.first)->target_ = target;
  return *ret.first;
}

class AnnotatedRegionSet::Creator : protected MixedModeVisitor {
 public:
  Creator(const Op& region_begin_op, const Op& region_end_op)
      : begin_op_(region_begin_op), end_op_(region_end_op) {}

  AnnotatedRegionSet Create(const Expr& expr) {
    VisitExpr(expr);
    return std::move(region_set_);
  }

  void AddToArgRegion(Expr expr, Array<Expr> args) {
    // Merge argument regions and add itself to the region.

    // Find the first open region.
    AnnotatedRegion region;
    for (auto arg : args) {
      const CallNode* end = arg.as<CallNode>();
      if (end && end->op == end_op_) {  // Ignore closed regions.
          continue;
      }

      region = region_set_->GetRegion(arg);
      if (region.defined()) {
          break;
      }
    }

    // Try to merge open regions.
    for (auto arg : args) {
      const CallNode* end = arg.as<CallNode>();
      if (end && end->op == end_op_) {  // Ignore closed regions.
          continue;
      }

      auto arg_region = region_set_->GetRegion(arg);
      CHECK_EQ(region.defined(), arg_region.defined())
          << "Arg regions are inconsistent: " << AsText(expr);
      if (region.defined() && region != arg_region) {
        region_set_->MergeRegions(arg_region, region);
      }
    }
    if (region.defined()) {
      region_set_->AddToRegion(region, expr);
    }
  }

  void VisitExpr_(const CallNode* call) {
    auto op_node = call->op.as<OpNode>();

    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      AddToArgRegion(GetRef<Call>(call), call->args);
    } else if (call->op == begin_op_) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);
      std::string target = call->attrs.as<CompilerAttrs>()->compiler;

      // Check if the argument already belongs to a region
      auto region = region_set_->GetRegion(GetRef<Call>(call));
      CHECK(!region.defined());

      // Create a new region.
      region = region_set_->MakeRegion(target);
      region->nodes_.insert(GetRef<Call>(call));
      region->ins_.push_back(GetRef<Call>(call));
    } else {
      CHECK_EQ(call->op, end_op_);
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);
      std::string target = call->attrs.as<CompilerAttrs>()->compiler;

      // Check if the argument already belongs to a region
      auto region = region_set_->GetRegion(call->args[0]);
      if (!region.defined()) {
        throw Error(ErrorBuilder() << "Cannot find the corresponding region for end annotation:\n"
                                   << AsText(GetRef<Call>(call), false));
      } else {
        // If the argument is belonged to a region, it must have the same target.
        // Otherwise we should see a region_begin op.
        CHECK_EQ(region->GetTarget(), target);
      }
      region->nodes_.insert(GetRef<Call>(call));
      region->outs_.push_back(GetRef<Call>(call));
    }
  }

  void VisitExpr_(const TupleNode* op) {
    AddToArgRegion(GetRef<Tuple>(op), op->fields);
  }

  void VisitExpr_(const TupleGetItemNode* g) {
    Array<Expr> args = {g->tuple};
    AddToArgRegion(GetRef<TupleGetItem>(g), args);
  }

  void VisitExpr_(const LetNode* op) {
    Array<Expr> args = {op->var, op->value, op->body};
    AddToArgRegion(GetRef<Let>(op), args);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IfNode* op) {
    Array<Expr> args = {op->cond, op->true_branch, op->false_branch};
    AddToArgRegion(GetRef<If>(op), args);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefCreateNode* op) {
    Array<Expr> args = {op->value};
    AddToArgRegion(GetRef<RefCreate>(op), args);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefReadNode* op) {
    Array<Expr> args = {op->ref};
    AddToArgRegion(GetRef<RefRead>(op), args);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const RefWriteNode* op) {
    Array<Expr> args = {op->ref};
    AddToArgRegion(GetRef<RefWrite>(op), args);
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
