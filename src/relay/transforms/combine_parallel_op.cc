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
 *
 * \file combine_parallel_op.cc
 * \brief Abstract class to combine parallel ops and their successive element-wise ops.
 */

#include "combine_parallel_op.h"

#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

BranchGroupFinder::BranchGroupFinder(const Op& op, FIsSupportedOp fis_supported_op,
                                     FAreCompatibleOps fare_compatible_ops)
    : cached_op_(op),
      fis_supported_op_(fis_supported_op),
      fare_compatible_ops_(fare_compatible_ops) {}

std::vector<Group> BranchGroupFinder::Find(const Expr& expr) {
  this->VisitExpr(expr);

  std::vector<Group> groups;
  for (const auto& root : op_roots_) {
    const auto& children = children_map_.at(root);
    size_t ngroups = groups.size();
    for (const CallNode* child : children) {
      if (child->op != cached_op_) continue;

      auto&& branch = CreateBranch(child);
      // add the branch to a group, or create a new group
      auto it = std::find_if(groups.begin() + ngroups, groups.end(), [&](const Group& group) {
        ICHECK(!group.empty() && !group[0].empty());
        return fare_compatible_ops_(child, group[0][0]);
      });
      if (it != groups.end()) {
        it->push_back(branch);
      } else {
        groups.emplace_back();
        // each group has at least one branch
        groups.back().push_back(branch);
      }
    }
  }
  return groups;
}

// Create a branch starting from op.
Branch BranchGroupFinder::CreateBranch(const CallNode* op) {
  auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
  // each branch has at least one element, the first element is always op
  Branch branch{op};
  auto it = children_map_.find(GetRef<Expr>(branch.back()));
  while (it != children_map_.end() && it->second.size() == 1) {
    const CallNode* call = it->second[0];
    auto pattern = fpattern[Downcast<Op>(call->op)];
    if (pattern <= kBroadcast) {
      branch.push_back(call);
      it = children_map_.find(GetRef<Expr>(branch.back()));
    } else {
      break;
    }
  }
  return branch;
}

void BranchGroupFinder::VisitExpr_(const CallNode* n) {
  ExprVisitor::VisitExpr_(n);
  if (n->op == cached_op_ && fis_supported_op_(n)) {
    op_roots_.insert(n->args[0]);
    children_map_[n->args[0]].push_back(n);
  } else {
    for (size_t i = 0; i < n->args.size(); i++) {
      children_map_[n->args[i]].push_back(n);
    }
  }
}

ParallelOpCombiner::ParallelOpCombiner(const std::string& op_name, uint64_t min_num_branches)
    : cached_op_(Op::Get(op_name)), min_num_branches_(min_num_branches) {}

Expr ParallelOpCombiner::Combine(const Expr& expr) {
  auto groups = BranchGroupFinder(
                    cached_op_, [&](const CallNode* n) { return IsSupportedOp(n); },
                    [&](const CallNode* a, const CallNode* b) { return CanOpsBeCombined(a, b); })
                    .Find(expr);
  for (const Group& group : groups) {
    if (group.size() < min_num_branches_) {
      continue;
    }
    CombineBranches(group);
  }
  return ExprSubst(expr, std::move(subst_map_));
}

void ParallelOpCombiner::CombineBranches(const Group& branches) {
  Call combined = MakeCombinedOp(branches);
  auto it = std::min_element(branches.begin(), branches.end(),
                             [](const Branch& branch_a, const Branch& branch_b) {
                               return branch_a.size() < branch_b.size();
                             });
  size_t depth = it->size();
  size_t i;
  // starting from 1 to skip the op
  for (i = 1; i < depth; i++) {
    size_t parent_index;
    for (parent_index = 0; parent_index < branches[0][i]->args.size(); parent_index++) {
      if (branches[0][i]->args[parent_index].get() == branches[0][i - 1]) break;
    }
    ICHECK_NE(parent_index, branches[0][i]->args.size());
    if (!CheckLevel(branches, i, parent_index)) break;
    combined = MakeCombinedCallFromFollowingOps(combined, branches, i, parent_index);
  }
  UpdateGroupOutput(combined, branches, i - 1, &subst_map_);
}

bool ParallelOpCombiner::CheckLevel(const Group& branches, size_t depth, size_t parent_index) {
  const CallNode* call = branches[0][depth];
  tvm::StructuralEqual attrs_equal;
  // check if all branches in current depth can be combined
  for (auto it = branches.begin() + 1; it != branches.end(); it++) {
    const Branch& branch = *it;
    if (!branch[depth]->op.same_as(call->op) || !attrs_equal(branch[depth]->attrs, call->attrs) ||
        branch[depth]->args.size() != call->args.size()) {
      return false;
    }

    if (branch[depth]->args[parent_index].get() != branch[depth - 1]) return false;

    // Check args
    for (size_t i = 0; i < call->args.size(); i++) {
      if (i == parent_index) continue;

      if (!IsArgCompatible(call, branch[depth], i) ||
          !attrs_equal(call->attrs, branch[depth]->attrs)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace relay
}  // namespace tvm
