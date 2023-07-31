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
 * \file src/relay/transforms/merge_compiler_regions.cc
 *
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates regions of the operators for each target. It
 * is guaranteed that the regions will have a topological ordering so that
 * no data dependency issues exist.
 *
 * This pass only introduces annotations to indicate the regions.
 * partition_graph must subsequently be called to lift these regions out
 * as external functions.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../analysis/annotated_region_set.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {
namespace merge_compiler_region {

class RegionMerger : public MixedModeVisitor {
 public:
  explicit RegionMerger(AnnotatedRegionSet regions) : regions_(regions) {}

  void find_control_flow_regions(
      const Expr op,
      std::unordered_set<AnnotatedRegion, ObjectPtrHash, ObjectPtrEqual>& correlative_regions) {
    // Find correlative restriction regions from control flow.

    // In IfNode, find from condition, true_branch and false branch.
    const IfNode* if_node = op.as<IfNode>();
    if (if_node) {
      auto cond_region = regions_->GetRegion(if_node->cond);
      auto true_branch_region = regions_->GetRegion(if_node->true_branch);
      auto false_branch_region = regions_->GetRegion(if_node->false_branch);
      if (cond_region.defined()) {
        correlative_regions.insert(cond_region);
      } else {
        find_control_flow_regions(if_node->cond, correlative_regions);
      }
      if (true_branch_region.defined()) {
        correlative_regions.insert(true_branch_region);
      } else {
        find_control_flow_regions(if_node->true_branch, correlative_regions);
      }
      if (false_branch_region.defined()) {
        correlative_regions.insert(false_branch_region);
      } else {
        find_control_flow_regions(if_node->false_branch, correlative_regions);
      }
    }
  }

  void VisitExpr_(const CallNode* call) final {
    if (call->op == CompilerEndOp()) {
      auto region = regions_->GetRegion(GetRef<Call>(call));

      // Skip this region if it has been merged to the other region.
      if (merged_regions_.find(region->GetID()) != merged_regions_.end()) {
        return;
      }

      // Check the region target.
      auto compiler_attrs = call->attrs.as<CompilerAttrs>();
      ICHECK_EQ(region->GetTarget(), compiler_attrs->compiler);

      // Visit the unmerged parent regions.
      for (const auto& arg : region->GetInputs()) {
        // Region inputs must be begin annotation, and the region of
        // the begin annotation's argument is the parent region.
        auto begin = Downcast<Call>(arg);
        ICHECK_EQ(begin->op, CompilerBeginOp());
        auto parent_region = regions_->GetRegion(begin->args[0]);

        // Skip this region if it has been merged.
        if (!parent_region.defined()) {
          continue;
        } else if (merged_regions_.find(parent_region->GetID()) == merged_regions_.end()) {
          VisitExpr(begin->args[0]);
        }
      }

      // Collect unmerged parent regions.
      std::unordered_set<AnnotatedRegion, ObjectPtrHash, ObjectPtrEqual> mergeable_regions;
      // Collect correlative regions to propagate restrictions.
      std::unordered_set<AnnotatedRegion, ObjectPtrHash, ObjectPtrEqual> correlative_regions;
      for (const auto& arg : region->GetInputs()) {
        auto begin = Downcast<Call>(arg);
        ICHECK_EQ(begin->op, CompilerBeginOp());
        auto parent_region = regions_->GetRegion(begin->args[0]);
        if (parent_region.defined()) {
          mergeable_regions.insert(parent_region);
          correlative_regions.insert(parent_region);
        } else {
          find_control_flow_regions(begin->args[0], correlative_regions);
        }
      }

      // Propogate all the parent restrictions to the current region.
      auto& region_restrictions = region_restrictions_[region->GetID()];
      for (const auto& parent_region : correlative_regions) {
        auto parent_restrictions = region_restrictions_[parent_region->GetID()];
        region_restrictions.insert(parent_restrictions.begin(), parent_restrictions.end());
      }

      for (const auto& parent_region : mergeable_regions) {
        // Skip the parent region with a different target.
        if (parent_region->GetTarget() != compiler_attrs->compiler) {
          region_restrictions.insert(parent_region->GetID());
          continue;
        }

        // Skip the parent region if it is in the restriction set.
        if (region_restrictions.find(parent_region->GetID()) != region_restrictions.end()) {
          continue;
        }

        // Merge the parent region to the current one.
        regions_->MergeRegions(parent_region, region);

        // Replace the parent region ID with the current region for all
        // other regions' restriction sets.
        for (const auto& r : regions_) {
          auto& restrictions = region_restrictions_[r->GetID()];
          if (restrictions.find(parent_region->GetID()) != restrictions.end()) {
            restrictions.erase(parent_region->GetID());
            restrictions.insert(region->GetID());
          }
        }
      }
      merged_regions_.insert(region->GetID());
    }
  }

 private:
  AnnotatedRegionSet regions_;
  std::unordered_set<int> merged_regions_;
  std::unordered_map<int, std::unordered_set<int>> region_restrictions_;
};

class MergeAnnotations : public ExprRewriter {
 public:
  explicit MergeAnnotations(AnnotatedRegionSet regions) : regions_(regions) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    // Merge annotations which are now internal to a region.
    // This happens if we see a compiler begin next to a
    // compiler end and they're both in the same region.
    if (call->op == CompilerBeginOp() && call->args[0]->IsInstance<CallNode>()) {
      auto arg = Downcast<Call>(call->args[0]);
      if (arg->op == CompilerEndOp()) {
        auto region1 = regions_->GetRegion(GetRef<Call>(call));
        auto region2 = regions_->GetRegion(arg);
        if (region1 == region2) {
          auto post_arg = post.as<CallNode>()->args[0];
          return post_arg.as<CallNode>()->args[0];
        }
      }
    }
    return post;
  }

 private:
  AnnotatedRegionSet regions_;
};

Expr MergeCompilerRegions(const Expr& expr) {
  // Create regions using the annotations.
  AnnotatedRegionSet regions = AnnotatedRegionSet::Create(expr, CompilerBeginOp(), CompilerEndOp());

  // Analyze the graph to explore the opportunities of merging regions.
  RegionMerger merger(regions);
  merger.VisitExpr(expr);

  // Remove annotations that are not in the region boundaries.
  MergeAnnotations merge_anno(regions);
  return PostOrderRewrite(expr, &merge_anno);
}

}  // namespace merge_compiler_region

namespace transform {

Pass MergeCompilerRegions() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> part_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(merge_compiler_region::MergeCompilerRegions(f));
      };
  auto merged = CreateFunctionPass(part_func, 0, "MergeCompilerRegions", {});
  return Sequential({merged, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.MergeCompilerRegions")
    .set_body_typed(transform::MergeCompilerRegions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
