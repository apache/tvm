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

/*
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
#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../analysis/annotated_region_set.h"


namespace tvm {
namespace relay {
namespace partitioning {

// Cache compiler_begin and compiler_end annotation ops for equivalence check to
// reduce registry lookup overhead.
static const Op& compiler_begin_op = Op::Get("annotation.compiler_begin");
static const Op& compiler_end_op = Op::Get("annotation.compiler_end");

/*! \brief This is a pre-requisite pass to merge-supported pass.
 *  The AnnotateRestDefault pass will put "default" Compiler Annotations to
 *  nodes that are not annotated already. This is there to ensure that the
 *  user will not leave un-annotated nodes MergeCompilerRegions pass is run.
 *  Why? Because, MergeCompilerRegions pass assumes every node to be annotated.
 */
class AnnotateRestDefault : public ExprMutator {
 public:
  explicit AnnotateRestDefault(const Expr& expr) {
      regions_ = AnnotatedRegionSet::Create(expr, compiler_begin_op, compiler_end_op);
  }

  Expr Annotate(const Expr& expr) {
    // Its a function that is being passed on to annotate
    func_ = Downcast<Function>(expr);

    // Corner Case CC1 : If the last node does not belong
    // to a region nede to add a compiler_end
    auto region = regions_->GetRegion(func_->body);
    auto mutated_expr = this->VisitExpr(expr);
    if (!region.defined()) {
      func_ = Downcast<Function>(mutated_expr);
      // CC1 : add that compiler end after mutation
      auto body = AddCompilerEnd_(func_->body);
      func_ = Function(func_->params, body,
                       body->checked_type_, {}, DictAttrs());
      return Downcast<Expr>(func_);
    }
    return mutated_expr;
  }

  /*! \brief This function adds compiler ends to nodes that
   * have a region AND they should not be arguments of the
   * original function
   * \param expr The expression to add a compiler end to.
   * \return expr The expression with or without a compiler end added.
   */
  Expr AddCompilerEnd(const Expr& expr) {
    auto region = regions_->GetRegion(expr);
    auto visited_expr = VisitExpr(expr);

    // The compiler ends are added to nodes that does have a region
    // AND they should not be arguments of the original function
    if (!region.defined() &&
       std::find(func_->params.begin(),
                 func_->params.end(), visited_expr)
                   == func_->params.end()) {
      return AddCompilerEnd_(visited_expr);
    }
    return visited_expr;
  }

  Expr AddCompilerEnd_(const Expr& expr) {
    const auto* end_op =
      runtime::Registry::Get("relay.op.annotation._make.compiler_end");
    CHECK(end_op);
    Expr end = (*end_op)(expr, target_);
    return end;
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    auto ret = GetRef<Call>(call);

    Array<Expr> args;

    // Add compiler ends if the parent is supported
    for (auto arg : call->args) {
      args.push_back(AddCompilerEnd(arg));
    }

    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      // Skip annotatation ops, only add default compiler to actual compute nodes

      auto region = regions_->GetRegion(ret);
      if (!region.defined()) {
        // if the current node does not belong to annotated region
        // annotate the all incoming edges (args)
        // with "default" compile_begin and compiler_end annotations.
        tvm::Array<tvm::relay::Expr> compiler_begins;
        for (auto arg : args) {
          const auto* begin_op =
            runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
          CHECK(begin_op);
          Expr begin = (*begin_op)(arg, target_);
          compiler_begins.push_back(begin);
        }
        Expr update_call = Call(call->op, compiler_begins, call->attrs);
        return update_call;
      }
    }
    return Call(call->op, args, call->attrs);
  };

  Expr VisitExpr_(const TupleNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto tup = Downcast<Tuple>(new_e);
    Array<Expr> new_fields;
    for (auto field : tup->fields) {
      new_fields.push_back(AddCompilerEnd(field));
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const TupleGetItemNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto get = Downcast<TupleGetItem>(new_e);
    return TupleGetItem(AddCompilerEnd(get->tuple), get->index);
  }

  Expr VisitExpr_(const LetNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto let = Downcast<Let>(new_e);
    return Let(
      let->var,
      AddCompilerEnd(let->value),
      AddCompilerEnd(let->body));
  }

  Expr VisitExpr_(const IfNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto iff = Downcast<If>(new_e);
    return If(
      AddCompilerEnd(iff->cond),
      AddCompilerEnd(iff->true_branch),
      AddCompilerEnd(iff->false_branch));
  }

  Expr VisitExpr_(const RefCreateNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto create = Downcast<RefCreate>(new_e);
    return RefCreate(AddCompilerEnd(create->value));
  }

  Expr VisitExpr_(const RefReadNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto read = Downcast<RefRead>(new_e);
    return RefRead(AddCompilerEnd(read->ref));
  }

  Expr VisitExpr_(const RefWriteNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto write = Downcast<RefWrite>(new_e);
    return RefWrite(
      AddCompilerEnd(write->ref),
      AddCompilerEnd(write->value));
  }

 private:
    AnnotatedRegionSet regions_;
    const std::string target_ = "default";
    Function func_;
};

class MergeAnnotations : public ExprMutator {
 public:
  explicit MergeAnnotations(AnnotatedRegionSet regions) : regions_(regions) {}

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op == compiler_begin_op) {
      if (call->args[0]->IsInstance<CallNode>()) {
        auto arg = Downcast<Call>(call->args[0]);
        if (arg->op == compiler_end_op) {
          auto region1 = regions_->GetRegion(GetRef<Call>(call));
          auto region2 = regions_->GetRegion(arg);
          if (region1 == region2) {
            return ExprMutator::VisitExpr(arg->args[0]);
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  AnnotatedRegionSet regions_;
};

class RegionMerger : public ExprVisitor {
 public:
  explicit RegionMerger(AnnotatedRegionSet regions) : regions_(regions) {}

  void VisitExpr_(const CallNode* call) final {
    if (call->op == compiler_end_op) {
      auto region = regions_->GetRegion(GetRef<Call>(call));
      // set the region target
      auto compiler_attrs = call->attrs.as<CompilerAttrs>();
      region_targets_[region->GetID()] = compiler_attrs->compiler;
      std::vector<AnnotatedRegion> mergeable_regions;
      // first look at the region args to determine the parent regions
      for (const auto& arg : region->GetInputs()) {
        // all args should be begin annotations
        auto begin = Downcast<Call>(arg);
        CHECK_EQ(begin->op, compiler_begin_op);
        // the arguments of the begin annotations will be in the parent regions
        auto parent_region = regions_->GetRegion(begin->args[0]);
        // if there is no parent region, move on
        if (!parent_region.defined()) continue;
        // merge the parent region if it hasn't been done already
        if (merged_regions_.find(parent_region->GetID()) == merged_regions_.end()) {
          VisitExpr(begin->args[0]);
        }
        mergeable_regions.push_back(parent_region);
      }
      auto& region_restrictions = region_restrictions_[region->GetID()];
      for (const auto& parent_region : mergeable_regions) {
        // add all the parent restrictions to the current region
        auto parent_restrictions = region_restrictions_[parent_region->GetID()];
        region_restrictions.insert(parent_restrictions.begin(),
                                     parent_restrictions.end());
      }
      for (const auto& parent_region : mergeable_regions) {
        bool merged = false;
        // check the parent region has the same target
        if (region_targets_[parent_region->GetID()] == compiler_attrs->compiler) {
          // check the parent region isn't in the restrictions
          if (region_restrictions.find(parent_region->GetID()) == region_restrictions.end()) {
            // merge the parent region into the current region
            regions_->MergeRegions(parent_region, region);
            // update the restrictions of all other regions to reflect the change in id
            for (const auto& r : regions_) {
              auto& restrictions = region_restrictions_[r->GetID()];
              if (restrictions.find(parent_region->GetID()) != restrictions.end()) {
                restrictions.erase(parent_region->GetID());
                restrictions.insert(region->GetID());
              }
            }
            merged = true;
          }
        }
        // if the parent wasn't merged, add it as a restriction to the current region
        if (!merged)
          region_restrictions.insert(parent_region->GetID());
      }
      merged_regions_.insert(region->GetID());
    }
    ExprVisitor::VisitExpr_(call);
  }

 private:
  AnnotatedRegionSet regions_;
  std::unordered_set<int> merged_regions_;
  std::map<int, std::unordered_set<int>> region_restrictions_;
  std::map<int, std::string> region_targets_;
};


Expr MergeCompilerRegions(const Expr& expr) {
  // Annotate all the nodes that aren't annotated as 'default'.
  AnnotateRestDefault anno_default(expr);
  auto expr_all_annotated = anno_default.Annotate(expr);

  // Create regions using the annotations.
  AnnotatedRegionSet regions = AnnotatedRegionSet::Create(expr_all_annotated,
                                                          compiler_begin_op, compiler_end_op);

  // By now, all the nodes have some sort of annotation.
  // Region merger is an ExprVisitor that will update the
  // AnnotatedRegionSet, merging all the regions that can be merged.
  RegionMerger merger(regions);
  merger.VisitExpr(expr_all_annotated);

  // This updates the expression to remove annotations that are now
  // 'internal' to a merged region.
  MergeAnnotations merge_anno(regions);
  return merge_anno.Mutate(expr_all_annotated);
}

}  // namespace partitioning

namespace transform {

Pass MergeCompilerRegions() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> part_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(partitioning::MergeCompilerRegions(f));
      };
  auto partitioned = CreateFunctionPass(part_func, 0, "MergeCompilerRegions", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.MergeCompilerRegions")
.set_body_typed(transform::MergeCompilerRegions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
