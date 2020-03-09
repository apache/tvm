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
 * \file src/relay/pass/merge_supported.cc
 *
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates subgraphs of the operators for each target. It
 * is guaranteed that the subgraphs will have a topological ordering so that
 * no data dependency issues exist.
 *
 * This pass only introduces annotations to indicate the subgraphs.
 * partition_graph must subsequently be called to lift these subgraphs out
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
#include <utility>
#include <vector>

#include "../analysis/subgraph_set.h"


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
 *  user will not leave un-annotated nodes MergeSupported pass is run.
 *  Why? Because, MergeSupported pass assumes every node to be annotated.
 */
class AnnotateRestDefault : public ExprMutator {
 public:
  explicit AnnotateRestDefault(const Expr& expr) {
      subgraphs_ = SubgraphSet::Create(expr, compiler_begin_op, compiler_end_op);
  }

  Expr Annotate(const Expr& expr) {
    // Its a function that is being passed on to annotate
    func_ = Downcast<Function>(expr);

    // Corner Case CC1 : If the last node does not belong
    // to a subgraph nede to add a compiler_end
    auto subgraph = subgraphs_->GetSubgraph(func_->body);
    auto mutated_expr = this->VisitExpr(expr);
    if (!subgraph.defined()) {
      func_ = Downcast<Function>(mutated_expr);
      // CC1 : add that compiler end after mutation
      auto body = AddCompilerEnd_(func_->body);
      func_ = Function(func_->params, body,
                       body->checked_type_, {}, DictAttrs());
      return Downcast<Expr>(func_);
    } else {
      return mutated_expr;
    }
  }

    /*! \brief This function adds compiler ends to nodes nodes that
     * does have a subgraph AND they should not be arguments of the
     * original function
     * @param expr
     * @return expr
     */
  Expr AddCompilerEnd(const Expr& expr) {
    auto subgraph = subgraphs_->GetSubgraph(expr);
    auto visited_expr = VisitExpr(expr);

    // The compiler ends are added to nodes that does have a subgraph
    // AND they should not be arguments of the original function
    if (!subgraph.defined() &&
       std::find(func_->params.begin(),
                 func_->params.end(), visited_expr)
                   == func_->params.end()) {
      return AddCompilerEnd_(visited_expr);
    } else {
      return visited_expr;
    }
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

      auto subgraph = subgraphs_->GetSubgraph(ret);
      if (!subgraph.defined()) {
        // if the current node does not belong to annotated subgraph
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
        Expr update_call = CallNode::make(call->op, compiler_begins, call->attrs);
        return update_call;
      } else {
        return CallNode::make(call->op, args, call->attrs);
      }
    } else {
      return CallNode::make(call->op, args, call->attrs);
    }
  };

  Expr VisitExpr_(const TupleNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto tup = Downcast<Tuple>(new_e);
    Array<Expr> new_fields;
    for (auto field : tup->fields) {
      new_fields.push_back(AddCompilerEnd(field));
    }
    return TupleNode::make(new_fields);
  }

  Expr VisitExpr_(const TupleGetItemNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto get = Downcast<TupleGetItem>(new_e);
    return TupleGetItemNode::make(
      AddCompilerEnd(get->tuple),
      get->index);
  }

  Expr VisitExpr_(const LetNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto let = Downcast<Let>(new_e);
    return LetNode::make(
      let->var,
      AddCompilerEnd(let->value),
      AddCompilerEnd(let->body));
  }

  Expr VisitExpr_(const IfNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto iff = Downcast<If>(new_e);
    return IfNode::make(
      AddCompilerEnd(iff->cond),
      AddCompilerEnd(iff->true_branch),
      AddCompilerEnd(iff->false_branch));
  }

  Expr VisitExpr_(const RefCreateNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto create = Downcast<RefCreate>(new_e);
    return RefCreateNode::make(AddCompilerEnd(create->value));
  }

  Expr VisitExpr_(const RefReadNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto read = Downcast<RefRead>(new_e);
    return RefReadNode::make(AddCompilerEnd(read->ref));
  }

  Expr VisitExpr_(const RefWriteNode *op) {
    auto new_e = ExprMutator::VisitExpr_(op);
    auto write = Downcast<RefWrite>(new_e);
    return RefWriteNode::make(
      AddCompilerEnd(write->ref),
      AddCompilerEnd(write->value));
  }

 private:
    SubgraphSet subgraphs_;
    const std::string target_ = "default";
    Function func_;
};

class MergeAnnotations : public ExprMutator {
 public:
  explicit MergeAnnotations(SubgraphSet subgraphs) : subgraphs_(subgraphs) {}

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op == compiler_begin_op) {
      if (call->args[0]->IsInstance<CallNode>()) {
        auto arg = Downcast<Call>(call->args[0]);
        if (arg->op == compiler_end_op) {
          auto subgraph1 = subgraphs_->GetSubgraph(GetRef<Call>(call));
          auto subgraph2 = subgraphs_->GetSubgraph(arg);
          if (subgraph1 == subgraph2) {
            return ExprMutator::VisitExpr(arg->args[0]);
          }
        }
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  SubgraphSet subgraphs_;
};

class SubgraphMerger : public ExprVisitor {
 public:
  explicit SubgraphMerger(SubgraphSet subgraphs) : subgraphs_(subgraphs) {}

  void VisitExpr_(const CallNode* call) final {
    if (call->op == compiler_end_op) {
      auto subgraph = subgraphs_->GetSubgraph(GetRef<Call>(call));
      // set the subgraph target
      auto compiler_attrs = call->attrs.as<CompilerAttrs>();
      subgraph_targets_[subgraph->id] = compiler_attrs->compiler;
      std::vector<Subgraph> mergeable_subgraphs;
      // first look at the subgraph args to determine the parent subgraphs
      for (const auto& arg : subgraph->args) {
        // all args should be begin annotations
        auto begin = Downcast<Call>(arg);
        // the arguments of the begin annotations will be in the parent subgraphs
        auto parent_subgraph = subgraphs_->GetSubgraph(begin->args[0]);
        // if there is no parent subgraph, move on
        if (!parent_subgraph.defined()) continue;
        // merge the parent subgraph if it hasn't been done already
        if (merged_subgraphs_.find(parent_subgraph->id) == merged_subgraphs_.end()) {
          VisitExpr(begin->args[0]);
        }
        mergeable_subgraphs.push_back(parent_subgraph);
      }
      // subgraph_restrictions_[subgraph->id] = std::unordered_set<int>();
      auto& subgraph_restrictions = subgraph_restrictions_[subgraph->id];
      for (const auto& parent_subgraph : mergeable_subgraphs) {
        auto parent_restrictions = subgraph_restrictions_[parent_subgraph->id];
        // add all the parent restrictions to the current subgraph
        subgraph_restrictions.insert(parent_restrictions.begin(),
                                     parent_restrictions.end());
      }
      for (const auto& parent_subgraph : mergeable_subgraphs) {
        bool merged = false;
        // check the parent subgraph has the same target
        if (subgraph_targets_[parent_subgraph->id] == compiler_attrs->compiler) {
          // check the parent subgraph isn't in the restrictions
          if (subgraph_restrictions.find(parent_subgraph->id) == subgraph_restrictions.end()) {
            // merge the parent subgraph into the current subgraph
            subgraphs_->MergeSubgraph(subgraph, parent_subgraph);
            // update the restrictions of all other subgraphs to reflect the change in id
            for (const auto& s : subgraphs_) {
              auto& restrictions = subgraph_restrictions_[s->id];
              if (restrictions.find(parent_subgraph->id) != restrictions.end()) {
                restrictions.erase(parent_subgraph->id);
                restrictions.insert(subgraph->id);
              }
            }
            merged = true;
          }
        }
        // if the parent wasn't merged, add it as a restriction to the current subgraph
        if (!merged)
          subgraph_restrictions.insert(parent_subgraph->id);
      }
      merged_subgraphs_.insert(subgraph->id);
    }
    ExprVisitor::VisitExpr_(call);
  }

 private:
  SubgraphSet subgraphs_;
  std::unordered_set<int> merged_subgraphs_;
  std::map<int, std::unordered_set<int>> subgraph_restrictions_;
  std::map<int, std::string> subgraph_targets_;
};


Expr MergeSupported(const Expr& expr) {
  // Annotate the nodes that are not annotated, if any.
  AnnotateRestDefault anno_default(expr);
  auto expr_all_annotated = anno_default.Annotate(expr);

  // Create subgraphs using the annotations.
  SubgraphSet subgraphs = SubgraphSet::Create(expr_all_annotated,
                                              compiler_begin_op, compiler_end_op);

  // By now, all the nodes have some sort of annotation.
  // Subgraph merger is ExprVisitor that will update the
  // SubgraphSet.
  SubgraphMerger merger(subgraphs);
  merger.VisitExpr(expr_all_annotated);

  // This will use updated (merged)
  // SubgraphSet : subgraphs_withdefault to remove annotations
  // within each subgraph
  MergeAnnotations merge_anno(subgraphs);
  return merge_anno.Mutate(expr_all_annotated);
}

}  // namespace partitioning

namespace transform {

Pass MergeSupported() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> part_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(partitioning::MergeSupported(f));
      };
  auto partitioned = CreateFunctionPass(part_func, 0, "MergeSupported", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.MergeSupported")
.set_body_typed(transform::MergeSupported);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
