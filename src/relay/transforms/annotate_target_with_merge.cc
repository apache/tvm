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
 * \file src/relay/pass/annotate_target_with_merge.cc
 * \brief Wraps a call with compiler_begin and compiler_end to indicate that
 * the op of this call node will use external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace annotate_target_with_merge {

/*!
 * \brief The annotation group properties.
 */
struct AnnotateGroup {
  /*! \brief The group ID. */
  int id;

  /*! \brief Output node of this group. */
  Expr out;

  /*! \brief Nodes in this group. */
  std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes;

  /*! \brief The target for this group. */
  std::string target;

  /*! \brief Block gorup IDs. */
  std::unordered_set<int> blocks;
};

// A visitor to determine where to insert annotations for which target.
class AnalyzeAnnotateTargets : public ExprVisitor {
 public:
  explicit AnalyzeAnnotateTargets(Array<tvm::PrimExpr> targets) {
    for (auto target : targets) {
      auto* str_target = target.as<tir::StringImmNode>();
      this->targets_.push_back(str_target->value);
    }
  }

  std::shared_ptr<AnnotateGroup> CreateAnnotateGroup(const Expr node,
                                                     std::vector<std::string>& supported_targets) {
    auto ret = this->groups_.emplace(std::make_shared<AnnotateGroup>());
    auto group = *ret.first;
    group->id = this->curr_group_id_++;
    if (!supported_targets.empty()) {
      // A heuristic to offload this node to the first offloadable target.
      group->target = supported_targets.front();
    }
    group->nodes.insert(node);
    group->out = node;
    return group;
  }

  std::shared_ptr<AnnotateGroup> GetAnnotateGroup(const Expr node) {
    for (auto candidate : this->groups_) {
      if (candidate->nodes.find(node) != candidate->nodes.end()) {
        return candidate;
      }
    }
    return nullptr;
  }

  void MergeAnnotateGroup(std::shared_ptr<AnnotateGroup> group1,
                          std::shared_ptr<AnnotateGroup> group2) {
    if (group1->id == group2->id) {
      return;
    } else if (group1->target != group2->target) {
      return;
    }

    // Merge group 2 to group 1 and erase group 2.
    // FIXME (@comaniac): We do not update the "out" in this function and
    // let caller deal with it for simplify.
    group1->nodes.insert(group2->nodes.begin(), group2->nodes.end());
    group1->blocks.insert(group2->blocks.begin(), group2->blocks.end());
    this->groups_.erase(group2);
  }

  void AddToAnnotateGroup(std::shared_ptr<AnnotateGroup> group, const Expr expr) {
    CHECK(!GetAnnotateGroup(expr)); // The expr cannot be in the other group.
    group->nodes.insert(expr);
    group->out = expr;
  }

  void PrintGroups() {
    for (auto group : this->groups_) {
      std::cerr << "Group " << group->id << std::endl;
      std::cerr << "\tNodes #: " << group->nodes.size() << std::endl;
      std::cerr << "\tOut: " << AsText(group->out, false) << std::endl;
    }
  }

  std::unordered_set<std::shared_ptr<AnnotateGroup>> GetGroups() {
    std::unordered_set<std::shared_ptr<AnnotateGroup>> groups;
    for (auto group : this->groups_) {
      if (group->target != "default") {
        groups.insert(group);
      }
    }
    return groups;
  }

  void JoinArgAnnotateGroup(const Array<Expr> args,
                            std::vector<std::string> supported_targets,
                            Expr node_expr) {
    // Try to join groups that arguments belong to.
    std::shared_ptr<AnnotateGroup> joined_group = nullptr;
    std::unordered_set<int32_t> blocked_ids;

    // Traverse arguments.
    std::unordered_set<std::shared_ptr<AnnotateGroup>> arg_groups;
    for (const auto& it : args) {
      VisitExpr(it);
      auto group = GetAnnotateGroup(it);
      if (group) {
        arg_groups.insert(group);
      }
    }

    // Check for blocked groups.
    for (auto group : arg_groups) {
      for (auto rid : group->blocks) {
        blocked_ids.insert(rid);
      }
    }

    // Determine the first highest priority target group that does not be blocked.
    int curr_priority = supported_targets.size();
    for (auto group : arg_groups) {
      if (blocked_ids.find(group->id) != blocked_ids.end()) {
        continue;
      }
      auto it = std::find(supported_targets.begin(), supported_targets.end(), group->target);
      if (it != supported_targets.cend()) {
        auto priority = std::distance(supported_targets.begin(), it);
        if (priority < curr_priority) {
          joined_group = group;
          curr_priority = priority;
        }
      }
    }

    if (joined_group && (joined_group->target != "default" || supported_targets.size() == 1)) {
      // Join the group.
      AddToAnnotateGroup(joined_group, node_expr);
    } else {
      // Failed to join any argument groups. Create a new one.
      joined_group = CreateAnnotateGroup(node_expr, supported_targets);

      // Inherit block IDs from arguments.
      joined_group->blocks.insert(blocked_ids.begin(), blocked_ids.end());
    }

    // Merge other non-blocked groups with the same target, or block other groups from re-joining.
    for (auto group : arg_groups) {
      if (joined_group->id == group->id) {
        continue;
      } else if (joined_group->target == group->target &&
                 blocked_ids.find(group->id) == blocked_ids.end()) {
        MergeAnnotateGroup(joined_group, group);
      } else {
        joined_group->blocks.insert(group->id);
      }
    }
  }

  void VisitExpr_(const CallNode* call) {
    Op op = Downcast<Op>(call->op);
    CHECK(op.defined());

    // Supported targets for this node. The order implies the priority.
    std::vector<std::string> supported_targets;

    // Check which targets this op can be offloaded.
    for (auto target : this->targets_) {
      static auto fannotate = Op::GetAttr<FTVMAnnotateTarget>("target." + target);
      if (fannotate.count(op) && fannotate[op](call->attrs, call->args)) {
        supported_targets.push_back(target);
      }
    }
    if (supported_targets.empty()) {
      LOG(WARNING) << op->name << " is not registered in any targets. It will be executed on CPU.";
    }
    supported_targets.push_back("default"); // Make default as the last option.
    JoinArgAnnotateGroup(call->args, supported_targets, GetRef<Call>(call));
  }

  void VisitExpr_(const TupleNode* op) final {
    JoinArgAnnotateGroup(op->fields, {"default"}, GetRef<Tuple>(op));
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    JoinArgAnnotateGroup(Array<Expr>({op->tuple}), {"default"}, GetRef<TupleGetItem>(op));
  }

  void VisitExpr_(const LetNode* op) final {
    std::vector<Expr> args = {op->var, op->value, op->body};
    JoinArgAnnotateGroup(Array<Expr>(args), {"default"}, GetRef<Let>(op));
  }

  void VisitExpr_(const IfNode* op) final {
    std::vector<Expr> args = {op->cond, op->true_branch, op->false_branch};
    JoinArgAnnotateGroup(args, {"default"}, GetRef<If>(op));
  }

  void VisitExpr_(const RefCreateNode* op) final {
    JoinArgAnnotateGroup(Array<Expr>({op->value}), {"default"}, GetRef<RefCreate>(op));
  }

  void VisitExpr_(const RefReadNode* op) final {
    JoinArgAnnotateGroup(Array<Expr>({op->ref}), {"default"}, GetRef<RefRead>(op));
  }

  void VisitExpr_(const RefWriteNode* op) final {
    std::vector<Expr> args = {op->ref, op->value};
    JoinArgAnnotateGroup(Array<Expr>(args), {"default"}, GetRef<RefWrite>(op));
  }

 private:
  int curr_group_id_ = 0; 
  std::vector<std::string> targets_;
  std::unordered_set<std::shared_ptr<AnnotateGroup>> groups_;
};

class AnnotateTargetWithMergeWrapper : public ExprMutator {
 public:
  explicit AnnotateTargetWithMergeWrapper(std::unordered_set<std::shared_ptr<AnnotateGroup>> groups)
      : groups_(groups) {}


  Expr VisitExpr_(const CallNode* call) {
    // TODO(@zhiics, @comaniac) Handle composite functions.

    tvm::Array<tvm::relay::Expr> new_args;
    auto group = GetAnnotateGroup(GetRef<Call>(call));
    for (const auto& it : call->args) {
      auto new_arg = VisitExpr(it);
      auto arg_group = GetAnnotateGroup(it);

      // Insert compiler_end to close the group that the argument belongs to if
      // the argument belongs to a different group.
      if (arg_group && (!group || arg_group->id != group->id)) {
        const auto* end_op = runtime::Registry::Get("relay.op.annotation._make.compiler_end");
        CHECK(end_op);
        new_arg = (*end_op)(new_arg, arg_group->target);
      }

      // Insert compiler_begin to start this group if the argument belongs to a different group.
      if (group) {
        if (!arg_group || arg_group->id != group->id) {
          const auto* begin_op = runtime::Registry::Get("relay.op.annotation._make.compiler_begin");
          CHECK(begin_op);
          new_arg = (*begin_op)(new_arg, group->target);
        }
      }
      new_args.push_back(new_arg);
    }
    return CallNode::make(call->op, new_args, call->attrs);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    // Insert compiler_end to the last node of this function if it belongs to a group.
    auto group = GetAnnotateGroup(op->body);
    if (group) {
      Array<Var> params;
      for (auto param : op->params) {
        Var new_param = Downcast<Var>(VisitExpr(param));
        params.push_back(new_param);
      }
      auto body = VisitExpr(op->body);
      Call last_call = Downcast<Call>(body);
      Expr update_call = CallNode::make(last_call->op, last_call->args, last_call->attrs);
      const auto* end_op = runtime::Registry::Get("relay.op.annotation._make.compiler_end");
      CHECK(end_op);
      Expr new_body = (*end_op)(update_call, group->target);
      return Function(params, new_body, op->ret_type, op->type_params, op->attrs);
    }
    return ExprMutator::VisitExpr_(op);
  }

 private:
  // TODO(@comaniac): Move this function to a common place to share with AnalyzeAnnotateTargets.
  std::shared_ptr<AnnotateGroup> GetAnnotateGroup(const Expr node) {
    for (auto candidate : this->groups_) {
      if (candidate->nodes.find(node) != candidate->nodes.end()) {
        return candidate;
      }
    }
    return nullptr;
  }

  std::unordered_set<std::shared_ptr<AnnotateGroup>> groups_;
};

Expr AnnotateTargetWithMerge(const Expr& expr, Array<tvm::PrimExpr> targets) {
  auto analyzer = AnalyzeAnnotateTargets(targets);
  analyzer.VisitExpr(expr);
  return AnnotateTargetWithMergeWrapper(analyzer.GetGroups()).Mutate(expr);
}

}  // namespace annotate_target_with_merge

namespace transform {

Pass AnnotateTargetWithMerge(Array<tvm::PrimExpr> targets) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::annotate_target_with_merge::AnnotateTargetWithMerge(f, targets));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "AnnotateTargetWithMergeFunc",
                                      {tir::StringImmNode::make("InferType")});
  return transform::Sequential({func_pass, InferType()}, "AnnotateTargetWithMerge");
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateTargetWithMerge")
.set_body_typed(AnnotateTargetWithMerge);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
