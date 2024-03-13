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
 * \file src/relax/transform/merge_composite_functions.cc
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 * function.
 *
 * The new function will be annotated with kCodegen and kGlobalSymbol attributes, and it is
 * intented to be offloaded to an external backend.
 *
 * A group for one composite function can be merged into another group for one of its arguments,
 * which we call the parent group for brevity, if the following conditions are met:
 * - The argument is the result of calling a composite function offloaded to the same backend
 * - Merging into the parent group would not create a cyclic dependency with other parent groups
 *
 * For example, in the subgraph below the bottom group cannot be merged into the left parent group,
 * since the right parent group for X depends on an output from the left parent group.
 *
 *  O = Offloaded to A
 *  X = Offloaded to B
 *
 * Correct partitioning:
 *
 *     O         O
 *    / \       /	            \
 *   O   X --> O    +     +    X
 *    \ /             \ /
 *     O               O
 *
 * The algorithm proceeds by assigning a group to each subexpression in the function according to
 * its dataflow. On encountering a call node whose callee is a composite function, we check the
 * two conditions above to see if we can merge this call node into one of its parent groups, and
 * if we can merge some of its parent groups.
 *
 * To detect cyclic dependencies between groups, we propagate dependency relations, both direct
 * and indirect ones, as we flow through the function. The propagation of indirect dependencies
 * is important since the dependency relation is transitive.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

#include "../../support/arena.h"
#include "utils.h"

namespace tvm {
namespace relax {

using relay::GraphPartitioner;

namespace {

using Group = GraphPartitioner::Group;

/*! \brief Assign group to each subexpression in a function according to its
 * dataflow, and returns a mapping from a subexpression to its group. */
class CompositeGroupsBuilder : public MemoizedExprTranslator<Group*> {
 public:
  using GroupMap = std::unordered_map<const Object*, Group*>;
  using MemoizedExprTranslator<Group*>::VisitExpr_;

  CompositeGroupsBuilder(IRModule mod, support::Arena* arena) : mod_(mod), arena_(arena) {}

  GroupMap Run(Function func) {
    for (const auto& param : func->params) {
      memo_[param] = arena_->make<Group>();
    }

    PostOrderVisit(func, [this](Expr e) {
      // Make default groups for dataflow nodes other than CallNode.
      // Groups for CallNode are created in its visitor.
      if (e->IsInstance<ConstantNode>() || e->IsInstance<ShapeExprNode>() ||
          e->IsInstance<TupleNode>() || e->IsInstance<TupleGetItemNode>() ||
          e->IsInstance<PrimValueNode>()) {
        memo_[e] = arena_->make<Group>();
      }
    });

    VisitExpr(func->body);

    GroupMap group_map;
    for (const auto& [expr, group] : memo_) {
      group_map[expr.get()] = group->FindRoot();
    }

    return group_map;
  }

  Group* VisitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingNode>()) {
      return VisitBinding_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
    }
  }

  void VisitBindingBlock_(const BindingBlockNode* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
  }

  void VisitBindingBlock(const BindingBlock& block) {
    if (const auto* node = block.as<DataflowBlockNode>()) {
      VisitBindingBlock_(node);
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      VisitBindingBlock_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
    }
  }

  Group* VisitExpr_(const SeqExprNode* op) {
    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }
    return VisitExpr(op->body);
  }

  Group* VisitExpr_(const CallNode* call) {
    std::vector<Group*> groups_to_merge = GetGroupsToMerge(call);
    Group* group;

    if (groups_to_merge.size() == 0) {
      // Create new group if there is nothing to merge with
      group = CreateNewGroup(call);
    } else {
      auto it = groups_to_merge.cbegin();
      // Assign the first mergable group to current node
      // to reduce the number of groups created
      group = *it++;
      group->num_nodes += 1;

      // Merge all groups
      for (; it != groups_to_merge.cend(); ++it) {
        MergeGroup(*it, group);
      }
    }

    UpdateGroupDependencies(group, call->args);
    return group;
  }

 private:
  Optional<String> GetCodegenName(const Expr& callee) {
    auto const* gvar = callee.as<GlobalVarNode>();
    if (!gvar) {
      return NullOpt;
    }

    auto composite_name_opt =
        mod_->Lookup(GetRef<GlobalVar>(gvar))->GetAttr<String>(attr::kComposite);
    if (!composite_name_opt) {
      return NullOpt;
    }

    return relax::GetCodegenName(composite_name_opt.value());
  }

  Optional<String> GetCodegenName(Group* group) {
    return Downcast<Optional<String>>(group->attrs.Get(attr::kCodegen));
  }

  Group* CreateNewGroup(const CallNode* call) {
    Group* group = arena_->make<Group>();
    if (Optional<String> codegen_name = GetCodegenName(call->op)) {
      group->attrs.Set(attr::kCodegen, codegen_name.value());
    }
    return group;
  }

  void MergeGroup(Group* from, Group* to) {
    ICHECK_EQ(GetCodegenName(from), GetCodegenName(to));

    Group* from_root = from->FindRoot();
    Group* to_root = to->FindRoot();
    if (from_root == to_root) {
      return;
    }

    from_root->parent = to_root;
    to_root->num_nodes += from_root->num_nodes;

    // Update the group_deps_, maintaining the invariant that
    // all groups in the map are root groups.
    group_deps_[to_root].merge(group_deps_[from_root]);
    group_deps_.erase(from_root);
    for (auto& it : group_deps_) {
      if (it.second.count(from_root)) {
        it.second.erase(from_root);
        it.second.insert(to_root);
      }
    }
  }

  std::unordered_set<Group*> GetParentGroupDependencies(const Array<Expr>& args) {
    // Collect groups that parent groups depend on
    std::unordered_set<Group*> dependencies;

    for (const auto& arg : args) {
      for (auto dep : group_deps_[memo_[arg]->FindRoot()]) {
        dependencies.insert(dep);
      }
    }

    return dependencies;
  }

  void UpdateGroupDependencies(Group* group, const Array<Expr>& args) {
    Group* group_root = group->FindRoot();

    for (const auto& arg : args) {
      auto arg_group_root = memo_[arg]->FindRoot();
      if (arg_group_root == group_root) {
        // If arg and the current node are in the same group,
        // there is nothing to update.
        continue;
      }
      // Add the group of arg as dependency
      group_deps_[group_root].insert(arg_group_root);
      // Propagate dependencies of arg
      for (auto dep : group_deps_[arg_group_root]) {
        group_deps_[group_root].insert(dep);
      }
    }
  }

  std::vector<Group*> GetGroupsToMerge(const CallNode* call) {
    Optional<String> codegen_name = GetCodegenName(call->op);
    if (!codegen_name.defined()) {
      return {};
    }

    std::vector<Group*> groups_to_merge;
    std::unordered_set<Group*> parent_dependencies = GetParentGroupDependencies(call->args);

    for (const auto& arg : call->args) {
      auto arg_group = memo_[arg];
      Optional<String> arg_codegen_name = GetCodegenName(arg_group);
      if (arg_codegen_name == codegen_name && !parent_dependencies.count(arg_group->FindRoot())) {
        // If there is a parent group with the same target, which none of the parent dependency
        // groups depends on, merging "this" call node into the parent group will not form a cyclic
        // dependency.
        groups_to_merge.push_back(arg_group);
      }
    }

    return groups_to_merge;
  }

  IRModule mod_;
  support::Arena* arena_;
  // Map from group to its dependencies. All groups in this map, whether it's
  // the key or in value, should be root node (that is, group->parent == nullptr).
  std::unordered_map<Group*, std::unordered_set<Group*>> group_deps_;
};

/*! \brief Inline definitions of composite functions at the global level into their call sites.
  This is necessary to make functions created by MergeCompositeFunctions self-contained - each
  external backend compiler does not need to refer to the original containing module.
 */
class CompositeInliner : public ExprMutator {
 public:
  explicit CompositeInliner(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitExpr_;

  Function Run(Function func) {
    inlined_functions_ = Map<Function, Function>();
    auto new_body = VisitExpr(ToNonDataflow(func->body));
    auto new_func = Function(func->params, new_body, func->ret_struct_info, func->is_pure,
                             func->attrs, func->span);
    return new_func;
  }

  Expr VisitExpr_(const CallNode* call) {
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto gvar = Downcast<GlobalVar>(call->op);
      auto func = Downcast<Function>(mod_->Lookup(gvar));
      if (func->GetAttr<String>(attr::kComposite)) {
        if (!inlined_functions_.count(func)) {
          auto new_func = CopyWithNewVars(func);
          new_func = WithoutAttr(new_func, tvm::relax::attr::kPrimitive);
          inlined_functions_.Set(func, new_func);
        }
        return Call(inlined_functions_[func], call->args);
      }
    }

    return ExprMutator::VisitExpr_(call);
  }

 private:
  IRModule mod_;
  Map<Function, Function> inlined_functions_;
};

/*!
 * \brief Wrap each created composite function with another function, whose body consists
 * only of a call to the composite function, and annotate the outer function with kCodegen
 * and kGlobalSymbol attributes.
 */
class CompositeFunctionAnnotator : public ExprMutator {
 public:
  explicit CompositeFunctionAnnotator(IRModule mod, IRModule new_mod)
      : ExprMutator(new_mod), mod_(new_mod), inliner(mod) {
    mod_.CopyOnWrite();
  }
  using ExprMutator::VisitExpr_;

  IRModule update() {
    auto gvar = mod_->GetGlobalVar("main");
    auto func = Downcast<Function>(mod_->Lookup(gvar));
    builder_->UpdateFunction(gvar, Downcast<Function>(VisitExpr(func)));
    return builder_->GetContextIRModule();
  }

  Expr VisitExpr_(const CallNode* call) {
    if (call->op->IsInstance<GlobalVarNode>()) {
      GlobalVar cur_var = Downcast<GlobalVar>(call->op);
      auto func = Downcast<Function>(mod_->Lookup(cur_var));
      if (auto codegen_name = func->GetAttr<String>(attr::kCodegen)) {
        GlobalVar new_var;
        if (var_map_.count(cur_var) > 0) {
          // if we visited before, we don't need to create the new function,
          // use the one we stored.
          new_var = var_map_[cur_var];
        } else {
          // if it is first time, create the new function with a new name.
          // remove old function from the irmoulde under construction.
          auto old_var = builder_->GetContextIRModule()->GetGlobalVar(cur_var->name_hint);
          builder_->GetContextIRModule()->Remove(old_var);

          // rename the function.
          String new_func_name = cur_var->name_hint + "_" + codegen_name.value();
          Function new_func = inliner.Run(Downcast<Function>(func));
          new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol, new_func_name);
          new_func = WithoutAttr(std::move(new_func), tvm::relax::attr::kPrimitive);
          // add a function with a new name.
          new_var = builder_->AddFunction(new_func, new_func_name);
          var_map_[cur_var] = new_var;
        }
        // we call new var instead of the old one.
        // we don't have to update args since we are just updating the function to call,
        // without any change in the arguments.
        return Call(new_var, call->args);
      }
    }
    return GetRef<Call>(call);
  }

 private:
  IRModule mod_;
  CompositeInliner inliner;
  std::unordered_map<GlobalVar, GlobalVar, ObjectPtrHash, ObjectPtrEqual> var_map_;
};

}  // namespace

IRModule MergeCompositeFunctions(IRModule mod) {
  auto gvar = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(gvar));
  support::Arena arena;
  auto group_map = CompositeGroupsBuilder(mod, &arena).Run(func);
  auto new_mod = MakeGroupedFunctions(mod, group_map);
  new_mod = CompositeFunctionAnnotator(mod, new_mod).update();

  // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
  return DeadCodeElimination(new_mod, {"main"});
}

namespace transform {

Pass MergeCompositeFunctions() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule mod, PassContext pc) { return relax::MergeCompositeFunctions(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,              //
                          /*opt_level=*/0,                          //
                          /*pass_name=*/"MergeCompositeFunctions",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.MergeCompositeFunctions")
    .set_body_typed(MergeCompositeFunctions);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
