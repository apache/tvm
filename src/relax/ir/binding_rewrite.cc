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
 * \file src/relax/ir/binding_rewrite.cc
 * \brief Implementation of binding rewriters.
 */

#include <tvm/relax/binding_rewrite.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <functional>
#include <iterator>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(DataflowBlockRewriteNode);
DataflowBlockRewrite::DataflowBlockRewrite(DataflowBlock dfb, Function root_fn) {
  auto n = make_object<DataflowBlockRewriteNode>();
  n->dfb_ = dfb;
  n->root_fn_ = root_fn;
  n->original_fn_ptr_ = root_fn.get();
  auto p = FunctionUseDef(root_fn);
  n->to_users_ = std::move(p.first);
  n->fn_outputs_ = std::move(p.second);
  n->name_table_ = NameTable(n->to_users_.begin(), n->to_users_.end(),
                             [](const auto& p) { return p.first->name_hint(); });

  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowBlockRewrite")
    .set_body_typed([](DataflowBlock dfb, Function root_fn) {
      return DataflowBlockRewrite(dfb, root_fn);
    });

void DataflowBlockRewriteNode::ReplaceAllUses(Var old_var, Var new_var) {
  class ReplaceAllUsePass : public ExprMutator {
    Var old_var, new_var;
    const DataflowBlockNode* const to_catch;

   public:
    const DataflowBlockNode* caught = nullptr;

    ReplaceAllUsePass(Var old_var, Var new_var, const DataflowBlockNode* to_catch)
        : old_var(old_var), new_var(new_var), to_catch(to_catch) {}

    using ExprMutator::VisitExpr_;

    Expr VisitExpr_(const VarNode* op) override {
      return (op == old_var.get()) ? new_var : GetRef<Expr>(op);
    }

    Expr VisitExpr_(const DataflowVarNode* op) override {
      return (op == old_var.get()) ? new_var : GetRef<Expr>(op);
    }

    BindingBlock VisitBindingBlock_(const DataflowBlockNode* op) override {
      BindingBlock res = ExprMutator::VisitBindingBlock_(op);
      if (op == to_catch) caught = static_cast<const DataflowBlockNode*>(res.get());
      return res;
    }
  };

  ICHECK(to_users_.find(old_var) != to_users_.end()) << "Cannot find " << old_var;
  ICHECK(to_users_.find(new_var) != to_users_.end()) << "Cannot find " << new_var;

  // replace uses inside the DataflowBlock.
  ReplaceAllUsePass replacer(old_var, new_var, dfb_.get());
  root_fn_ = Downcast<Function>(replacer.VisitExpr_(root_fn_.get()));
  dfb_ = GetRef<DataflowBlock>(replacer.caught);

  // update udchain
  // old_var -> old_var users | changed to {}
  // new_var -> {?}           | changed to old_var users
  for (Var user : to_users_[old_var]) {
    auto new_var_uses = to_users_[new_var];
    if (new_var_uses.end() == std::find(new_var_uses.begin(), new_var_uses.end(), user)) {
      new_var_uses.push_back(user);
    }
  }

  to_users_.Set(old_var, {});

  auto it_old_output = std::find(fn_outputs_.begin(), fn_outputs_.end(), old_var);
  if (it_old_output != fn_outputs_.end()) {
    fn_outputs_.Set(std::distance(fn_outputs_.begin(), it_old_output), new_var);
  }
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_replace_all_uses")
    .set_body_typed([](DataflowBlockRewrite rwt, Var old_var, Var new_var) {
      rwt->ReplaceAllUses(old_var, new_var);
    });

class UpdateDFB : public ExprMutator {
 private:
  DataflowBlock old_dfb, new_dfb;

 public:
  UpdateDFB(DataflowBlock old_dfb, DataflowBlock new_dfb)
      : old_dfb(std::move(old_dfb)), new_dfb(std::move(new_dfb)) {}

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* op) override {
    return old_dfb.get() == op ? new_dfb : old_dfb;
  }
};

void DataflowBlockRewriteNode::Add(Binding binding) {
  auto p = [binding] {
    if (auto vb = binding.as<VarBindingNode>()) {
      return std::make_pair(vb->var, vb->value);
    } else if (auto mc = binding.as<MatchCastNode>()) {
      return std::make_pair(mc->var, mc->value);
    }
    LOG(FATAL) << "Unsupported binding type";
    return std::make_pair(Var{}, Expr{});
  }();
  Var var = p.first;
  Expr val = p.second;

  ICHECK(0 == to_users_.count(var)) << var << " has been defined so cannot be added.";

  // Add this VarBinding statement after the definition of uses.
  std::set<const VarNode*> used_vars = [val] {
    class UsedVars : public ExprVisitor {
     public:
      std::set<const VarNode*> used_vars;
      void VisitExpr_(const VarNode* op) override { used_vars.insert(op); }
      void VisitExpr_(const DataflowVarNode* op) override { used_vars.insert(op); }
    } uvar{};
    uvar.VisitExpr(val);
    return std::move(uvar.used_vars);
  }();

  size_t line_last_req_def = 0;
  for (size_t i = 0; i < dfb_.value()->bindings.size(); ++i) {
    auto line = dfb_.value()->bindings[i];
    if (used_vars.find(line->var.get()) != used_vars.cend()) line_last_req_def = i;
  }

  auto old_dfb = dfb_.value();

  dfb_ = [old_dfb, binding, line_last_req_def, this] {
    auto new_dfb = dfb_.value();
    new_dfb.CopyOnWrite()->bindings.insert(dfb_.value()->bindings.begin() + 1 + line_last_req_def,
                                           binding);
    return new_dfb;
  }();

  auto updater = UpdateDFB(old_dfb, dfb_.value());
  root_fn_ = Downcast<Function>(updater.VisitExpr_(root_fn_.get()));

  for (const VarNode* v : used_vars) to_users_.Get(GetRef<Var>(v)).value().push_back(var);
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_add_binding")
    .set_body_typed([](DataflowBlockRewrite rwt, Binding vb) { rwt->Add(vb); });

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_add")
    .set_body_typed([](DataflowBlockRewrite rwt, Expr expr, Optional<String> name, bool is_dfvar) {
      if (name.get()) {
        rwt->Add(name.value(), expr, is_dfvar);
      } else {
        rwt->Add(expr, is_dfvar);
      }
    });

class RemoveUnusedVars : public ExprMutator {
 public:
  std::set<Var> unused_vars;
  Optional<DataflowBlock> caught_rewrite = NullOpt;

  RemoveUnusedVars(Map<Var, Array<Var>> users, Array<Var> fn_outputs)
      : unused_vars([&] {
          std::vector<Var> unused;

          // iterative dataflow algorithm.
          size_t prev_size;
          do {
            prev_size = unused.size();

            std::vector<Var> used;
            used.reserve(users.size());
            for (const auto& kv : users) {
              // var -> [users...]
              // var is unused iff
              //   user -> empty
              //   var is not output var
              if (kv.second.empty() &&  // kv.first is not used by fn outputs.
                  fn_outputs.end() == std::find(fn_outputs.begin(), fn_outputs.end(), kv.first)) {
                unused.push_back(kv.first);
              } else {
                used.push_back(kv.first);
              }
            }

            for (size_t i = prev_size; i < unused.size(); ++i) {
              users.erase(unused[i]);
              // remove def site.
              for (const auto& used_var : used) {
                ICHECK(users.count(used_var));
                Array<Var> var_users = users[used_var];
                // remove the unused var from the use site.
                auto it = std::find(var_users.begin(), var_users.end(), unused[i]);
                if (it != var_users.end()) {
                  var_users.erase(it);
                  users.Set(used_var, std::move(var_users));
                }
              }
            }
          } while (prev_size != unused.size());  // changed? => continue.

          return std::set<Var>(unused.begin(), unused.end());
        }()) {}

  RemoveUnusedVars(std::pair<Map<Var, Array<Var>>, Array<Var>> users_and_outputs)
      : RemoveUnusedVars(std::move(users_and_outputs.first), std::move(users_and_outputs.second)) {}
  RemoveUnusedVars(Function fn) : RemoveUnusedVars(FunctionUseDef(fn)) {}
  RemoveUnusedVars(std::set<Var> unused_vars) : unused_vars(std::move(unused_vars)) {}

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    auto prev_dfb = GetRef<DataflowBlock>(block);
    builder_->BeginDataflowBlock();
    for (Binding binding : block->bindings) {
      if (!unused_vars.count(binding->var)) {
        VisitBinding(binding);
      }
    }
    auto new_dfb = builder_->EndBlock();
    if (caught_rewrite == prev_dfb) caught_rewrite = Downcast<DataflowBlock>(new_dfb);
    return std::move(new_dfb);
  }
};

void DataflowBlockRewriteNode::RemoveUnused(Var unused, bool allow_undef) {
  // first need to check if this var is used.
  if (0 == to_users_.count(unused)) {  // no def.
    if (allow_undef) return;
    LOG(FATAL) << unused << " undefined. Set allow_undef=True to allow 'removing' undefined var";
  }

  ICHECK(to_users_[unused].empty())
      << unused << " is used by " << to_users_[unused].size() << " vars";

  auto old_dfb = dfb_.value();

  RemoveUnusedVars remover({unused});
  dfb_ = Downcast<DataflowBlock>(remover.VisitBindingBlock_(old_dfb.get()));

  auto updater = UpdateDFB(old_dfb, dfb_.value());
  root_fn_ = Downcast<Function>(updater.VisitExpr_(root_fn_.get()));

  to_users_.erase(unused);  // update use-def chain.
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_remove_unused")
    .set_body_typed([](DataflowBlockRewrite rwt, Var unused, bool allow_undef) {
      rwt->RemoveUnused(unused, allow_undef);
    });

void DataflowBlockRewriteNode::RemoveAllUnused() {
  RemoveUnusedVars remover(to_users_, fn_outputs_);
  remover.caught_rewrite = dfb_.value();

  // this could also clean unused variables in other DataflowBlock.
  root_fn_ = Downcast<Function>(remover.VisitExpr_(root_fn_.get()));

  // DataflowBlock could be None.
  dfb_ = remover.caught_rewrite.value();

  // clean up use-def chain.
  for (const auto& unused : remover.unused_vars) to_users_.erase(unused);
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_remove_all_unused")
    .set_body_typed([](DataflowBlockRewrite rwt) { rwt->RemoveAllUnused(); });

Function RemoveAllUnused(Function fn) {
  RemoveUnusedVars remover(fn);
  return Downcast<Function>(remover.VisitExpr_(fn.get()));
}

TVM_REGISTER_GLOBAL("relax.analysis.remove_all_unused").set_body_typed(RemoveAllUnused);

IRModule DataflowBlockRewriteNode::MutateIRModule(IRModule irmod) {
  BlockBuilder builder = BlockBuilder::Create(irmod);

  for (auto& p : irmod->functions) {
    if (original_fn_ptr_ == p.second.get()) {
      builder->UpdateFunction(p.first, root_fn_.value());
      break;
    }
  }

  return builder->GetContextIRModule();
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_mutate_irmodule")
    .set_body_typed([](DataflowBlockRewrite rwt, IRModule irmod) {
      return rwt->MutateIRModule(irmod);
    });

}  // namespace relax
}  // namespace tvm
