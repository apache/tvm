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
 * \file src/relax/transform/kill_after_last_use.cc
 * \brief Kill storage/tensor objects after last use, if not already killed
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <set>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

class UnusedTrivialBindingRemover : public ExprMutator {
 public:
  static Expr Apply(Expr expr) {
    struct UsedCollector : ExprVisitor {
      void VisitExpr_(const VarNode* val) override { used.insert(val); }
      void VisitExpr_(const DataflowVarNode* val) override {
        VisitExpr_(static_cast<const VarNode*>(val));
      }

      void VisitBinding_(const VarBindingNode* binding, const VarNode* val) override {
        has_trivial_binding.insert(binding->var.get());
        ExprVisitor::VisitBinding_(binding, val);
      }
      void VisitBinding_(const MatchCastNode* binding) override {
        if (binding->value.as<VarNode>() &&
            StructuralEqual()(GetStructInfo(binding->var), GetStructInfo(binding->value))) {
          has_trivial_binding.insert(binding->var.get());
        }
        ExprVisitor::VisitBinding_(binding);
      }
      void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* val) override {
        VisitBinding_(binding, static_cast<const VarNode*>(val));
      }

      std::unordered_set<const VarNode*> used;
      std::unordered_set<const VarNode*> has_trivial_binding;
    };

    UsedCollector collector;
    collector(expr);

    auto to_remove = std::move(collector.has_trivial_binding);
    for (const auto& used : collector.used) {
      to_remove.erase(used);
    }

    UnusedTrivialBindingRemover remover(to_remove);
    return remover(expr);
  }

 private:
  explicit UnusedTrivialBindingRemover(std::unordered_set<const VarNode*> to_remove)
      : to_remove_(std::move(to_remove)) {}

  void VisitBinding(const Binding& binding) override {
    if (!to_remove_.count(binding->var.get())) {
      ExprMutator::VisitBinding(binding);
    }
  }

  std::unordered_set<const VarNode*> to_remove_;
};

class CollectLastUsage : public ExprVisitor {
 public:
  struct LastUsage {
    std::vector<const VarNode*> tensors;
    std::vector<const VarNode*> storage;
    std::vector<const VarNode*> objects;
  };
  using Result = std::unordered_map<const VarNode*, LastUsage>;

  static Result Collect(const Expr& expr) {
    CollectLastUsage visitor;
    visitor(expr);

    Result output;
    for (const auto* var : visitor.binding_order_) {
      if (auto it = visitor.last_usage_of_.find(var); it != visitor.last_usage_of_.end()) {
        const auto* last_usage_point = it->second;
        bool is_output = last_usage_point == nullptr;
        bool already_killed = visitor.killed_objects_.count(var);

        // Currently, the VM requires that objects to be killed
        // objects only exist in VM registers.  This requires
        // KillAfterLastUse to have more knowledge about the VM
        // implementation than should exist at this stage of lowering.
        // In the future, this may be handled more easily at the
        // CodeGenVM level.
        bool stored_in_vm_register =
            !(visitor.constant_tensors_.count(var) || var->struct_info_.as<FuncStructInfoNode>() ||
              var->struct_info_.as<ShapeStructInfoNode>() ||
              var->struct_info_.as<PrimStructInfoNode>());

        if (!is_output && !already_killed) {
          if (visitor.storage_objects_.count(var)) {
            output[last_usage_point].storage.push_back(var);
          } else if (var->struct_info_.as<TensorStructInfoNode>() && stored_in_vm_register) {
            output[last_usage_point].tensors.push_back(var);
          } else if (stored_in_vm_register) {
            output[last_usage_point].objects.push_back(var);
          }
        }
      }
    }

    return output;
  }

  void VisitBinding(const Binding& binding) override {
    auto cache = current_binding_;
    current_binding_ = binding->var.get();
    binding_order_.push_back(current_binding_);
    ExprVisitor::VisitBinding(binding);
    current_binding_ = cache;
  }

  void VisitExpr_(const VarNode* op) override {
    ExprVisitor::VisitExpr_(op);
    // Overwrite any previous usage, such that after the visitor
    // completes, last_usage_of_ contains the last usage point.  If
    // this occurs in an output, then current_binding_ will be
    // nullptr.
    last_usage_of_[op] = current_binding_;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) override {
    static const Op& vm_alloc_storage = Op::Get("relax.vm.alloc_storage");
    static const Op& mem_alloc_storage = Op::Get("relax.memory.alloc_storage");

    static const Op& mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
    static const Op& mem_kill_storage = Op::Get("relax.memory.kill_storage");
    static const Op& vm_kill_object = Op::Get("relax.vm.kill_object");

    if (val->op.same_as(vm_alloc_storage) || val->op.same_as(mem_alloc_storage)) {
      storage_objects_.insert(binding->var.get());
    } else if (val->op.same_as(mem_kill_tensor) || val->op.same_as(mem_kill_storage) ||
               val->op.same_as(vm_kill_object)) {
      CHECK_EQ(val->args.size(), 1)
          << "Operator " << val->op << " should have one argument, "
          << "but instead found " << val->args.size() << " arguments: " << val->args;
      auto killed_object = val->args[0].as<VarNode>();
      ICHECK(killed_object) << "Internal error: non-normalized expression " << GetRef<Call>(val);
      killed_objects_.insert(killed_object);
    } else {
      // Only recursively visit if it isn't one of the special cases.
      ExprVisitor::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val) override {
    constant_tensors_.insert(binding->var.get());
  }

 private:
  // The current binding being visited, or nullptr if no binding is
  // being visited.
  const VarNode* current_binding_{nullptr};

  // Order of bindings, to ensure consistent order of destruction, in
  // case a Binding is the last usage for more than one variable.
  std::vector<const VarNode*> binding_order_;

  // Map from a variable to the last variable binding that makes use
  // of it.
  std::unordered_map<const VarNode*, const VarNode*> last_usage_of_;

  // Storage objects, eligible for R.vm.kill_object.  This cannot be
  // determined solely from the StructInfo, because the
  // `R.*.alloc_storage` operators return ObjectStructInfo
  std::unordered_set<const VarNode*> storage_objects_;

  // Constants, which do not have a VM register, and may *not* have
  // R.builtin.kill_tensor called on them.
  std::unordered_set<const VarNode*> constant_tensors_;

  // Set of objects that already have a call node to kill them.  Should not have a duplicate
  std::unordered_set<const VarNode*> killed_objects_;

  // Trivial var-to-var bindings.
  std::unordered_map<const VarNode*, const VarNode*> trivial_bindings_;
};

class KillInserter : public ExprMutator {
 private:
  Expr VisitExpr_(const FunctionNode* op) override {
    last_usage_ = CollectLastUsage::Collect(GetRef<Expr>(op));
    auto mutated = ExprMutator::VisitExpr_(op);
    last_usage_.clear();
    return mutated;
  }

  Expr VisitExpr_(const SeqExprNode* op) override {
    last_usage_ = CollectLastUsage::Collect(GetRef<Expr>(op));
    auto mutated = ExprMutator::VisitExpr_(op);
    last_usage_.clear();
    return mutated;
  }

  void VisitBinding(const Binding& binding) override {
    ExprMutator::VisitBinding(binding);
    if (auto it = last_usage_.find(binding->var.get()); it != last_usage_.end()) {
      static const Op& mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
      for (const auto& tensor_obj : it->second.tensors) {
        builder_->Emit(Call(mem_kill_tensor, {GetRef<Expr>(tensor_obj)}), /*name_hint=*/"_");
      }

      static const Op& mem_kill_storage = Op::Get("relax.memory.kill_storage");
      for (const VarNode* storage_obj : it->second.storage) {
        builder_->Emit(Call(mem_kill_storage, {GetRef<Expr>(storage_obj)}), /*name_hint=*/"_");
      }

      static const Op& vm_kill_object = Op::Get("relax.vm.kill_object");
      for (const VarNode* obj : it->second.objects) {
        builder_->Emit(Call(vm_kill_object, {GetRef<Expr>(obj)}), /*name_hint=*/"_");
      }
    }
  }

  CollectLastUsage::Result last_usage_;
};

Expr KillAfterLastUse(Expr expr) {
  expr = CanonicalizeBindings(expr);
  expr = UnusedTrivialBindingRemover::Apply(expr);

  KillInserter mutator;
  return mutator(expr);
}

namespace transform {

Pass KillAfterLastUse() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function func, IRModule m, PassContext pc) {
        return Downcast<Function>(relax::KillAfterLastUse(std::move(func)));
      };
  return CreateFunctionPass(pass_func, /*opt_level=*/0, "KillAfterLastUse", {});
}

TVM_REGISTER_GLOBAL("relax.transform.KillAfterLastUse").set_body_typed(KillAfterLastUse);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
