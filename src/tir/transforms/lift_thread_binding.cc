/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

std::pair<std::unordered_map<Stmt, std::vector<std::pair<IterVar, Map<String, ObjectRef>>>,
                             ObjectPtrHash, ObjectPtrEqual>,
          Map<Var, Var>>
FindLoopLCA(const Stmt& root) {
  class LCAFinder : public StmtVisitor {
   public:
    void VisitStmt_(const ForNode* op) final {
      stack.push_back(GetRef<Stmt>(op));
      StmtVisitor::VisitStmt_(op);
      if (op->kind == ForKind::kThreadBinding) {
        UpdateLCA(op);
      }
      stack.pop_back();
    }

    void UpdateLCA(const ForNode* loop) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      {
        Map<String, ObjectRef>* tgt = &annotations[thread_tag];
        for (const auto& kv : loop->annotations) {
          tgt->Set(kv.first, kv.second);
        }
      }
      IterVar& iter_var = iters[thread_tag];
      if (!iter_var.defined()) {
        iter_var = IterVar(Range::FromMinExtent(loop->min, loop->extent),  //
                           loop->loop_var.copy_with_name(thread_tag),      //
                           loop->thread_binding.value()->iter_type,        //
                           thread_tag);
        lca[thread_tag] = stack;
        var_subst.Set(loop->loop_var, iter_var->var);
        return;
      }
      var_subst.Set(loop->loop_var, iter_var->var);
      std::vector<Stmt>& path = lca[thread_tag];
      uint32_t i = 0;
      for (; i < stack.size() && i < path.size(); ++i) {
        if (!stack[i].same_as(path[i])) {
          break;
        }
      }
      path.resize(i);
    }

    std::unordered_map<std::string, std::vector<Stmt>> lca;
    std::unordered_map<std::string, IterVar> iters;
    std::unordered_map<std::string, Map<String, ObjectRef>> annotations;
    Map<Var, Var> var_subst;
    std::vector<Stmt> stack;
  };
  LCAFinder finder;
  finder(root);
  std::unordered_map<Stmt, std::vector<std::pair<IterVar, Map<String, ObjectRef>>>, ObjectPtrHash,
                     ObjectPtrEqual>
      result;
  std::vector<std::string> sorted_thread_tags;
  for (const auto& kv : finder.lca) {
    sorted_thread_tags.push_back(kv.first);
  }
  std::sort(sorted_thread_tags.begin(), sorted_thread_tags.end(),
            [](const std::string& lhs, const std::string& rhs) {
              return lhs.size() > rhs.size();
              runtime::ThreadScope lhs_scope = runtime::ThreadScope::Create(lhs);
              runtime::ThreadScope rhs_scope = runtime::ThreadScope::Create(rhs);
              if (lhs_scope.rank != rhs_scope.rank) {
                return lhs_scope.rank < rhs_scope.rank;
              }
              return lhs_scope.dim_index < rhs_scope.dim_index;
            });
  for (const auto& thread_tag : sorted_thread_tags) {
    Stmt lca = finder.lca[thread_tag].back();
    const IterVar& iter = finder.iters[thread_tag];
    const Map<String, ObjectRef>& annotations = finder.annotations[thread_tag];
    result[lca].emplace_back(iter, annotations);
  }
  return {result, finder.var_subst};
}

/*!
 * \brief Substitute expr via BlockRealize value bindings and convert each block into opaque
 *        blocks.
 */
class ThreadBindingLifter : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const ForNode* _op) final {
    For op = GetRef<For>(_op);
    bool is_kernel_root = false;
    if (op->kind == ForKind::kThreadBinding) {
      if (iter_lca.empty()) {
        is_kernel_root = true;
        SetKernelRoot(_op);
      }
    }
    For new_op = Downcast<For>(StmtExprMutator::VisitStmt_(_op));
    Stmt body = std::move(new_op.CopyOnWrite()->body);
    if (auto it = iter_lca.find(op); it != iter_lca.end()) {
      for (const auto& [iter_var, annotation] : it->second) {
        body = For(iter_var->var, iter_var->dom->min, iter_var->dom->extent,
                   ForKind::kThreadBinding, std::move(body),
                   IterVar(Range(nullptr), Var(iter_var->thread_tag, iter_var->var->dtype),
                           kThreadIndex, iter_var->thread_tag),
                   annotation);
      }
    }
    if (is_kernel_root) {
      iter_lca.clear();
      var_subst.clear();
    }
    if (op->kind == ForKind::kThreadBinding) {
      return body;
    } else {
      new_op.CopyOnWrite()->body = std::move(body);
      return new_op;
    }
  }

  void SetKernelRoot(const ForNode* op) {
    auto result = FindLoopLCA(GetRef<Stmt>(op));
    this->iter_lca = std::move(result.first);
    this->var_subst = std::move(result.second);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_subst.find(GetRef<Var>(op));
    if (it != var_subst.end()) {
      return (*it).second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  std::unordered_map<Stmt, std::vector<std::pair<IterVar, Map<String, ObjectRef>>>, ObjectPtrHash,
                     ObjectPtrEqual>
      iter_lca;
  Map<Var, Var> var_subst;
};

PrimFunc LiftThreadBinding(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = ThreadBindingLifter()(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LiftThreadBinding() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LiftThreadBinding(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LiftThreadBinding", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LiftThreadBinding").set_body_typed(LiftThreadBinding);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
