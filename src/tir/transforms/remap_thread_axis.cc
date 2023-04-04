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
 * \file remap_thread_axis.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

namespace tvm {
namespace tir {

// Mutator to change the read pattern
class ThreadAxisRewriter : private StmtExprMutator {
 public:
  explicit ThreadAxisRewriter(const std::unordered_map<std::string, IterVar>& tmap) : tmap_(tmap) {}

  Stmt Rewrite(Stmt stmt) { return operator()(std::move(stmt)); }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      auto it = tmap_.find(iv->thread_tag);
      if (it != tmap_.end()) {
        const IterVar& new_iv = it->second;
        const VarNode* v = iv->var.get();
        if (!vmap_.count(v)) {
          vmap_[v] = new_iv->var;
        } else {
          ICHECK(vmap_[v].same_as(new_iv->var));
        }
        Stmt body = this->VisitStmt(op->body);
        return AttrStmt(new_iv, op->attr_key, op->value, body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = vmap_.find(op);
    if (it != vmap_.end()) return it->second;
    return StmtExprMutator::VisitExpr_(op);
  }
  // The thread map
  const std::unordered_map<std::string, IterVar>& tmap_;
  // variable map
  std::unordered_map<const VarNode*, Var> vmap_;
};

PrimFunc RemapThreadAxis(PrimFunc func, Map<runtime::String, IterVar> thread_map) {
  std::unordered_map<std::string, IterVar> tmap;
  for (const auto& kv : thread_map) {
    tmap[kv.first] = kv.second;
  }

  if (auto opt = func->GetAttr<Array<IterVar>>(tir::attr::kKernelLaunchParams)) {
    ICHECK(opt != nullptr) << "Require attribute " << tir::attr::kKernelLaunchParams;
    auto launch_params = opt.value();
    // replace the thread axis attribute
    for (size_t i = 0; i < launch_params.size(); ++i) {
      auto it = tmap.find(launch_params[i]->thread_tag);
      if (it != tmap.end()) {
        launch_params.Set(i, it->second);
      }
    }

    func = WithAttr(std::move(func), tir::attr::kKernelLaunchParams, launch_params);
  }

  auto* n = func.CopyOnWrite();
  n->body = ThreadAxisRewriter(tmap).Rewrite(std::move(n->body));
  return func;
}

namespace transform {

Pass RemapThreadAxis(Map<runtime::String, IterVar> thread_map) {
  auto pass_func = [thread_map](PrimFunc f, IRModule m, PassContext ctx) {
    return RemapThreadAxis(std::move(f), thread_map);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemapThreadAxis", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemapThreadAxis").set_body_typed(RemapThreadAxis);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
