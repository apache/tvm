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
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_pass.h>
#include <unordered_map>


namespace tvm {
namespace ir {

// Mutator to change the read pattern
class ThreadAxisRewriter : private StmtExprMutator {
 public:
  explicit ThreadAxisRewriter(
      const std::unordered_map<std::string, IterVar>& tmap)
      : tmap_(tmap) {
  }

  Stmt Rewrite(Stmt stmt) {
    return operator()(std::move(stmt));
  }

 private:
  Stmt VisitStmt_(const AttrStmt* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK_NE(iv->thread_tag.length(), 0U);
      auto it = tmap_.find(iv->thread_tag);
      if (it != tmap_.end()) {
        const IterVar& new_iv = it->second;
        const Variable* v = iv->var.get();
        if (!vmap_.count(v)) {
          vmap_[v] = new_iv->var;
        } else {
          CHECK(vmap_[v].same_as(new_iv->var));
        }
        Stmt body = this->VisitStmt(op->body);
        return AttrStmt::make(
            new_iv, op->attr_key, op->value, body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Expr VisitExpr_(const Variable* op) final {
    auto it = vmap_.find(op);
    if (it != vmap_.end()) return it->second;
    return StmtExprMutator::VisitExpr_(op);
  }
  // The thread map
  const std::unordered_map<std::string, IterVar>& tmap_;
  // variable map
  std::unordered_map<const Variable*, Var> vmap_;
};

LoweredFunc
RemapThreadAxis(LoweredFunc f, Map<Expr, IterVar> thread_map) {
  std::unordered_map<std::string, IterVar> tmap;
  for (const auto& kv : thread_map) {
    const StringImm* str = kv.first.as<StringImm>();
    CHECK(str != nullptr);
    tmap[str->value] = kv.second;
  }

  CHECK_EQ(f->func_type, kDeviceFunc);
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  // replace the thread axis
  for (size_t i = 0; i < n->thread_axis.size(); ++i) {
    auto it = tmap.find(n->thread_axis[i]->thread_tag);
    if (it != tmap.end()) {
      n->thread_axis.Set(i, it->second);
    }
  }
  n->body = ThreadAxisRewriter(tmap).Rewrite(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
