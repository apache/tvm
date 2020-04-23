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

#include <tvm/tir/expr.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

class AssertSkipper : public StmtMutator {
 public:
  Stmt VisitStmt_(const AssertStmtNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AssertStmtNode>();
    return op->body;
  }
};

Stmt SkipAssert(Stmt stmt) {
  return AssertSkipper()(std::move(stmt));
}

namespace transform {

Pass SkipAssert() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = AssertSkipper()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SkipAssert", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SkipAssert")
.set_body_typed(SkipAssert);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
