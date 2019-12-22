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

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace ir {

class AssertSkipper : public IRMutator {
 public:
  Stmt Mutate_(const AssertStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AssertStmt>();
    return op->body;
  }
};

Stmt SkipAssert(Stmt stmt) {
  return AssertSkipper().Mutate(stmt);
}

LoweredFunc SkipAssert(LoweredFunc f) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = SkipAssert(f->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
