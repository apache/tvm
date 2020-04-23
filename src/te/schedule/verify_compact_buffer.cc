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
 * \file verify_compact_buffer.cc
 * \brief Verify if there was any compact buffer bound to a statement.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/tensor.h>
#include <tvm/te/schedule_pass.h>

#include <unordered_map>

namespace tvm {
namespace te {

class VerifyBuffer : public StmtVisitor {
 public:
  bool Verify(const Stmt& stmt) {
    this->VisitStmt(stmt);
    return is_compact_;
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    if (op->attr_key == tir::attr::buffer_bind_scope) {
      is_compact_ = true;
    }
  }

 private:
  bool is_compact_{false};
};

bool VerifyCompactBuffer(const Stmt& stmt) {
  VerifyBuffer verifier;
  return verifier.Verify(stmt);
}

TVM_REGISTER_GLOBAL("schedule.VerifyCompactBuffer")
.set_body_typed(VerifyCompactBuffer);

}  // namespace te
}  // namespace tvm
