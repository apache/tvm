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
 *  Copyright (c) 2019 by Contributors
 * \file verify_compact_buffer.cc
 * \brief Verify if there was any compact buffer bound to a statement.
 */
#include <tvm/buffer.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/tensor.h>

#include <unordered_map>

namespace tvm {
namespace ir {

class VerifyBuffer : public IRVisitor {
 public:
  bool Verify(const Stmt& stmt) {
    this->Visit(stmt);
    return is_compact_;
  }

  void Visit_(const AttrStmt* op) final {
    IRVisitor::Visit_(op);
    if (op->attr_key == attr::buffer_bind_scope) {
      is_compact_ = true;
    }
  }

 private:
  bool is_compact_{false};
};

bool VerifyCompactBuffer(Stmt stmt) {
  VerifyBuffer verifier;
  return verifier.Verify(stmt);
}

}  // namespace ir
}  // namespace tvm
