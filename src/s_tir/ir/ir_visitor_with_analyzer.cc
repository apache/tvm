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

#include "ir_visitor_with_analyzer.h"

namespace tvm {
namespace s_tir {
using namespace tirx;
ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const SBlockNode* op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    for (const auto& iter_var : op->iter_vars) {
      analyzer_->Bind(iter_var->var, iter_var->dom);
    }
    return s_tir::StmtExprVisitor::VisitBlock(this, op);
  });
}

}  // namespace s_tir
}  // namespace tvm
