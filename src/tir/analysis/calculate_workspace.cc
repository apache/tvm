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
 * \file tir/analysis/calculate_workspace.cc
 * \brief Calculate any intermediary memory required by PrimFuncs.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

class WorkspaceCalculator : public StmtExprVisitor {
 public:
  WorkspaceCalculator() = default;
  int operator()(const PrimFunc& func);

 private:
  void VisitStmt_(const AllocateNode* op) override;
  int CalculateExtentsSize(const AllocateNode* op);
  int current_size = 0;
  int max_size = 0;
};

int WorkspaceCalculator::operator()(const PrimFunc& func) {
  this->VisitStmt(func->body);
  return this->max_size;
}

int WorkspaceCalculator::CalculateExtentsSize(const AllocateNode* op) {
  int element_size_bytes = op->dtype.bytes();
  int num_elements = 1;
  for (const auto& ext : op->extents) {
    num_elements *= Downcast<IntImm>(ext)->value;
  }
  return num_elements * element_size_bytes;
}

void WorkspaceCalculator::VisitStmt_(const AllocateNode* op) {
  auto size = CalculateExtentsSize(op);
  current_size += size;
  if (current_size > max_size) {
    max_size = current_size;
  }
  StmtExprVisitor::VisitStmt(op->body);
  current_size -= size;
}

int CalculateWorkspaceBytes(const PrimFunc& func) {
  WorkspaceCalculator wc;
  return wc(func);
}

TVM_REGISTER_GLOBAL("tir.analysis.calculate_workspace_bytes")
    .set_body_typed(CalculateWorkspaceBytes);

}  // namespace tir
}  // namespace tvm
