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
#include "../utils.h"

namespace tvm {
namespace tir {

/******** Block-loop relation ********/

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref) {
  struct Collector : public StmtVisitor {
   public:
    static Array<StmtSRef> Collect(const ScheduleState& self, const Stmt& stmt) {
      Collector collector(self);
      collector(stmt);
      return std::move(collector.result_);
    }

   private:
    explicit Collector(const ScheduleState& self) : self_(self) {}

    void VisitStmt_(const BlockNode* block) final {
      auto it = self_->stmt2ref.find(block);
      ICHECK(it != self_->stmt2ref.end());
      result_.push_back(it->second);
    }

    const ScheduleState& self_;
    Array<StmtSRef> result_;
  };

  if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    return Collector::Collect(self, loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    return Collector::Collect(self, block->body);
  }
  ICHECK(false) << "Unreachable";
  throw;
}

}  // namespace tir
}  // namespace tvm
