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

Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name, const String& func_name) {
  struct Finder : public StmtVisitor {
    explicit Finder(const ScheduleState& self, const String& name) : self_(self), name_(name) {}

    void VisitStmt_(const BlockNode* block) override {
      if (block->name_hint == name_) {
        auto it = self_->stmt2ref.find(block);
        ICHECK(it != self_->stmt2ref.end());
        results_.push_back(it->second);
      }
      StmtVisitor::VisitStmt_(block);
    }

    const ScheduleState& self_;
    const String& name_;
    Array<StmtSRef> results_;
  };

  BaseFunc func = self->mod->Lookup(func_name);
  const auto* prim_func = TVM_TYPE_AS(prim_func, func, PrimFuncNode);
  Finder finder(self, name);
  finder(prim_func->body);
  return std::move(finder.results_);
}

Array<StmtSRef> GetLoops(const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

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
