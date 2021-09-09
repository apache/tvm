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

/******** InstructionKind Registration ********/

struct GetBlockTraits : public UnpackedInstTraits<GetBlockTraits> {
  static constexpr const char* kName = "GetBlock";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, String name, String func_name) {
    return sch->GetBlock(name, func_name);
  }

  static String UnpackedAsPython(Array<String> outputs, String name, String func_name) {
    PythonAPICall py("get_block");
    py.Input("name", name);
    py.Input("func_name", func_name);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct GetLoopsTraits : public UnpackedInstTraits<GetLoopsTraits> {
  static constexpr const char* kName = "GetLoops";
  static constexpr bool kIsPure = true;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static Array<LoopRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv) {
    return sch->GetLoops(block_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv) {
    PythonAPICall py("get_loops");
    py.Input("block", block_rv);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(GetBlockTraits);
TVM_REGISTER_INST_KIND_TRAITS(GetLoopsTraits);

}  // namespace tir
}  // namespace tvm
