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

/*! \brief Check if an IRModule has any dynamic loop. */
struct DynamicExtentFinder : private StmtVisitor {
 public:
  static bool Find(const IRModule& mod) {
    DynamicExtentFinder finder;
    for (const auto& kv : mod->functions) {
      const BaseFunc& func = kv.second;
      if (const auto* prim_func = func.as<PrimFuncNode>()) {
        finder(prim_func->body);
        if (finder.found_) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    if (!loop->extent->IsInstance<IntImmNode>()) {
      found_ = true;
    } else {
      StmtVisitor::VisitStmt_(loop);
    }
  }

  void VisitStmt(const Stmt& stmt) final {
    if (!found_) {
      StmtVisitor::VisitStmt(stmt);
    }
  }

  bool found_ = false;
};

}  // namespace tir

namespace meta_schedule {

/*! \brief Check if the IRModule has any loop with non-constant extent. */
class DisallowDynamicLoopNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final { return !tir::DynamicExtentFinder::Find(sch->mod()); }
  // Inherited from PostprocNode
  Postproc Clone() const {
    ObjectPtr<DisallowDynamicLoopNode> n = make_object<DisallowDynamicLoopNode>(*this);
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.DisallowDynamicLoop";
  TVM_DECLARE_FINAL_OBJECT_INFO(DisallowDynamicLoopNode, PostprocNode);
};

Postproc Postproc::DisallowDynamicLoop() {
  ObjectPtr<DisallowDynamicLoopNode> n = make_object<DisallowDynamicLoopNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(DisallowDynamicLoopNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDisallowDynamicLoop")
    .set_body_typed(Postproc::DisallowDynamicLoop);

}  // namespace meta_schedule
}  // namespace tvm
