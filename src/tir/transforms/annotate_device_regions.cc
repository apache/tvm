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
 * \file annotate_device_regions.cc
 * \brief Split device function from host.
 */
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class DeviceRegionAnnotater : public StmtMutator {
 public:
  static Stmt Apply(Target host_target, Target device_target, Stmt body) {
    DeviceRegionAnnotater mutator(device_target);
    body = mutator(body);

    bool same_host_and_device = host_target->str() == device_target->str();

    // If no region was found that must be on the device, but the
    // device and host differ (e.g. `T.target('c', host='llvm')`),
    // then the entire region should be annotated.  This preserves the
    // host-side handling of DLTensor arguments, while ensuring that
    // any device targets are used for the codegen.
    if (!mutator.found_target_region_ && !same_host_and_device) {
      body = AttrStmt(device_target, tvm::attr::kTarget, 0, body);
    }

    return body;
  }

 private:
  explicit DeviceRegionAnnotater(Target device_target) : device_target_(device_target) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      // If a target attribute already exists, use it as-is.
      found_target_region_ = true;
      return GetRef<Stmt>(op);
    } else if (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope ||
               op->attr_key == attr::device_scope) {
      // These attributes are only allowed in device-side code, so
      // they should be annotated with the function's default target.
      found_target_region_ = true;
      Stmt body = GetRef<Stmt>(op);
      return AttrStmt(device_target_, tvm::attr::kTarget, 0, body);
    } else {
      // All other annotations are ignored
      return StmtMutator::VisitStmt_(op);
    }
  }

  Target device_target_;
  bool found_target_region_{false};
};

namespace transform {

Pass AnnotateDeviceRegions() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) -> PrimFunc {
    auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(opt_target) << "AnnotateDeviceRegions: Require the target attribute";
    Target target = opt_target.value();

    if (auto opt_host = target->GetHost()) {
      auto new_body =
          DeviceRegionAnnotater::Apply(opt_host.value(), target.WithoutHost(), func->body);
      if (!new_body.same_as(func->body)) {
        func.CopyOnWrite()->body = new_body;
      }
    }
    return func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tir.AnnotateDeviceRegions", {});
}

TVM_REGISTER_GLOBAL("tir.transform.AnnotateDeviceRegions").set_body_typed(AnnotateDeviceRegions);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
