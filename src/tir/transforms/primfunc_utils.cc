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
 * \file primfunc_utils.cc
 * \brief Passes that serve as helper functions.
 */

#include <tvm/driver/driver_api.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {
transform::Pass BindTarget(Target target) {
  auto fpass = [target](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    if (f->GetAttr<Integer>(tvm::tir::attr::kIsHostFunc) == 1) {
      return WithAttr(std::move(WithoutAttr(std::move(f), tvm::tir::attr::kIsHostFunc)),
                      tvm::attr::kTarget, target->host.value_or(Target("llvm")));
    }
    return WithAttr(std::move(f), tvm::attr::kTarget, target);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tir.BindTarget", {});
}

transform::Pass AnnotateEntryFunc() {
  auto fpass = [](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    ICHECK(m->functions.size() == 1);
    return WithAttr(std::move(f), tir::attr::kIsEntryFunc, Bool(true));
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tir.AnnotateEntryFunc", {});
}

transform::Pass Filter(runtime::TypedPackedFunc<bool(PrimFunc)> fcond) {
  auto fpass = [fcond](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    if (fcond(f)) {
      return f;
    } else {
      return tir::PrimFunc(nullptr);
    }
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tir.Filter", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BindTarget").set_body_typed(BindTarget);
TVM_REGISTER_GLOBAL("tir.transform.AnnotateEntryFunc").set_body_typed(AnnotateEntryFunc);
TVM_REGISTER_GLOBAL("tir.transform.Filter").set_body_typed(Filter);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
