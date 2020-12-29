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
 * \file decorate_device_scope.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

Stmt DecorateDeviceScope(Stmt&& stmt) {
  Stmt body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::device_scope, 0, stmt);
  return body;
}

namespace transform {

Pass DecorateDeviceScope() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = DecorateDeviceScope(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.DecorateDeviceScope", {});
}

TVM_REGISTER_GLOBAL("tir.transform.DecorateDeviceScope").set_body_typed(DecorateDeviceScope);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
