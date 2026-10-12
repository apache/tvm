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
#include "../../tirx/transform/decorate_device_scope.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/transform.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace transform {

Pass DecorateDeviceScope() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    return tirx::DecorateDeviceScopeImpl(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.DecorateDeviceScope");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.DecorateDeviceScope", DecorateDeviceScope);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
