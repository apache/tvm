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
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/lowered_func.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace codegen {

TVM_REGISTER_GLOBAL("codegen._Build")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  if (args[0].IsObjectRef<tir::LoweredFunc>()) {
      *ret = Build({args[0]}, args[1]);
    } else {
      *ret = Build(args[0], args[1]);
    }
  });

TVM_REGISTER_GLOBAL("module._PackImportsToC")
.set_body_typed(PackImportsToC);

TVM_REGISTER_GLOBAL("module._PackImportsToLLVM")
.set_body_typed(PackImportsToLLVM);
}  // namespace codegen
}  // namespace tvm
