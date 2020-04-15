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
 * \file src/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/runtime/registry.h>
#include <tvm/ir/function.h>
// NOTE: reverse dependency on relay, tir/
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: We calls into the type specific WithAttr function
#include <tvm/tir/function.h>
#include <tvm/relay/function.h>


namespace tvm {

TVM_REGISTER_GLOBAL("ir.BaseFunc_Attrs")
.set_body_typed([](BaseFunc func) {
  return func->attrs;
});

TVM_REGISTER_GLOBAL("ir.BaseFuncCopy")
.set_body_typed([](BaseFunc func) {
  return func;
});

TVM_REGISTER_GLOBAL("ir.BaseFuncWithAttr")
.set_body_typed([](BaseFunc func, std::string key, ObjectRef value) -> BaseFunc {
  if (func->IsInstance<tir::PrimFuncNode>()) {
    return WithAttr(Downcast<tir::PrimFunc>(std::move(func)), key, value);
  } else if (func->IsInstance<relay::FunctionNode>()) {
    return WithAttr(Downcast<relay::Function>(std::move(func)), key, value);
  } else {
    LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
    return func;
  }
});


}  // namespace tvm
