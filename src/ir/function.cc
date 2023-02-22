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
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>

namespace tvm {

TVM_REGISTER_GLOBAL("ir.BaseFunc_Attrs").set_body_typed([](BaseFunc func) { return func->attrs; });

TVM_REGISTER_GLOBAL("ir.BaseFuncCopy").set_body_typed([](BaseFunc func) { return func; });

TVM_REGISTER_GLOBAL("ir.BaseFuncWithAttr")
    .set_body_typed([](BaseFunc func, String key, ObjectRef value) -> BaseFunc {
      if (func->IsInstance<tir::PrimFuncNode>()) {
        return WithAttr(Downcast<tir::PrimFunc>(std::move(func)), key, value);
      } else if (func->IsInstance<relay::FunctionNode>()) {
        return WithAttr(Downcast<relay::Function>(std::move(func)), key, value);
      } else if (func->IsInstance<relax::FunctionNode>()) {
        return WithAttr(Downcast<relax::Function>(std::move(func)), key, value);
      } else {
        LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
      }
    });

TVM_REGISTER_GLOBAL("ir.BaseFuncWithAttrs")
    .set_body_typed([](BaseFunc func, Map<String, ObjectRef> attr_map) -> BaseFunc {
      if (func->IsInstance<tir::PrimFuncNode>()) {
        return WithAttrs(Downcast<tir::PrimFunc>(std::move(func)), attr_map);
      }
      if (const auto* f = runtime::Registry::Get("relay.ir.FuncWithAttrs")) {
        if (Optional<BaseFunc> ret = (*f)(func, attr_map)) {
          return ret.value();
        }
      }
      if (const auto* f = runtime::Registry::Get("relax.FuncWithAttrs")) {
        if (Optional<BaseFunc> ret = (*f)(func, attr_map)) {
          return ret.value();
        }
      }
      LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
    });

TVM_REGISTER_GLOBAL("ir.BaseFuncWithoutAttr")
    .set_body_typed([](BaseFunc func, String key) -> BaseFunc {
      if (func->IsInstance<tir::PrimFuncNode>()) {
        return WithoutAttr(Downcast<tir::PrimFunc>(std::move(func)), key);
      } else if (func->IsInstance<relay::FunctionNode>()) {
        return WithoutAttr(Downcast<relay::Function>(std::move(func)), key);
      } else if (func->IsInstance<relax::FunctionNode>()) {
        return WithoutAttr(Downcast<relax::Function>(std::move(func)), key);
      } else {
        LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
        return func;
      }
    });

}  // namespace tvm
