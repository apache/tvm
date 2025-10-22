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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/function.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.BaseFunc_Attrs", [](BaseFunc func) { return func->attrs; })
      .def("ir.BaseFuncCopy", [](BaseFunc func) { return func; })
      .def("ir.BaseFuncWithAttr",
           [](ffi::RValueRef<BaseFunc> func_ref, ffi::String key, Any value) -> BaseFunc {
             BaseFunc func = *std::move(func_ref);
             if (func->IsInstance<tir::PrimFuncNode>()) {
               return WithAttr(Downcast<tir::PrimFunc>(std::move(func)), key, value);
             } else if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttr(Downcast<relax::Function>(std::move(func)), key, value);
             } else if (func->IsInstance<relax::ExternFuncNode>()) {
               return WithAttr(Downcast<relax::ExternFunc>(std::move(func)), key, value);
             } else {
               LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
             }
           })
      .def("ir.BaseFuncWithAttrs",
           [](ffi::RValueRef<BaseFunc> func_ref,
              ffi::Map<ffi::String, ffi::Any> attr_map) -> BaseFunc {
             BaseFunc func = *std::move(func_ref);
             if (func->IsInstance<tir::PrimFuncNode>()) {
               return WithAttrs(Downcast<tir::PrimFunc>(std::move(func)), attr_map);
             }
             if (const auto f = tvm::ffi::Function::GetGlobal("relax.FuncWithAttrs")) {
               if (auto ret = (*f)(func, attr_map).cast<ffi::Optional<BaseFunc>>()) {
                 return ret.value();
               }
             }
             if (func->IsInstance<relax::ExternFuncNode>()) {
               return WithAttrs(Downcast<relax::ExternFunc>(std::move(func)), attr_map);
             }
             LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
             TVM_FFI_UNREACHABLE();
           })
      .def("ir.BaseFuncWithoutAttr",
           [](ffi::RValueRef<BaseFunc> func_ref, ffi::String key) -> BaseFunc {
             BaseFunc func = *std::move(func_ref);
             if (func->IsInstance<tir::PrimFuncNode>()) {
               return WithoutAttr(Downcast<tir::PrimFunc>(std::move(func)), key);
             } else if (func->IsInstance<relax::FunctionNode>()) {
               return WithoutAttr(Downcast<relax::Function>(std::move(func)), key);
             } else {
               LOG(FATAL) << "Do not support function type " << func->GetTypeKey();
               TVM_FFI_UNREACHABLE();
             }
           });
}

}  // namespace tvm
