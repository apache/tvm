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
 *  Common build utilities
 * \file build_common.h
 */
#ifndef TVM_TARGET_BUILD_COMMON_H_
#define TVM_TARGET_BUILD_COMMON_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <string>

#include "../runtime/metadata.h"

namespace tvm {
namespace codegen {

inline ffi::Map<ffi::String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule& mod) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    ffi::Array<DLDataType> arg_types;
    ffi::Array<runtime::ArgExtraTags> arg_extra_tags;
    for (size_t i = 0; i < f->params.size(); ++i) {
      arg_types.push_back(f->params[i].dtype());
      auto is_tensormap = [](const tir::Var& var) -> bool {
        const auto* type = var->type_annotation.as<PointerTypeNode>();
        if (type == nullptr) {
          return false;
        }
        return type->element_type.as<TensorMapTypeNode>() != nullptr;
      };
      arg_extra_tags.push_back(is_tensormap(f->params[i]) ? runtime::ArgExtraTags::kTensorMap
                                                          : runtime::ArgExtraTags::kNone);
    }
    ffi::Array<ffi::String> launch_param_tags;
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(tir::attr::kKernelLaunchParams)) {
      for (const auto& tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    if (global_symbol) {
      fmap.Set(global_symbol.value(),
               runtime::FunctionInfo(global_symbol.value(), std::move(arg_types),
                                     std::move(launch_param_tags), std::move(arg_extra_tags)));
    }
  }
  return fmap;
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_BUILD_COMMON_H_
