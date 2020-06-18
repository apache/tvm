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

#include <tvm/target/codegen.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/ir/module.h>
#include <tvm/tir/function.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <unordered_map>
#include <string>
#include "../runtime/meta_data.h"

namespace tvm {
namespace codegen {

inline std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule& mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      info.arg_types.push_back(f->params[i].dtype());
    }
    if (auto opt = f->GetAttr<Array<tir::IterVar>>(tir::attr::kDeviceThreadAxis)) {
      auto thread_axis = opt.value();
      for (size_t i = 0; i < thread_axis.size(); ++i) {
        info.thread_axis_tags.push_back(thread_axis[i]->thread_tag);
      }
    }
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_BUILD_COMMON_H_
