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
 * \file src/relay/backend/aot/create_function_metadata.cc
 * \brief Create FunctionInfo metadata from a lowered TIR module.
 */
#include "./create_function_metadata.h"

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target_kind.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/usmp/utils.h>

#include "../utils.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

/*!
 * \brief Calculate FunctionInfo for all the PrimFuncs in a module.
 */
Map<String, backend::FunctionInfo> CalculateFunctionInfos(const IRModule& mod,
                                                          Integer workspace_byte_alignment,
                                                          Integer constant_byte_alignment) {
  Map<String, backend::FunctionInfo> function_metadata;
  for (const auto& kv : mod->functions) {
    GlobalVar global_var = kv.first;
    BaseFunc base_func = kv.second;
    if (base_func->IsInstance<tir::PrimFuncNode>()) {
      tir::PrimFunc pfunc = Downcast<tir::PrimFunc>(base_func);
      Optional<Target> tgt_opt = pfunc->GetAttr<Target>(tvm::attr::kTarget);
      ICHECK(tgt_opt) << "Target must be defined for all primfuncs.";
      Target tgt = tgt_opt.value();
      // Determine the size of input/output buffers
      auto params = pfunc->params;
      int64_t total_io_bytes = 0;
      for (const auto& param : params) {
        if (pfunc->buffer_map.find(param) != pfunc->buffer_map.end()) {
          auto buffer = pfunc->buffer_map[param];
          total_io_bytes += GetMemorySizeBytes(buffer->shape, buffer->dtype);
        }
      }
      const auto& ws = CalculateWorkspaceBytes(pfunc, workspace_byte_alignment);
      const auto& cs = CalculateConstantBytes(pfunc, constant_byte_alignment);
      backend::FunctionInfo finfo{
          {{tgt, ws}}, {{tgt, total_io_bytes}}, {{tgt, cs}}, {{tgt, pfunc}}, {}};
      function_metadata.Set(global_var->name_hint, finfo);
    }
  }
  return function_metadata;
}

Map<String, backend::FunctionInfo> CreateFunctionMetadata(const IRModule& mod,
                                                          Integer workspace_byte_alignment,
                                                          Integer constant_byte_alignment) {
  // First calculate the FunctionInfos from the buffers that are explicitly allocated
  auto function_metadata =
      CalculateFunctionInfos(mod, workspace_byte_alignment, constant_byte_alignment);
  // Now adjust the FunctionInfo for the main func to also include PoolInfo allocations
  // made by the USMP.
  Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
      mod->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
  backend::FunctionInfo main_func_info =
      function_metadata.Get(runtime::symbol::tvm_module_main).value();
  if (allocated_pool_infos) {
    for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
      for (const auto& tgt : allocated_pool_info->pool_info->targets) {
        VLOG(1) << "USMP requires target " << tgt->ToDebugString() << " to have pool size "
                << allocated_pool_info->allocated_size->value;
        size_t size = allocated_pool_info->allocated_size->value;
        if (allocated_pool_info->pool_info->IsInstance<ConstantPoolInfoNode>()) {
          size += main_func_info->constant_sizes.count(tgt)
                      ? main_func_info->constant_sizes[tgt]->value
                      : 0;
          main_func_info->constant_sizes.Set(tgt, size);
        } else if (allocated_pool_info->pool_info->IsInstance<WorkspacePoolInfoNode>()) {
          size += main_func_info->workspace_sizes.count(tgt)
                      ? main_func_info->workspace_sizes[tgt]->value
                      : 0;
          main_func_info->workspace_sizes.Set(tgt, size);
        } else {
          LOG(FATAL) << "Unknown pool type: " << allocated_pool_info->pool_info->GetTypeKey();
        }
      }
    }
  }
  function_metadata.Set(runtime::symbol::tvm_module_main, main_func_info);
  return function_metadata;
}

TVM_REGISTER_GLOBAL("relay.backend.aot.CreateFunctionMetadata")
    .set_body_typed(CreateFunctionMetadata);

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm
