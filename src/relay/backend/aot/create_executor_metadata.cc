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
 * \file src/relay/backend/aot/create_executor_metadata.cc
 * \brief Create the ExecutorCodegenMetadata from a compiled IRModule.
 */

#include "./create_executor_metadata.h"

#include "../utils.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

ExecutorCodegenMetadata CreateExecutorMetadata(const IRModule& mod, String mod_name,
                                               Executor executor, Integer workspace_byte_alignment,
                                               Integer constant_byte_alignment) {
  // Get relevant executor config information
  std::string interface_api = executor->GetAttr<String>("interface-api").value_or("packed");
  bool unpacked_api = executor->GetAttr<Bool>("unpacked-api").value_or(Bool(false));
  // Get the input vars
  auto tir_main_func = Downcast<tir::PrimFunc>(mod->Lookup(runtime::symbol::tvm_module_main));
  Array<tir::Var> inputs = tir_main_func->GetAttr<Array<tir::Var>>("input_vars").value();
  Array<TensorType> input_tensor_types;
  for (const auto& input : inputs) {
    auto buffer = tir_main_func->buffer_map.Get(input).value();
    input_tensor_types.push_back(TensorType(buffer->shape, buffer->dtype));
  }
  // Extract USMP metadata to pass onto metadata sources
  Map<tir::Var, tir::usmp::AllocatedPoolInfo> pool_var_info;
  std::vector<tir::Var> pool_vars;
  Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
      tir_main_func->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
  if (allocated_pool_infos) {
    for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
      int pool_var_index = allocated_pool_info->pool_var_idx.value()->value;
      pool_vars.push_back(tir_main_func->params[pool_var_index]);
      pool_var_info.Set(tir_main_func->params[pool_var_index], allocated_pool_info);
    }
  }
  Map<String, tir::usmp::PoolAllocation> io_pool_allocations =
      mod->GetAttr<Map<String, tir::usmp::PoolAllocation>>(tvm::attr::kIOTensorPoolAllocations)
          .value_or({});

  Array<tir::Var> outputs = tir_main_func->GetAttr<Array<tir::Var>>("output_vars").value();
  Array<TensorType> output_tensor_types;
  std::vector<String> output_var_names;
  for (const auto& output : outputs) {
    auto buffer = tir_main_func->buffer_map.Get(output).value();
    output_tensor_types.push_back(TensorType(buffer->shape, buffer->dtype));
    output_var_names.push_back(output->name_hint);
  }
  auto devices = tir_main_func->GetAttr<Array<String>>("devices").value_or({});

  return ExecutorCodegenMetadata(inputs, input_tensor_types, output_var_names, output_tensor_types,
                                 pool_vars, devices, runtime::kTvmExecutorAot, mod_name,
                                 interface_api, unpacked_api, workspace_byte_alignment,
                                 constant_byte_alignment, pool_var_info, io_pool_allocations);
}

TVM_REGISTER_GLOBAL("relay.backend.aot.CreateExecutorMetadata")
    .set_body_typed(CreateExecutorMetadata);

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm
