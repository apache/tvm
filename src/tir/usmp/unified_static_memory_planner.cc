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
 * \file tir/analysis/usmp/unified_static_memory_planner.cc
 * \brief This is the pass that integrates the USMP passes to
 * a single composite pass.
 */

#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/algorithms.h>
#include <tvm/tir/usmp/analysis.h>
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <algorithm>
#include <string>

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPEnableOption, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPAlgorithmOption, String);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPUseWorkspaceIO, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPCustomAlgorithmOption, String);

namespace tir {
namespace usmp {

static constexpr const char* kDefaultAlgo = "greedy_by_size";

static std::unordered_map<String, std::function<Map<BufferInfo, PoolAllocation>(
                                      const Array<BufferInfo>&, const Integer&)>>
    algorithms{{"greedy_by_size", algo::GreedyBySize},
               {"greedy_by_conflicts", algo::GreedyByConflicts},
               {"hill_climb", algo::HillClimb}};

IRModule PlanMemory(const IRModule& mod, String algo, bool use_workspace_io,
                    Optional<String> opt_custom_algo) {
  VLOG(1) << "workspace required = " << CalculateModuleWorkspaceSize(mod);
  IRModule module = mod->ShallowCopy();
  if (use_workspace_io) {
    module = transform::CreateAllocatesForIO()(module);
  }
  module = transform::AssignPoolInfo()(module);
  PrimFunc main_func = Downcast<PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
  BufferInfoAnalysis buffer_info_analysis = ExtractBufferInfo(main_func, module);
  Array<BufferInfo> buffer_info_arr =
      ConvertToArrayOfBufferInfo(buffer_info_analysis->buffer_info_stmts);
  decltype(algorithms)::mapped_type algorithm;
  if (opt_custom_algo) {
    String algo_func_name = "tir.usmp.algo." + opt_custom_algo.value();
    const runtime::PackedFunc* pfAlgo = runtime::Registry::Get(algo_func_name);
    CHECK(pfAlgo) << "The selected custom USMP algorithm : " << opt_custom_algo.value()
                  << " is not defined. Please register it as " << algo_func_name;
    algorithm = *pfAlgo;
  } else {
    CHECK(algorithms.count(algo))
        << "The selected USMP algorithm : " << algo
        << " is not defined. Please define it in the above algorithms map.";
    algorithm = algorithms[algo];
  }
  Map<BufferInfo, PoolAllocation> buffer_info_pool_allocations =
      algorithm(buffer_info_arr, buffer_info_analysis->memory_pressure);

  Map<Stmt, PoolAllocation> stmt_pool_allocations = AssignStmtPoolAllocations(
      buffer_info_analysis->buffer_info_stmts, buffer_info_pool_allocations);

  module = transform::ConvertPoolAllocationsToOffsets(stmt_pool_allocations)(module);
  if (use_workspace_io) {
    Map<String, PoolAllocation> io_pool_allocations =
        GetIOPoolAllocations(buffer_info_pool_allocations);
    module = WithAttr(module, tvm::attr::kIOTensorPoolAllocations, io_pool_allocations);
  }
  tir::PrimFunc tir_main_func =
      Downcast<tir::PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
  Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
      tir_main_func->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
  if (allocated_pool_infos) {
    for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
      VLOG(1) << "pool_size = " << allocated_pool_info->allocated_size;
    }
  }
  return module;
}

}  // namespace usmp

namespace transform {

tvm::transform::Pass UnifiedStaticMemoryPlanner() {
  auto usmp_main_pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    auto algorithm_str = ctx->GetConfig(kUSMPAlgorithmOption, String(usmp::kDefaultAlgo));
    auto use_workspace_io = ctx->GetConfig(kUSMPUseWorkspaceIO, Bool(false));
    auto custom_algorithm_str = ctx->GetConfig<String>(kUSMPCustomAlgorithmOption);
    tvm::relay::Executor executor_config =
        m->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor).value();
    String interface_api = executor_config->GetAttr<String>("interface-api").value_or("packed");
    tvm::relay::Runtime runtime_config =
        m->GetAttr<tvm::relay::Runtime>(tvm::attr::kRuntime).value();
    if (use_workspace_io.value()) {
      CHECK(interface_api == "c") << kUSMPUseWorkspaceIO
                                  << " option is only compatible with interface_api c.\n"
                                  << "Please use interface_api c to be able to enable "
                                  << kUSMPUseWorkspaceIO << "\n";
    }
    return Downcast<IRModule>(
        usmp::PlanMemory(m, algorithm_str.value_or(String(usmp::kDefaultAlgo)),
                         use_workspace_io.value_or(Bool(false)), custom_algorithm_str));
  };

  return tvm::transform::CreateModulePass(usmp_main_pass_func, 0,
                                          "tir.transform.UnifiedStaticMemoryPlanner", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UnifiedStaticMemoryPlanner")
    .set_body_typed(UnifiedStaticMemoryPlanner);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
