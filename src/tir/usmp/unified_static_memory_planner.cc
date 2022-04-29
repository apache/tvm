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

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/algorithms.h>
#include <tvm/tir/usmp/analysis.h>
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <string>

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPEnableOption, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUSMPAlgorithmOption, String);

namespace tir {
namespace usmp {

static constexpr const char* kDefaultAlgo = "greedy_by_size";

static std::unordered_map<String, std::function<Map<BufferInfo, PoolAllocation>(
                                      const Array<BufferInfo>&, const Integer&)>>
    algorithms{{"greedy_by_size", algo::GreedyBySize},
               {"greedy_by_conflicts", algo::GreedyByConflicts},
               {"hill_climb", algo::HillClimb}};

IRModule PlanMemory(const IRModule& mod, String algo) {
  VLOG(1) << "workspace required = " << CalculateModuleWorkspaceSize(mod);
  PrimFunc main_func = Downcast<PrimFunc>(mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
  BufferInfoAnalysis buffer_info_analysis = ExtractBufferInfo(main_func, mod);
  Array<BufferInfo> buffer_info_arr =
      CreateArrayBufferInfo(buffer_info_analysis->buffer_info_stmts);
  CHECK(algorithms.count(algo)) << "The selected USMP algorithm : " << algo
                                << " is not defined. Please define it in the above algorithms map.";
  Map<BufferInfo, PoolAllocation> buffer_info_pool_allocations =
      algorithms[algo](buffer_info_arr, buffer_info_analysis->memory_pressure);
  Map<Stmt, PoolAllocation> stmt_pool_allocations = AssignStmtPoolAllocations(
      buffer_info_analysis->buffer_info_stmts, buffer_info_pool_allocations);
  IRModule ret = transform::ConvertPoolAllocationsToOffsets(stmt_pool_allocations)(mod);
  tir::PrimFunc tir_main_func =
      Downcast<tir::PrimFunc>(ret->Lookup(::tvm::runtime::symbol::tvm_module_main));
  Optional<Array<tir::usmp::AllocatedPoolInfo>> allocated_pool_infos =
      tir_main_func->GetAttr<Array<tir::usmp::AllocatedPoolInfo>>(tvm::attr::kPoolArgs);
  if (allocated_pool_infos) {
    for (const tir::usmp::AllocatedPoolInfo& allocated_pool_info : allocated_pool_infos.value()) {
      VLOG(1) << "pool_size = " << allocated_pool_info->allocated_size;
    }
  }
  return ret;
}

}  // namespace usmp

namespace transform {

tvm::transform::Pass UnifiedStaticMemoryPlanner() {
  auto usmp_main_pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    auto algorithm_str = ctx->GetConfig(kUSMPAlgorithmOption, String(usmp::kDefaultAlgo));
    return Downcast<IRModule>(
        usmp::PlanMemory(m, algorithm_str.value_or(String(usmp::kDefaultAlgo))));
  };

  return tvm::transform::Sequential(
      {tvm::tir::usmp::transform::AssignPoolInfo(),
       tvm::transform::CreateModulePass(usmp_main_pass_func, 0,
                                        "tir.transform.UnifiedStaticMemoryPlanner", {})});
}

TVM_REGISTER_GLOBAL("tir.transform.UnifiedStaticMemoryPlanner")
    .set_body_typed(UnifiedStaticMemoryPlanner);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
