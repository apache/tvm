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
 * \file auto_scheduler/search_task.cc
 * \brief Meta information and hardware parameters for a search task.
 */

#include <tvm/auto_scheduler/search_task.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>

#include <utility>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(HardwareParamsNode);
TVM_REGISTER_NODE_TYPE(SearchTaskNode);

HardwareParams::HardwareParams(int num_cores, int vector_unit_bytes, int cache_line_bytes,
                               int max_shared_memory_per_block, int max_local_memory_per_block,
                               int max_threads_per_block, int max_vthread_extent, int warp_size) {
  auto node = make_object<HardwareParamsNode>();
  node->num_cores = num_cores;
  node->vector_unit_bytes = vector_unit_bytes;
  node->cache_line_bytes = cache_line_bytes;
  node->max_shared_memory_per_block = max_shared_memory_per_block;
  node->max_local_memory_per_block = max_local_memory_per_block;
  node->max_threads_per_block = max_threads_per_block;
  node->max_vthread_extent = max_vthread_extent;
  node->warp_size = warp_size;
  data_ = std::move(node);
}

HardwareParams HardwareParamsNode::GetDefaultHardwareParams(const Target& target,
                                                            const Target& target_host) {
  if (target->kind->device_type == kDLCPU) {
    return HardwareParams(tvm::runtime::threading::MaxConcurrency(), 64, 64, 0, 0, 0, 0, 0);
  } else if (target->kind->device_type == kDLGPU) {
    auto ctx = TVMContext{kDLGPU, 0};
    auto func = tvm::runtime::Registry::Get("device_api.gpu");
    ICHECK(func != nullptr) << "Cannot find GPU device_api in registry";
    auto device_api = static_cast<tvm::runtime::DeviceAPI*>(((*func)()).operator void*());

    tvm::runtime::TVMRetValue ret;
    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kMaxSharedMemoryPerBlock, &ret);
    int max_shared_memory_per_block = ret;

    // There is no explicit local memory limition in CUDA runtime,
    // so we can use INT32_MAX to disalbe the check on local_memory.
    int max_local_memory_per_block = INT32_MAX;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kMaxThreadsPerBlock, &ret);
    int max_threads_per_block = ret;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kWarpSize, &ret);
    int warp_size = ret;

    int max_vthread_extent = warp_size / 4;
    return HardwareParams(-1, 16, 64, max_shared_memory_per_block, max_local_memory_per_block,
                          max_threads_per_block, max_vthread_extent, warp_size);
  } else if (target->kind->device_type == kDLMetal) {
    // Reference: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    // This setting looks working for Metal GPUs later than A10
    int max_shared_memory_per_block = 32 * 1024;
    int max_local_memory_per_block = INT32_MAX;  // skip the check on local memory
    int max_threads_per_block = 1024;
    int warp_size = 8;
    int max_vthread_extent = warp_size / 4;
    return HardwareParams(-1, 16, 64, max_shared_memory_per_block, max_local_memory_per_block,
                          max_threads_per_block, max_vthread_extent, warp_size);
  } else {
    LOG(FATAL) << "No default hardware parameters for target: " << target;
  }
  return HardwareParams();
}

SearchTask::SearchTask(ComputeDAG compute_dag, String workload_key, Target target,
                       Target target_host, Optional<HardwareParams> hardware_params) {
  auto node = make_object<SearchTaskNode>();
  node->compute_dag = std::move(compute_dag);
  node->workload_key = std::move(workload_key);
  node->target = std::move(target);
  node->target_host = std::move(target_host);
  if (hardware_params) {
    node->hardware_params = hardware_params.value();
  } else {
    node->hardware_params =
        HardwareParamsNode::GetDefaultHardwareParams(node->target, node->target_host);
  }
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("auto_scheduler.HardwareParams")
    .set_body_typed([](int num_cores, int vector_unit_bytes, int cache_line_bytes,
                       int max_shared_memory_per_block, int max_local_memory_per_block,
                       int max_threads_per_block, int max_vthread_extent, int warp_size) {
      return HardwareParams(num_cores, vector_unit_bytes, cache_line_bytes,
                            max_shared_memory_per_block, max_local_memory_per_block,
                            max_threads_per_block, max_vthread_extent, warp_size);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchTask")
    .set_body_typed([](ComputeDAG compute_dag, String workload_key, Target target,
                       Target target_host, Optional<HardwareParams> hardware_params) {
      return SearchTask(compute_dag, workload_key, target, target_host, hardware_params);
    });

}  // namespace auto_scheduler
}  // namespace tvm
