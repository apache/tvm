/*!
 *  Copyright (c) 2020 by Contributors
 */
#include "search_task.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>

#include <string>
#include <utility>

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(HardwareParamsNode);
TVM_REGISTER_NODE_TYPE(SearchTaskNode);

HardwareParams HardwareParamsNode::make(int num_cores, int vector_unit_bytes,
                                        int cache_line_bytes,
                                        int max_unroll_vec,
                                        int max_innermost_split_factor) {
  auto node = make_object<HardwareParamsNode>();
  node->num_cores = num_cores;
  node->vector_unit_bytes = vector_unit_bytes;
  node->cache_line_bytes = cache_line_bytes;
  node->max_unroll_vec = max_unroll_vec;
  node->max_innermost_split_factor = max_innermost_split_factor;
  return HardwareParams(node);
}

HardwareParams HardwareParamsNode::GetDefaultHardwareParams(
    const Target& target, const Target& target_host) {
  if (target->target_name == "llvm") {
    return HardwareParamsNode::make(tvm::runtime::threading::MaxConcurrency(),
                                    32, 64, 16, 64);
  } else if (target->device_type == kDLGPU) {
    // TODO(jcf94): temp implementation, max vectorize size in GPU is related
    // to the data type
    auto hardware_params = HardwareParamsNode::make(100000, 16, 64, 4, 64);
    auto* p_hardware_params = hardware_params.CopyOnWrite();

    auto ctx = TVMContext{kDLGPU, 0};
    auto func = tvm::runtime::Registry::Get("device_api.gpu");
    CHECK(func != nullptr) << "Cannot find GPU device_api in registry";
    auto device_api =
        static_cast<tvm::runtime::DeviceAPI*>(((*func)()).operator void*());

    tvm::runtime::TVMRetValue ret;
    device_api->GetAttr(
        ctx, tvm::runtime::DeviceAttrKind::kMaxSharedMemoryPerBlock, &ret);
    p_hardware_params->max_shared_memory_per_block = ret;

    device_api->GetAttr(
        ctx, tvm::runtime::DeviceAttrKind::kMaxRegistersPerBlock, &ret);
    p_hardware_params->max_registers_per_block = ret;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kMaxThreadsPerBlock,
                        &ret);
    p_hardware_params->max_threads_per_block = ret;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kWarpSize, &ret);
    p_hardware_params->warp_size = ret;

    // Manually set now
    p_hardware_params->max_vthread_extent = 4;

    return hardware_params;
  } else if (target->device_type == kDLOpenCL) {
    // TODO(jcf94): temp implementation
    auto hardware_params = HardwareParamsNode::make(100000, 16, 64, 4, 64);
    auto p_hardware_params = hardware_params.CopyOnWrite();

    auto ctx = TVMContext{kDLOpenCL, 0};
    auto func = tvm::runtime::Registry::Get("device_api.opencl");
    CHECK(func != nullptr) << "Cannot find GPU device_api in registry";
    auto device_api =
        static_cast<tvm::runtime::DeviceAPI*>(((*func)()).operator void*());

    tvm::runtime::TVMRetValue ret;
    device_api->GetAttr(
        ctx, tvm::runtime::DeviceAttrKind::kMaxSharedMemoryPerBlock, &ret);
    p_hardware_params->max_shared_memory_per_block = ret;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kMaxThreadsPerBlock,
                        &ret);
    p_hardware_params->max_threads_per_block = ret;

    device_api->GetAttr(ctx, tvm::runtime::DeviceAttrKind::kWarpSize, &ret);
    p_hardware_params->warp_size = ret;

    // Manually set now
    p_hardware_params->max_vthread_extent = 4;

    return hardware_params;
  } else {
    LOG(FATAL) << "No default hardware parameters for target: " << target;
  }
  return HardwareParams();
}

SearchTask SearchTaskNode::make(ComputeDAG compute_dag,
                                std::string workload_key, Target target,
                                Target target_host,
                                HardwareParams hardware_params) {
  auto node = make_object<SearchTaskNode>();
  node->compute_dag = std::move(compute_dag);
  node->workload_key = std::move(workload_key);
  node->target = std::move(target);
  node->target_host = std::move(target_host);
  if (hardware_params.defined()) {
    node->hardware_params = std::move(hardware_params);
  } else {
    node->hardware_params = HardwareParamsNode::GetDefaultHardwareParams(
        node->target, node->target_host);
  }
  return SearchTask(node);
}

TVM_REGISTER_GLOBAL("ansor.HardwareParams")
    .set_body_typed([](int num_cores, int vector_unit_bytes,
                       int cache_line_bytes, int max_unroll_vec,
                       int max_innermost_split_factor) {
      return HardwareParamsNode::make(num_cores, vector_unit_bytes,
                                      cache_line_bytes, max_unroll_vec,
                                      max_innermost_split_factor);
    });

TVM_REGISTER_GLOBAL("ansor.SearchTask")
    .set_body_typed([](ComputeDAG compute_dag, std::string workload_key,
                       Target target, Target target_host,
                       HardwareParams hardware_params) {
      return SearchTaskNode::make(compute_dag, workload_key, target,
                                  target_host, hardware_params);
    });

}  // namespace ansor
}  // namespace tvm
