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
 * \file src/ir/memory_pools.cc
 * \brief The object definition for relay.build argument type of memory pools
 */

#include <tvm/ir/memory_pools.h>
#include <tvm/relay/executor.h>

namespace tvm {

PoolInfo::PoolInfo(String pool_name, Map<Target, String> target_access, Integer size_hint_bytes,
                   Integer clock_frequency_hz, Integer read_bandwidth_bytes_per_cycle,
                   Integer write_bandwidth_bytes_per_cycle, Integer read_latency_cycles,
                   Integer write_latency_cycles, Map<Target, Integer> target_burst_bytes,
                   Bool is_internal) {
  auto poolinfo_node = make_object<PoolInfoNode>();
  poolinfo_node->pool_name = pool_name;
  poolinfo_node->size_hint_bytes = size_hint_bytes;
  poolinfo_node->target_access = target_access;
  poolinfo_node->clock_frequency_hz = clock_frequency_hz;
  poolinfo_node->read_bandwidth_bytes_per_cycle = read_bandwidth_bytes_per_cycle;
  poolinfo_node->write_bandwidth_bytes_per_cycle = write_bandwidth_bytes_per_cycle;
  poolinfo_node->read_latency_cycles = read_latency_cycles;
  poolinfo_node->write_latency_cycles = write_latency_cycles;
  poolinfo_node->target_burst_bytes = target_burst_bytes;
  poolinfo_node->is_internal = is_internal;
  data_ = std::move(poolinfo_node);
}

TVM_REGISTER_NODE_TYPE(PoolInfoNode);
TVM_REGISTER_GLOBAL("ir.PoolInfo")
    .set_body_typed([](String pool_name, Map<Target, String> target_access, Integer size_hint_bytes,
                       Integer clock_frequency_hz, Integer read_bandwidth_bytes_per_cycle,
                       Integer write_bandwidth_bytes_per_cycle, Integer read_latency_cycles,
                       Integer write_latency_cycles, Map<Target, Integer> target_burst_bytes) {
      return PoolInfo(pool_name, target_access, size_hint_bytes, clock_frequency_hz,
                      read_bandwidth_bytes_per_cycle, write_bandwidth_bytes_per_cycle,
                      read_latency_cycles, write_latency_cycles, target_burst_bytes, Bool(false));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PoolInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PoolInfoNode*>(ref.get());
      p->stream << "PoolInfoNode(\n"
                << "  pool_name=" << node->pool_name << ",\n  target_access=" << node->target_access
                << ",\n  size_hint_bytes=" << node->size_hint_bytes
                << ",\n  clock_frequency_hz=" << node->clock_frequency_hz
                << ",\n  read_bandwidth_bytes_per_cycle=" << node->read_bandwidth_bytes_per_cycle
                << ",\n  write_bandwidth_bytes_per_cycle=" << node->write_bandwidth_bytes_per_cycle
                << ",\n  read_latency_cycles=" << node->read_latency_cycles
                << ",\n  write_latency_cycles=" << node->write_latency_cycles
                << ",\n  target_burst_bytes=" << node->target_burst_bytes << ")";
    });

WorkspaceMemoryPools::WorkspaceMemoryPools(Array<PoolInfo> pools) {
  auto workspace_memory_pools_node = make_object<WorkspaceMemoryPoolsNode>();
  workspace_memory_pools_node->pools = pools;
  data_ = std::move(workspace_memory_pools_node);
}

TVM_REGISTER_NODE_TYPE(WorkspaceMemoryPoolsNode);
TVM_REGISTER_GLOBAL("ir.WorkspaceMemoryPools").set_body_typed([](Array<PoolInfo> pools) {
  return WorkspaceMemoryPools(pools);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WorkspaceMemoryPoolsNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const WorkspaceMemoryPoolsNode*>(ref.get());
      p->stream << "WorkspaceMemoryPoolsNode(\n"
                << "pools=" << node->pools << ")";
    });

}  // namespace tvm
