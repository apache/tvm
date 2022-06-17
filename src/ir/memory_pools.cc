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

PoolInfo::PoolInfo(String pool_name, Integer size_hint_bytes, Integer clock_frequency_hz,
                   Integer read_bandwidth_bytes_per_cycle, Integer write_bandwidth_bytes_per_cycle,
                   Integer read_latency_cycles, Integer write_latency_cycles,
                   Map<Target, Integer> target_burst_bytes, Bool is_internal) {
  auto poolinfo_node = make_object<PoolInfoNode>();
  poolinfo_node->pool_name = pool_name;
  poolinfo_node->size_hint_bytes = size_hint_bytes;
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PoolInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PoolInfoNode*>(ref.get());
      p->stream << "PoolInfoNode(\n"
                << "  pool_name=" << node->pool_name
                << ",\n  size_hint_bytes=" << node->size_hint_bytes
                << ",\n  clock_frequency_hz=" << node->clock_frequency_hz
                << ",\n  read_bandwidth_bytes_per_cycle=" << node->read_bandwidth_bytes_per_cycle
                << ",\n  write_bandwidth_bytes_per_cycle=" << node->write_bandwidth_bytes_per_cycle
                << ",\n  read_latency_cycles=" << node->read_latency_cycles
                << ",\n  write_latency_cycles=" << node->write_latency_cycles
                << ",\n  target_burst_bytes=" << node->target_burst_bytes << ")";
    });

PoolInfoProperties::PoolInfoProperties(Integer size_hint_bytes, Integer clock_frequency_hz,
                                       Integer read_bandwidth_bytes_per_cycle,
                                       Integer write_bandwidth_bytes_per_cycle,
                                       Integer read_latency_cycles, Integer write_latency_cycles,
                                       Map<Target, Integer> target_burst_bytes, Bool is_internal) {
  auto poolinfo_properties_node = make_object<PoolInfoPropertiesNode>();
  poolinfo_properties_node->size_hint_bytes = size_hint_bytes;
  poolinfo_properties_node->clock_frequency_hz = clock_frequency_hz;
  poolinfo_properties_node->read_bandwidth_bytes_per_cycle = read_bandwidth_bytes_per_cycle;
  poolinfo_properties_node->write_bandwidth_bytes_per_cycle = write_bandwidth_bytes_per_cycle;
  poolinfo_properties_node->read_latency_cycles = read_latency_cycles;
  poolinfo_properties_node->write_latency_cycles = write_latency_cycles;
  poolinfo_properties_node->target_burst_bytes = target_burst_bytes;
  poolinfo_properties_node->is_internal = is_internal;
  data_ = std::move(poolinfo_properties_node);
}

TVM_REGISTER_NODE_TYPE(PoolInfoPropertiesNode);
TVM_REGISTER_GLOBAL("ir.PoolInfoProperties")
    .set_body_typed([](Integer size_hint_bytes, Integer clock_frequency_hz,
                       Integer read_bandwidth_bytes_per_cycle,
                       Integer write_bandwidth_bytes_per_cycle, Integer read_latency_cycles,
                       Integer write_latency_cycles, Map<Target, Integer> target_burst_bytes) {
      return PoolInfoProperties(size_hint_bytes, clock_frequency_hz, read_bandwidth_bytes_per_cycle,
                                write_bandwidth_bytes_per_cycle, read_latency_cycles,
                                write_latency_cycles, target_burst_bytes, Bool(false));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PoolInfoPropertiesNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PoolInfoPropertiesNode*>(ref.get());
      p->stream << "PoolInfoPropertiesNode(\n"
                << "  size_hint_bytes=" << node->size_hint_bytes
                << ",\n  clock_frequency_hz=" << node->clock_frequency_hz
                << ",\n  read_bandwidth_bytes_per_cycle=" << node->read_bandwidth_bytes_per_cycle
                << ",\n  write_bandwidth_bytes_per_cycle=" << node->write_bandwidth_bytes_per_cycle
                << ",\n  read_latency_cycles=" << node->read_latency_cycles
                << ",\n  write_latency_cycles=" << node->write_latency_cycles
                << ",\n  target_burst_bytes=" << node->target_burst_bytes << ")";
    });

WorkspacePoolInfo::WorkspacePoolInfo(String pool_name, Array<Target> targets,
                                     PoolInfoProperties properties) {
  auto poolinfo_node = make_object<WorkspacePoolInfoNode>();
  poolinfo_node->pool_name = pool_name;
  poolinfo_node->size_hint_bytes = properties->size_hint_bytes;
  poolinfo_node->targets = targets;
  poolinfo_node->clock_frequency_hz = properties->clock_frequency_hz;
  poolinfo_node->read_bandwidth_bytes_per_cycle = properties->read_bandwidth_bytes_per_cycle;
  poolinfo_node->write_bandwidth_bytes_per_cycle = properties->write_bandwidth_bytes_per_cycle;
  poolinfo_node->read_latency_cycles = properties->read_latency_cycles;
  poolinfo_node->write_latency_cycles = properties->write_latency_cycles;
  poolinfo_node->target_burst_bytes = properties->target_burst_bytes;
  poolinfo_node->is_internal = properties->is_internal;
  data_ = std::move(poolinfo_node);
}

TVM_REGISTER_NODE_TYPE(WorkspacePoolInfoNode);
TVM_REGISTER_GLOBAL("ir.WorkspacePoolInfo")
    .set_body_typed([](String pool_name, Array<Target> targets, PoolInfoProperties properties) {
      return WorkspacePoolInfo(pool_name, targets, properties);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WorkspacePoolInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const WorkspacePoolInfoNode*>(ref.get());
      p->stream << "WorkspacePoolInfoNode(\n"
                << "  pool_name=" << node->pool_name << ",\n  targets=" << node->targets
                << ",\n  size_hint_bytes=" << node->size_hint_bytes
                << ",\n  clock_frequency_hz=" << node->clock_frequency_hz
                << ",\n  read_bandwidth_bytes_per_cycle=" << node->read_bandwidth_bytes_per_cycle
                << ",\n  write_bandwidth_bytes_per_cycle=" << node->write_bandwidth_bytes_per_cycle
                << ",\n  read_latency_cycles=" << node->read_latency_cycles
                << ",\n  write_latency_cycles=" << node->write_latency_cycles
                << ",\n  target_burst_bytes=" << node->target_burst_bytes
                << ",\n  is_internal=" << node->is_internal << ")"
                << "\n";
    });

ConstantInfo::ConstantInfo(String name_hint, Integer byte_offset, runtime::NDArray data) {
  auto constant_info_node = make_object<ConstantInfoNode>();
  constant_info_node->name_hint = name_hint;
  constant_info_node->byte_offset = byte_offset;
  constant_info_node->data = data;
  data_ = std::move(constant_info_node);
}

TVM_REGISTER_NODE_TYPE(ConstantInfoNode);
TVM_REGISTER_GLOBAL("ir.ConstantInfo")
    .set_body_typed([](String name_hint, Integer byte_offset, runtime::NDArray data) {
      return ConstantInfo(name_hint, byte_offset, data);
    });
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstantInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstantInfoNode*>(ref.get());
      p->stream << "ConstantInfoNode(\n"
                << "name_hint=" << node->name_hint << ",\n byte_offset=" << node->byte_offset
                << ",\n data=" << node->data << ")";
    });

ConstantPoolInfo::ConstantPoolInfo(String pool_name, Array<Target> targets,
                                   Array<ConstantInfo> constant_info_array,
                                   PoolInfoProperties properties) {
  auto constant_poolinfo_node = make_object<ConstantPoolInfoNode>();
  constant_poolinfo_node->pool_name = pool_name;
  constant_poolinfo_node->constant_info_array = constant_info_array;
  constant_poolinfo_node->targets = targets;

  constant_poolinfo_node->size_hint_bytes = properties->size_hint_bytes;
  constant_poolinfo_node->clock_frequency_hz = properties->clock_frequency_hz;
  constant_poolinfo_node->read_bandwidth_bytes_per_cycle =
      properties->read_bandwidth_bytes_per_cycle;
  constant_poolinfo_node->write_bandwidth_bytes_per_cycle =
      properties->write_bandwidth_bytes_per_cycle;
  constant_poolinfo_node->read_latency_cycles = properties->read_latency_cycles;
  constant_poolinfo_node->write_latency_cycles = properties->write_latency_cycles;
  constant_poolinfo_node->target_burst_bytes = properties->target_burst_bytes;
  constant_poolinfo_node->is_internal = properties->is_internal;
  data_ = std::move(constant_poolinfo_node);
}

TVM_REGISTER_NODE_TYPE(ConstantPoolInfoNode);
TVM_REGISTER_GLOBAL("ir.ConstantPoolInfo")
    .set_body_typed([](String pool_name, Array<Target> targets,
                       Array<ConstantInfo> constant_info_array, PoolInfoProperties properties) {
      return ConstantPoolInfo(pool_name, targets, constant_info_array, properties);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstantPoolInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstantPoolInfoNode*>(ref.get());
      p->stream << "ConstantPoolInfoNode(\n"
                << "  pool_name=" << node->pool_name << ",\n  targets=" << node->targets
                << ",\n  constant_info_array=" << node->constant_info_array
                << ",\n  size_hint_bytes=" << node->size_hint_bytes
                << ",\n  clock_frequency_hz=" << node->clock_frequency_hz
                << ",\n  read_bandwidth_bytes_per_cycle=" << node->read_bandwidth_bytes_per_cycle
                << ",\n  write_bandwidth_bytes_per_cycle=" << node->write_bandwidth_bytes_per_cycle
                << ",\n  read_latency_cycles=" << node->read_latency_cycles
                << ",\n  write_latency_cycles=" << node->write_latency_cycles
                << ",\n  target_burst_bytes=" << node->target_burst_bytes
                << ",\n  is_internal=" << node->is_internal << ")"
                << "\n";
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

ConstantMemoryPools::ConstantMemoryPools(Array<ConstantPoolInfo> pools) {
  auto constant_memory_pools_node = make_object<ConstantMemoryPoolsNode>();
  constant_memory_pools_node->pools = pools;
  data_ = std::move(constant_memory_pools_node);
}

TVM_REGISTER_NODE_TYPE(ConstantMemoryPoolsNode);
TVM_REGISTER_GLOBAL("ir.ConstantMemoryPools").set_body_typed([](Array<ConstantPoolInfo> pools) {
  return ConstantMemoryPools(pools);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstantMemoryPoolsNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstantMemoryPoolsNode*>(ref.get());
      p->stream << "ConstantMemoryPoolsNode(\n"
                << "pools=" << node->pools << ")";
    });
}  // namespace tvm
