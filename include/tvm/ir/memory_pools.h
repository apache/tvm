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
 * \file tvm/ir/memory_pools.h
 * \brief The object definition for relay.build argument type of memory pools
 */
#ifndef TVM_IR_MEMORY_POOLS_H_
#define TVM_IR_MEMORY_POOLS_H_

#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>

namespace tvm {

/*!
 * \brief Describes a pool of memory accessible by one or more targets.
 */
struct PoolInfoNode : public Object {
  /*! \brief The name of the memory pool */
  String pool_name;
  /*! \brief The expected size hint to be used by the allocator.
   * The size_hint_bytes is set to kUnrestrictedPoolSizeHint
   * to indicate the pool is not size restricted.
   */
  Integer size_hint_bytes;
  /*! \brief The accessibility from each Target */
  Map<Target, String> target_access;  // 'rw' or 'ro'
  /*! \brief The clock frequency of the memory in Hz */
  Integer clock_frequency_hz;
  /*! \brief The read bandwidth in bytes/cycle */
  Integer read_bandwidth_bytes_per_cycle;
  /*! \brief The write bandwidth in bytes/cycle */
  Integer write_bandwidth_bytes_per_cycle;
  /*! \brief The read latency in cycles */
  Integer read_latency_cycles;
  /*! \brief The write latency in cycles */
  Integer write_latency_cycles;
  /*! \brief The burst length in bytes for each Target */
  Map<Target, Integer> target_burst_bytes;
  /*! \brief Whether pool is internally generated.
   * The internal pools will be generated as part of
   * the entry point code generation of the executor
   */
  bool is_internal = false;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pool_name", &pool_name);
    v->Visit("size_hint_bytes", &size_hint_bytes);
    v->Visit("target_access", &target_access);
    v->Visit("clock_frequency_hz", &clock_frequency_hz);
    v->Visit("read_bandwidth_bytes_per_cycle", &read_bandwidth_bytes_per_cycle);
    v->Visit("write_bandwidth_bytes_per_cycle", &write_bandwidth_bytes_per_cycle);
    v->Visit("read_latency_cycles", &read_latency_cycles);
    v->Visit("write_latency_cycles", &write_latency_cycles);
    v->Visit("target_burst_bytes", &target_burst_bytes);
    v->Visit("is_internal", &is_internal);
  }

  bool SEqualReduce(const PoolInfoNode* other, SEqualReducer equal) const {
    return equal(pool_name, other->pool_name) && equal(size_hint_bytes, other->size_hint_bytes) &&
           equal(target_access, other->target_access) &&
           equal(target_access, other->target_access) &&
           equal(clock_frequency_hz, other->clock_frequency_hz) &&
           equal(read_bandwidth_bytes_per_cycle, other->read_bandwidth_bytes_per_cycle) &&
           equal(write_bandwidth_bytes_per_cycle, other->write_bandwidth_bytes_per_cycle) &&
           equal(read_latency_cycles, other->read_latency_cycles) &&
           equal(write_latency_cycles, other->write_latency_cycles) &&
           equal(target_burst_bytes, other->target_burst_bytes) &&
           equal(is_internal, other->is_internal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(pool_name);
    hash_reduce(size_hint_bytes);
    hash_reduce(target_access);
    hash_reduce(clock_frequency_hz);
    hash_reduce(read_bandwidth_bytes_per_cycle);
    hash_reduce(write_bandwidth_bytes_per_cycle);
    hash_reduce(read_latency_cycles);
    hash_reduce(write_latency_cycles);
    hash_reduce(target_burst_bytes);
    hash_reduce(is_internal);
  }

  static constexpr const char* _type_key = "ir.PoolInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(PoolInfoNode, Object);
};

/*!
 * \brief The string parameter to indicate read and write access to a pool
 * This needs to be kept in sync with PoolInfo.READ_WRITE_ACCESS in
 * python/tvm/ir/memory_pools.py
 */
static constexpr const char* kTargetPoolReadWriteAccess = "rw";

/*!
 * \brief The string parameter to indicate read only access to a pool
 * This needs to be kept in sync with PoolInfo.READ_ONLY_ACCESS in
 * python/tvm/ir/memory_pools.py
 */
static constexpr const char* kTargetPoolReadOnlyAccess = "ro";

/*! \brief The PoolSize is unrestricted for the memory planner */
static const int kUnrestrictedPoolSizeHint = -1;

/*! \brief The clock frequency is not known */
static const int kUnknownClockFrequency = -1;

/*! \brief The read bandwidth is not known */
static const int kUnknownReadBandwidth = -1;

/*! \brief The write bandwidth is not known */
static const int kUnknownWriteBandwidth = -1;

class PoolInfo : public ObjectRef {
 public:
  TVM_DLL PoolInfo(String pool_name, Map<Target, String> target_access,
                   Integer size_hint_bytes = kUnrestrictedPoolSizeHint,
                   Integer clock_frequency_hz = kUnknownClockFrequency,
                   Integer read_bandwidth_bytes_per_cycle = kUnknownReadBandwidth,
                   Integer write_bandwidth_bytes_per_cycle = kUnknownWriteBandwidth,
                   Integer read_latency_cycles = 0, Integer write_latency_cycles = 0,
                   Map<Target, Integer> target_burst_bytes = {}, Bool is_internal = Bool(false));
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PoolInfo, ObjectRef, PoolInfoNode);
};

struct WorkspaceMemoryPoolsNode : public Object {
  Array<PoolInfo> pools;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pools", &pools); }

  bool SEqualReduce(const WorkspaceMemoryPoolsNode* other, SEqualReducer equal) const {
    return equal(pools, other->pools);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(pools); }

  static constexpr const char* _type_key = "ir.WorkspaceMemoryPools";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkspaceMemoryPoolsNode, Object);
};

class WorkspaceMemoryPools : public ObjectRef {
 public:
  TVM_DLL WorkspaceMemoryPools(Array<PoolInfo> pools);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(WorkspaceMemoryPools, ObjectRef, WorkspaceMemoryPoolsNode);
};

}  // namespace tvm

#endif  // TVM_IR_MEMORY_POOLS_H_
