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
 * \file src/runtime/contrib/clml/clml_memory_planner.cc
 * \brief Various memory planning methods.
 */
#ifdef TVM_GRAPH_EXECUTOR_CLML
#include "clml_memory_planner.h"

#include <map>
#include <utility>

#include "clml_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

/*!
 * Release memory after use.
 *
 */
void FreeMemory(CachedLayer* layer, int nid) {
  LOG_MEM << "FreeMemory:" << nid;
  if (layer->storage_ref_map.find(nid) != layer->storage_ref_map.end()) {
    LOG_MEM << "Ref Cnt:" << layer->storage_ref_map[nid];
    layer->storage_ref_map[nid]--;
    if (0 == layer->storage_ref_map[nid]) {
      LOG_MEM << "Ref Cnt Nill";
      // Look into on-chip allocation
      for (auto it = layer->on_chip_pool_alloc_info.begin();
           it != layer->on_chip_pool_alloc_info.end(); it++) {
        if (it->second == nid) {
          LOG_MEM << "Free Segment:" << it->first << " Nid:" << nid;
          layer->in_chip_total_free += layer->on_chip_pool_size[it->first];
          layer->in_chip_total_alloc -= layer->on_chip_pool_size[it->first];
          layer->on_chip_pool_alloc_info.erase(it->first);
          return;
        }
      }
      // Look into DDR allocation
      if (layer->ddr_alloc_plan.find(nid) != layer->ddr_alloc_plan.end()) {
        LOG_MEM << "Free DDR segment from local pool";
        layer->ddr_storage_ref_map[layer->ddr_alloc_plan[nid]].second = false;
        return;
      }
      LOG_MEM << "*** Not a managed memory buffer";
    }
  } else {
    LOG_MEM << "Not in storage ref map :" << nid;
  }
}

/*!
 * \brief Partition and allocate
 *
 */
size_t PartitionAndAllocate(CachedLayer* layer, size_t segment_start, size_t size, bool is_left) {
  LOG_MEM << "PartitionAndAllocate:" << segment_start << " Size:" << size
          << " Is Begin:" << is_left;
  size_t segment_size = layer->on_chip_pool_size[segment_start];
  size_t left_space = segment_size - size;

  layer->in_chip_total_free -= size;
  layer->in_chip_total_alloc += size;

  if (is_left) {
    // Start allocation
    layer->on_chip_pool_size[segment_start] = size;
    if (left_space) {
      layer->on_chip_pool_size.insert({segment_start + size, left_space});
    }
    return segment_start;
  } else {
    // End allocation
    if (left_space) {
      layer->on_chip_pool_size[segment_start] = left_space;
    }
    layer->on_chip_pool_size.insert({segment_start + left_space, size});
    return segment_start + left_space;
  }
}

/*!
 * \brief Ping-Pong allocation with in best fit
 *
 */
size_t PingPongAllocate(CachedLayer* layer, const std::map<size_t, size_t>& segments, size_t size) {
  /*
   * segments contains all free segments details (start, size) that can fit the requirement
   * PingPong Allocation Strategy:
   * Here we find the smallest segment among all.
   * We allocate at begining or end of this segment based on the ping-pong flag.
   * Ping-pong allocation helps to have largest possible free segment at center
   * for most of the graphs.
   *
   */
  ssize_t free_start;
  ssize_t free_size;
  ssize_t last_found_size = CLMLWorkspace::Global()->onchip_mem_size + 1;

  for (auto it = segments.begin(); it != segments.end(); it++) {
    if (it->second < last_found_size) {
      free_start = it->first;
      free_size = it->second;
      last_found_size = it->second;
      LOG_MEM << "Mem Found:" << free_start << " Size:" << free_size;
    }
  }

  LOG_MEM << "Alloc On-chip Mem:" << free_start << " Size:" << free_size
          << " PingPong:" << layer->alloc_ping_pong;

  // Allocate on-chip memory
  layer->alloc_ping_pong ^= 1;
  return PartitionAndAllocate(layer, free_start, size, layer->alloc_ping_pong);
}

/*!
 * \brief Allocate on-chip memory.
 *
 */
size_t RequestOnChipMemory(CachedLayer* layer, size_t size) {
  LOG_MEM << "Request On-Chip Mem:" << size;
  // Optimize for any fragmented parts
  bool any_merge = true;
  while (any_merge) {
    any_merge = false;
    for (auto it = layer->on_chip_pool_size.begin(); it != layer->on_chip_pool_size.end(); it++) {
      if ((layer->on_chip_pool_alloc_info.find(it->first) ==
           layer->on_chip_pool_alloc_info.end()) &&
          (layer->on_chip_pool_alloc_info.find(it->first + it->second) ==
           layer->on_chip_pool_alloc_info.end()) &&
          (it->first + it->second < CLMLWorkspace::Global()->onchip_mem_size)) {
        size_t left_begin = it->first;
        size_t left_size = it->second;
        size_t right_size = layer->on_chip_pool_size[it->first + it->second];
        LOG_MEM << "Merge:" << left_begin << " Size:" << left_size << " with :" << right_size;
        layer->on_chip_pool_size[left_begin] = left_size + right_size;
        layer->on_chip_pool_size.erase(left_begin + left_size);
        any_merge = true;
        break;
      }
    }
  }

  // Look for any best fit free fragment
  std::map<size_t, size_t> feasible_segments;
  for (auto it = layer->on_chip_pool_size.begin(); it != layer->on_chip_pool_size.end(); it++) {
    if (layer->on_chip_pool_alloc_info.find(it->first) == layer->on_chip_pool_alloc_info.end()) {
      if (it->second >= size) {
        LOG_MEM << "Mem Pool:" << it->first << " - " << it->first + it->second << ":" << it->second
                << " - Free";
        feasible_segments.insert({it->first, it->second});
      } else {
        LOG_MEM << "Mem Pool:" << it->first << " - " << it->first + it->second << ":" << it->second
                << " - Doesn't fit";
      }
    } else {
      LOG_MEM << "Mem Pool:" << it->first << " - " << it->first + it->second << ":" << it->second
              << " - Busy";
    }
  }
  if (0 == feasible_segments.size()) {
    LOG_MEM << "No Suitable Mem Found:" << size << " Free Size:" << layer->in_chip_total_free;
    if (size <= layer->in_chip_total_free) {
      LOG_STATS << "*** ALERT ***: Couldn't allocate due to fragmentation:" << size
                << " Total Free:" << layer->in_chip_total_free;
      layer->on_chip_alert_fail += size;
    }
    return -1;
  }

  return PingPongAllocate(layer, feasible_segments, size);
}

/*!
 * \brief Allocate DDR memory for requested size.
 *
 */
cl_mem RequestDDRMemory(CachedLayer* layer, size_t size) {
  // Look for local storage map for a best fit
  auto cws = CLMLWorkspace::Global();
  cl_mem memptr = nullptr;
  size_t best_fit = INT_MAX;
  for (auto it = layer->ddr_storage_ref_map.begin(); it != layer->ddr_storage_ref_map.end(); it++) {
    if ((it->second.first >= size) && (false == it->second.second)) {
      if (best_fit > it->second.first) {
        memptr = it->first;
        best_fit = it->second.first;
      }
    }
  }

  if (memptr) {
    LOG_MEM << "Reuse from local pool";
    layer->ddr_storage_ref_map[memptr].second = true;
    return memptr;
  }
  // No available buffer in local pool, look for global pool
  for (auto it = cws->ddr_global_pool.begin(); it != cws->ddr_global_pool.end(); it++) {
    if ((it->second.first >= size) &&
        (layer->ddr_storage_ref_map.find(it->first) == layer->ddr_storage_ref_map.end())) {
      // Found a buffer in global pool. Insert in local pool and then use.
      if (best_fit > it->second.first) {
        memptr = it->first;
        best_fit = it->second.first;
      }
    }
  }

  if (memptr) {
    LOG_MEM << "Reuse from global pool";
    cws->ddr_global_pool[memptr].second += 1;
    layer->ddr_storage_ref_map.insert(
        {memptr, std::make_pair(cws->ddr_global_pool[memptr].first, true)});
    return memptr;
  }

  // Allocate a fresh buffer in global then use in local pool.
  LOG_MEM << "Allocating fresh buffer in global pool";
  memptr = AllocateDDRTensorMemory(size);
  cws->ddr_global_pool.insert({memptr, std::make_pair(size, 1)});
  layer->ddr_storage_ref_map.insert({memptr, std::make_pair(size, true)});

  return memptr;
}

/*!
 * \brief Release memory from global pool.
 *
 */
void ReleaseDDRMemory(cl_mem memptr) {
  cl_int result;
  auto cws = CLMLWorkspace::Global();
  cws->ddr_global_pool[memptr].second -= 1;
  if (0 == cws->ddr_global_pool[memptr].second) {
    LOG_MEM << "Release DDR mem from global pool";
    result = clReleaseMemObject(memptr);
    ICHECK(result == CL_SUCCESS) << "clReleaseMemObject:" << result;
    cws->ddr_global_pool.erase(memptr);
  }
}

}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm
#endif
