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

#include <gtest/gtest.h>
#include <tvm/runtime/container/optional.h>

#if defined(TVM_GRAPH_EXECUTOR_CLML)
#include "../src/runtime/contrib/clml/clml_memory_planner.h"
#include "../src/runtime/contrib/clml/clml_runtime.h"
#include "../src/runtime/opencl/opencl_common.h"
#include "../src/runtime/texture.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

void InitMemoryPlan(tvm::runtime::contrib::CachedLayer& layer) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  layer.on_chip_pool_size.clear();
  layer.on_chip_pool_size.insert({0, cws->onchip_mem_size});
  layer.on_chip_pool_alloc_info.clear();
  layer.alloc_ping_pong = true;
  layer.in_chip_total_free = cws->onchip_mem_size;
  layer.in_chip_total_alloc = 0;
  layer.on_chip_alert_fail = 0;

  for (auto it = cws->ddr_global_pool.begin(); it != cws->ddr_global_pool.end(); it++) {
    clReleaseMemObject(it->first);
  }
  cws->ddr_global_pool.clear();
}

void PlanMemory(tvm::runtime::contrib::CachedLayer& layer, int total_nodes,
                std::map<int, uint32_t>& tensor_sizes,
                std::map<int, std::vector<int>>& input_tensors) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  for (int nid = 0; nid < total_nodes; ++nid) {
    uint32_t size = tensor_sizes[nid];
    size_t offset = -1;
    if (cws->is_on_chip_memory) {
      LOG(WARNING) << "Requesting On-chip:" << nid;
      offset = RequestOnChipMemory(&layer, size);
    }
    if (-1 != offset) {
      LOG(WARNING) << "On Chip not found:" << nid;
      layer.on_chip_pool_alloc_info.insert({offset, nid});
      layer.on_chip_alloc_plan.insert({nid, std::make_pair(size, offset)});
    } else {
      LOG(WARNING) << "Requesting DDR memory:" << nid;
      layer.on_chip_reject.insert({nid, size});
      // DDR Allocation
      auto ddr_mem = RequestDDRMemory(&layer, size);
      layer.ddr_alloc_plan.insert({nid, ddr_mem});
    }

    // Now free up the input tensors on-chip memory for reuse.
    for (auto& input_node : input_tensors[nid]) {
      FreeMemory(&layer, input_node);
    }
  }

#if 0
  // Stats dump
  size_t in_chip_total_alloc = 0;
  size_t total_reject = 0;
  for (auto it = layer.on_chip_alloc_plan.begin(); it != layer.on_chip_alloc_plan.end(); it++) {
    LOG(WARNING) << " On-chip Alloc:" << it->first << " Size:" << it->second.first
              << " Offset:" << it->second.second;
    in_chip_total_alloc += it->second.first;
  }

  for (auto it = layer.on_chip_reject.begin(); it != layer.on_chip_reject.end(); it++) {
    LOG(WARNING) << "Reject:" << it->first << " Size:" << it->second;
    total_reject += it->second;
  }
  LOG(WARNING) << "Total On-chip Alloc:" << in_chip_total_alloc + total_reject
            << " On-Chip:" << in_chip_total_alloc << " Reject:" << total_reject;

  for (auto it = cws->ddr_global_pool.begin(); it != cws->ddr_global_pool.end(); it++) {
    LOG(WARNING) << "DDR Global pool - size:" << it->second.first << " Ref:" << it->second.second;
  }
  for (auto it = layer.ddr_storage_ref_map.begin();
       it != layer.ddr_storage_ref_map.end(); it++) {
    LOG(WARNING) << "DDR Local pool - size:" << it->second.first << " Ref cnt:" << it->second.second;
  }
#endif
}

void CompareOnChipPlan(tvm::runtime::contrib::CachedLayer& layer,
                       const std::vector<int>& on_chip_plan) {
  for (auto& nid : on_chip_plan) {
    EXPECT_EQ(layer.on_chip_alloc_plan.find(nid) == layer.on_chip_alloc_plan.end(), false);
  }
}

void CompareDDRPlan(tvm::runtime::contrib::CachedLayer& layer, const std::vector<int>& ddr_plan) {
  for (auto& nid : ddr_plan) {
    EXPECT_EQ(layer.ddr_alloc_plan.find(nid) == layer.ddr_alloc_plan.end(), false);
  }
}

TEST(CLMLMemoryPlanner, sequential_all_on_chip) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 1});
  layer.storage_ref_map.insert({1, 1});
  layer.storage_ref_map.insert({2, 1});
  layer.storage_ref_map.insert({3, 1});
  layer.storage_ref_map.insert({4, 1});
  layer.storage_ref_map.insert({5, 1});
  layer.storage_ref_map.insert({6, 1});
  layer.storage_ref_map.insert({7, 1});
  layer.storage_ref_map.insert({8, 1});
  layer.storage_ref_map.insert({9, 1});

  layer.life_span.insert({0, 1});
  layer.life_span.insert({1, 2});
  layer.life_span.insert({2, 3});
  layer.life_span.insert({3, 4});
  layer.life_span.insert({4, 5});
  layer.life_span.insert({5, 6});
  layer.life_span.insert({6, 7});
  layer.life_span.insert({7, 8});
  layer.life_span.insert({8, 9});
  layer.life_span.insert({9, 10});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, 1024000});
  tensor_sizes.insert({1, 1024000});
  tensor_sizes.insert({2, 1024000});
  tensor_sizes.insert({3, 1024000});
  tensor_sizes.insert({4, 1024000});
  tensor_sizes.insert({5, 1024000});
  tensor_sizes.insert({6, 1024000});
  tensor_sizes.insert({7, 1024000});
  tensor_sizes.insert({8, 1024000});
  tensor_sizes.insert({9, 1024000});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}},  {1, {0}}, {2, {1}}, {3, {2}}, {4, {3}},
      {5, {4}}, {6, {5}}, {7, {6}}, {8, {7}}, {9, {8}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  CompareDDRPlan(layer, std::vector<int>({}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 0);
}

TEST(CLMLMemoryPlanner, sequential_mixed) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 1});
  layer.storage_ref_map.insert({1, 1});
  layer.storage_ref_map.insert({2, 1});
  layer.storage_ref_map.insert({3, 1});
  layer.storage_ref_map.insert({4, 1});
  layer.storage_ref_map.insert({5, 1});

  layer.life_span.insert({0, 1});
  layer.life_span.insert({1, 2});
  layer.life_span.insert({2, 3});
  layer.life_span.insert({3, 4});
  layer.life_span.insert({4, 5});
  layer.life_span.insert({5, 6});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, 1024000});
  tensor_sizes.insert({1, 1024000});
  tensor_sizes.insert({2, cws->onchip_mem_size + 1});
  tensor_sizes.insert({3, 1024000});
  tensor_sizes.insert({4, cws->onchip_mem_size + 1});
  tensor_sizes.insert({5, 1024000});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}}, {1, {0}}, {2, {1}}, {3, {2}}, {4, {3}}, {5, {4}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({0, 1, 3, 5}));
  CompareDDRPlan(layer, std::vector<int>({2, 4}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 1);
}

TEST(CLMLMemoryPlanner, sequential_all_ddr) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 1});
  layer.storage_ref_map.insert({1, 1});
  layer.storage_ref_map.insert({2, 1});
  layer.storage_ref_map.insert({3, 1});
  layer.storage_ref_map.insert({4, 1});
  layer.storage_ref_map.insert({5, 1});

  layer.life_span.insert({0, 1});
  layer.life_span.insert({1, 2});
  layer.life_span.insert({2, 3});
  layer.life_span.insert({3, 4});
  layer.life_span.insert({4, 5});
  layer.life_span.insert({5, 6});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, cws->onchip_mem_size + 1});
  tensor_sizes.insert({1, cws->onchip_mem_size + 1});
  tensor_sizes.insert({2, cws->onchip_mem_size + 1});
  tensor_sizes.insert({3, cws->onchip_mem_size + 1});
  tensor_sizes.insert({4, cws->onchip_mem_size + 1});
  tensor_sizes.insert({5, cws->onchip_mem_size + 1});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}}, {1, {0}}, {2, {1}}, {3, {2}}, {4, {3}}, {5, {4}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({}));
  CompareDDRPlan(layer, std::vector<int>({0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 2);
}

TEST(CLMLMemoryPlanner, branched_all_on_alive_on_chip) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 9});
  layer.storage_ref_map.insert({1, 8});
  layer.storage_ref_map.insert({2, 7});
  layer.storage_ref_map.insert({3, 6});
  layer.storage_ref_map.insert({4, 5});
  layer.storage_ref_map.insert({5, 4});
  layer.storage_ref_map.insert({6, 3});
  layer.storage_ref_map.insert({7, 2});
  layer.storage_ref_map.insert({8, 1});
  layer.storage_ref_map.insert({9, 1});

  layer.life_span.insert({0, 9});
  layer.life_span.insert({1, 9});
  layer.life_span.insert({2, 9});
  layer.life_span.insert({3, 9});
  layer.life_span.insert({4, 9});
  layer.life_span.insert({5, 9});
  layer.life_span.insert({6, 9});
  layer.life_span.insert({7, 9});
  layer.life_span.insert({8, 9});
  layer.life_span.insert({9, 10});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, 102400});
  tensor_sizes.insert({1, 102400});
  tensor_sizes.insert({2, 102400});
  tensor_sizes.insert({3, 102400});
  tensor_sizes.insert({4, 102400});
  tensor_sizes.insert({5, 102400});
  tensor_sizes.insert({6, 102400});
  tensor_sizes.insert({7, 102400});
  tensor_sizes.insert({8, 102400});
  tensor_sizes.insert({9, 102400});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}},
      {1, {0}},
      {2, {0, 1}},
      {3, {0, 1, 2}},
      {4, {0, 1, 2, 3}},
      {5, {0, 1, 2, 3, 4}},
      {6, {0, 1, 2, 3, 4, 5}},
      {7, {0, 1, 2, 3, 4, 5, 6}},
      {8, {0, 1, 2, 3, 4, 5, 6, 7}},
      {9, {0, 1, 2, 3, 4, 5, 6, 7, 8}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  CompareDDRPlan(layer, std::vector<int>({}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 0);
}

TEST(CLMLMemoryPlanner, branched_all_on_alive_mixed) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 9});
  layer.storage_ref_map.insert({1, 8});
  layer.storage_ref_map.insert({2, 7});
  layer.storage_ref_map.insert({3, 6});
  layer.storage_ref_map.insert({4, 5});
  layer.storage_ref_map.insert({5, 4});
  layer.storage_ref_map.insert({6, 3});
  layer.storage_ref_map.insert({7, 2});
  layer.storage_ref_map.insert({8, 1});
  layer.storage_ref_map.insert({9, 1});

  layer.life_span.insert({0, 9});
  layer.life_span.insert({1, 9});
  layer.life_span.insert({2, 9});
  layer.life_span.insert({3, 9});
  layer.life_span.insert({4, 9});
  layer.life_span.insert({5, 9});
  layer.life_span.insert({6, 9});
  layer.life_span.insert({7, 9});
  layer.life_span.insert({8, 9});
  layer.life_span.insert({9, 10});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, 102400});
  tensor_sizes.insert({1, 102400});
  tensor_sizes.insert({2, cws->onchip_mem_size + 1});
  tensor_sizes.insert({3, 102400});
  tensor_sizes.insert({4, cws->onchip_mem_size + 1});
  tensor_sizes.insert({5, 102400});
  tensor_sizes.insert({6, cws->onchip_mem_size + 1});
  tensor_sizes.insert({7, 102400});
  tensor_sizes.insert({8, 102400});
  tensor_sizes.insert({9, 102400});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}},
      {1, {0}},
      {2, {0, 1}},
      {3, {0, 1, 2}},
      {4, {0, 1, 2, 3}},
      {5, {0, 1, 2, 3, 4}},
      {6, {0, 1, 2, 3, 4, 5}},
      {7, {0, 1, 2, 3, 4, 5, 6}},
      {8, {0, 1, 2, 3, 4, 5, 6, 7}},
      {9, {0, 1, 2, 3, 4, 5, 6, 7, 8}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({0, 1, 3, 5, 7, 8, 9}));
  CompareDDRPlan(layer, std::vector<int>({2, 4, 6}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 3);
}

TEST(CLMLMemoryPlanner, branched_all_on_alive_all_ddr) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 9});
  layer.storage_ref_map.insert({1, 8});
  layer.storage_ref_map.insert({2, 7});
  layer.storage_ref_map.insert({3, 6});
  layer.storage_ref_map.insert({4, 5});
  layer.storage_ref_map.insert({5, 4});
  layer.storage_ref_map.insert({6, 3});
  layer.storage_ref_map.insert({7, 2});
  layer.storage_ref_map.insert({8, 1});
  layer.storage_ref_map.insert({9, 1});

  layer.life_span.insert({0, 9});
  layer.life_span.insert({1, 9});
  layer.life_span.insert({2, 9});
  layer.life_span.insert({3, 9});
  layer.life_span.insert({4, 9});
  layer.life_span.insert({5, 9});
  layer.life_span.insert({6, 9});
  layer.life_span.insert({7, 9});
  layer.life_span.insert({8, 9});
  layer.life_span.insert({9, 10});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, cws->onchip_mem_size + 1});
  tensor_sizes.insert({1, cws->onchip_mem_size + 1});
  tensor_sizes.insert({2, cws->onchip_mem_size + 1});
  tensor_sizes.insert({3, cws->onchip_mem_size + 1});
  tensor_sizes.insert({4, cws->onchip_mem_size + 1});
  tensor_sizes.insert({5, cws->onchip_mem_size + 1});
  tensor_sizes.insert({6, cws->onchip_mem_size + 1});
  tensor_sizes.insert({7, cws->onchip_mem_size + 1});
  tensor_sizes.insert({8, cws->onchip_mem_size + 1});
  tensor_sizes.insert({9, cws->onchip_mem_size + 1});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}},
      {1, {0}},
      {2, {0, 1}},
      {3, {0, 1, 2}},
      {4, {0, 1, 2, 3}},
      {5, {0, 1, 2, 3, 4}},
      {6, {0, 1, 2, 3, 4, 5}},
      {7, {0, 1, 2, 3, 4, 5, 6}},
      {8, {0, 1, 2, 3, 4, 5, 6, 7}},
      {9, {0, 1, 2, 3, 4, 5, 6, 7, 8}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({}));
  CompareDDRPlan(layer, std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 10);
}

TEST(CLMLMemoryPlanner, skip_connections_mixed) {
  tvm::runtime::contrib::CLMLWorkspace* cws = tvm::runtime::contrib::CLMLWorkspace::Global();

  tvm::runtime::contrib::CachedLayer layer;
  InitMemoryPlan(layer);

  layer.storage_ref_map.insert({0, 2});
  layer.storage_ref_map.insert({1, 1});
  layer.storage_ref_map.insert({2, 2});
  layer.storage_ref_map.insert({3, 1});
  layer.storage_ref_map.insert({4, 2});
  layer.storage_ref_map.insert({5, 1});
  layer.storage_ref_map.insert({6, 2});
  layer.storage_ref_map.insert({7, 1});
  layer.storage_ref_map.insert({8, 1});
  layer.storage_ref_map.insert({9, 1});

  layer.life_span.insert({0, 2});
  layer.life_span.insert({1, 2});
  layer.life_span.insert({2, 4});
  layer.life_span.insert({3, 4});
  layer.life_span.insert({4, 6});
  layer.life_span.insert({5, 6});
  layer.life_span.insert({6, 8});
  layer.life_span.insert({7, 8});
  layer.life_span.insert({8, 9});
  layer.life_span.insert({9, 10});

  std::map<int, uint32_t> tensor_sizes;
  tensor_sizes.insert({0, 1024000});
  tensor_sizes.insert({1, 1024000});
  tensor_sizes.insert({2, cws->onchip_mem_size + 1});
  tensor_sizes.insert({3, cws->onchip_mem_size + 1});
  tensor_sizes.insert({4, 1024000});
  tensor_sizes.insert({5, 1024000});
  tensor_sizes.insert({6, cws->onchip_mem_size + 1});
  tensor_sizes.insert({7, cws->onchip_mem_size + 1});
  tensor_sizes.insert({8, 1024000});
  tensor_sizes.insert({9, cws->onchip_mem_size + 1});

  std::map<int, std::vector<int>> input_tensors{
      {0, {}},  {1, {0}},    {2, {0, 1}}, {3, {2}},    {4, {2, 3}},
      {5, {4}}, {6, {4, 5}}, {7, {6}},    {8, {6, 7}}, {9, {8}},
  };

  PlanMemory(layer, input_tensors.size(), tensor_sizes, input_tensors);

  CompareOnChipPlan(layer, std::vector<int>({0, 1, 4, 5, 8}));
  CompareDDRPlan(layer, std::vector<int>({2, 3, 6, 7, 9}));
  EXPECT_EQ(cws->ddr_global_pool.size(), 2);
}

#endif  // TVM_GRAPH_EXECUTOR_CLML
