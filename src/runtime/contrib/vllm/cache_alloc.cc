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
#include <cuda_runtime.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace vllm {

Array<NDArray> AllocateKVCache(int head_size, int num_layers, int num_heads, int block_size,
                               int num_blocks) {
  Array<NDArray> cache;
  int element_size = 2;
  int vec_size = 16 / element_size;

  int device_id;
  cudaGetDevice(&device_id);

  DLDevice dev{DLDeviceType::kDLCUDA, device_id};

  for (int i = 0; i < num_layers; ++i) {
    NDArray key_blocks =
        NDArray::Empty({num_blocks, num_heads, head_size / vec_size, block_size, vec_size},
                       runtime::DataType::Float(16), dev);
    NDArray value_blocks = NDArray::Empty({num_blocks, num_heads, head_size, block_size},
                                          runtime::DataType::Float(16), dev);
    cache.push_back(key_blocks);
    cache.push_back(value_blocks);
  }

  return cache;
}

TVM_REGISTER_GLOBAL("tvm.contrib.vllm.allocate_kv_cache").set_body_typed(AllocateKVCache);

}  // namespace vllm
}  // namespace runtime
}  // namespace tvm
