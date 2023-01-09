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
// Acknowledgement: l2flush struct in nvbench project.
// Reference:
// https://github.com/NVIDIA/nvbench/blob/1a13a2e724b8aa8aee27649ac6878babb63862a6/nvbench/detail/l2flush.cuh
#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "cuda_common.h"

namespace tvm {

namespace runtime {

class L2Flush {
 public:
  L2Flush() : initialized_(false), l2_size_(0), l2_buffer_(nullptr) {}

  ~L2Flush() {
    if (l2_size_ > 0) {
      CUDA_CALL(cudaFree(l2_buffer_));
    }
  }

  void Flush() {
    if (!initialized_) {
      // initialize l2_buffer_ and l2_size_
      initialized_ = true;
      int device_id;
      CUDA_CALL(cudaGetDevice(&device_id));
      CUDA_CALL(cudaDeviceGetAttribute(&l2_size_, cudaDevAttrL2CacheSize, device_id));
      if (l2_size_ > 0) {
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&l2_buffer_), l2_size_));
      }
    }
    cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
    if (l2_size_ > 0) {
      CUDA_CALL(cudaMemsetAsync(l2_buffer_, 0, l2_size_, stream));
    }
  }

  static L2Flush* ThreadLocal();

 private:
  bool initialized_ = false;
  int l2_size_;
  int* l2_buffer_;
};

typedef dmlc::ThreadLocalStore<L2Flush> L2FlushStore;

L2Flush* L2Flush::ThreadLocal() { return L2FlushStore::Get(); }

TVM_REGISTER_GLOBAL("l2_cache_flush_cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK(L2Flush::ThreadLocal() != nullptr) << "L2Flush::ThreadLocal do not exist.";
  L2Flush::ThreadLocal()->Flush();
});

}  // namespace runtime
}  // namespace tvm
