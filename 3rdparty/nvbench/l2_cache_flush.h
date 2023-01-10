/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 * \file l2_cache_flush.h
 * \brief Functions to flush L2 cache using CUDA's API, adopted from nvbench.
 */
#ifndef L2_CACHE_FLUSH_H_
#define L2_CACHE_FLUSH_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/logging.h>

namespace tvm {
namespace runtime {

#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }

class L2Flush {
 public:
  L2Flush() : initialized_(false), l2_size_(0), l2_buffer_(nullptr) {}

  ~L2Flush() {
    if (l2_size_ > 0) {
      CUDA_CALL(cudaFree(l2_buffer_));
    }
  }

  void Flush(cudaStream_t stream) {
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

}  // namespace runtime
}  // namespace tvm

#endif  // L2_CACHE_FLUSH_H_
