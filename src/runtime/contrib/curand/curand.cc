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
#include <curand.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"
#include "./helper_cuda_kernels.h"

namespace tvm {
namespace runtime {
namespace curand {

#define TVM_CURAND_CALL(func)                                    \
  {                                                              \
    curandStatus_t e = (func);                                   \
    ICHECK(e == CURAND_STATUS_SUCCESS) << "cuRAND error: " << e; \
  }

class CURandGenerator {
 public:
  CURandGenerator() { TVM_CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); }
  ~CURandGenerator() { TVM_CURAND_CALL(curandDestroyGenerator(gen)); }

  void Generate32bit(void* ptr, int64_t n) {
    TVM_CURAND_CALL(curandGenerateNormal(gen, static_cast<float*>(ptr), n, 0.0f, 5.0f));
    cudaDeviceSynchronize();
  }

  void Generate64bit(void* ptr, int64_t n) {
    TVM_CURAND_CALL(curandGenerateNormalDouble(gen, static_cast<double*>(ptr), n, 0.0f, 5.0f));
  }

  curandGenerator_t gen;
};

DeviceAPI* GetCUDADeviceAPI() {
  const PackedFunc* get_cuda_api = runtime::Registry::Get("device_api.cuda");
  ICHECK(get_cuda_api) << "ValueError: TVM is not built with USE_CUDA=ON";
  void* ret = (*get_cuda_api)();
  runtime::DeviceAPI* cuda_api = static_cast<runtime::DeviceAPI*>(ret);
  return cuda_api;
}

int64_t GetTensorSize(DLTensor* tensor) {
  int64_t tensor_size = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    tensor_size *= tensor->shape[i];
  }
  return tensor_size;
}

struct DeferredFunc {
 public:
  explicit DeferredFunc(std::function<void()> func) : func_(func) {}
  ~DeferredFunc() { func_(); }

 private:
  std::function<void()> func_;
};

void RandomFill(DLTensor* tensor) {
  static DeviceAPI* cuda_api = GetCUDADeviceAPI();
  CHECK(tensor->device.device_type == DLDeviceType::kDLCUDA)
      << "ValueError: cuRAND only works on CUDA devices";
  int64_t tensor_size = GetTensorSize(tensor);
  int64_t actual_size = tensor_size % 2 == 0 ? tensor_size : tensor_size + 1;
  if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 16) {
    // curand only works for size % 2 = 0
    void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(float));
    {
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate32bit(data, actual_size);
      ConvertFp32toFp16(/*src=*/data, /*dst=*/tensor->data, /*num=*/tensor_size);
    }
  } else if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 32) {
    if (tensor_size % 2 == 1) {
      void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(float));
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate32bit(data, actual_size);
      cudaMemcpy(tensor->data, data, tensor_size * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
      CURandGenerator().Generate32bit(tensor->data, actual_size);
    }
  } else if (tensor->dtype.code == DLDataTypeCode::kDLFloat && tensor->dtype.bits == 64) {
    if (tensor_size % 2 == 1) {
      void* data = cuda_api->AllocWorkspace(tensor->device, actual_size * sizeof(double));
      DeferredFunc defer([data, tensor]() { cuda_api->FreeWorkspace(tensor->device, data); });
      CURandGenerator().Generate64bit(data, actual_size);
      cudaMemcpy(tensor->data, data, tensor_size * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
      CURandGenerator().Generate64bit(tensor->data, actual_size);
    }
  } else {
    LOG(FATAL) << "ValueError: Unsupported dtype: " << tensor->dtype;
  }
  TVMSynchronize(tensor->device.device_type, tensor->device.device_type, nullptr);
}

TVM_REGISTER_GLOBAL("runtime.contrib.curand.RandomFill").set_body_typed(RandomFill);

}  // namespace curand
}  // namespace runtime
}  // namespace tvm
