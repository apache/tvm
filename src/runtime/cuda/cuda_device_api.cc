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
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/thread_local.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>

#include <cstring>

#include "cuda_common.h"

namespace tvm {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { CUDA_CALL(cudaSetDevice(dev.device_id)); }
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    int value = 0;
    switch (kind) {
      case kExist: {
        int count;
        auto err = cudaGetDeviceCount(&count);
        value = (err == cudaSuccess && static_cast<int>(dev.device_id < count));
        break;
      }
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, dev.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, dev.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, dev.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, dev.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, dev.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        CUDA_DRIVER_CALL(cuDeviceGetName(&name[0], name.size(), dev.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, dev.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, dev.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, dev.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, dev.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, dev.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, dev.device_id));
        break;
      }
      case kGcnArch:
        return;
      case kApiVersion: {
        *rv = CUDA_VERSION;
        return;
      }
      case kDriverVersion:
        return;
      case kL2CacheSizeBytes: {
        // Get size of device l2 cache size in bytes.
        int l2_size = 0;
        CUDA_CALL(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev.device_id));
        *rv = l2_size;
        return;
      }
      case kTotalGlobalMemory: {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, dev.device_id));
        int64_t total_global_memory = prop.totalGlobalMem;
        *rv = total_global_memory;
        return;
      }
      case kAvailableGlobalMemory: {
        size_t free_mem, total_mem;
        CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
        *rv = static_cast<int64_t>(free_mem);
        return;
      }
      case kImagePitchAlignment:
        return;
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    ICHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLCUDAHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      CUDA_CALL(cudaMallocHost(&ret, nbytes));
    } else {
      CUDA_CALL(cudaSetDevice(dev.device_id));
      size_t free_mem, total_mem;
      CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
      VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
              << " bytes currently free out of " << total_mem << " bytes available";
      CUDA_CALL(cudaMalloc(&ret, nbytes));
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    if (std::uncaught_exceptions() && cudaPeekAtLastError() == cudaErrorIllegalAddress) {
      // For most CUDA calls, an error from an API call will be
      // immediately reported, and raised as an exception.  However,
      // errors raised from async kernel execution leave the CUDA
      // driver in an inconsistent state.  These errors are "sticky",
      // and are never cleared. (See [0] for more details.)
      //
      // If we are currently unwinding the stack due to a thrown
      // exception, and the CUDA driver is in an unrecoverable error,
      // do not attempt to free the CUDA allocations.  Performing any
      // CUDA API call while in this state will throw an additional
      // exception, causing a segfault.  In this case, it is better to
      // allow the original error to continue propagating.
      //
      // [0] https://forums.developer.nvidia.com/t/cuda-errors-determine-sticky-ness/271625
      return;
    }

    if (dev.device_type == kDLCUDAHost) {
      VLOG(1) << "freeing host memory";
      CUDA_CALL(cudaFreeHost(ptr));
    } else {
      CUDA_CALL(cudaSetDevice(dev.device_id));
      VLOG(1) << "freeing device memory";
      CUDA_CALL(cudaFree(ptr));
    }
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (dev_from.device_type == kDLCUDAHost) {
      dev_from.device_type = kDLCPU;
    }

    if (dev_to.device_type == kDLCUDAHost) {
      dev_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, cu_stream);
      }
    } else if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCPU) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t retval;
    CUDA_CALL(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    CUDAThreadEntry::ThreadLocal()->stream = static_cast<cudaStream_t>(stream);
  }

  TVMStreamHandle GetCurrentStream(Device dev) final {
    return static_cast<TVMStreamHandle>(CUDAThreadEntry::ThreadLocal()->stream);
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, cudaMemcpyKind kind,
                      cudaStream_t stream) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
  }
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry() : pool(kDLCUDA, CUDADeviceAPI::Global()) {}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() { return CUDAThreadStore::Get(); }

TVM_FFI_REGISTER_GLOBAL("device_api.cuda").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_FFI_REGISTER_GLOBAL("device_api.cuda_host")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      DeviceAPI* ptr = CUDADeviceAPI::Global();
      *rv = static_cast<void*>(ptr);
    });

class CUDATimerNode : public TimerNode {
 public:
  virtual void Start() {
    // This initial cudaEventRecord is sometimes pretty slow (~100us). Does
    // cudaEventRecord do some stream synchronization?
    CUDA_CALL(cudaEventRecord(start_, CUDAThreadEntry::ThreadLocal()->stream));
  }
  virtual void Stop() { CUDA_CALL(cudaEventRecord(stop_, CUDAThreadEntry::ThreadLocal()->stream)); }
  virtual int64_t SyncAndGetElapsedNanos() {
    CUDA_CALL(cudaEventSynchronize(stop_));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~CUDATimerNode() {
    CUDA_CALL(cudaEventDestroy(start_));
    CUDA_CALL(cudaEventDestroy(stop_));
  }
  CUDATimerNode() {
    CUDA_CALL(cudaEventCreate(&start_));
    CUDA_CALL(cudaEventCreate(&stop_));
  }

  static constexpr const char* _type_key = "runtime.cuda.CUDATimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDATimerNode, TimerNode);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

TVM_REGISTER_OBJECT_TYPE(CUDATimerNode);

TVM_FFI_REGISTER_GLOBAL("profiling.timer.cuda").set_body_typed([](Device dev) {
  return Timer(make_object<CUDATimerNode>());
});

TVM_DLL String GetCudaFreeMemory() {
  size_t free_mem, total_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
  std::stringstream ss;
  ss << "Current CUDA memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return ss.str();
}

TVM_FFI_REGISTER_GLOBAL("runtime.GetCudaFreeMemory").set_body_typed(GetCudaFreeMemory);

TVM_FFI_REGISTER_GLOBAL("runtime.get_cuda_stream").set_body_typed([]() {
  return static_cast<void*>(CUDAThreadEntry::ThreadLocal()->stream);
});

TVM_DLL int GetCudaDeviceCount() {
  int count;
  CUDA_CALL(cudaGetDeviceCount(&count));
  return count;
}

TVM_FFI_REGISTER_GLOBAL("runtime.GetCudaDeviceCount").set_body_typed(GetCudaDeviceCount);

#if (CUDA_VERSION >= 12000)
/**
 * \brief FFI wrapper for cuTensorMapEncodeTiled.
 *
 * This function registers a global function `runtime.cuTensorMapEncodeTiled` that can be
 * called from other parts of the TVM runtime (e.g., Python). It wraps the CUDA Driver API
 * function `cuTensorMapEncodeTiled`, which initializes a tensor map descriptor (CUtensorMap).
 *
 * \param tensor_map (handle): A `void*` pointer to the CUtensorMap object to be initialized.
 * \param tensor_dtype (DataType): The TVM data type of the tensor.
 * \param tensor_rank (int): The rank (number of dimensions) of the tensor.
 * \param tensor_ptr (handle): A `void*` pointer to the start of the tensor in global memory.
 * \param global_shape (int...): `tensor_rank` integer arguments for the global tensor dimensions.
 * \param global_strides (int...): `tensor_rank - 1` integer arguments for the global tensor
 * strides. The stride for the innermost dimension is not provided as it's assumed to be contiguous.
 * \param shared_shape (int...): `tensor_rank` integer arguments for the shape of the tile (box)
 * in shared memory.
 * \param shared_strides (int...): `tensor_rank` integer arguments for the strides of the tile (box)
 * in shared memory.
 * \param interleaved_kind (int): An integer corresponding to the CUtensorMapInterleave enum.
 * \param swizzle_kind (int): An integer corresponding to the CUtensorMapSwizzle enum.
 * \param l2_promotion_kind (int): An integer corresponding to the CUtensorMapL2promotion enum.
 * \param oob_fill_kind (int): An integer corresponding to the CUtensorMapFloatOOBfill enum.
 */
TVM_FFI_REGISTER_GLOBAL("runtime.cuTensorMapEncodeTiled")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      CHECK_GE(args.size(), 4) << "init_cuTensorMap expects at least 4 arguments";
      size_t arg_cnt = 0;
      CUtensorMap* tensor_map = static_cast<CUtensorMap*>(args[arg_cnt++].cast<void*>());
      runtime::DataType tensor_dtype = args[arg_cnt++].cast<runtime::DataType>();
      uint32_t tensor_rank = static_cast<uint32_t>(args[arg_cnt++].cast<int32_t>());
      void* tensor_ptr = static_cast<void*>(args[arg_cnt++].cast<void*>());

      CHECK_EQ(args.size(), 4 + tensor_rank * 4 + 3)
          << "cuTensorMapEncodeTiled expects " << 4 + tensor_rank * 4 + 3 << " arguments"
          << "tensor_map, tensor_dtype, tensor_rank, tensor_ptr, global_shape(" << tensor_rank
          << "), global_strides(" << tensor_rank - 1 << "), shared_shape(" << tensor_rank
          << "), shared_strides(" << tensor_rank << "), interleaved_kind, swizzle_kind"
          << ", l2_promotion_kind, oob_fill_kind";

      std::vector<cuuint64_t> global_shape(tensor_rank);
      std::vector<cuuint64_t> global_strides(tensor_rank);
      std::vector<uint32_t> shared_shape(tensor_rank);
      std::vector<uint32_t> shared_strides(tensor_rank);
      for (size_t i = 0; i < tensor_rank; ++i) {
        global_shape[i] = static_cast<cuuint64_t>(args[arg_cnt++].cast<int64_t>());
      }
      for (size_t i = 0; i < tensor_rank - 1; ++i) {
        global_strides[i] = static_cast<cuuint64_t>(args[arg_cnt++].cast<int64_t>());
        CHECK_EQ(global_strides[i] % 16, 0) << "global strides must be multiple of 16";
      }
      for (size_t i = 0; i < tensor_rank; ++i) {
        shared_shape[i] = static_cast<uint32_t>(args[arg_cnt++].cast<int32_t>());
        CHECK_GE(shared_shape[i], 0) << "boxDim must be non-negative";
        CHECK_LE(shared_shape[i], 256) << "boxDim must be less than or equal to 256";
      }
      for (size_t i = 0; i < tensor_rank; ++i) {
        shared_strides[i] = static_cast<uint32_t>(args[arg_cnt++].cast<int32_t>());
      }
      auto interleaved_kind = static_cast<CUtensorMapInterleave>(args[arg_cnt++].cast<int>());
      auto swizzle_kind = static_cast<CUtensorMapSwizzle>(args[arg_cnt++].cast<int>());
      auto l2_promotion_kind = static_cast<CUtensorMapL2promotion>(args[arg_cnt++].cast<int>());
      auto oob_fill_kind = static_cast<CUtensorMapFloatOOBfill>(args[arg_cnt++].cast<int>());

      ICHECK_EQ(tensor_dtype.lanes(), 1)
          << "Expect tensor_dtype to have lanes=1, but get " << tensor_dtype;
      CUtensorMapDataType cu_dtype;
      switch (tensor_dtype.code()) {
        case DataType::kInt:
          // int
          switch (tensor_dtype.bits()) {
            case 8:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
              break;
            case 32:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_INT32;
              break;
            case 64:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_INT64;
              break;
            default:
              LOG(FATAL) << "Unsupported data type " << runtime::DLDataTypeToString(tensor_dtype);
          }
          break;
        case DataType::kUInt:
          // unsigned int
          switch (tensor_dtype.bits()) {
            case 8:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
              break;
            case 16:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT16;
              break;
            case 32:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
              break;
            case 64:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT64;
              break;
            default:
              LOG(FATAL) << "Unsupported data type " << runtime::DLDataTypeToString(tensor_dtype);
          }
          break;
        case DataType::kFloat:
          // float
          switch (tensor_dtype.bits()) {
            case 16:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
              break;
            case 32:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
              break;
            case 64:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
              break;
            default:
              LOG(FATAL) << "Unsupported data type " << runtime::DLDataTypeToString(tensor_dtype);
          }
          break;
        case DataType::kBFloat:
          // bfloat
          switch (tensor_dtype.bits()) {
            case 16:
              cu_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
              break;
            default:
              LOG(FATAL) << "Unsupported data type " << runtime::DLDataTypeToString(tensor_dtype);
          }
          break;
        case DataType::kFloat8_e4m3fn:
          // NV float8 e4m3
          cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
          break;
        case DataType::kFloat8_e5m2:
          // NV float8 e5m2
          cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
          break;
        default:
          LOG(FATAL) << "Unsupported data type " << runtime::DLDataTypeToString(tensor_dtype);
      }

      // sanity checks per cuTensorMapEncodeTiled requirements
      // see
      // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
      CHECK_EQ((reinterpret_cast<uint64_t>(tensor_ptr) & 0b1111), 0);    // 16-byte alignment
      CHECK_EQ((reinterpret_cast<uint64_t>(tensor_map) & 0b111111), 0);  // 64-byte alignment
      CHECK_LE(tensor_rank, 5) << "cuTensorMapEncodeTiled only supports up to 5D tensors";

      if (swizzle_kind == CU_TENSOR_MAP_SWIZZLE_32B) {
        CHECK_LE(shared_shape[0] * tensor_dtype.bytes(), 32)
            << "CU_TENSOR_MAP_SWIZZLE_32B implies the bounding box inner dimension will be <= 32.";
      } else if (swizzle_kind == CU_TENSOR_MAP_SWIZZLE_64B) {
        CHECK_LE(shared_shape[0] * tensor_dtype.bytes(), 64)
            << "CU_TENSOR_MAP_SWIZZLE_64B implies the bounding box inner dimension will be <= 64.";
      } else if (swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B) {
        CHECK_LE(shared_shape[0] * tensor_dtype.bytes(), 128)
            << "CU_TENSOR_MAP_SWIZZLE_128B implies the bounding box inner dimension will be <= "
               "128.";
      }

      const cuuint64_t* global_shape_ptr = global_shape.data();
      const cuuint64_t* global_strides_ptr = global_strides.data();
      const uint32_t* shared_shape_ptr = shared_shape.data();
      const uint32_t* shared_strides_ptr = shared_strides.data();

      CUresult res =
          cuTensorMapEncodeTiled(tensor_map, cu_dtype, tensor_rank, tensor_ptr, global_shape_ptr,
                                 global_strides_ptr, shared_shape_ptr, shared_strides_ptr,
                                 interleaved_kind, swizzle_kind, l2_promotion_kind, oob_fill_kind);
      const char* errstr;
      cuGetErrorString(res, &errstr);
      if (res != CUDA_SUCCESS) {
        // get error string
        const char* error_string = nullptr;
        cuGetErrorString(res, &error_string);
        std::cerr << "Error in cuTensorMapEncodeTiled: " << error_string << std::endl;
        std::cout << "cu_dtype: " << cu_dtype << "\n";
        std::cout << "TMA Desc Addr:   " << tensor_map << "\n";
        std::cout << "TMA Interleave:  " << interleaved_kind << "\n";
        std::cout << "TMA L2Promotion: " << l2_promotion_kind << "\n";
        std::cout << "TMA OOBFill:     " << oob_fill_kind << "\n";
        std::cout << "SMEM Swizzle:    " << swizzle_kind << "\n";
        std::cout << "tensor rank: " << tensor_rank << "\n";
        std::cout << "global prob shape: ";
        for (size_t i = 0; i < tensor_rank; i++) {
          std::cout << global_shape[i] << " ";
        }
        std::cout << "\n";
        std::cout << "global prob stride: ";
        for (size_t i = 0; i < tensor_rank; i++) {
          std::cout << global_strides[i] << " ";
        }
        std::cout << "\n";
        std::cout << "smem box shape: ";
        for (size_t i = 0; i < tensor_rank; i++) {
          std::cout << shared_shape[i] << " ";
        }
        std::cout << "\n";
        std::cout << "smem box stride: ";
        for (size_t i = 0; i < tensor_rank; i++) {
          std::cout << shared_strides[i] << " ";
        }
        std::cout << "\n";
        CHECK_EQ(res, CUDA_SUCCESS) << "Error in cuTensorMapEncodeTiled: " << errstr;
      }
    });
#endif  // CUDA_VERSION >= 12000

}  // namespace runtime
}  // namespace tvm
