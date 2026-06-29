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
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <cstring>

#include "../../../runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

#ifndef CUDA_DRIVER_CALL
#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      TVM_FFI_THROW(CUDAError) << "" #x " failed with error: " << msg;  \
    }                                                                   \
  }
#endif

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id)); }
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
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, dev.device_id));
        break;
      }
      case kWarpSize: {
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, dev.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, dev.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, dev.device_id));
        os << value << ".";
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, dev.device_id));
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
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, dev.device_id));
        break;
      }
      case kMultiProcessorCount: {
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, dev.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, dev.device_id));
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, dev.device_id));
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, dev.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock: {
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, dev.device_id));
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
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev.device_id));
        *rv = l2_size;
        return;
      }
      case kTotalGlobalMemory: {
        cudaDeviceProp prop;
        TVM_FFI_CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev.device_id));
        int64_t total_global_memory = prop.totalGlobalMem;
        *rv = total_global_memory;
        return;
      }
      case kAvailableGlobalMemory: {
        size_t free_mem, total_mem;
        TVM_FFI_CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
        *rv = static_cast<int64_t>(free_mem);
        return;
      }
      case kImagePitchAlignment:
        return;
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    TVM_FFI_ICHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLCUDAHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      TVM_FFI_CHECK_CUDA_ERROR(cudaMallocHost(&ret, nbytes));
    } else {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      size_t free_mem, total_mem;
      TVM_FFI_CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
      VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
              << " bytes currently free out of " << total_mem << " bytes available";
      TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&ret, nbytes));
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
      TVM_FFI_CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    } else {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      VLOG(1) << "freeing device memory";
      TVM_FFI_CHECK_CUDA_ERROR(cudaFree(ptr));
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
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, cu_stream);
      }
    } else if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCPU) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLCUDA) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      TVM_FFI_THROW(InternalError) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
    cudaStream_t retval;
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventCreate(&evt));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventRecord(evt, src_stream));
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamWaitEvent(dst_stream, evt, 0));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return ThreadLocalWorkspacePool()->AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    ThreadLocalWorkspacePool()->FreeWorkspace(dev, data);
  }

  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

 private:
  static WorkspacePool* ThreadLocalWorkspacePool();

  static void GPUCopy(const void* from, void* to, size_t size, cudaMemcpyKind kind,
                      cudaStream_t stream) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpyAsync(to, from, size, kind, stream));
  }
};

WorkspacePool* CUDADeviceAPI::ThreadLocalWorkspacePool() {
  static thread_local WorkspacePool pool(kDLCUDA, CUDADeviceAPI::Global());
  return &pool;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.cuda",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    DeviceAPI* ptr = CUDADeviceAPI::Global();
                    *rv = static_cast<void*>(ptr);
                  })
      .def_packed("device_api.cuda_host", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = CUDADeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

class CUDATimerNode : public TimerNode {
 public:
  virtual void Start() {
    // This initial cudaEventRecord is sometimes pretty slow (~100us). Does
    // cudaEventRecord do some stream synchronization?
    int device_id;
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    stream_ = TVMFFIEnvGetStream(kDLCUDA, device_id);
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventRecord(start_, static_cast<cudaStream_t>(stream_)));
  }
  virtual void Stop() {
    int device_id;
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventRecord(stop_, static_cast<cudaStream_t>(stream_)));
  }
  virtual int64_t SyncAndGetElapsedNanos() {
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
    float milliseconds = 0;
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~CUDATimerNode() {
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventDestroy(start_));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
  }
  CUDATimerNode() {
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventCreate(&start_));
    TVM_FFI_CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.cuda.CUDATimerNode", CUDATimerNode, TimerNode);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  TVMStreamHandle stream_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.cuda",
                        [](Device dev) { return Timer(ffi::make_object<CUDATimerNode>()); });
}

TVM_RUNTIME_DLL ffi::String GetCudaFreeMemory() {
  size_t free_mem, total_mem;
  TVM_FFI_CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
  std::stringstream ss;
  ss << "Current CUDA memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return ss.str();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GetCudaFreeMemory", GetCudaFreeMemory)
      .def("runtime.get_cuda_stream", []() {
        // TODO(tvm-team): remove once confirms all dep such as flashinfer
        // migrated to TVMFFIEnvGetStream
        int device_id;
        TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
        return static_cast<void*>(TVMFFIEnvGetStream(kDLCUDA, device_id));
      });
}

TVM_RUNTIME_DLL int GetCudaDeviceCount() {
  int count;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDeviceCount(&count));
  return count;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.GetCudaDeviceCount", GetCudaDeviceCount);
}

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
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("runtime.cuTensorMapEncodeTiled", [](ffi::PackedArgs args,
                                                                    ffi::Any* rv) {
    TVM_FFI_ICHECK_GE(args.size(), 4) << "init_cuTensorMap expects at least 4 arguments";
    size_t arg_cnt = 0;
    CUtensorMap* tensor_map = static_cast<CUtensorMap*>(args[arg_cnt++].cast<void*>());
    DLDataType tensor_dtype = args[arg_cnt++].cast<DLDataType>();
    int32_t raw_tensor_rank = args[arg_cnt++].cast<int32_t>();
    TVM_FFI_ICHECK_GT(raw_tensor_rank, 0) << "tensorRank must be non-zero";
    TVM_FFI_ICHECK_LE(raw_tensor_rank, 5)
        << "cuTensorMapEncodeTiled only supports up to 5D tensors";
    uint32_t tensor_rank = static_cast<uint32_t>(raw_tensor_rank);
    void* tensor_ptr = static_cast<void*>(args[arg_cnt++].cast<void*>());

    // The base arg list ends with oob_fill_kind. An OPTIONAL trailing int
    // ``force_cu_dtype`` (>= 0) overrides the dtype-derived CUtensorMapDataType
    // (e.g. 11 == TFLOAT32 for an fp32 gmem buffer feeding a tf32 MMA, so the TMA
    // hardware RN-truncates fp32->tf32 on load). -1 / absent = derive from
    // tensor_dtype. Omitting it keeps older callers backward-compatible.
    int base_arg_count = static_cast<int>(4 + tensor_rank * 4 + 3);
    TVM_FFI_ICHECK(args.size() == base_arg_count || args.size() == base_arg_count + 1)
        << "cuTensorMapEncodeTiled expects " << base_arg_count
        << " (or +1 force_cu_dtype) arguments"
        << "tensor_map, tensor_dtype, tensor_rank, tensor_ptr, global_shape(" << tensor_rank
        << "), global_strides(" << tensor_rank - 1 << "), shared_shape(" << tensor_rank
        << "), shared_strides(" << tensor_rank << "), interleaved_kind, swizzle_kind"
        << ", l2_promotion_kind, oob_fill_kind[, force_cu_dtype]";

    std::vector<cuuint64_t> global_shape(tensor_rank);
    std::vector<cuuint64_t> global_strides(
        std::max<size_t>(tensor_rank > 0 ? tensor_rank - 1 : 0, 1));
    std::vector<uint32_t> box_dim(tensor_rank);
    std::vector<uint32_t> element_strides(tensor_rank);
    for (size_t i = 0; i < tensor_rank; ++i) {
      int64_t value = args[arg_cnt++].cast<int64_t>();
      TVM_FFI_ICHECK_GT(value, 0) << "globalDim[" << i << "] must be non-zero";
      TVM_FFI_ICHECK_LE(static_cast<uint64_t>(value), uint64_t{1} << 32)
          << "globalDim[" << i << "] must be less than or equal to 2^32";
      global_shape[i] = static_cast<cuuint64_t>(value);
    }
    for (size_t i = 0; i < tensor_rank - 1; ++i) {
      int64_t value = args[arg_cnt++].cast<int64_t>();
      TVM_FFI_ICHECK_GE(value, 0) << "globalStrides[" << i << "] must be non-negative";
      global_strides[i] = static_cast<cuuint64_t>(value);
      TVM_FFI_ICHECK_EQ(global_strides[i] % 16, 0) << "global strides must be multiple of 16";
      TVM_FFI_ICHECK_LT(global_strides[i], uint64_t{1} << 40)
          << "globalStrides[" << i << "] must be less than 2^40";
    }
    for (size_t i = 0; i < tensor_rank; ++i) {
      int32_t value = args[arg_cnt++].cast<int32_t>();
      TVM_FFI_ICHECK_GT(value, 0) << "boxDim[" << i << "] must be non-zero";
      TVM_FFI_ICHECK_LE(value, 256) << "boxDim[" << i << "] must be less than or equal to 256";
      box_dim[i] = static_cast<uint32_t>(value);
    }
    for (size_t i = 0; i < tensor_rank; ++i) {
      int32_t value = args[arg_cnt++].cast<int32_t>();
      TVM_FFI_ICHECK_GT(value, 0) << "elementStrides[" << i << "] must be non-zero";
      TVM_FFI_ICHECK_LE(value, 8) << "elementStrides[" << i << "] must be less than or equal to 8";
      element_strides[i] = static_cast<uint32_t>(value);
    }
    auto interleaved_kind = static_cast<CUtensorMapInterleave>(args[arg_cnt++].cast<int>());
    auto swizzle_kind = static_cast<CUtensorMapSwizzle>(args[arg_cnt++].cast<int>());
    auto l2_promotion_kind = static_cast<CUtensorMapL2promotion>(args[arg_cnt++].cast<int>());
    auto oob_fill_kind = static_cast<CUtensorMapFloatOOBfill>(args[arg_cnt++].cast<int>());
    int force_cu_dtype =
        (arg_cnt < static_cast<size_t>(args.size())) ? args[arg_cnt++].cast<int>() : -1;

    TVM_FFI_ICHECK_EQ(tensor_dtype.lanes, 1)
        << "Expect tensor_dtype to have lanes=1, but get " << tensor_dtype;
    uint64_t tensor_dtype_bytes = (static_cast<uint64_t>(tensor_dtype.bits) + 7) / 8;
    CUtensorMapDataType cu_dtype;
    switch (tensor_dtype.code) {
      case kDLInt:
        // int
        switch (tensor_dtype.bits) {
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
            TVM_FFI_THROW(InternalError)
                << "Unsupported data type " << ffi::DLDataTypeToString(tensor_dtype);
        }
        break;
      case kDLUInt:
        // unsigned int
        switch (tensor_dtype.bits) {
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
            TVM_FFI_THROW(InternalError)
                << "Unsupported data type " << ffi::DLDataTypeToString(tensor_dtype);
        }
        break;
      case kDLFloat:
        // float
        switch (tensor_dtype.bits) {
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
            TVM_FFI_THROW(InternalError)
                << "Unsupported data type " << ffi::DLDataTypeToString(tensor_dtype);
        }
        break;
      case kDLBfloat:
        // bfloat
        switch (tensor_dtype.bits) {
          case 16:
            cu_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
            break;
          default:
            TVM_FFI_THROW(InternalError)
                << "Unsupported data type " << ffi::DLDataTypeToString(tensor_dtype);
        }
        break;
      case kDLFloat8_e4m3fn:
        // NV float8 e4m3
        cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        break;
      case kDLFloat8_e5m2:
        // NV float8 e5m2
        cu_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        break;
      case kDLFloat4_e2m1fn:
#if (CUDA_VERSION >= 12080)
        // Packed FP4 in GMEM, unpacked into SMEM/TMEM-facing tiles.
        cu_dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
        break;
#else
        TVM_FFI_THROW(InternalError)
            << "float4_e2m1fn TensorMap requires CUDA support for "
               "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B";
#endif
      default:
        TVM_FFI_THROW(InternalError)
            << "Unsupported data type " << ffi::DLDataTypeToString(tensor_dtype);
    }
    // Caller override (e.g. TFLOAT32 == 11 for an fp32 buffer in a tf32 GEMM):
    // bypass the dtype-derived mapping so the descriptor uses the requested
    // CUtensorMapDataType. Same byte size, different on-load rounding semantics.
    if (force_cu_dtype >= 0) {
      cu_dtype = static_cast<CUtensorMapDataType>(force_cu_dtype);
    }

    auto is_valid_interleave = interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_NONE ||
                               interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_16B ||
                               interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_32B;
    TVM_FFI_ICHECK(is_valid_interleave)
        << "Unsupported interleave enum value: " << static_cast<int>(interleaved_kind);

    auto is_valid_swizzle =
        swizzle_kind == CU_TENSOR_MAP_SWIZZLE_NONE || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_32B ||
        swizzle_kind == CU_TENSOR_MAP_SWIZZLE_64B || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B;
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
    is_valid_swizzle = is_valid_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
#endif
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
    is_valid_swizzle =
        is_valid_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B;
#endif
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
    is_valid_swizzle = is_valid_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
#endif
    TVM_FFI_ICHECK(is_valid_swizzle)
        << "Unsupported swizzle enum value: " << static_cast<int>(swizzle_kind);

    auto is_valid_l2_promotion = l2_promotion_kind == CU_TENSOR_MAP_L2_PROMOTION_NONE ||
                                 l2_promotion_kind == CU_TENSOR_MAP_L2_PROMOTION_L2_64B ||
                                 l2_promotion_kind == CU_TENSOR_MAP_L2_PROMOTION_L2_128B ||
                                 l2_promotion_kind == CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    TVM_FFI_ICHECK(is_valid_l2_promotion)
        << "Unsupported l2Promotion enum value: " << static_cast<int>(l2_promotion_kind);

    auto is_valid_oob_fill = oob_fill_kind == CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE ||
                             oob_fill_kind == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
    TVM_FFI_ICHECK(is_valid_oob_fill)
        << "Unsupported oobFill enum value: " << static_cast<int>(oob_fill_kind);

    bool is_packed_16u4_align8 = false;
#ifdef CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
    is_packed_16u4_align8 = cu_dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#endif
    bool is_packed_16u4_align16 = false;
#ifdef CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
    is_packed_16u4_align16 = cu_dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
#endif
    bool is_packed_16u6_align16 = false;
#ifdef CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
    is_packed_16u6_align16 = cu_dtype == CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
#endif
    auto is_packed_align16 = is_packed_16u4_align16 || is_packed_16u6_align16;
    auto is_packed_dtype = is_packed_16u4_align8 || is_packed_align16;
    auto is_floating_dtype = cu_dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT16 ||
                             cu_dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32 ||
                             cu_dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT64 ||
                             cu_dtype == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
#ifdef CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ
    is_floating_dtype = is_floating_dtype || cu_dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ;
#endif
#ifdef CU_TENSOR_MAP_DATA_TYPE_TFLOAT32
    is_floating_dtype = is_floating_dtype || cu_dtype == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
#endif
#ifdef CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ
    is_floating_dtype = is_floating_dtype || cu_dtype == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ;
#endif

    auto is_128b_swizzle = swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B;
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
    is_128b_swizzle = is_128b_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
#endif
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
    is_128b_swizzle =
        is_128b_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B;
#endif
#ifdef CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
    is_128b_swizzle = is_128b_swizzle || swizzle_kind == CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
#endif

    // Host-side validation for documented cuTensorMapEncodeTiled requirements.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    TVM_FFI_ICHECK_EQ((reinterpret_cast<uint64_t>(tensor_ptr) & 0b1111), 0);    // 16-byte alignment
    TVM_FFI_ICHECK_EQ((reinterpret_cast<uint64_t>(tensor_map) & 0b111111), 0);  // 64-byte alignment

    if (interleaved_kind != CU_TENSOR_MAP_INTERLEAVE_NONE) {
      TVM_FFI_ICHECK_GE(tensor_rank, 3U)
          << "tensorRank must be greater than or equal to 3 when interleave is not NONE";
    }
    if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_32B || is_packed_align16) {
      TVM_FFI_ICHECK_EQ((reinterpret_cast<uint64_t>(tensor_ptr) & 0b11111), 0)
          << "globalAddress must be 32-byte aligned";
    }
    if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_32B || is_packed_align16) {
      for (size_t i = 0; i < global_strides.size(); ++i) {
        TVM_FFI_ICHECK_EQ(global_strides[i] % 32, 0)
            << "globalStrides[" << i << "] must be a multiple of 32";
      }
    }
    if (is_packed_align16) {
      TVM_FFI_ICHECK_EQ(global_shape[0] % 128, 0)
          << "globalDim[0] must be a multiple of 128 for packed 16U4/16U6 align16 formats";
      TVM_FFI_ICHECK_EQ(box_dim[0], 128U)
          << "boxDim[0] must be 128 for packed 16U4/16U6 align16 formats";
    }
    if (is_packed_16u4_align8) {
      TVM_FFI_ICHECK_EQ(global_shape[0] % 2, 0)
          << "globalDim[0] must be a multiple of 2 for packed 16U4 align8 format";
    }
    if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_NONE && !is_packed_dtype) {
      uint64_t inner_box_bytes = static_cast<uint64_t>(box_dim[0]) * tensor_dtype_bytes;
      TVM_FFI_ICHECK_EQ(inner_box_bytes % 16, 0)
          << "boxDim[0] * elementSizeInBytes(tensorDataType) must be a multiple of 16 bytes";
    }
    if (oob_fill_kind == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA) {
      TVM_FFI_ICHECK(is_floating_dtype)
          << "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA requires a floating-point "
             "tensorDataType";
      TVM_FFI_ICHECK(!is_packed_dtype)
          << "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA is not supported for packed "
             "tensorDataType";
    }

    if (is_packed_16u6_align16 && is_128b_swizzle) {
      TVM_FFI_ICHECK_EQ(interleaved_kind, CU_TENSOR_MAP_INTERLEAVE_NONE)
          << "packed 16U6 align16 formats require interleave NONE for 128B swizzles";
    }

    if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_NONE && !is_packed_dtype &&
        swizzle_kind == CU_TENSOR_MAP_SWIZZLE_32B) {
      TVM_FFI_ICHECK_LE(box_dim[0] * tensor_dtype_bytes, 32)
          << "CU_TENSOR_MAP_SWIZZLE_32B implies the bounding box inner dimension will be <= 32.";
    } else if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_NONE && !is_packed_dtype &&
               swizzle_kind == CU_TENSOR_MAP_SWIZZLE_64B) {
      TVM_FFI_ICHECK_LE(box_dim[0] * tensor_dtype_bytes, 64)
          << "CU_TENSOR_MAP_SWIZZLE_64B implies the bounding box inner dimension will be <= 64.";
    } else if (interleaved_kind == CU_TENSOR_MAP_INTERLEAVE_NONE && !is_packed_dtype &&
               is_128b_swizzle) {
      TVM_FFI_ICHECK_LE(box_dim[0] * tensor_dtype_bytes, 128)
          << "CU_TENSOR_MAP_SWIZZLE_128B implies the bounding box inner dimension will be <= "
             "128.";
    }

    const cuuint64_t* global_shape_ptr = global_shape.data();
    const cuuint64_t* global_strides_ptr = global_strides.data();
    const uint32_t* shared_shape_ptr = box_dim.data();
    const uint32_t* shared_strides_ptr = element_strides.data();

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
      for (size_t i = 0; i < global_strides.size(); i++) {
        std::cout << global_strides[i] << " ";
      }
      std::cout << "\n";
      std::cout << "smem box shape: ";
      for (size_t i = 0; i < tensor_rank; i++) {
        std::cout << box_dim[i] << " ";
      }
      std::cout << "\n";
      std::cout << "smem box stride: ";
      for (size_t i = 0; i < tensor_rank; i++) {
        std::cout << element_strides[i] << " ";
      }
      std::cout << "\n";
      TVM_FFI_ICHECK_EQ(res, CUDA_SUCCESS) << "Error in cuTensorMapEncodeTiled: " << errstr;
    }
  });
}
#endif  // CUDA_VERSION >= 12000

}  // namespace runtime
}  // namespace tvm
