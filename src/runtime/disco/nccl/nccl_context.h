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

#ifndef TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_
#define TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/registry.h>

#include "../../../support/process_id.h"
#include "../utils.h"

/* `TVM_NCCL_RCCL_SWITCH` is set to 0 for NCCL, 1 for RCCL */
#ifndef TVM_NCCL_RCCL_SWITCH
#define TVM_NCCL_RCCL_SWITCH 0
#endif
#if TVM_NCCL_RCCL_SWITCH == 0
#include <nccl.h>

#include "../../cuda/cuda_common.h"
#include "msccl.h"
#else
#include <rccl/rccl.h>

#include "../../rocm/rocm_common.h"
#endif

namespace tvm {
namespace runtime {
namespace nccl {

#define NCCL_CALL(cmd)                                                      \
  do {                                                                      \
    auto r = (cmd);                                                         \
    if (r != ncclSuccess) {                                                 \
      LOG(FATAL) << TVM_DISCO_CCL_NAME "Errror: " << ncclGetErrorString(r); \
    }                                                                       \
  } while (0)

#define MSCCL_CALL(cmd)                                                      \
  do {                                                                       \
    auto r = (cmd);                                                          \
    if (r != mscclSuccess) {                                                 \
      LOG(FATAL) << TVM_DISCO_CCL_NAME "Errror: " << mscclGetErrorString(r); \
    }                                                                        \
  } while (0)

#if TVM_NCCL_RCCL_SWITCH == 0

#define TVM_DISCO_DEVICE_NAME "cuda"
#define TVM_DISCO_CCL_NAME "nccl"

using deviceStream_t = cudaStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLCUDA;
inline void SetDevice(int device_id) { CUDA_CALL(cudaSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) { CUDA_CALL(cudaStreamSynchronize(stream)); }
inline void StreamCreate(deviceStream_t* stream) { CUDA_CALL(cudaStreamCreate(stream)); }
inline void StreamDestroy(deviceStream_t stream) { CUDA_CALL(cudaStreamDestroy(stream)); }

#else

#define TVM_DISCO_DEVICE_NAME "rocm"
#define TVM_DISCO_CCL_NAME "rccl"

using deviceStream_t = hipStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLROCM;
inline void SetDevice(int device_id) { ROCM_CALL(hipSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) { ROCM_CALL(hipStreamSynchronize(stream)); }
inline void StreamCreate(deviceStream_t* stream) { ROCM_CALL(hipStreamCreate(stream)); }
inline void StreamDestroy(deviceStream_t stream) { ROCM_CALL(hipStreamDestroy(stream)); }

#endif

/*! \brief Convert DataType to ncclDataType. */
inline ncclDataType_t AsNCCLDataType(runtime::DataType dtype) {
  if (dtype == DataType::Int(8)) {
    return ncclInt8;
  }
  if (dtype == DataType::UInt(8)) {
    return ncclUint8;
  }
  if (dtype == DataType::Int(32)) {
    return ncclInt32;
  }
  if (dtype == DataType::UInt(32)) {
    return ncclUint32;
  }
  if (dtype == DataType::Int(64)) {
    return ncclInt64;
  }
  if (dtype == DataType::UInt(64)) {
    return ncclUint64;
  }
  if (dtype == DataType::Float(16)) {
    return ncclFloat16;
  }
  if (dtype == DataType::Float(32)) {
    return ncclFloat32;
  }
  if (dtype == DataType::Float(64)) {
    return ncclFloat64;
  }
  if (dtype == DataType::BFloat(16)) {
    return ncclBfloat16;
  }
  LOG(FATAL) << "ValueError: Unsupported data type " << dtype;
  throw;
}

}  // namespace nccl

template <typename CCLType>
struct ccl;

struct nccl_t {};
template <>
struct ccl<nccl_t> {
  using Comm_t = ncclComm_t;
  using UniqueId = ncclUniqueId;
  using DataType_t = ncclDataType_t;
  using RedOp_t = ncclRedOp_t;

  static constexpr auto GetUniqueId = [](ncclUniqueId* id) { NCCL_CALL(ncclGetUniqueId(id)); };
  static constexpr auto InitCommRank = [](void* comm, int num_workers, const ncclUniqueId& id,
                                          int worker_id) {
    NCCL_CALL(ncclCommInitRank(reinterpret_cast<ncclComm_t*>(comm), num_workers, id, worker_id));
  };
  static constexpr auto AllReduce = [](const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
    NCCL_CALL(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream));
  };
  static void CommDestroy(ncclComm_t& comm) { NCCL_CALL(ncclCommDestroy(comm)); }
  static constexpr size_t CCL_UNIQUE_ID_BYTES = NCCL_UNIQUE_ID_BYTES;
  static constexpr char type[] = "nccl";
};

struct msccl_t {};
template <>
struct ccl<msccl_t> {
  using Comm_t = mscclComm_t;
  using UniqueId = mscclUniqueId;
  using DataType_t = mscclDataType_t;
  using RedOp_t = mscclRedOp_t;

  static constexpr auto GetUniqueId = [](mscclUniqueId* id) { MSCCL_CALL(mscclGetUniqueId(id)); };
  static constexpr auto InitCommRank = [](void* comm, int num_workers, const mscclUniqueId& id,
                                          int worker_id) {
    MSCCL_CALL(mscclCommInitRank(reinterpret_cast<mscclComm_t*>(comm), num_workers, id, worker_id));
  };

  static constexpr auto AllReduce = [](const void* sendbuff, void* recvbuff, size_t count,
                                       mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                                       cudaStream_t stream) {
    CHECK(op == mscclSum) << "MSCCLPP AllReduce currently only supports mscclRedOp_t: [mscclSum]";
    MSCCL_CALL(mscclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream));
  };
  static void CommDestroy(mscclComm_t& comm) { MSCCL_CALL(mscclCommDestroy(comm)); }
  static constexpr size_t CCL_UNIQUE_ID_BYTES = MSCCL_UNIQUE_ID_BYTES;
  static constexpr char type[] = "msccl";
};

template <typename T>
struct CCLThreadLocalContext {
  DiscoWorker* worker;
  int device_id;
  nccl::deviceStream_t default_stream = nullptr;
  typename ccl<T>::Comm_t comm;

  void Clear() {
    ccl<T>::CommDestroy(comm);
    if (default_stream != nullptr) {
      nccl::StreamDestroy(default_stream);
    }
  }

  nccl::deviceStream_t GetDefaultStream() {
    const auto* func = tvm::runtime::Registry::Get("runtime.get_" TVM_DISCO_DEVICE_NAME "_stream");
    ICHECK(func != nullptr);
    nccl::deviceStream_t stream = static_cast<nccl::deviceStream_t>((*func)().operator void*());
    return stream == nullptr ? default_stream : stream;
  }

  static CCLThreadLocalContext* Get();
};

template <typename T>
CCLThreadLocalContext<T>* CCLThreadLocalContext<T>::Get() {
  thread_local static CCLThreadLocalContext<T> ctx;
  return &ctx;
}

namespace nccl {
using CCLThreadLocalContext = CCLThreadLocalContext<nccl_t>;
}  // namespace nccl
namespace msccl {
using CCLThreadLocalContext = CCLThreadLocalContext<msccl_t>;
}  // namespace msccl

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_
