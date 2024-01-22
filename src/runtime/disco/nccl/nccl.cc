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

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "../../../support/process_id.h"
#include "../utils.h"

/* `TVM_NCCL_RCCL_SWITCH` is set to 0 for NCCL, 1 for RCCL */
#ifndef TVM_NCCL_RCCL_SWITCH
#define TVM_NCCL_RCCL_SWITCH 0
#endif
#if TVM_NCCL_RCCL_SWITCH == 0
#include <nccl.h>

#include "../../cuda/cuda_common.h"
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

inline ncclRedOp_t AsNCCLRedOp(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::kSum:
      return ncclSum;
    case ReduceKind::kProd:
      return ncclProd;
    case ReduceKind::kMin:
      return ncclMin;
    case ReduceKind::kMax:
      return ncclMax;
    case ReduceKind::kAvg:
      return ncclAvg;
  }
  LOG(FATAL) << "ValueError: Unknown ReduceKind: " << static_cast<int>(kind);
  throw;
}

struct CCLThreadLocalContext {
  DiscoWorker* worker;
  int device_id;
  deviceStream_t default_stream = nullptr;
  ncclComm_t comm;

  void Clear() {
    NCCL_CALL(ncclCommDestroy(comm));
    if (default_stream != nullptr) {
      StreamDestroy(default_stream);
    }
  }

  deviceStream_t GetDefaultStream() {
    const auto* func = tvm::runtime::Registry::Get("runtime.get_" TVM_DISCO_DEVICE_NAME "_stream");
    ICHECK(func != nullptr);
    deviceStream_t stream = static_cast<deviceStream_t>((*func)().operator void*());
    return stream == nullptr ? default_stream : stream;
  }

  static CCLThreadLocalContext* Get() {
    thread_local static CCLThreadLocalContext ctx;
    return &ctx;
  }
};

void InitCCL(Session sess, IntTuple device_ids) {
  DRef func = sess->GetGlobalFunc("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker");
  DLOG(INFO) << "Initializing " TVM_DISCO_CCL_NAME " with devices: " << device_ids;
  ncclUniqueId id;
  TVMByteArray array;
  NCCL_CALL(ncclGetUniqueId(&id));
  array.data = id.internal;
  array.size = NCCL_UNIQUE_ID_BYTES;
  sess->CallPacked(func, device_ids, array);
}

void InitCCLPerWorker(IntTuple device_ids, std::string unique_id_bytes) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker != nullptr);
  CHECK_EQ(unique_id_bytes.size(), NCCL_UNIQUE_ID_BYTES)
      << "ValueError: The length of unique_id must be " << NCCL_UNIQUE_ID_BYTES << ", but got "
      << unique_id_bytes.size() << ".";
  // Step up local context of NCCL
  int device_id = device_ids[worker->worker_id];
  SetDevice(device_id);
#if TVM_NCCL_RCCL_SWITCH == 0
  StreamCreate(&ctx->default_stream);
#endif
  Device device{TVM_DISCO_DEVICE_TYPE, device_id};
  worker->default_device = device;
  worker->ccl = TVM_DISCO_CCL_NAME;
  ctx->worker = worker;
  ctx->device_id = device_id;
  // Initialize the communicator
  ncclUniqueId id;
  std::memcpy(id.internal, unique_id_bytes.data(), NCCL_UNIQUE_ID_BYTES);
  NCCL_CALL(ncclCommInitRank(&ctx->comm, worker->num_workers, id, worker->worker_id));
}

void AllReduce(NDArray send, ReduceKind reduce_kind, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  deviceStream_t stream = ctx->GetDefaultStream();
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*op=*/AsNCCLRedOp(reduce_kind), ctx->comm, stream));
}

void AllGather(NDArray send, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  deviceStream_t stream = ctx->GetDefaultStream();
  NCCL_CALL(ncclAllGather(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)), ctx->comm, stream));
}

void BroadcastFromWorker0(NDArray send, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ICHECK(send.Shape()->Product() == recv.Shape()->Product());
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  deviceStream_t stream = ctx->GetDefaultStream();
  NCCL_CALL(ncclBroadcast(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*root=*/0, ctx->comm, stream));
}

void ScatterFromWorker0(Optional<NDArray> send, NDArray recv) {
  CHECK(recv.defined()) << "ValueError: buffer `recv` must not be None";
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  deviceStream_t stream = ctx->GetDefaultStream();
  if (worker_id == 0) {
    CHECK(send.defined()) << "ValueError: buffer `send` must be provided when worker_id == 0.";
    NDArray buffer = send.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_workers, 0) << "ValueError: Scattering evenly requires that the number "
                                        "of elements in the buffer to be "
                                        "divisible by the number of workers, but got numel = "
                                     << numel << " and " << num_workers << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_workers;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, recv.Shape()->Product())
        << "ValueError: The number of elements in buffer `recv` must be the same as each shard "
           "of "
           "buffer `send`. `send.size` is "
        << numel << ", but `recv.size` is " << recv.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_workers; ++i) {
      NCCL_CALL(ncclSend(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, stream));
      data += bytes_per_shard;
    }
  } else {
    if (send.defined()) {
      LOG(WARNING) << "Buffer `send` must be None when worker_id != 0, but got "
                      "send = "
                   << send.get() << ". This will be ignored.";
    }
    NCCL_CALL(ncclGroupStart());
  }
  int64_t numel = recv.Shape()->Product();
  DataType dtype(recv->dtype);
  NCCL_CALL(ncclRecv(recv->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void GatherToWorker0(NDArray send, Optional<NDArray> recv) {
  CHECK(send.defined()) << "ValueError: buffer `send` must not be None";
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  deviceStream_t stream = ctx->GetDefaultStream();
  if (worker_id == 0) {
    CHECK(recv.defined()) << "ValueError: buffer `recv` must be provided when worker_id == 0.";
    NDArray buffer = recv.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_workers, 0) << "ValueError: Gathering evenly requires that the number "
                                        "of elements in the buffer to be "
                                        "divisible by the number of workers, but got numel = "
                                     << numel << " and " << num_workers << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_workers;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, send.Shape()->Product())
        << "ValueError: The number of elements in buffer `send` must be the same as each shard "
           "of "
           "buffer `recv`. `recv.size` is "
        << numel << ", but `send.size` is " << send.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_workers; ++i) {
      NCCL_CALL(ncclRecv(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, stream));
      data += bytes_per_shard;
    }
  } else {
    if (recv.defined()) {
      LOG(WARNING) << "ValueError: buffer `recv` must be None when worker_id != 0. However, got "
                      "recv = "
                   << recv.get() << ". This will be ignored.";
    }
    NCCL_CALL(ncclGroupStart());
  }
  int64_t numel = send.Shape()->Product();
  DataType dtype(send->dtype);
  NCCL_CALL(ncclSend(send->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void RecvFromWorker0(NDArray buffer) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  CHECK_NE(ctx->worker->worker_id, 0)
      << "ValueError: Worker 0 is not allowed to call RecvFromWorker0.";
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()), 0,
                     ctx->comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void SyncWorker() {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ICHECK(ctx->worker != nullptr);
  deviceStream_t stream = ctx->GetDefaultStream();
  StreamSynchronize(stream);
}

TVM_REGISTER_GLOBAL("runtime.disco.compiled_ccl").set_body_typed([]() -> String {
  return TVM_DISCO_CCL_NAME;
});
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl").set_body_typed(InitCCL);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker")
    .set_body_typed(InitCCLPerWorker);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".allreduce")
    .set_body_typed([](NDArray send, int kind, NDArray recv) {
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      nccl::AllReduce(send, static_cast<ReduceKind>(kind), recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".allgather")
    .set_body_typed([](NDArray send, NDArray recv) { nccl::AllGather(send, recv); });
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".broadcast_from_worker0")
    .set_body_typed(BroadcastFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".scatter_from_worker0")
    .set_body_typed(ScatterFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".gather_to_worker0")
    .set_body_typed(GatherToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_worker0")
    .set_body_typed(RecvFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".sync_worker").set_body_typed(SyncWorker);

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
