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
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <nccl.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "../../../support/process_id.h"
#include "../../cuda/cuda_common.h"
#include "./utils.h"

namespace tvm {
namespace runtime {
namespace nccl {

struct NCCLThreadLocalContext {
  DiscoWorker* worker;
  int device_id;
  cudaStream_t comm_stream;
  cudaStream_t compute_stream = nullptr;
  ncclComm_t comm;

  void Clear() {
    NCCL_CALL(ncclCommDestroy(comm));
    CUDA_CALL(cudaStreamDestroy(comm_stream));
  }

  static NCCLThreadLocalContext* Get() {
    thread_local static NCCLThreadLocalContext ctx;
    return &ctx;
  }
};

void InitCCL(Session sess, ShapeTuple device_ids) {
  DRef func = sess->GetGlobalFunc("runtime.disco.nccl.init_ccl_per_worker");
  LOG(INFO) << "Initializing NCCL with devices: " << device_ids;
  ncclUniqueId id;
  TVMByteArray array;
  NCCL_CALL(ncclGetUniqueId(&id));
  array.data = id.internal;
  array.size = NCCL_UNIQUE_ID_BYTES;
  sess->CallPacked(func, device_ids, array);
}

void InitCCLPerWorker(ShapeTuple device_ids, std::string unique_id_bytes) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  ICHECK(worker != nullptr);
  CHECK_EQ(unique_id_bytes.size(), NCCL_UNIQUE_ID_BYTES)
      << "ValueError: The length of unique_id must be " << NCCL_UNIQUE_ID_BYTES << ", but got "
      << unique_id_bytes.size() << ".";
  // Step up local context of NCCL
  int device_id = device_ids[worker->worker_id];
  CUDA_CALL(cudaSetDevice(device_id));
  CUDA_CALL(cudaStreamCreate(&ctx->comm_stream));
  Device device{DLDeviceType::kDLCUDA, device_id};
  worker->default_device = device;
  worker->ccl = "nccl";
  ctx->worker = worker;
  ctx->device_id = device_id;
  // Initialize the communicator
  ncclUniqueId id;
  std::memcpy(id.internal, unique_id_bytes.data(), NCCL_UNIQUE_ID_BYTES);
  NCCL_CALL(ncclCommInitRank(&ctx->comm, worker->num_workers, id, worker->worker_id));
}

void AllReduce(NDArray send, ReduceKind reduce_kind, NDArray recv) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  Device device = ctx->worker->default_device;
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->compute_stream, ctx->comm_stream);
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*op=*/AsNCCLRedOp(reduce_kind), ctx->comm, ctx->comm_stream));
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->comm_stream, ctx->compute_stream);
}

void BroadcastFromWorker0(NDArray send, NDArray recv) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  ICHECK(send.Shape()->Product() == recv.Shape()->Product());
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  Device device = ctx->worker->default_device;
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->compute_stream, ctx->comm_stream);
  NCCL_CALL(ncclBroadcast(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*root=*/0, ctx->comm, ctx->comm_stream));
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->comm_stream, ctx->compute_stream);
}

void ScatterFromWorker0(Optional<NDArray> send, NDArray recv) {
  CHECK(recv.defined()) << "ValueError: buffer `recv` must not be None";
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  Device device = ctx->worker->default_device;
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->compute_stream, ctx->comm_stream);
  if (worker_id == 0) {
    CHECK(send.defined()) << "ValueError: buffer `send` must be provided when worker_id == 0.";
    NDArray buffer = send.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_workers, 0)
        << "ValueError: Scattering evenly requires that the number of elements in the buffer to be "
           "divisible by the number of workers, but got numel = "
        << numel << " and " << num_workers << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_workers;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, recv.Shape()->Product())
        << "ValueError: The number of elements in buffer `recv` must be the same as each shard of "
           "buffer `send`. `send.size` is "
        << numel << ", but `recv.size` is " << recv.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_workers; ++i) {
      NCCL_CALL(
          ncclSend(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, ctx->comm_stream));
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
  NCCL_CALL(ncclRecv(recv->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, ctx->comm_stream));
  NCCL_CALL(ncclGroupEnd());
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->comm_stream, ctx->compute_stream);
}

void GatherToWorker0(NDArray send, Optional<NDArray> recv) {
  CHECK(send.defined()) << "ValueError: buffer `send` must not be None";
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  Device device = ctx->worker->default_device;
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->compute_stream, ctx->comm_stream);
  if (worker_id == 0) {
    CHECK(recv.defined()) << "ValueError: buffer `recv` must be provided when worker_id == 0.";
    NDArray buffer = recv.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_workers, 0)
        << "ValueError: Gathering evenly requires that the number of elements in the buffer to be "
           "divisible by the number of workers, but got numel = "
        << numel << " and " << num_workers << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_workers;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, send.Shape()->Product())
        << "ValueError: The number of elements in buffer `send` must be the same as each shard of "
           "buffer `recv`. `recv.size` is "
        << numel << ", but `send.size` is " << send.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_workers; ++i) {
      NCCL_CALL(
          ncclRecv(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, ctx->comm_stream));
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
  NCCL_CALL(ncclSend(send->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, ctx->comm_stream));
  NCCL_CALL(ncclGroupEnd());
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->comm_stream, ctx->compute_stream);
}

void RecvFromWorker0(NDArray buffer) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  CHECK_NE(ctx->worker->worker_id, 0)
      << "ValueError: Worker 0 is not allowed to call RecvFromWorker0.";
  Device device = ctx->worker->default_device;
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->compute_stream, ctx->comm_stream);
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()), 0,
                     ctx->comm, ctx->comm_stream));
  NCCL_CALL(ncclGroupEnd());
  DeviceAPI::Get(device)->SyncStreamFromTo(device, ctx->comm_stream, ctx->compute_stream);
}

void SyncWorker() {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  ICHECK(ctx->worker != nullptr);
  CUDA_CALL(cudaStreamSynchronize(ctx->compute_stream));
}

TVM_REGISTER_GLOBAL("runtime.disco.nccl.init_ccl").set_body_typed(InitCCL);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.init_ccl_per_worker").set_body_typed(InitCCLPerWorker);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.allreduce")
    .set_body_typed([](NDArray send, int kind, NDArray recv) {
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      AllReduce(send, static_cast<ReduceKind>(kind), recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco.nccl.broadcast_from_worker0")
    .set_body_typed(BroadcastFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.scatter_from_worker0").set_body_typed(ScatterFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.gather_to_worker0").set_body_typed(GatherToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.recv_from_worker0").set_body_typed(RecvFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.nccl.sync_worker").set_body_typed(SyncWorker);

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
