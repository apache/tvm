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
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/registry.h>

#include <mutex>
#include <sstream>
#include <vector>

#include "../../cuda/cuda_common.h"
#include "./utils.h"

namespace tvm {
namespace runtime {
namespace nccl {

struct NCCLGlobalContext {
  std::vector<ncclComm_t> communicators;

  static NCCLGlobalContext* Get() {
    static NCCLGlobalContext ctx;
    return &ctx;
  }

  void Initialize(const std::vector<int>& device_ids) {
    {
      std::ostringstream os;
      bool is_first = true;
      for (int device_id : device_ids) {
        if (!is_first) {
          os << ",";
        } else {
          is_first = false;
        }
        os << device_id;
      }
      LOG(INFO) << "Initializing NCCL with devices: " << os.str() << ".";
    }
    // TODO(@junrushao): support more flexible communicator pattern for generic SPMD usecases
    DiscoWorker* worker = DiscoWorker::ThreadLocal();
    int num_workers = worker->num_workers;
    CHECK_EQ(device_ids.size(), num_workers)
        << "ValueError: There are " << num_workers << " worker(s), but " << device_ids.size()
        << " device id(s) are provided.";
    ncclUniqueId id;
    NCCL_CALL(ncclGetUniqueId(&id));
    NCCL_CALL(ncclGroupStart());
    for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
      int device_id = device_ids[worker_id];
      ncclComm_t comm;
      CUDA_CALL(cudaSetDevice(device_id));
      NCCL_CALL(ncclCommInitRank(&comm, num_workers, id, worker_id));
      this->communicators.push_back(comm);
    }
    NCCL_CALL(ncclGroupEnd());
  }
};

struct NCCLThreadLocalContext {
  DiscoWorker* worker;
  int device_id;
  cudaStream_t stream;
  ncclComm_t comm;

  void Clear() {
    NCCL_CALL(ncclCommDestroy(comm));
    CUDA_CALL(cudaStreamDestroy(stream));
  }

  static NCCLThreadLocalContext* Get() {
    thread_local static NCCLThreadLocalContext ctx;
    return &ctx;
  }
};

void InitCCL(const std::vector<int>& device_ids) {
  // Set up global context only once
  static std::once_flag flag;
  std::call_once(flag, [&]() { NCCLGlobalContext::Get()->Initialize(device_ids); });
  // Set up thread-local context for each thread
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  int device_id = device_ids[worker->worker_id];
  CUDA_CALL(cudaSetDevice(device_id));
  Device device{DLDeviceType::kDLCUDA, device_id};
  worker->default_device = device;
  worker->ccl = "nccl";
  ctx->worker = worker;
  ctx->device_id = device_id;
  ctx->comm = NCCLGlobalContext::Get()->communicators[worker->worker_id];
  CUDA_CALL(cudaStreamCreate(&ctx->stream));
  DeviceAPI::Get(device)->SetStream(device, ctx->stream);
}

NDArray AllReduce(NDArray send, ReduceKind reduce_kind) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  NDArray recv = NDArray::Empty(shape, send->dtype, send->device);
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*op=*/AsNCCLRedOp(reduce_kind), ctx->comm, ctx->stream));
  return recv;
}

NDArray BroadcastFromWorker0(NDArray buffer) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  ShapeTuple shape = buffer.Shape();
  int64_t numel = shape->Product();
  NCCL_CALL(ncclBroadcast(buffer->data, buffer->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(buffer->dtype)),
                          /*root=*/0, ctx->comm, ctx->stream));
  return buffer;
}

void ScatterFromWorker0(Optional<NDArray> send, NDArray recv) {
  CHECK(recv.defined()) << "ValueError: buffer `recv` must not be None";
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
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
      NCCL_CALL(ncclSend(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, ctx->stream));
      data += bytes_per_shard;
    }
  } else {
    if (send.defined()) {
      LOG(WARNING) << "ValueError: buffer `send` must be None when worker_id != 0. However, got "
                      "send = "
                   << send.get() << ". This will be ignored.";
    }
    NCCL_CALL(ncclGroupStart());
  }
  int64_t numel = recv.Shape()->Product();
  DataType dtype(recv->dtype);
  NCCL_CALL(ncclRecv(recv->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, ctx->stream));
  NCCL_CALL(ncclGroupEnd());
}

void GatherToWorker0(NDArray send, Optional<NDArray> recv) {
  CHECK(send.defined()) << "ValueError: buffer `send` must not be None";
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
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
      NCCL_CALL(ncclRecv(data, numel_per_shard, AsNCCLDataType(dtype), i, ctx->comm, ctx->stream));
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
  NCCL_CALL(ncclSend(send->data, numel, AsNCCLDataType(dtype), 0, ctx->comm, ctx->stream));
  NCCL_CALL(ncclGroupEnd());
}

void RecvFromWorker0(NDArray buffer) {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  CHECK_NE(ctx->worker->worker_id, 0)
      << "ValueError: Worker 0 is not allowed to call RecvFromWorker0.";
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()), 0,
                     ctx->comm, ctx->stream));
  NCCL_CALL(ncclGroupEnd());
}

void SyncWorker() {
  NCCLThreadLocalContext* ctx = NCCLThreadLocalContext::Get();
  CUDA_CALL(cudaStreamSynchronize(ctx->stream));
}

TVM_REGISTER_GLOBAL("runtime.disco.nccl.init_ccl")
    .set_body([](TVMArgs args, TVMRetValue* rv) -> void {
      std::vector<int> device_ids;
      for (int i = 0; i < args.num_args; ++i) {
        device_ids.push_back(args[i].operator int());
      }
      InitCCL(device_ids);
    });
TVM_REGISTER_GLOBAL("runtime.disco.nccl.allreduce").set_body_typed([](NDArray send, int kind) {
  CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
  return AllReduce(send, static_cast<ReduceKind>(kind));
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
