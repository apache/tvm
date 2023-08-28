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
#include <vector>

#include "../../cuda/cuda_common.h"
#include "./utils.h"

namespace tvm {
namespace runtime {
namespace nccl {

struct NCCLGlobalContext {
  // TODO(@junrushao): support more flexible communicator pattern for generic SPMD usecases
  std::vector<int> device_ids;
  std::vector<ncclComm_t> communicators;
  std::vector<cudaStream_t> streams;

  ~NCCLGlobalContext() {}

  void Clear() {
    for (ncclComm_t comm : this->communicators) {
      NCCL_CALL(ncclCommDestroy(comm));
    }
    device_ids.clear();
    communicators.clear();
  }

  static NCCLGlobalContext* Get() {
    static NCCLGlobalContext ctx;
    return &ctx;
  }

  void Initialize(std::vector<int> device_ids) {
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
      cudaStream_t stream;
      CUDA_CALL(cudaSetDevice(device_id));
      CUDA_CALL(cudaStreamCreate(&stream));
      NCCL_CALL(ncclCommInitRank(&comm, num_workers, id, worker_id));
      this->streams.push_back(stream);
      this->communicators.push_back(comm);
    }
    NCCL_CALL(ncclGroupEnd());
    this->device_ids = std::move(device_ids);
  }

  static ncclComm_t ThreadLocalCommunicator() {
    thread_local static ncclComm_t comm =
        NCCLGlobalContext::Get()->communicators[DiscoWorker::ThreadLocal()->worker_id];
    return comm;
  }

  static cudaStream_t ThreadLocalStream() {
    thread_local static cudaStream_t stream =
        NCCLGlobalContext::Get()->streams[DiscoWorker::ThreadLocal()->worker_id];
    return stream;
  }
};

inline int64_t GetNumel(const ShapeTuple& shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    numel *= d;
  }
  return numel;
}

NDArray AllReduce(NDArray send, ReduceKind reduce_kind) {
  ShapeTuple shape = send.Shape();
  int64_t numel = GetNumel(shape);
  NDArray recv = NDArray::Empty(shape, send->dtype, send->device);
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          /*op=*/AsNCCLRedOp(reduce_kind),
                          /*comm=*/NCCLGlobalContext::ThreadLocalCommunicator(),
                          /*stream=*/NCCLGlobalContext::ThreadLocalStream()));
  return recv;
}

NDArray BroadcastFromZero(NDArray buffer) {
  ShapeTuple shape = buffer.Shape();
  int64_t numel = GetNumel(shape);
  NCCL_CALL(ncclBroadcast(buffer->data, buffer->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(buffer->dtype)),  //
                          /*root=*/0, NCCLGlobalContext::ThreadLocalCommunicator(),
                          NCCLGlobalContext::ThreadLocalStream()));
  return buffer;
}

TVM_REGISTER_GLOBAL("runtime.disco.nccl.init").set_body([](TVMArgs args, TVMRetValue* rv) -> void {
  // Parse the inputs into device ids
  std::vector<int> device_ids;
  for (int i = 0; i < args.num_args; ++i) {
    device_ids.push_back(args[i].operator int());
  }
  // Set the `default_device` and `ccl` for the current worker
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  worker->default_device = Device{DLDeviceType::kDLCUDA, device_ids[worker->worker_id]};
  worker->ccl = "nccl";
  // Setup global context only once
  static std::once_flag flag;
  std::call_once(flag, [&]() { NCCLGlobalContext::Get()->Initialize(device_ids); });
});

TVM_REGISTER_GLOBAL("runtime.disco.nccl.allreduce")
    .set_body_typed([](NDArray send, ShapeTuple reduce_kind) {
      return AllReduce(send, static_cast<ReduceKind>(IntegerFromShapeTuple(reduce_kind)));
    });
TVM_REGISTER_GLOBAL("runtime.disco.nccl.broadcast_from_worker0").set_body_typed(BroadcastFromZero);

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
