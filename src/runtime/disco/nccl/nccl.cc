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

#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "../../../support/process_id.h"
#include "../utils.h"
#include "nccl_context.h"

namespace tvm {
namespace runtime {
namespace nccl {

CCLThreadLocalContext* CCLThreadLocalContext::Get() {
  thread_local static CCLThreadLocalContext ctx;
  return &ctx;
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

  CHECK(!ctx->global_comm) << "Cannot initialize CCL, "
                           << "the previous thread-global comm still exists, "
                           << "and has not been destructed";
  CHECK(!ctx->group_comm) << "Cannot initialize CCL, "
                          << "the previous thread-group comm still exists, "
                          << "and has not been destructed";
  CHECK(!ctx->default_stream) << "Cannot initialize CCL, "
                              << "the previous thread-global stream still exists, "
                              << "and has not been destructed";
  CHECK(!ctx->worker) << "Cannot initialize CCL, "
                      << "the previous thread-global worker still exists, "
                      << "and has not been destructed";

  // Step up local context of NCCL
  int group_size = worker->num_workers / worker->num_groups;
  int device_id = device_ids[worker->local_worker_id];
  SetDevice(device_id);
#if TVM_NCCL_RCCL_SWITCH == 0
  StreamCreate(&ctx->default_stream);
#endif
  Device device{TVM_DISCO_DEVICE_TYPE, device_id};
  if (worker->default_device.device_type == DLDeviceType::kDLCPU) {
    worker->default_device = device;
  } else {
    ICHECK(worker->default_device.device_type == device.device_type &&
           worker->default_device.device_id == device.device_id)
        << "The default device of the worker is inconsistent with the device used for CCL. "
        << "The default device is " << worker->default_device << ", but the device used for CCL is "
        << device << ".";
  }
  worker->ccl = TVM_DISCO_CCL_NAME;
  ctx->worker = worker;
  ctx->device_id = device_id;
  // Initialize the communicator
  ncclUniqueId id;
  std::memcpy(id.internal, unique_id_bytes.data(), NCCL_UNIQUE_ID_BYTES);
  NCCL_CALL(ncclCommInitRank(&ctx->global_comm, worker->num_workers, id, worker->worker_id));
  if (worker->num_groups == 1) {
    ctx->group_comm = ctx->global_comm;
  } else {
    NCCL_CALL(ncclCommSplit(ctx->global_comm, worker->worker_id / group_size,
                            worker->worker_id % group_size, &ctx->group_comm, NULL));
  }
}

void AllReduce(NDArray send, ReduceKind reduce_kind, bool in_group, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  deviceStream_t stream = ctx->GetDefaultStream();
  DataType dtype = DataType(send->dtype);
  if (dtype == DataType::NVFloat8E4M3() || dtype == DataType::NVFloat8E5M2()) {
    LOG(FATAL) << "Float8 data type cannot be allreduced, as nccl does not support this data type.";
  }
  NCCL_CALL(ncclAllReduce(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(dtype),
                          /*op=*/AsNCCLRedOp(reduce_kind),
                          in_group ? ctx->group_comm : ctx->global_comm, stream));
}

void AllGather(NDArray send, bool in_group, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  ShapeTuple shape = send.Shape();
  int64_t numel = shape->Product();
  deviceStream_t stream = ctx->GetDefaultStream();
  NCCL_CALL(ncclAllGather(send->data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(send->dtype)),
                          in_group ? ctx->group_comm : ctx->global_comm, stream));
}

void BroadcastFromWorker0(Optional<NDArray> send, bool in_group, NDArray recv) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int group_size = ctx->worker->num_workers / ctx->worker->num_groups;
  bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);

  const void* send_data = [&]() -> const void* {
    if (is_sender) {
      CHECK(send.defined());
      CHECK(send.value().Shape()->Product() == recv.Shape()->Product());
      return send.value()->data;
    } else {
      return nullptr;
    }
  }();
  int64_t numel = recv.Shape()->Product();

  deviceStream_t stream = ctx->GetDefaultStream();
  NCCL_CALL(ncclBroadcast(send_data, recv->data, numel,
                          /*datatype=*/AsNCCLDataType(DataType(recv->dtype)),
                          /*root=*/0, in_group ? ctx->group_comm : ctx->global_comm, stream));
}

void ScatterFromWorker0(Optional<NDArray> send, bool in_group, NDArray recv) {
  CHECK(recv.defined()) << "ValueError: buffer `recv` must not be None";
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  int group_size = num_workers / ctx->worker->num_groups;
  bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
  int num_receiver = in_group ? group_size : num_workers;
  deviceStream_t stream = ctx->GetDefaultStream();
  if (is_sender) {
    CHECK(send.defined()) << "ValueError: buffer `send` must be provided when worker_id == 0.";
    NDArray buffer = send.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_receiver, 0) << "ValueError: Scattering evenly requires that the number "
                                         "of elements in the buffer to be "
                                         "divisible by the number of workers, but got numel = "
                                      << numel << " and " << num_receiver << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_receiver;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, recv.Shape()->Product())
        << "ValueError: The number of elements in buffer `recv` must be the same as each shard "
           "of "
           "buffer `send`. `send.size` is "
        << numel << ", but `recv.size` is " << recv.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_receiver; ++i) {
      NCCL_CALL(ncclSend(data, numel_per_shard, AsNCCLDataType(dtype), i,
                         in_group ? ctx->group_comm : ctx->global_comm, stream));
      data += bytes_per_shard;
    }
  } else {
    if (send.defined()) {
      LOG(WARNING) << "ValueError: buffer `send` must be None when (worker_id != 0 && !in_group) "
                      "or (worker_id % group_size != 0 && in_group). However, got send = "
                   << send.get() << ". This will be ignored.";
    }
    NCCL_CALL(ncclGroupStart());
  }
  int64_t numel = recv.Shape()->Product();
  DataType dtype(recv->dtype);
  NCCL_CALL(ncclRecv(recv->data, numel, AsNCCLDataType(dtype), 0,
                     in_group ? ctx->group_comm : ctx->global_comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void GatherToWorker0(NDArray send, bool in_group, Optional<NDArray> recv) {
  CHECK(send.defined()) << "ValueError: buffer `send` must not be None";
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  int worker_id = ctx->worker->worker_id;
  int num_workers = ctx->worker->num_workers;
  int group_size = num_workers / ctx->worker->num_groups;
  bool is_sender = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
  int num_receiver = in_group ? group_size : num_workers;
  deviceStream_t stream = ctx->GetDefaultStream();
  if (is_sender) {
    CHECK(recv.defined()) << "ValueError: buffer `recv` must be provided when worker_id == 0.";
    NDArray buffer = recv.value();
    int64_t numel = buffer.Shape()->Product();
    CHECK_EQ(numel % num_receiver, 0) << "ValueError: Gathering evenly requires that the number "
                                         "of elements in the buffer to be "
                                         "divisible by the number of workers, but got numel = "
                                      << numel << " and " << num_receiver << " workers.";
    DataType dtype(buffer->dtype);
    int64_t numel_per_shard = numel / num_receiver;
    int64_t bytes_per_shard = numel_per_shard * dtype.bytes();
    CHECK_EQ(numel_per_shard, send.Shape()->Product())
        << "ValueError: The number of elements in buffer `send` must be the same as each shard "
           "of "
           "buffer `recv`. `recv.size` is "
        << numel << ", but `send.size` is " << send.Shape()->Product() << ".";
    NCCL_CALL(ncclGroupStart());
    uint8_t* data = static_cast<uint8_t*>(buffer->data);
    for (int i = 0; i < num_receiver; ++i) {
      NCCL_CALL(ncclRecv(data, numel_per_shard, AsNCCLDataType(dtype), i,
                         in_group ? ctx->group_comm : ctx->global_comm, stream));
      data += bytes_per_shard;
    }
  } else {
    if (recv.defined()) {
      LOG(WARNING) << "ValueError: buffer `recv` must be None when (worker_id != 0 && !in_group) "
                      "or (worker_id % group_size != 0 && in_group). However, got recv = "
                   << recv.get() << ". This will be ignored.";
    }
    NCCL_CALL(ncclGroupStart());
  }
  int64_t numel = send.Shape()->Product();
  DataType dtype(send->dtype);
  NCCL_CALL(ncclSend(send->data, numel, AsNCCLDataType(dtype), 0,
                     in_group ? ctx->group_comm : ctx->global_comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void RecvFromWorker0(NDArray buffer) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  CHECK_NE(ctx->worker->worker_id, 0)
      << "ValueError: Worker 0 is not allowed to call RecvFromWorker0.";
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()), 0,
                     ctx->global_comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void SendToNextGroup(NDArray buffer) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  int worker_id = ctx->worker->worker_id;
  int group_size = ctx->worker->num_workers / ctx->worker->num_groups;
  int receiver_id = worker_id + group_size;
  CHECK_LT(receiver_id, ctx->worker->num_workers)
      << "The current group is already the last group and there is no such a next group.";
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclSend(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()),
                     receiver_id, ctx->global_comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void RecvFromPrevGroup(NDArray buffer) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  int worker_id = ctx->worker->worker_id;
  int group_size = ctx->worker->num_workers / ctx->worker->num_groups;
  int sender_id = worker_id - group_size;
  CHECK_GE(sender_id, 0)
      << "The current group is already the first group and there is no such a previous group.";
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()),
                     sender_id, ctx->global_comm, stream));
  NCCL_CALL(ncclGroupEnd());
}

void SendToWorker(NDArray buffer, int receiver_id) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  int worker_id = ctx->worker->worker_id;
  CHECK(receiver_id >= 0 && receiver_id < ctx->worker->num_workers)
      << "Invalid receiver id " << receiver_id << ". The world size is "
      << ctx->worker->num_workers;
  CHECK_NE(worker_id, receiver_id) << "Cannot send to worker itself.";
  NCCL_CALL(ncclSend(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()),
                     receiver_id, ctx->global_comm, stream));
}

void RecvFromWorker(NDArray buffer, int sender_id) {
  CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
  deviceStream_t stream = ctx->GetDefaultStream();
  int worker_id = ctx->worker->worker_id;
  CHECK(sender_id >= 0 && sender_id < ctx->worker->num_workers)
      << "Invalid sender id " << sender_id << ". The world size is " << ctx->worker->num_workers;
  CHECK_NE(worker_id, sender_id) << "Cannot receive from the worker itself.";
  NCCL_CALL(ncclRecv(buffer->data, buffer.Shape()->Product(), AsNCCLDataType(buffer.DataType()),
                     sender_id, ctx->global_comm, stream));
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
    .set_body_typed([](NDArray send, int kind, bool in_group, NDArray recv) {
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      nccl::AllReduce(send, static_cast<ReduceKind>(kind), in_group, recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".allgather")
    .set_body_typed([](NDArray send, bool in_group, NDArray recv) {
      nccl::AllGather(send, in_group, recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".broadcast_from_worker0")
    .set_body_typed(BroadcastFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".scatter_from_worker0")
    .set_body_typed(ScatterFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".gather_to_worker0")
    .set_body_typed(GatherToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_worker0")
    .set_body_typed(RecvFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".send_to_next_group")
    .set_body_typed(SendToNextGroup);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_prev_group")
    .set_body_typed(RecvFromPrevGroup);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".send_to_worker")
    .set_body_typed(SendToWorker);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".recv_from_worker")
    .set_body_typed(RecvFromWorker);
TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".sync_worker").set_body_typed(SyncWorker);

TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME
                    ".test_send_to_next_group_recv_from_prev_group")
    .set_body_typed([](NDArray buffer) {
      CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
      CHECK_EQ(ctx->worker->num_workers, 4) << "The test requires the world size to be 4.";
      CHECK_EQ(ctx->worker->num_groups, 2) << "The test requires the group size to be 2.";
      int group_size = ctx->worker->num_workers / ctx->worker->num_groups;
      int group_id = ctx->worker->worker_id / group_size;
      if (group_id == 0) {
        tvm::runtime::nccl::SendToNextGroup(buffer);
      } else {
        tvm::runtime::nccl::RecvFromPrevGroup(buffer);
      }
    });

TVM_REGISTER_GLOBAL("runtime.disco." TVM_DISCO_CCL_NAME ".test_worker2_sends_to_worker0")
    .set_body_typed([](NDArray buffer) {
      CCLThreadLocalContext* ctx = CCLThreadLocalContext::Get();
      CHECK_EQ(ctx->worker->num_workers, 4) << "The test requires the world size to be 4.";
      CHECK_EQ(ctx->worker->num_groups, 2) << "The test requires the group size to be 2.";
      if (ctx->worker->worker_id == 2) {
        tvm::runtime::nccl::SendToWorker(buffer, 0);
      } else if (ctx->worker->worker_id == 0) {
        tvm::runtime::nccl::RecvFromWorker(buffer, 2);
      }
    });

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
