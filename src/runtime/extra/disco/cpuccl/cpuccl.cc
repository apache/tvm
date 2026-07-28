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

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>


namespace tvm {
namespace runtime {
namespace cpuccl {

#ifndef TVM_DISCO_DEVICE_NAME
#define TVM_DISCO_DEVICE_NAME "cpu"
#endif
#ifndef TVM_DISCO_CCL_NAME
#define TVM_DISCO_CCL_NAME "cpuccl"
#endif

/*! \brief Number of bytes taken by a single element of `dtype`. */
inline int64_t DTypeBytes(DLDataType dtype) { return (dtype.bits * dtype.lanes + 7) / 8; }

template <typename T>
static void DivideInPlace(T* data, int64_t n, int div) {
  for (int64_t i = 0; i < n; ++i) data[i] /= static_cast<T>(div);
}

static void DivideInPlaceBytes(void* data, int64_t numel, DLDataType dtype, int div) {
  if (dtype == DLDataType{kDLFloat, 32, 1})
    DivideInPlace(static_cast<float*>(data), numel, div);
  else if (dtype == DLDataType{kDLFloat, 64, 1})
    DivideInPlace(static_cast<double*>(data), numel, div);
  else if (dtype == DLDataType{kDLInt, 32, 1})
    DivideInPlace(static_cast<int32_t*>(data), numel, div);
  else if (dtype == DLDataType{kDLInt, 64, 1})
    DivideInPlace(static_cast<int64_t*>(data), numel, div);
  else
    TVM_FFI_THROW(ValueError) << "CPU collective: kAvg unsupported dtype " << dtype;
}

template <typename T>
static void ReduceInPlace(T* dst, const T* src, int64_t n, ReduceKind op) {
  switch (op) {
    case ReduceKind::kSum:
      for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
      break;
    case ReduceKind::kMax:
      for (int64_t i = 0; i < n; ++i) dst[i] = std::max(dst[i], src[i]);
      break;
    case ReduceKind::kMin:
      for (int64_t i = 0; i < n; ++i) dst[i] = std::min(dst[i], src[i]);
      break;
    case ReduceKind::kProd:
      for (int64_t i = 0; i < n; ++i) dst[i] *= src[i];
      break;
    default:
      TVM_FFI_THROW(ValueError) << "CPU collective: unsupported ReduceKind "
                                << static_cast<int>(op);
  }
}

static void ReduceBytes(void* dst, const void* src, int64_t numel, DLDataType dtype,
                        ReduceKind op) {
  if (dtype == DLDataType{kDLFloat, 32, 1})
    ReduceInPlace(static_cast<float*>(dst), static_cast<const float*>(src), numel, op);
  else if (dtype == DLDataType{kDLFloat, 64, 1})
    ReduceInPlace(static_cast<double*>(dst), static_cast<const double*>(src), numel, op);
  else if (dtype == DLDataType{kDLInt, 32, 1})
    ReduceInPlace(static_cast<int32_t*>(dst), static_cast<const int32_t*>(src), numel, op);
  else if (dtype == DLDataType{kDLInt, 64, 1})
    ReduceInPlace(static_cast<int64_t*>(dst), static_cast<const int64_t*>(src), numel, op);
  else
    TVM_FFI_THROW(ValueError) << "CPU collective: unsupported dtype " << dtype;
}

void InitCCL(Session sess, ffi::Shape device_ids) {
  DRef func = sess->GetGlobalFunc("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker");
  DLOG(INFO) << "Initializing " TVM_DISCO_CCL_NAME " with devices: " << device_ids;
  sess->CallPacked(func, device_ids);
}

void InitCCLPerWorker(ffi::Shape device_ids) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  TVM_FFI_ICHECK(worker != nullptr) << "Must be called on a DiscoWorker thread";
  if (worker->num_workers > 1) {
    TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
        << "CPU ring not built; create session with build_ring=true";
  }
  worker->ccl = TVM_DISCO_CCL_NAME;
}

void SyncWorker() {}

void AllReduce(Tensor send, ReduceKind reduce_kind, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();

  if (worker->num_workers > 1) {
    TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
        << "Ring not initialized";
  }

  DLDataType dtype = send->dtype;
  int64_t dtype_bytes = DTypeBytes(dtype);
  int num_workers = worker->num_workers;
  int rank = worker->worker_id;
  int64_t numel = send.Shape().Product();
  int64_t bytes = numel * dtype_bytes;

  std::memcpy(recv->data, send->data, static_cast<size_t>(bytes));
  char* buf = static_cast<char*>(recv->data);

  ReduceKind acc_kind = (reduce_kind == ReduceKind::kAvg) ? ReduceKind::kSum : reduce_kind;

  if (num_workers == 1) return;

  TVM_FFI_CHECK_EQ(numel % num_workers, 0, ValueError)
      << "cpuccl AllReduce: element count (" << numel << ") must be divisible by num_workers ("
      << num_workers << "). A non-divisible size leaves empty chunks in the ring reduce-scatter "
      << "and deadlocks. Pad/reshape so numel % num_workers == 0.";

  int64_t chunk_elem = (numel + num_workers - 1) / num_workers;
  int64_t chunk_bytes = chunk_elem * dtype_bytes;
  std::vector<char> tmp(static_cast<size_t>(chunk_bytes));

  // Phase A: reduce-scatter
  for (int r = 0; r < num_workers - 1; ++r) {
    int si = ((rank - r) % num_workers + num_workers) % num_workers;
    int ri = ((rank - r - 1) % num_workers + num_workers) % num_workers;
    int64_t s_off = si * chunk_bytes;
    int64_t r_off = ri * chunk_bytes;
    int64_t s_size = std::min(chunk_bytes, bytes - s_off);
    int64_t r_size = std::min(chunk_bytes, bytes - r_off);
    if (s_size <= 0 || r_size <= 0) continue;

    worker->ring_out->Send(buf + s_off, static_cast<size_t>(s_size));
    worker->ring_in->Recv(tmp.data(), static_cast<size_t>(r_size));
    ReduceBytes(buf + r_off, tmp.data(), r_size / dtype_bytes, dtype, acc_kind);
  }

  // Phase B: allgather
  for (int r = 0; r < num_workers - 1; ++r) {
    int si = ((rank + 1 - r) % num_workers + num_workers) % num_workers;
    int ri = ((rank - r) % num_workers + num_workers) % num_workers;
    int64_t s_off = si * chunk_bytes;
    int64_t r_off = ri * chunk_bytes;
    int64_t s_size = std::min(chunk_bytes, bytes - s_off);
    int64_t r_size = std::min(chunk_bytes, bytes - r_off);
    if (s_size <= 0 || r_size <= 0) continue;

    worker->ring_out->Send(buf + s_off, static_cast<size_t>(s_size));
    worker->ring_in->Recv(buf + r_off, static_cast<size_t>(r_size));
  }

  if (reduce_kind == ReduceKind::kAvg) {
    DivideInPlaceBytes(buf, numel, dtype, num_workers);
  }
}

void AllGather(Tensor send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
      << "Ring not initialized";

  int num_workers = worker->num_workers;
  int rank = worker->worker_id;
  int64_t chunk_bytes = send.Shape().Product() * DTypeBytes(send->dtype);
  char* buf = static_cast<char*>(recv->data);

  std::memcpy(buf + rank * chunk_bytes, send->data, static_cast<size_t>(chunk_bytes));

  if (num_workers == 1) return;

  for (int r = 0; r < num_workers - 1; ++r) {
    int si = ((rank - r) % num_workers + num_workers) % num_workers;
    int ri = ((rank - r - 1) % num_workers + num_workers) % num_workers;

    worker->ring_out->Send(buf + si * chunk_bytes, static_cast<size_t>(chunk_bytes));
    worker->ring_in->Recv(buf + ri * chunk_bytes, static_cast<size_t>(chunk_bytes));
  }
}

void BroadcastFromWorker0(ffi::Optional<Tensor> send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();

  if (worker->num_workers > 1) {
    TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
        << "Ring not initialized";
  }

  int rank = worker->worker_id;
  int num_workers = worker->num_workers;
  int64_t bytes = recv.Shape().Product() * DTypeBytes(recv->dtype);

  if (rank == 0) {
    TVM_FFI_CHECK(send.has_value(), ValueError)
        << "Worker 0 must provide send buffer for broadcast";
    std::memcpy(recv->data, send.value()->data, static_cast<size_t>(bytes));

    if (num_workers > 1) {
      worker->ring_out->Send(recv->data, static_cast<size_t>(bytes));
    }
  } else {
    worker->ring_in->Recv(recv->data, static_cast<size_t>(bytes));

    if (rank < num_workers - 1) {
      worker->ring_out->Send(recv->data, static_cast<size_t>(bytes));
    }
  }
}

void ScatterFromWorker0(ffi::Optional<Tensor> send, bool /*in_group*/, Tensor recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();

  if (worker->num_workers > 1) {
    TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
        << "Ring not initialized";
  }

  int rank = worker->worker_id;
  int num_workers = worker->num_workers;
  int64_t recv_bytes = recv.Shape().Product() * DTypeBytes(recv->dtype);

  if (num_workers == 1) {
    TVM_FFI_CHECK(send.has_value(), ValueError)
        << "Worker 0 must provide send buffer for scatter";
    std::memcpy(recv->data, send.value()->data, static_cast<size_t>(recv_bytes));
    return;
  }

  if (rank == 0) {
    TVM_FFI_CHECK(send.has_value(), ValueError)
        << "Worker 0 must provide send buffer for scatter";
    char* src = static_cast<char*>(send.value()->data);
    std::memcpy(recv->data, src, static_cast<size_t>(recv_bytes));
    int64_t tail = recv_bytes * (num_workers - 1);
    worker->ring_out->Send(src + recv_bytes, static_cast<size_t>(tail));
  } else {
    int64_t incoming = recv_bytes * (num_workers - rank);
    std::vector<char> buf(static_cast<size_t>(incoming));
    worker->ring_in->Recv(buf.data(), static_cast<size_t>(incoming));
    std::memcpy(recv->data, buf.data(), static_cast<size_t>(recv_bytes));

    if (rank < num_workers - 1) {
      int64_t forward_bytes = incoming - recv_bytes;
      worker->ring_out->Send(buf.data() + recv_bytes, static_cast<size_t>(forward_bytes));
    }
  }
}

void GatherToWorker0(Tensor send, bool /*in_group*/, ffi::Optional<Tensor> recv) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();

  int rank = worker->worker_id;
  int num_workers = worker->num_workers;
  int64_t chunk_bytes = send.Shape().Product() * DTypeBytes(send->dtype);

  if (num_workers == 1) {
    TVM_FFI_CHECK(recv.has_value(), ValueError)
        << "Worker 0 must provide recv buffer for gather";
    std::memcpy(recv.value()->data, send->data, static_cast<size_t>(chunk_bytes));
    return;
  }

  TVM_FFI_ICHECK(worker->ring_in != nullptr && worker->ring_out != nullptr)
      << "Ring not initialized";

  if (rank == 0) {
    TVM_FFI_CHECK(recv.has_value(), ValueError)
        << "Worker 0 must provide recv buffer for gather";
    char* dst = static_cast<char*>(recv.value()->data);
    std::memcpy(dst, send->data, static_cast<size_t>(chunk_bytes));
    int64_t incoming = chunk_bytes * (num_workers - 1);
    worker->ring_in->Recv(dst + chunk_bytes, static_cast<size_t>(incoming));
  } else {
    int64_t prior = chunk_bytes * (rank - 1);
    std::vector<char> buf(static_cast<size_t>(prior + chunk_bytes));
    if (rank > 1) {
      worker->ring_in->Recv(buf.data(), static_cast<size_t>(prior));
    }
    std::memcpy(buf.data() + prior, send->data, static_cast<size_t>(chunk_bytes));
    worker->ring_out->Send(buf.data(), buf.size());
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl", InitCCL)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".init_ccl_per_worker", InitCCLPerWorker)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".allreduce", AllReduce)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".allgather", AllGather)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".broadcast_from_worker0", BroadcastFromWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".scatter_from_worker0", ScatterFromWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".gather_to_worker0", GatherToWorker0)
      .def("runtime.disco." TVM_DISCO_CCL_NAME ".sync_worker", SyncWorker);
}

}  // namespace cpuccl
}  // namespace runtime
}  // namespace tvm
