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
#ifndef TVM_RUNTIME_DISCO_BUILTIN_H_
#define TVM_RUNTIME_DISCO_BUILTIN_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Possible kinds of reduction operations.
 */
enum class ReduceKind : int32_t {
  kSum = 0,
  kProd = 1,
  kMin = 2,
  kMax = 3,
  kAvg = 4,
};

/*! \brief Converts `ReduceKind` to string */
inline std::string ReduceKind2String(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::kSum:
      return "kSum";
    case ReduceKind::kProd:
      return "kProd";
    case ReduceKind::kMin:
      return "kMin";
    case ReduceKind::kMax:
      return "kMax";
    case ReduceKind::kAvg:
      return "kAvg";
  }
  LOG(FATAL) << "ValueError: Unknown ReduceKind: " << static_cast<int>(kind);
}

/*!
 * \brief Load a runtime Module, then create and initialize a RelaxVM
 * \param path The path to the runtime Module (a DSO file) to be loaded
 * \param device The default device used to initialize the RelaxVM
 * \return The RelaxVM as a runtime Module
 */
Module LoadVMModule(std::string path, Device device);
/*!
 * \brief Create an uninitialized empty NDArray
 * \param shape The shape of the NDArray
 * \param dtype The dtype of the NDArray
 * \param device The device the NDArray is created on. If None, use the thread local default device
 * \return The NDArray created
 */
NDArray DiscoEmptyNDArray(ShapeTuple shape, DataType dtype, Device device);
/*!
 * \brief Perform an allreduce operation using the underlying communication library
 * \param send The array send to perform allreduce on
 * \param reduce_kind The kind of reduction operation (e.g. sum, avg, min, max)
 * \param recv The array receives the outcome of allreduce
 */
void AllReduce(NDArray send, ReduceKind reduce_kind, NDArray recv);
/*!
 * \brief Perform an allgather operation using the underlying communication library
 * \param send The array send to perform allgather on
 * \param recv The array receives the outcome of allgather
 */
void AllGather(NDArray send, NDArray recv);
/*!
 * \brief Perform a broadcast operation from worker-0
 * \param send The buffer to be broadcasted
 * \param recv The buffer receives the broadcasted array
 */
TVM_DLL void BroadcastFromWorker0(NDArray send, NDArray recv);
/*!
 * \brief Perform a scatter operation from worker-0, chunking the given buffer into equal parts.
 * \param send For worker-0, it must be provided, and otherwise, the buffer must be None.
 * The buffer will be divided into equal parts and sent to each worker accordingly.
 * \param recv The receiving buffer, which must not be None.
 */
TVM_DLL void ScatterFromWorker0(Optional<NDArray> send, NDArray recv);
/*!
 * \brief Perform a gather operation to worker-0.
 * \param send The sending buffer, which must not be None.
 * \param recv For worker-0, it must be provided, and otherwise, the buffer must be None. The
 * receiving buffer will be divided into equal parts and receive from each worker accordingly.
 */
void GatherToWorker0(NDArray send, Optional<NDArray> recv);
/*!
 * \brief Receive a buffer from worker-0. No-op if the current worker is worker-0.
 * \param buffer The buffer to be received
 */
void RecvFromWorker0(NDArray buffer);
/*! \brief Get the local worker id */
int WorkerId();
/*!
 * \brief Called by the worker thread. Waiting until the worker completes all its tasks.
 * As a specific example, on a CUDA worker, it blocks until all kernels are launched and
 * cudaStreamSynchronize is complete.
 */
void SyncWorker();

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_BUILTIN_H_
