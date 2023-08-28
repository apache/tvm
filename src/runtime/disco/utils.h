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
#ifndef TVM_RUNTIME_DISCO_UTILS_H_
#define TVM_RUNTIME_DISCO_UTILS_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/disco/session.h>

#include <string>
#include <vector>

#include "./worker.h"

namespace tvm {
namespace runtime {

inline Device UseDefaultDeviceIfNone(Device device) {
  if (device.device_type == 0 && device.device_id == 0) {
    return DiscoWorker::ThreadLocal()->default_device;
  }
  return device;
}

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
 * \brief Converts a 1-d shape tuple to an integer.
 * \note At the time of scaffolding Disco, RelaxVM has not provided mature support for standalone
 * integers. A common workaround is to use a 1-d shape tuple as an integer.
 */
inline int64_t IntegerFromShapeTuple(const ShapeTuple& shape) {
  CHECK_EQ(shape.size(), 1) << "ValueError: shape tuple must be 1-d to be converted to integer.";
  return shape[0];
}

/*!
 * \brief Get the shape of a result tensor if it is scattered along a given axis.
 * \param shape The shape of the input tensor.
 * \param dim The axis along which the tensor is scattered.
 * \param num_shards The number of shards.
 * \return The shape of the result tensor.
 */
inline ShapeTuple ShardShape(const ShapeTuple& shape, int dim, int num_shards) {
  CHECK(0 <= dim && dim < static_cast<int>(shape.size()))
      << "ValueError: Cannot scatter at dim " << dim << ", because "
      << "shape is " << shape << ".";
  CHECK_EQ(shape[dim] % num_shards, 0)
      << "ValueError: The shape " << shape << " cannot be scattered at dim " << dim << " into "
      << num_shards << " shards.";
  std::vector<ShapeTupleObj::index_type> result{shape.begin(), shape.end()};
  result[dim] /= num_shards;
  return ShapeTuple(result);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_UTILS_H_
