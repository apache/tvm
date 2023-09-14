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
#ifndef TVM_RUNTIME_DISCO_NCCL_UTILS_H_
#define TVM_RUNTIME_DISCO_NCCL_UTILS_H_

#include <nccl.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/disco/session.h>

#include "../utils.h"

namespace tvm {
namespace runtime {
namespace nccl {

#define NCCL_CALL(cmd)                                       \
  do {                                                       \
    ncclResult_t r = cmd;                                    \
    if (r != ncclSuccess) {                                  \
      LOG(FATAL) << "NCCLErrror: " << ncclGetErrorString(r); \
    }                                                        \
  } while (0)

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

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_NCCL_UTILS_H_
