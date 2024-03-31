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

/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#include "runtime.h"

#include <cuda.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tl {

using namespace runtime;

struct TensorMapArgs {
  CUtensorMap* map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void* globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;
};

static TensorMapArgs ExtractTensormapArgs(TVMArgs args) {
  TensorMapArgs T;
  int idx = 0;
  ICHECK(args.num_args >= 8);
  T.map = reinterpret_cast<CUtensorMap*>(static_cast<void*>(args[idx++]));
  T.type = static_cast<CUtensorMapDataType>(static_cast<int64_t>(args[idx++]));
  T.tensorRank = static_cast<cuuint32_t>(static_cast<int64_t>(args[idx++]));
  T.globalAddress = args[idx++];
  ICHECK(T.tensorRank >= 1 && T.tensorRank <= 5);
  ICHECK(args.num_args == static_cast<int>(8 + T.tensorRank * 4));
  for (size_t i = 0; i < T.tensorRank; i++) {
    T.globalDim[i] = static_cast<cuuint64_t>(args[idx++]);
  }
  for (size_t i = 0; i < T.tensorRank; i++) {
    T.globalStride[i] = static_cast<cuuint64_t>(args[idx++]);
  }
  for (size_t i = 0; i < T.tensorRank; i++) {
    T.boxDim[i] = static_cast<cuuint64_t>(args[idx++]);
  }
  for (size_t i = 0; i < T.tensorRank; i++) {
    T.elementStrides[i] = static_cast<cuuint64_t>(args[idx++]);
  }
  T.interleave = static_cast<CUtensorMapInterleave>(static_cast<int64_t>(args[idx++]));
  T.swizzle = static_cast<CUtensorMapSwizzle>(static_cast<int64_t>(args[idx++]));
  T.l2Promotion = static_cast<CUtensorMapL2promotion>(static_cast<int64_t>(args[idx++]));
  T.oobFill = static_cast<CUtensorMapFloatOOBfill>(static_cast<int64_t>(args[idx++]));
  return T;
}

template <typename T>
static std::string ArrayToStr(const T* ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0) ss << ", ";
    ss << ptr[i];
  }
  ss << "]";
  return ss.str();
}

static std::string ToDebugString(const TensorMapArgs& T) {
  std::stringstream ss;
  ss << "TMA Desc Addr:   " << T.map << std::endl
     << "format         " << T.type << std::endl
     << "dim            " << T.tensorRank << std::endl
     << "gmem_address   " << T.globalAddress << std::endl
     << "globalDim      " << ArrayToStr(T.globalDim, T.tensorRank) << std::endl
     << "globalStrides  " << ArrayToStr(T.globalStride, T.tensorRank) << std::endl
     << "boxDim         " << ArrayToStr(T.boxDim, T.tensorRank) << std::endl
     << "elementStrides " << ArrayToStr(T.elementStrides, T.tensorRank) << std::endl
     << "interleave     " << T.interleave << std::endl
     << "swizzle        " << T.swizzle << std::endl
     << "l2Promotion    " << T.l2Promotion << std::endl
     << "oobFill        " << T.oobFill << std::endl;
  return ss.str();
}

// set device api
TVM_REGISTER_GLOBAL(tvm_tensormap_create).set_body([](TVMArgs args, TVMRetValue* ret) {
  TensorMapArgs T = ExtractTensormapArgs(args);
  CUresult result = cuTensorMapEncodeTiled(T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
                                           T.globalStride + 1, T.boxDim, T.elementStrides,
                                           T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
  if (result != CUDA_SUCCESS) {
    LOG_FATAL << "Failed to initialize the TMA descriptor " << result << std::endl
              << ToDebugString(T);
  }
  *ret = static_cast<int>(result);
});

}  // namespace tl
}  // namespace tvm
