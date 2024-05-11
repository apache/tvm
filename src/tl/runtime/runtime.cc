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

  static TensorMapArgs Extract(TVMArgs args) {
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

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << std::endl
       << "format         " << type << std::endl
       << "dim            " << tensorRank << std::endl
       << "gmem_address   " << globalAddress << std::endl
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << std::endl
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << std::endl
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << std::endl
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << std::endl
       << "interleave     " << interleave << std::endl
       << "swizzle        " << swizzle << std::endl
       << "l2Promotion    " << l2Promotion << std::endl
       << "oobFill        " << oobFill << std::endl;
    return ss.str();
  }
};

// set device api
TVM_REGISTER_GLOBAL(tvm_tensormap_create_tiled).set_body([](TVMArgs args, TVMRetValue* ret) {
  TensorMapArgs T = TensorMapArgs::Extract(args);
  CUresult result = cuTensorMapEncodeTiled(
      T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim, T.globalStride + 1, T.boxDim,
      T.elementStrides, T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
  if (result != CUDA_SUCCESS) {
    LOG_FATAL << "Failed to initialize the TMA descriptor " << result << std::endl
              << T.ToDebugString();
  }
  *ret = static_cast<int>(result);
});

struct TensorMapIm2ColArgs {
  CUtensorMap* map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void* globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t elementStrides[5];
  int pixelBoxLowerCorner[3], pixelBoxUpperCorner[3];
  cuuint32_t smem_box_channel, smem_box_pixel;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapIm2ColArgs Extract(TVMArgs args) {
    TensorMapIm2ColArgs T;
    int idx = 0;
    ICHECK(args.num_args >= 8);
    T.map = reinterpret_cast<CUtensorMap*>(static_cast<void*>(args[idx++]));
    T.type = static_cast<CUtensorMapDataType>(static_cast<int64_t>(args[idx++]));
    T.tensorRank = static_cast<cuuint32_t>(static_cast<int64_t>(args[idx++]));
    T.globalAddress = args[idx++];
    ICHECK(T.tensorRank >= 3 && T.tensorRank <= 5);
    ICHECK(args.num_args == static_cast<int>(6 + T.tensorRank * 5));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = static_cast<cuuint64_t>(args[idx++]);
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = static_cast<cuuint64_t>(args[idx++]);
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = static_cast<cuuint64_t>(args[idx++]);
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxLowerCorner[i] = static_cast<int>(args[idx++]);
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxUpperCorner[i] = static_cast<int>(args[idx++]);
    }
    T.smem_box_pixel = static_cast<cuuint64_t>(args[idx++]);
    T.smem_box_channel = static_cast<cuuint64_t>(args[idx++]);
    T.interleave = static_cast<CUtensorMapInterleave>(static_cast<int64_t>(args[idx++]));
    T.swizzle = static_cast<CUtensorMapSwizzle>(static_cast<int64_t>(args[idx++]));
    T.l2Promotion = static_cast<CUtensorMapL2promotion>(static_cast<int64_t>(args[idx++]));
    T.oobFill = static_cast<CUtensorMapFloatOOBfill>(static_cast<int64_t>(args[idx++]));
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << std::endl
       << "format         " << type << std::endl
       << "dim            " << tensorRank << std::endl
       << "gmem_address   " << globalAddress << std::endl
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << std::endl
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << std::endl
       << "smem_box_pixel " << smem_box_pixel << std::endl
       << "smem_box_channel " << smem_box_channel << std::endl
       << "pixelBoxLowerCorner  " << ArrayToStr(pixelBoxLowerCorner, tensorRank - 2) << std::endl
       << "pixelBoxUpperCorner  " << ArrayToStr(pixelBoxUpperCorner, tensorRank - 2) << std::endl
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << std::endl
       << "interleave     " << interleave << std::endl
       << "swizzle        " << swizzle << std::endl
       << "l2Promotion    " << l2Promotion << std::endl
       << "oobFill        " << oobFill << std::endl;
    return ss.str();
  }
};

TVM_REGISTER_GLOBAL(tvm_tensormap_create_im2col).set_body([](TVMArgs args, TVMRetValue* ret) {
  TensorMapIm2ColArgs T = TensorMapIm2ColArgs::Extract(args);
  CUresult result = cuTensorMapEncodeIm2col(
      T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim, T.globalStride + 1,
      T.pixelBoxLowerCorner, T.pixelBoxUpperCorner, T.smem_box_channel, T.smem_box_pixel,
      T.elementStrides, T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
  if (result != CUDA_SUCCESS) {
    LOG_FATAL << "Failed to initialize the TMA descriptor " << result << std::endl
              << T.ToDebugString();
  }
  *ret = static_cast<int>(result);
});

}  // namespace tl
}  // namespace tvm
