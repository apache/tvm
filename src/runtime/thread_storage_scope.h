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
 * \file thread_storage_scope.h
 * \brief Thread and storage scope helpers.
 */
#ifndef TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <tvm/ffi/error.h>

#include <cstddef>
#include <functional>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Memory hierachy rank in the storage system
 * \note The global rank and shared rank have one to one
 *       correspondence to the thread rank.
 */
enum class StorageRank {
  /*! \brief global memory */
  kGlobal = 0,
  /*! \brief shared memory among thread group */
  kShared = 1,
  /*!
   * \brief reserved for warp memory.
   *  This is only used by programming model.
   *  There is no such memory usually in GPU.
   *  Instead, we can simulate it by registers and shuffle.
   */
  kWarp = 2,
  /*! \brief thread local memory */
  kLocal = 3,
  /*! \brief wmma scope memory of matrix_a */
  kWMMAMatrixA = 4,
  /*! \brief wmma scope memory of matrix_b */
  kWMMAMatrixB = 5,
  /*! \brief wmma scope memory of accumulator */
  kWMMAAccumulator = 6,
  /*! \brief global scope texture memory */
  kTexture = 7,
  /*! \brief global scope amx tmm memory */
  kAMXTMM = 8,
  /*! \brief mma scope memory of matrix_a */
  kMMAMatrixA = 9,
  /*! \brief mma scope memory of matrix_b */
  kMMAMatrixB = 10,
  /*! \brief mma scope memory of accumulator */
  kMMAMatrixC = 11,
  /*! \brief Metal SIMD group memory */
  kMetalSimdGroup = 12,
  /*! \brief Metal cooperative_tensor memory (MetalPerformancePrimitives) */
  kMetalCooperativeTensor = 13,
  /*! \brief Trainium sbuf */
  kTrnSbuf = 14,
  /*! \brief Trainium psum */
  kTrnPsum = 15,
};

/*!
 * \param thread_scope_rank The thread scope rank
 * \return default storage rank given the thread scope
 */
inline StorageRank DefaultStorageRank(int thread_scope_rank) {
  switch (thread_scope_rank) {
    case -1:
      return StorageRank::kGlobal;
    case 0:
      return StorageRank::kShared;
    case 1:
      return StorageRank::kLocal;
    default: {
      TVM_FFI_THROW(InternalError) << "unknown rank";
    }
  }
}

/*! \brief class to represent storage scope */
struct StorageScope {
  /*! \brief The rank of the storage */
  StorageRank rank{StorageRank::kGlobal};
  /*! \brief tag for special purpose memory. */
  std::string tag;
  // comparator
  inline bool operator==(const StorageScope& other) const {
    return rank == other.rank && tag == other.tag;
  }
  inline bool operator!=(const StorageScope& other) const { return !(*this == other); }
  inline std::string to_string() const {
    std::string ret;
    switch (rank) {
      case StorageRank::kGlobal:
        return "global" + tag;
      case StorageRank::kShared:
        return "shared" + tag;
      case StorageRank::kWarp:
        return "warp" + tag;
      case StorageRank::kLocal:
        return "local" + tag;
      case StorageRank::kWMMAMatrixA:
        return "wmma.matrix_a" + tag;
      case StorageRank::kWMMAMatrixB:
        return "wmma.matrix_b" + tag;
      case StorageRank::kWMMAAccumulator:
        return "wmma.accumulator" + tag;
      case StorageRank::kTexture:
        return "texture" + tag;
      case StorageRank::kMMAMatrixA:
        return "m16n8k8.matrixA" + tag;
      case StorageRank::kMMAMatrixB:
        return "m16n8k8.matrixB" + tag;
      case StorageRank::kMMAMatrixC:
        return "m16n8k8.matrixC" + tag;
      case StorageRank::kMetalSimdGroup:
        return "metal.simdgroup" + tag;
      case StorageRank::kMetalCooperativeTensor:
        return "metal.cooperative_tensor" + tag;
      default:
        TVM_FFI_THROW(InternalError) << "unknown storage scope";
        return "";
    }
  }
  /*!
   * \brief Create storage scope from string
   * \param s The string to be parsed.
   * \return The storage scope.
   */
  static StorageScope Create(const std::string& s) {
    StorageScope r;
    if (s.empty()) {
      r.rank = StorageRank::kGlobal;
    } else if (s.compare(0, 6, "global") == 0) {
      r.rank = StorageRank::kGlobal;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 6, "shared") == 0) {
      r.rank = StorageRank::kShared;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 4, "warp") == 0) {
      r.rank = StorageRank::kWarp;
      r.tag = s.substr(4, std::string::npos);
    } else if (s.compare(0, 5, "local") == 0) {
      r.rank = StorageRank::kLocal;
      r.tag = s.substr(5, std::string::npos);
    } else if (s.compare(0, 13, "wmma.matrix_a") == 0) {
      r.rank = StorageRank::kWMMAMatrixA;
      r.tag = s.substr(13, std::string::npos);
    } else if (s.compare(0, 13, "wmma.matrix_b") == 0) {
      r.rank = StorageRank::kWMMAMatrixB;
      r.tag = s.substr(13, std::string::npos);
    } else if (s.compare(0, 16, "wmma.accumulator") == 0) {
      r.rank = StorageRank::kWMMAAccumulator;
      r.tag = s.substr(16, std::string::npos);
    } else if (s.compare(0, 7, "texture") == 0) {
      r.rank = StorageRank::kTexture;
      r.tag = s.substr(7, std::string::npos);
    } else if (s.compare(0, 7, "amx.tmm") == 0) {
      r.rank = StorageRank::kAMXTMM;
      r.tag = s.substr(7, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixA") == 0) {
      r.rank = StorageRank::kMMAMatrixA;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixB") == 0) {
      r.rank = StorageRank::kMMAMatrixB;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "m16n8k8.matrixC") == 0) {
      r.rank = StorageRank::kMMAMatrixC;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 15, "metal.simdgroup") == 0) {
      r.rank = StorageRank::kMetalSimdGroup;
      r.tag = s.substr(15, std::string::npos);
    } else if (s.compare(0, 24, "metal.cooperative_tensor") == 0) {
      r.rank = StorageRank::kMetalCooperativeTensor;
      r.tag = s.substr(24, std::string::npos);
    } else if (s.compare(0, 8, "trn.sbuf") == 0) {
      r.rank = StorageRank::kTrnSbuf;
      r.tag = s.substr(8, std::string::npos);
    } else if (s.compare(0, 8, "trn.psum") == 0) {
      r.rank = StorageRank::kTrnPsum;
      r.tag = s.substr(8, std::string::npos);
    } else {
      TVM_FFI_THROW(InternalError) << "unknown storage scope " << s;
    }
    return r;
  }
};

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::runtime::StorageScope> {
  std::size_t operator()(const ::tvm::runtime::StorageScope& k) const {
    return static_cast<size_t>(k.rank);
  }
};
}  // namespace std
#endif  // TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
