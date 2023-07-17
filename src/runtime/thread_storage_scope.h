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
 * \brief Extract launch parameters configuration from TVMArgs.
 */
#ifndef TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <tvm/runtime/metadata.h>
#include <tvm/runtime/packed_func.h>

#include <string>
#include <vector>

#include "meta_data.h"

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
      LOG(FATAL) << "unknown rank";
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
      default:
        LOG(FATAL) << "unknown storage scope";
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
    } else {
      LOG(FATAL) << "unknown storage scope " << s;
    }
    return r;
  }
};

/*! \brief class to represent thread scope */
struct ThreadScope {
  /*! \brief The rank of thread scope */
  int rank{0};
  /*! \brief the dimension index under the rank */
  int dim_index{0};
  /*!
   * \brief Create storage scope from string
   * \param s The string to be parsed.
   * \return The storage scope.
   */
  static ThreadScope Create(const std::string& s) {
    ThreadScope r;
    if (s.compare(0, 7, "vthread") == 0 || s == "cthread") {
      // virtual thread at the same level as local
      r.rank = 1;
      r.dim_index = -1;
    } else if (s.compare(0, 9, "blockIdx.") == 0) {
      r.rank = 0;
      r.dim_index = static_cast<int>(s[9] - 'x');
    } else if (s.compare(0, 10, "threadIdx.") == 0) {
      r.rank = 1;
      r.dim_index = static_cast<int>(s[10] - 'x');
    } else {
      LOG(FATAL) << "Unknown threadscope " << s;
    }
    return r;
  }
};

/*! \brief workload specification */
struct ThreadWorkLoad {
  // array, first three are thread configuration.
  size_t work_size[6];
  // Dynamic shared memory allocation size in bytes.
  size_t dyn_shmem_size{0};
  /*!
   * \param i The block dimension.
   * \return i-th block dim
   */
  inline size_t block_dim(size_t i) const { return work_size[i + 3]; }
  /*!
   * \param i The grid dimension.
   * \return i-th grid dim
   */
  inline size_t grid_dim(size_t i) const { return work_size[i]; }
};
/*! \brief Launch parameters configuration */
class LaunchParamConfig {
 public:
  void Init(size_t base, const std::vector<std::string>& launch_param_tags) {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      const std::string& tag = launch_param_tags[i];
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        ICHECK_EQ(i, launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
      } else {
        ThreadScope ts = ThreadScope::Create(tag);
        arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
        filled[ts.rank * 3 + ts.dim_index] = true;
      }
    }
    work_dim_ = 1;
    for (int i = 0; i < 3; ++i) {
      if (filled[i] || filled[i + 3]) {
        work_dim_ = i + 1;
      }
    }
  }
  // extract workload from arguments.
  ThreadWorkLoad Extract(TVMArgs x) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      // Dynamic shapes can result in 0 dim size. Guard to ensure that the dim size is at least 1.
      size_t size = static_cast<size_t>(x.values[base_ + i].v_int64);
      if (size > 0) {
        w.work_size[arg_index_map_[i]] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(x.values[base_ + arg_index_map_.size()].v_int64);
    }
    return w;
  }
  // return the work dim
  size_t work_dim() const { return work_dim_; }

 private:
  /*! \brief base axis */
  size_t base_;
  /*! \brief The worker dimension */
  size_t work_dim_;
  /*! \brief The index mapping. */
  std::vector<uint32_t> arg_index_map_;
  /*! \brief Whether or not use dynamic shared memory. */
  bool use_dyn_shared_memory_{false};
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
