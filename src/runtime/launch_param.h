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
 * \file launch_param.h
 * \brief Extract launch parameters configuration from ffi::PackedArgs.
 */
#ifndef TVM_RUNTIME_LAUNCH_PARAM_H_
#define TVM_RUNTIME_LAUNCH_PARAM_H_

#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "metadata.h"

namespace tvm {
namespace runtime {

/*! \brief workload specification */
struct ThreadWorkLoad {
  // work_size layout: [0-2] grid, [3-5] block, [6-8] cluster, [9-11] preferred_cluster
  size_t work_size[12];
  // Dynamic shared memory allocation size in bytes.
  size_t dyn_shmem_size{0};
  /*!
   * \param i The block dimension.
   * \return i-th block dim
   */
  inline size_t BlockDim(size_t i) const { return work_size[i + 3]; }
  /*!
   * \param i The grid dimension.
   * \return i-th grid dim
   */
  inline size_t GridDim(size_t i) const { return work_size[i]; }
  /*!
   * \param i The cluster dimension.
   * \return i-th cluster dim
   */
  inline size_t ClusterDim(size_t i) const { return work_size[i + 6]; }
  /*!
   * \param i The preferred cluster dimension.
   * \return i-th preferred cluster dim
   */
  inline size_t PreferredClusterDim(size_t i) const { return work_size[i + 9]; }
};

/*! \brief Launch parameters configuration */
class LaunchParamConfig {
 public:
  void Init(size_t base, const ffi::Array<ffi::String>& launch_param_tags) {
    base_ = base;
    use_cluster_launch_ = false;
    std::vector<bool> filled(12, false);
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      std::string tag(launch_param_tags[i]);
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        TVM_FFI_ICHECK_EQ(i, launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
      } else if (tag == launch_param::kUseProgramaticDependentLaunch) {
        use_programmatic_dependent_launch_ = true;
      } else if (tag == launch_param::kUseCooperativeLaunch) {
        use_cooperative_launch_ = true;
      } else if (tag == launch_param::kUseRequiredBlockDimension) {
        use_required_block_dimension_ = true;
      } else {
        static constexpr const char* kDimensionTags[] = {"blockIdx.x",
                                                         "blockIdx.y",
                                                         "blockIdx.z",
                                                         "threadIdx.x",
                                                         "threadIdx.y",
                                                         "threadIdx.z",
                                                         "clusterCtaIdx.x",
                                                         "clusterCtaIdx.y",
                                                         "clusterCtaIdx.z",
                                                         "preferredClusterCtaIdx.x",
                                                         "preferredClusterCtaIdx.y",
                                                         "preferredClusterCtaIdx.z"};
        uint32_t index = 0;
        while (index < 12 && tag != kDimensionTags[index]) {
          ++index;
        }
        TVM_FFI_ICHECK_LT(index, 12) << "Unsupported launch parameter tag: " << tag;
        if (index >= 6 && index < 9) {
          use_cluster_launch_ = true;
        }
        arg_index_map_.push_back(index);
        filled[index] = true;
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
  ThreadWorkLoad Extract(ffi::PackedArgs args) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 12, 1);
    const TVMFFIAny* raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());

    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      // Dynamic shapes can result in 0 dim size. Guard to ensure that the dim size is at least 1.
      size_t size = static_cast<size_t>(raw_args[base_ + i].v_int64);
      if (size > 0) {
        w.work_size[arg_index_map_[i]] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(raw_args[base_ + arg_index_map_.size()].v_int64);
    }
    return w;
  }
  // return the work dim
  size_t work_dim() const { return work_dim_; }

  bool use_programmatic_dependent_launch() const { return use_programmatic_dependent_launch_; }

  bool use_cooperative_launch() const { return use_cooperative_launch_; }

  bool use_required_block_dimension() const { return use_required_block_dimension_; }

  bool use_cluster_launch() const { return use_cluster_launch_; }

 private:
  /*! \brief base axis */
  size_t base_;
  /*! \brief The worker dimension */
  size_t work_dim_;
  /*! \brief The index mapping. */
  std::vector<uint32_t> arg_index_map_;
  /*! \brief Whether or not use dynamic shared memory. */
  bool use_dyn_shared_memory_{false};
  /*! \brief Whether or not use programmatic dependent launch. */
  bool use_programmatic_dependent_launch_{false};
  /*! \brief Whether or not use cooperative launch. */
  bool use_cooperative_launch_{false};
  /*! \brief Whether CUDA should use the kernel's statically required block dimension. */
  bool use_required_block_dimension_{false};
  /*! \brief Whether the kernel declares a cluster-to-CTA scope. */
  bool use_cluster_launch_{false};
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_LAUNCH_PARAM_H_
