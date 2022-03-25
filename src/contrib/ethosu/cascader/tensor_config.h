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
 * \file src/contrib/ethosu/cascader/tensor_config.h
 * \brief TensorConfig object for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_TENSOR_CONFIG_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_TENSOR_CONFIG_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "graph.h"
#include "stripe_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class MemoryRegionNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \brief The name of the region */
  std::string name;
  /*! \brief The size of the region */
  int size;
  /*! \brief The read bandwidth of the region in bytes per cycle */
  int read_bandwidth;
  /*! \brief The write bandwidth of the region in bytes per cycle */
  int write_bandwidth;
  /*! \brief The read bandwidth of the region in bytes per cycle */
  int read_latency;
  /*! \brief The write bandwidth of the region in bytes per cycle */
  int write_latency;
  /*! \brief Length of memory burst */
  int burst_length;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.MemoryRegion";
  TVM_DECLARE_FINAL_OBJECT_INFO(MemoryRegionNode, Object)
};

class MemoryRegion : public ObjectRef {
 public:
  MemoryRegion(std::string name, int size, int read_bandwidth, int write_bandwidth,
               int read_latency, int write_latency, int burst_length) {
    auto n = make_object<MemoryRegionNode>();
    n->name = name;
    n->size = size;
    n->read_bandwidth = read_bandwidth;
    n->write_bandwidth = write_bandwidth;
    n->read_latency = read_latency;
    n->write_latency = write_latency;
    n->burst_length = burst_length;
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(MemoryRegion, ObjectRef, MemoryRegionNode);
};

/*! \brief The 'state' of a TensorConfig as used in the Plan generation algorithm.
 * BOUNDARY - Should describe a Plan input/output Tensor.
 * INTERIOR - Should describe an intermediate Tensor in a 'closed' Plan.
 */
enum TensorConfigState { BOUNDARY, INTERIOR };

/*! \brief Node to represent a TensorConfig */
class TensorConfigNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \return The Tensor the config applies to */
  const Tensor GetTensor() const { return tensor_; }
  /*! \return The region where the tensor is allocated */
  MemoryRegion GetHomeRegion() const { return home_region_; }
  /*!
   * \return The state of the TensorConfig.
   * \note The TensorConfigState is only used as part of the Plan generation algorithm. For a Plan
   * to be 'closed' (and therefore not subject to any further merging), all the TensorConfigs that
   * describe Plan input or output Tensors must be in the 'BOUNDARY' state with the rest being
   * 'INTERIOR'. If any of the input or output tensors are described by an 'INTERIOR' TensorConfig,
   * then the Plan is 'open' and should be merged with other 'open' Plans until the result becomes
   * 'closed'.
   */
  TensorConfigState GetState() const { return state_; }
  /*!
   * \return The mode in which the buffer should be realized
   * \note There are multiple buffering strategies by which a tensor may be realized (computed).
   * These affect the amount of recomputation necessary as well as the size of buffer required to
   * store the tensor. See 'BufferMode' for a description of the allowable buffering modes.
   */
  BufferMode GetBufferMode() const { return buffer_mode_; }
  /*!
   * \return Whether to copy the tensor.
   * \note While a tensor will originally reside in its home region, the TensorConfig may optionally
   * specify that the tensor should be copied (according to the StripeConfigs) into another
   * MemoryRegion. As an example for where this may be used, if a weights tensor initially resides
   * in slow Flash memory then necessarily the home region will be Flash. However, if the weights
   * values are used multiple times by a Part, it may be more performant to choose to copy the
   * weights into a faster memory like SRAM.
   */
  bool DoCopy() const { return copy_tensor_; }
  /*! \return The region to copy the tensor to */
  MemoryRegion GetCopyRegion() const {
    if (!copy_tensor_) {
      return home_region_;
    }
    return copy_region_;
  }
  /*!
   * \return The StripeConfigs with which to compute the tensor.
   * \note The StripeConfigs determine the order in which the elements of the tensor should be
   * computed, including potentially computing them multiple times (recompute). Multiple
   * StripeConfigs are used over just a single StripeConfig for the case where the tensor is
   * consumed by two different Parts executing themselves with different StripeConfigs. In this
   * case, there is a StripeConfig per consumer of the tensor.
   */
  const std::vector<StripeConfig> GetStripeConfigs() const { return stripe_configs_; }
  /*!
   * \return The size of the buffer needed for the TensorConfig.
   * \note The size of buffer necessary to store a tensor being produced using the TensorConfig is
   * not necessarily just the size of the tensor. In Plans, a tensor may be being produced and
   * consumed in 'stripes' which are smaller than the full tensor. Therefore, the buffer necessary
   * to store the tensor may only need to be as large as the stripe. The precise size of the buffer
   * will depend both on the BufferMode and StripeConfigs (as well as, of course, the Tensor).
   */
  int GetBufferSize() const;
  /*! \return The hash of the TensorConfigNode */
  size_t GetHash() const { return hash_; }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.TensorConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorConfigNode, Object);

 protected:
  friend class TensorConfig;

  /*! \brief Compute the hash of the TensorConfigNode */
  void ComputeHash_();

  /*! \return The size of the recompute buffer needed*/
  int GetRecomputeBufferSize_() const;
  /*! \return The size of the rolling buffer needed*/
  int GetRollingBufferSize_() const;

  /*! \brief The Tensor the config applies to */
  Tensor tensor_;
  /*! \brief The region where the tensor is allocated */
  MemoryRegion home_region_;
  /*! \return The state of the TensorConfig */
  TensorConfigState state_;
  /*! \brief The mode in which the buffer should be realized */
  BufferMode buffer_mode_;
  /*! \return The StripeConfigs with which to compute the tensor */
  std::vector<StripeConfig> stripe_configs_;
  /*! \brief Whether to copy the tensor */
  bool copy_tensor_;
  /*! \brief The region to copy the tensor to */
  MemoryRegion copy_region_;
  /*! \brief The hash of the TensorConfigNode */
  size_t hash_{0};
};

/*!
 * \brief A class which describes how to realize a Tensor.
 * \note The TensorConfig describes both how a Tensor is scheduled (the order in which it's
 * produced/consumed) and how its allocated in memory (which region it should reside in and whether
 * it should be copied). For further detail on how TensorConfig stores this information, consult the
 * documentation of TensorConfigNode.
 */
class TensorConfig : public ObjectRef {
 public:
  TensorConfig(const Tensor& tensor, const MemoryRegion& home_region, TensorConfigState state,
               BufferMode buffer_mode, const std::vector<StripeConfig>& stripe_configs,
               bool copy_tensor, const MemoryRegion& copy_region);
  /*!
   * \brief Check if two TensorConfigs are equal to each other.
   * \param other TensorConfig to be checked.
   * \return Whether the two TensorConfigs equal each other.
   */
  bool operator==(const TensorConfig& other) const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorConfig, ObjectRef, TensorConfigNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

// Hash and equal function for TensorConfig
namespace std {

/*! \brief The equal_to function for tvm::contrib::ethosu::cascader::TensorConfig */
template <>
struct equal_to<::tvm::contrib::ethosu::cascader::TensorConfig> {
  bool operator()(const ::tvm::contrib::ethosu::cascader::TensorConfig& lhs,
                  const ::tvm::contrib::ethosu::cascader::TensorConfig& rhs) const {
    return lhs == rhs;
  }
};

/*! \brief The hash function for tvm::contrib::ethosu::cascader::TensorConfig */
template <>
struct hash<::tvm::contrib::ethosu::cascader::TensorConfig> {
  std::size_t operator()(
      const ::tvm::contrib::ethosu::cascader::TensorConfig& tensor_config) const {
    return tensor_config->GetHash();
  }
};

}  // namespace std

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_TENSOR_CONFIG_H_
