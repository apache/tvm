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
 * \file src/contrib/ethosu/cascader/parts/ethosu.h
 * \brief Arm(R) Ethos(TM)-U NPU Part object
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PARTS_ETHOSU_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PARTS_ETHOSU_H_

#include <tvm/runtime/object.h>

#include <vector>

#include "../block_config.h"
#include "../graph.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*! \brief Node to represent an EthosuPart */
class EthosuPartNode : public PartNode {
 public:
  /*!
   * \brief Get the optimal BlockConfig to use given a StripeConfig
   * \param output_stripe_config The output StripeConfig.
   */
  const BlockConfig GetBlockConfig(const StripeConfig& output_stripe_config);
  /*!
   * \brief Get the preferred alignment in each axis for a stripe of the Part.
   * \note This is used to bias the selection of StripeConfigs towards those that are integer
   * multiples of a tensor intrinsic used to compute the Part.
   */
  const std::vector<int> GetStripeAlignHint() const final { return output_quantum_; }
  /*!
   * \brief Get the performance information for a given output stripe config.
   * \param output_stripe_config The output stripe config to compute the performance for.
   * \param buffer_mode The mode of buffering, rolling or recompute.
   * \return The performance information containing the compute cycles and read/write bytes.
   */
  const PerformanceInfo GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                           BufferMode buffer_mode) final;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.EthosuPart";
  TVM_DECLARE_FINAL_OBJECT_INFO(EthosuPartNode, PartNode);

 protected:
  friend class EthosuPart;

  /*!
   * \brief Get the size of input required (per input tensor) to compute a stripe given a
   * block_shape
   * \param block_shape The shape of the block(s) the stripe is split into
   * \param stripe_shape The shape of the full stripe to compute.
   * \return The bytes required per input tensor.
   */
  const std::vector<int64_t> GetBytesRead(const std::vector<int>& block_shape,
                                          const std::vector<int>& full_shape);

  /*!
   * \brief Get cost heuristic of using a given block config with the associated stripe config
   * \param block_config The block config that is being checked for the cost
   * \param output_stripe_config The striping configuration associated with the operator
   * \return A cost heuristic representative of the choice
   */
  float CalculateCost(const BlockConfig& block_config, const StripeConfig& output_stripe_config);

  /*! \brief List of block configs that are valid for this part */
  std::vector<BlockConfig> valid_block_configs_;
  /*! \brief The output volume that is atomically computed */
  std::vector<int> output_quantum_;
  /*! \brief Index for output height dimension */
  int height_idx_;
  /*! \brief Index for output width dimension */
  int width_idx_;
  /*! \brief Index of weight tensor, -1 if the Part has no weights */
  int weight_tensor_idx_;
  /*! \brief Number of sub-kernels the kernel has been split into */
  int subkernels_;
};

/*!
 * \brief A class to describe a Part to be executed on an Arm(R) Ethos(TM)-U NPU.
 * \note EthosuParts must be provided with an output quantum and the cycles taken to
 * compute an output quantum which depend on the operator the NPU is computing.
 */
class EthosuPart : public Part {
 public:
  EthosuPart(const TESubgraph& subgraph, const std::vector<Propagator> propagators,
             const std::vector<int>& output_quantum, int subkernels,
             const std::vector<BlockConfig>& valid_block_configs, int weight_tensor_idx);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EthosuPart, Part, EthosuPartNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PARTS_ETHOSU_H_
