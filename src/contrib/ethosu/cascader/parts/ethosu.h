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

#include "../graph.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*! \brief Node to represent an EthosuPart */
class EthosuPartNode : public PartNode {
 public:
  /*!
   * \brief Get the optimal block shape to use.
   * \param output_stripe_config The output StripeConfig.
   * \param is_rolling Whether the output config should be computed as a rolling buffer.
   */
  const std::vector<int> GetBlockShape(const StripeConfig& output_stripe_config, bool is_rolling);
  /*!
   * \brief Get the preferred alignment in each axis for a stripe of the Part.
   * \note This is used to bias the selection of StripeConfigs towards those that are integer
   * multiples of a tensor intrinsic used to compute the Part.
   */
  const std::vector<int> GetStripeAlignHint() const final { return output_quantum_; }
  /*!
   * \brief Get the performance information for a given output stripe config.
   * \param output_stripe_config The output stripe config to compute the performance for.
   * \param is_rolling Whether the output config should be computed as a rolling buffer.
   * \return The performance information containing the compute cycles and read/write bytes.
   */
  const PerformanceInfo GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                           bool is_rolling) final;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.EthosuPart";
  TVM_DECLARE_FINAL_OBJECT_INFO(EthosuPartNode, PartNode);

 protected:
  friend class EthosuPart;

  /*!
   * \brief Get the size of input required (per input tensor) to compute a block.
   * \param block_shape The shape of the block to compute.
   * \return The bytes required per input tensor.
   */
  const std::vector<int> GetBlockInputBytes_(const std::vector<int>& block_shape);

  /*! \brief The output volume that is atomically computed */
  std::vector<int> output_quantum_;
  /*! \brief The cycles taken to compute a single output quantum */
  int quantum_cycles_;
};

/*!
 * \brief A class to describe a Part to be executed on an Arm(R) Ethos(TM)-U NPU.
 * \note EthosuParts must be provided with an output quantum and the cycles taken to
 * compute an output quantum which depend on the operator the NPU is computing.
 */
class EthosuPart : public Part {
 public:
  EthosuPart(const TESubgraph& subgraph, const std::vector<Propagator> propagators,
             const std::vector<int> output_quantum, int quantum_cycles);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EthosuPart, Part, EthosuPartNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PARTS_ETHOSU_H_
