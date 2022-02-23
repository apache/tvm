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
 * \file src/contrib/ethosu/cascader/parts/inline.h
 * \brief Inline Part object
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PARTS_INLINE_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PARTS_INLINE_H_

#include <tvm/runtime/object.h>

#include <vector>

#include "../graph.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*! \brief Node to represent an inlined Part */
class InlinePartNode : public PartNode {
 public:
  /*!
   * \brief Get the performance information for a given output stripe config.
   * \param output_stripe_config The output stripe config to compute the performance for.
   * \param is_rolling Whether the output config should be computed as a rolling buffer.
   * \return The performance information containing the compute cycles and read/write bytes.
   */
  const PerformanceInfo GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                           BufferMode buffer_mode) final;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.InlinePart";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlinePartNode, PartNode);

 protected:
  friend class InlinePart;
};

/*!
 * \brief A class to describe a inlined Part in a Cascader graph.
 * \note Inlined Parts have a few special properties. First by IsInline being true,
 * the Cascader will not allocate any space for the outputs of the Part. This is because
 * they will be directly consumed as they are produced by the following Part. Second, they
 * are assumed to be 'free' and require no cycles to execute. Lastly, as they are 'free'
 * the compute quantum is arbitrary, but by convention it is a single tensor element.
 *
 * Examples of inline Parts include strided_slice, reshape and concatenate - all of which
 * get absorbed into the DMA functionality of Ethos-U compute primitives.
 */
class InlinePart : public Part {
 public:
  InlinePart(const TESubgraph& subgraph, const std::vector<Propagator> propagators);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(InlinePart, Part, InlinePartNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PARTS_INLINE_H_
