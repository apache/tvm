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
 * \file src/relay/collage/gather_partition_specs.h
 * \brief Gather the relevant \p PartitionSpecs from the available \p Targets.
 */
#ifndef TVM_RELAY_COLLAGE_GATHER_PARTITION_SPECS_H_
#define TVM_RELAY_COLLAGE_GATHER_PARTITION_SPECS_H_

#include <tvm/target/compilation_config.h>

#include "./partition_spec.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief The 'styles' of BYOC integrations. Used to influence how their corresponding
 * partition rule is constructed.
 */
enum BYOCStyle {
  /*!
   * \brief The BYOC patterns pick out 'ideal' candidates directly, either because:
   *  - the BYOC toolchain does not perform any fusion so each matched sub-expression maps 1:1 to a
   *    BYOC-provided operator, or
   *  - the BYOC toolchain does perform fusion, however the patterns have been written to pick out
   *    fusable sub-graphs.
   */
  kNoFusionBYOCStyle,

  /*!
   * \brief The BYOC patterns pick out supported operators, but the BYOC backend may perform
   * fusion over those operators in much the same way TVM does.
   */
  kTVMFusionBYOCStyle,

  /*!
   * \brief The BYOC patterns pick out supported operators, but the BYOC backend may perform
   * arbitrary fusion over those operators.
   */
  kArbitraryFusionBYOCStyle,
};

/*!
 * \brief Returns all the partition specifications gathered from the \p Targets in \p config.
 */
Array<PartitionSpec> GatherPartitionSpecs(const CompilationConfig& config);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_GATHER_PARTITION_SPECS_H_
