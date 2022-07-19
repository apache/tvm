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
 * \file relay/collage/collage_partitioner.h
 * \brief Search for an optimal partitioning of a Relay model.
 *
 * See:
 *   Collage: Automated Integration of Deep Learning Backends
 *   Byungsoo Jeon, Sunghyun Park, Peiyuan Liao, Sheng Xu, Tianqi Chen, Zhihao Jia
 *   https://arxiv.org/pdf/2111.00655.pdf
 */
#ifndef TVM_RELAY_COLLAGE_COLLAGE_PARTITIONER_H_
#define TVM_RELAY_COLLAGE_COLLAGE_PARTITIONER_H_

#include <tvm/relay/transform.h>

#include "./cost_estimator.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Explores the space of all possible (sub-graph, target) pairs which cover the
 * model, and applies the globally optimal choice (assuming partition costs are additive).
 */
transform::Pass CollagePartition(CompilationConfig config, CostEstimator cost_estimator);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COLLAGE_PARTITIONER_H_
