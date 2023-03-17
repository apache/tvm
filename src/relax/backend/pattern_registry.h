
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
 * \file relax/backend/contrib/pattern_registry.h
 * \brief Functions related to registering and retrieving patterns for
 * functions handled by backends.
 */
#ifndef TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_
#define TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {
namespace backend {

using transform::FusionPattern;

/*!
 * \brief Register patterns which will be used to partition the DataflowBlock
 *        into subgraphs that are supported by external backends.
 * \param patterns Patterns to be registered. Patterns that appear later in the list have
 *        higher priority when partitioning DataflowBlock.
 */
void RegisterPatterns(Array<FusionPattern> patterns);

/*!
 * \brief Remove patterns from the registry by their name.
 * \param names The name of patterns to be removed
 */
void RemovePatterns(Array<String> names);

/*!
 * \brief Find patterns whose name starts with a particular prefix.
 * \param prefx The pattern name prefix.
 * \return Matched patterns, ordered by priority from high to low.
 */
Array<FusionPattern> GetPatternsWithPrefix(const String& prefix);

/*!
 * \brief Find the pattern with a particular name.
 * \param name The pattern name.
 * \return The matched pattern. NullOpt if not found.
 */
Optional<FusionPattern> GetPattern(const String& name);

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_
