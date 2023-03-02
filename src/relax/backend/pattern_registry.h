
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
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {
namespace backend {

/*!
 * \brief An entry in the pattern registry. This represents a single pattern that
 * can be used to identify expressions that can be handled by external
 * backends, like CUTLASS and TensorRT.
 */
class PatternRegistryEntryNode : public Object {
 public:
  /*!
   * \brief The name of pattern. Usually it starts with the name of backend, like
   * 'cutlass.matmul'.
   */
  String name;
  /*!
   * \brief The dataflow pattern that will be used to match expressions that can
   * be handled by external backends.
   */
  DFPattern pattern;
  /*!
   * \brief The mapping from arg name to its pattern. It can be used to extract
   * arg expression from match result. All DFPattern in this map should be part of
   * the `pattern`.
   */
  Map<String, DFPattern> arg_patterns;

  /*!
   * \brief The function to check whether the match result is accepted.
   *
   * It should have signature
   * bool(const Map<DFPattern, Expr>& match_result, const Expr& matched_expr)
   */
  PackedFunc check;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("pattern", &pattern);
    v->Visit("arg_patterns", &arg_patterns);
    v->Visit("check", &check);
  }

  static constexpr const char* _type_key = "relax.backend.PatternRegistryEntry";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternRegistryEntryNode, Object);
};

class PatternRegistryEntry : public ObjectRef {
 public:
  PatternRegistryEntry(String name, DFPattern pattern, Map<String, DFPattern> arg_patterns,
                       PackedFunc check);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PatternRegistryEntry, ObjectRef,
                                            PatternRegistryEntryNode);
};

/*!
 * \brief Register patterns which will be used to partition the DataflowBlock
 *        into subgraphs that are supported by external backends.
 * \param patterns Patterns to be registered. Patterns that appear later in the list have
 *        higher priority when partitioning DataflowBlock.
 */
void RegisterPatterns(Array<PatternRegistryEntry> entries);

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
Array<PatternRegistryEntry> GetPatternsWithPrefix(const String& prefix);

/*!
 * \brief Find the pattern with a particular name.
 * \param name The pattern name.
 * \return The matched pattern. NullOpt if not found.
 */
Optional<PatternRegistryEntry> GetPattern(const String& name);

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_
