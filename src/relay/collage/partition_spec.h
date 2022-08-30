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
 * \file src/relay/collage/partition_spec.h
 * \brief Combine a \p PartitionRule with a \p Target.
 */

#ifndef TVM_RELAY_COLLAGE_PARTITION_SPEC_H_
#define TVM_RELAY_COLLAGE_PARTITION_SPEC_H_

#include <tvm/relay/function.h>
#include <tvm/runtime/container/string.h>
#include <tvm/target/target.h>

#include <string>
#include <vector>

#include "./partition_rule.h"
#include "./sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Type of functions for checking the validity of partitions before they proceed to lowering
 * and codegen. The argument is the function extracted from the overall expression to represent
 * the partition. The result is a non-empty error message string if the candidate should be
 * rejected.
 */
using TValidateSubGraphFunc = TypedPackedFunc<String(const Function& function)>;

/*!
 * \brief The default validation function. Always returns the empty string, ie no error.
 */
String DefaultValidateSubGraphFunc(const Function& function);

/*!
 * \brief Pairs a \p PartitionRule with one or more \p Targets it can be used for.
 */
class PartitionSpecNode : public Object {
 public:
  /*!
   * \brief Specification name to distinguish this spec from all others. Typically the BYOC
   * 'compiler' name, "tvm", or "host".
   */
  String spec_name_;

  /*!
   * \brief The target all candidate partitions should be compiled for.
   *
   * It's tempting to support multiple targets here since. Eg the partitioning rules for
   * TVM are the same irrespective of whether the target is "cuda" or "llvm", so it would make
   * sense to build the candidate partitions first without committing to any target, then 'stamp'
   * them for each target as the final step.
   *
   * However, we want to make sure any predicate in \p DFPatternPartitionRuleNode instances
   * can have access to the current target instance. Eg the predicate may need to consult
   * build-time configuration to decide what operators, shapes etc are actually supported.
   * That implies the specific target is known when the candidate partitions are being constructed.
   *
   * So for now we'll just force each spec to have exactly one target.
   */
  Target target_;

  /*!
   * \brief The partition rule to use to gather candidates.
   */
  PartitionRule rule_;

  /*!
   * \brief The validation function to apply to each candidate's the extracted function before
   * proceeding to lowering/codegen.
   */
  TValidateSubGraphFunc validate_sub_graph_func_ = DefaultValidateSubGraphFunc;

  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Returns all the candidate partitions found by this specification. The candidates
   * will be for a specific target, but will not yet have an extracted function or cost.
   */
  std::vector<CandidatePartition> AllCandidates(const DataflowGraph& dataflow_graph) const;

  std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.PartitionSpec";
  TVM_DECLARE_FINAL_OBJECT_INFO(PartitionSpecNode, Object);
};

class PartitionSpec : public ObjectRef {
 public:
  PartitionSpec(String spec_name, Target target, PartitionRule rule,
                TValidateSubGraphFunc validate_sub_graph_func = DefaultValidateSubGraphFunc);

  TVM_DEFINE_OBJECT_REF_METHODS(PartitionSpec, ObjectRef, PartitionSpecNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_PARTITION_SPEC_H_
