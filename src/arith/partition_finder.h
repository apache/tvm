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
 * \file partition_finder.h
 */
#ifndef TVM_ARITH_PARTITION_FINDER_H_
#define TVM_ARITH_PARTITION_FINDER_H_

#include <tvm/arith/int_set.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include <utility>

#include "./interval_set.h"

namespace tvm {
namespace arith {

class PartitionDecisionNode : public Object {
 public:
  IntervalSet interval;
  Map<PrimExpr, Bool> cond_map;

  static constexpr const char* _type_key = "PartitionDecision";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(PartitionDecisionNode, Object);
};

/*! \brief Managed reference to PartitionCandidateNode. */
class PartitionDecision : public ObjectRef {
 public:
  // constructor
  PartitionDecision(IntervalSet interval, Map<PrimExpr, Bool> cond_map);
  TVM_DEFINE_OBJECT_REF_METHODS(PartitionDecision, ObjectRef, PartitionDecisionNode);
};

Optional<PartitionDecision> SearchBestPartition(
    const Var& var, arith::IntervalSet loop_range,
    const std::unordered_map<const VarNode*, IntSet>& hint_map,
    const std::unordered_map<const VarNode*, IntSet>& relax_map, bool partition_likely_cond_only,
    bool deduce_min_max, arith::Analyzer* analyzer, ObjectRef stmt_or_expr);

ConstIntBound EstimatePartitionedConstIntBound(
    const PrimExpr& e, const std::vector<Var>& vars,
    const std::unordered_map<const VarNode*, arith::IntSet>& dom_map);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_PARTITION_FINDER_H_