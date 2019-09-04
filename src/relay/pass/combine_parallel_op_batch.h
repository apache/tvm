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
 * Copyright (c) 2019 by Contributors
 * \file combine_parallel_op_batch.cc
 * \brief Combine parallel ops into a single batch op.
 */
#ifndef TVM_RELAY_PASS_COMBINE_PARALLEL_OP_BATCH_H_
#define TVM_RELAY_PASS_COMBINE_PARALLEL_OP_BATCH_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "./expr_subst.h"
#include "./pattern_util.h"
#include "./combine_parallel_op.h"

namespace tvm {
namespace relay {

class ParallelOpBatchCombiner : public ParallelOpCombiner {
 public:
  ParallelOpBatchCombiner(const std::string& op_name,
                          const std::string& batch_op_name,
                          uint64_t min_num_branches);

 protected:
  virtual bool IsSupportedOp(const CallNode* n);

  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b);

  Call MakeCombinedOp(const Group& branches) final;

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) final;

  Call MakeCombinedCallFromFollowingOps(const Expr& data,
                                        const Group& branches,
                                        size_t depth,
                                        size_t parent_index) final;

  void UpdateGroupOutput(const Expr& data,
                         const Group& branches,
                         size_t depth,
                         ExprSubstMap* subst_map) final;

 private:
  std::string batch_op_name_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_COMBINE_PARALLEL_OP_BATCH_H_
