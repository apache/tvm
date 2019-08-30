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
 *
 * \file combine_parallel_op.h
 * \brief Abstract class to combine parallel ops and their successive element-wise ops.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <unordered_map>
#include <unordered_set>
#include "./expr_subst.h"
#include "./pattern_util.h"


namespace tvm {
namespace relay {

using Branch = std::vector<const CallNode*>;
using Group = std::vector<Branch>;
using FIsSupportedOp = std::function<bool (const CallNode* n)>;
using FAreCompatibleOps = std::function<bool (const CallNode* a, const CallNode* b)>;
using ExprSubstMap = std::unordered_map<Expr, Expr, NodeHash, NodeEqual>;

/*
  Class to find parallel branches starting with op as shown below and then 
  group branches by kernel shape and attributes of op. 
  Op can be followed by zero or more elemwise or broadcast ops.
  Intermediate nodes have exactly one successor. It is possible that branches meet at a point,
  which should be handled in ParallelOpCombiner.

         data
        /    \
      op      op
      |        |
  elem-wise elem-wise
      |        |
*/
class BranchGroupFinder : private ExprVisitor {
 public:
  BranchGroupFinder(const std::string& op_name,
                    FIsSupportedOp fis_supported_op,
                    FAreCompatibleOps fare_compatible_ops);

  std::vector<Group> Find(const Expr& expr);

 private:
  std::string op_name_;
  FIsSupportedOp fis_supported_op_;
  FAreCompatibleOps fare_compatible_ops_;
  std::unordered_set<Expr, NodeHash, NodeEqual> op_roots_;
  std::unordered_map<Expr, std::vector<const CallNode*>, NodeHash, NodeEqual> children_map_;

  // Create a branch starting from op.
  Branch CreateBranch(const CallNode* op);

  void VisitExpr_(const CallNode* n) final;
};

/*
  Abstract class to find and combine parallel ops and the element-wise ops that follow.
*/
class ParallelOpCombiner {
 public:
  explicit ParallelOpCombiner(const std::string& op_name, uint64_t min_num_branches);

  Expr Combine(const Expr& expr);
 
 protected:
  // Returns true if the op represented by CallNode n is supported to be the
  // root of a branch to be combined. Otherwise, returns false.
  virtual bool IsSupportedOp(const CallNode* n) = 0;

  // Returns true if ops represented by CallNodes a and b can be combined.
  // Otherwise, returns false.
  virtual bool AreCompatibleOps(const CallNode* a, const CallNode* b) = 0;

  // Create Call that consists of the combined ops. This usually involves concatenating
  // or stacking inputs, then creating a new call.
  virtual Call MakeCombinedOp(const Group& branches) = 0;

  // Returns true if arguments of a and b at index index can be combined.
  virtual bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) = 0;

  // Create combined call of other ops in depth-th level. This usually involves concatenating
  // or stacking inputs, then creating a new call.
  virtual Call MakeCombinedCall(const Expr& data, const Group& branches, size_t depth, size_t parent_index) = 0;

  // Replace output of each branch with slices of the combined output.
  virtual void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth, ExprSubstMap& subst_map) = 0;

 private:
  std::string op_name_;
  uint64_t min_num_branches_;
  ExprSubstMap subst_map_;

  void CombineBranches(const Group& branches);

  bool CheckLevel(const Group& branches, size_t depth, size_t parent_index);
};

}  // namespace relay
}  // namespace tvm
