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
 *
 * \file combine_parallel_op.h
 * \brief Abstract class to combine parallel ops and their successive element-wise ops.
 */
#ifndef TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_H_
#define TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

using Branch = std::vector<const CallNode*>;
using Group = std::vector<Branch>;
using FIsSupportedOp = std::function<bool(const CallNode* n)>;
using FAreCompatibleOps = std::function<bool(const CallNode* a, const CallNode* b)>;
using ExprSubstMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;

/*
 * Class to find parallel branches starting with op that are
 * grouped if they are able to be combined. They are eligible to
 * be combined if they have the same input data.
 * Op can be followed by zero or more elemwise or broadcast ops,
 * which are included in the group.
 * Intermediate nodes have exactly one successor. It is possible that branches meet at a point,
 * which should be handled in ParallelOpCombiner.
 *
 *        data
 *       /    \
 *     op      op
 *     |        |
 * elem-wise elem-wise
 *     |        |
 */
class BranchGroupFinder : private ExprVisitor {
 public:
  /*
   * \brief Constructor
   * \param op The op that indicates the start of each group
   * \param fis_supported_op function that returns true if op
   *                         is supported for combining
   * \param fare_compatible_ops function that returns true if
   *                            two ops are compatible for combining
   */
  BranchGroupFinder(const Op& op, FIsSupportedOp fis_supported_op,
                    FAreCompatibleOps fare_compatible_ops);

  /*
   * \brief Finds all groups that can be combined.
   * \param expr Relay expression that represents function
   *             to look at for groups to be combined
   * \return Vector of groups which can be combined.
   */
  std::vector<Group> Find(const Expr& expr);

 private:
  /* \brief Cache the op for finding parallel branches */
  const Op& cached_op_;

  /* \brief function to return true if op is eligible to be combined,
   *         false otherwise
   */
  FIsSupportedOp fis_supported_op_;

  /* \brief function to return true if two parallel ops are eligible
   *         to be combined, false otherwise
   */
  FAreCompatibleOps fare_compatible_ops_;

  /* \brief ops that are on the first (logically, leftmost) branch
   *         of parallel ops and are eligible to be combined
   */
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> op_roots_;

  /* \brief map of Expr to CallNodes that follow it  */
  std::unordered_map<Expr, std::vector<const CallNode*>, ObjectPtrHash, ObjectPtrEqual>
      children_map_;

  /*
   * \brief Creates new branch from op and its children that have
   *        elementwise or broadcast patterns
   * \return New branch
   */
  Branch CreateBranch(const CallNode* op);

  /*
   * \brief Expression visitor function
   */
  void VisitExpr_(const CallNode* n) final;
};

/*
 * Abstract class to find and combine parallel ops and the elementwise ops that follow.
 */
class ParallelOpCombiner {
 public:
  /*! \brief virtual destructor */
  virtual ~ParallelOpCombiner() {}
  /*
   * \brief Constructor.
   * \param op_name name of op to combine
   * \param min_num_branches min number of parallel branches beginning with op
   *                         to start combining
   */
  explicit ParallelOpCombiner(const std::string& op_name, uint64_t min_num_branches);

  /*
   * \brief Combines ops and following elementwise or broadcast ops
   * \param expr function to modify
   * \return new function with combined ops
   */
  Expr Combine(const Expr& expr);

 protected:
  /*
   * \brief Checks if node is supported to be combined
   * \param n node in question
   * \return True if the op represented by n is supported to be the root of a branch
   *         to be combined. False otherwise.
   */
  virtual bool IsSupportedOp(const CallNode* n) = 0;

  /*
   * \brief Checks if two ops can be combined
   * \param a node a
   * \param b node b
   * \return True if a and b can be combined. False otherwise.
   */
  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b) = 0;

  /*
   * \brief Makes combined op from parallel ops in branches. This usually involves
   *        concatenating or stacking inputs, then creating a new call.
   * \param branches branches that are to be combined
   * \return new call with branches combined.
   */
  virtual Call MakeCombinedOp(const Group& branches) = 0;

  /*
   * \brief Checks if argument of op following combined ops are able to be combined
   * \param a node a
   * \param b node b
   * \param index index of argument in question
   * \return True if argument of a and b and index can be combined
   */
  virtual bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) = 0;

  /*
   * \brief Create combined call from ops that follow the initial combined op at the depth-th level.
   *        This usually involves concatenating or stacking inputs, then creating a new call.
   *        Only called if IsArgCompatbile returns true for each arg.
   * \param data combined op
   * \param branches branches of parallel ops to be combined
   * \param depth depth at which to combine ops
   * \param parent_index index of arg that corresponds to original input that was shared among
   *                     all combined ops
   * \return new combined call
   */
  virtual Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches,
                                                size_t depth, size_t parent_index) = 0;

  /*
   * \brief Updates map of expr to substitute with combined expr. This usually involves
   *        slicing or splitting data.
   * \param data combined op
   * \param branches branches of parallel ops to be combined
   * \param depth depth at which to substitute
   * \param subst_map map of Expr to replace with Expr to replace it with
   */
  virtual void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                                 ExprSubstMap* subst_map) = 0;

 private:
  /* \brief Cache the op to be combined */
  const Op& cached_op_;

  /* \brief minimum number of parallel branches to combine */
  uint64_t min_num_branches_;

  /* \brief map of Expr to Expr to substitute it with after running pass */
  ExprSubstMap subst_map_;

  /*
   * \brief Combine parallel branches and updates subst_map_ with Exprs
   *        to be substituted
   * \param branches branches to be combined
   */
  void CombineBranches(const Group& branches);

  /*
   * \brief Combine parallel branches and updates subst_map_ with Exprs
   *        to be substituted
   * \param branches parallel branches to potentially be combined
   * \param depth depth at which to look at op
   * \param parent_index index of arg that corresponds to original input that was shared among
   *                     all combined ops
   * \return true if parallel ops at depth can be combined, false otherwise
   */
  bool CheckLevel(const Group& branches, size_t depth, size_t parent_index);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_H_
