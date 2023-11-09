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
 * \file combine_parallel_op_batch.h
 * \brief Combine parallel ops into a single batch op.
 */
#ifndef TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_BATCH_H_
#define TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_BATCH_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "./combine_parallel_op.h"
#include "./expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*
 * Class to find and combine parallel ops and following element-wise
 * and broadcast ops into a single batch op. Ops can be combined
 * if they have the same input data. Batch op is formed by
 * stacking inputs. Final results are retrieved by splitting output.
 * For example:
 *
 *               data
 *         /              \
 *    dense (2,2)         dense (2,2)
 *        |                 |
 *   elemwise/bcast (2,2)  elemwise/bcast (2,2)
 *
 *   Would become:
 *
 *            data
 *             |
 *     batch_matmul+elemwise/bcast (2,2,2)
 */
class ParallelOpBatchCombiner : public ParallelOpCombiner {
 public:
  /*
   * \brief Constructor.
   * \param op_name name of op to combine
   * \param batch_op_name name of op that combined branches will be joined into
   * \param min_num_branches min number of parallel branches beginning with op
   *                         to start combining
   */
  ParallelOpBatchCombiner(const std::string& op_name, const std::string& batch_op_name,
                          uint64_t min_num_branches);

 protected:
  /*
   * \brief Checks if node is supported to be combined
   * \param n node in question
   * \return True by default
   */
  virtual bool IsSupportedOp(const CallNode* n);

  /*
   * \brief Checks if two ops can be combined
   * \param a node a
   * \param b node b
   * \return True if shapes and dtypes of all args of a and b are the same
   */
  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b);

  /*
   * \brief Makes combined op from parallel ops in branches. This usually involves
   *        concatenating or stacking inputs, then creating a new call.
   * \param branches branches that are to be combined
   * \return new call with branches combined as batch op by stacking args
   */
  virtual Call MakeCombinedOp(const Group& branches);

  /*
   * \brief Checks if argument of op following combined ops are able to be combined
   * \param a node a
   * \param b node b
   * \param index index of argument in question
   * \return True if shapes and dtypes of args[index] a and b are the same
   */
  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) final;

  /*
   * \brief Create combined call from ops that follow the initial combined op at the depth-th level.
   *        This usually involves concatenating or stacking inputs, then creating a new call.
   *        Only called if IsArgCompatbile returns true for each arg.
   * \param data combined op
   * \param branches branches of parallel ops to be combined
   * \param depth depth at which to combine ops
   * \param parent_index index of arg that corresponds to original input that was shared among
   *                     all combined ops
   * \return new combined call as batch op by stacking args
   */
  Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches, size_t depth,
                                        size_t parent_index) final;

  /*
   * \brief Updates map of expr to substitute with combined expr. This usually involves
   *        slicing or splitting data.
   * \param data combined op
   * \param branches branches of parallel ops to be combined
   * \param depth depth at which to substitute
   * \param subst_map map of Expr to replace with Expr to replace it with
   */
  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         ExprSubstMap* subst_map) final;

 private:
  /* \brief name of op to replace combined ops with. for example,
   *         for combining parallel dense, this will be set to
   *         nn.batch_matmul
   */
  std::string batch_op_name_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_COMBINE_PARALLEL_OP_BATCH_H_
