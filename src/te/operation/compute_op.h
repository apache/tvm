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
 * \brief Helper utilities to implement compute_op.
 * \file compute_op.h
 */
#ifndef TVM_TE_OPERATION_COMPUTE_OP_H_
#define TVM_TE_OPERATION_COMPUTE_OP_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace te {
// loop nest structure for general compute
// This the loop nest structured used in compute.
// Does not include the loop body.
struct ComputeLoopNest {
  // The common number of loops between init and main
  size_t num_common_loop;
  // predicates for the initialize loop
  std::vector<PrimExpr> init_predicates;
  // Initialization nest involved.
  std::vector<std::vector<Stmt>> init_nest;
  // Value map for the init code
  std::unordered_map<IterVar, PrimExpr> init_vmap;
  // Predicates for the main update loop
  std::vector<PrimExpr> main_predicates;
  // The general loop nest
  std::vector<std::vector<Stmt>> main_nest;
  // Value map for the IterVar.
  std::unordered_map<IterVar, PrimExpr> main_vmap;

  /*!
   * \brief constructor to build ComputeOpNest
   * \param self The pointer to compute op.
   * \param stage The scxhedule stage.
   * \param dom_map The domain map.
   * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
   * \return The constructed loop nest
   */
  static ComputeLoopNest Create(const BaseComputeOpNode* self, const Stage& stage,
                                const std::unordered_map<IterVar, Range>& dom_map,
                                bool debug_keep_trivial_loop);
};

/*!
 * \brief Build body of compute for cross thread reduction pattern.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeCrossThreadReduction(const ComputeOpNode* self, const Stage& stage,
                              const std::unordered_map<IterVar, Range>& dom_map,
                              bool debug_keep_trivial_loop);

/*!
 * \brief Build body of compute for tensorization.
 * \param self The pointer to ComputeOpNode
 * \param stage The schedule stage.
 * \param dom_map The domain map.
 * \param debug_keep_trivial_loop Whether keep trivial loops with extent of 1
 * \return The created statement.
 */
Stmt MakeTensorize(const ComputeOpNode* self, const Stage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map, bool debug_keep_trivial_loop);

/*!
 * \brief Transform the update part when there is no init func in tensorizing
 * \param stage The stage for tensorizing.
 * \param dom_map The range of each iter var.
 * \param n The loop nest structured used in compute.
 * \param body The body func in tensorize intrin
 * \param update The update func in tensorize intrin
 * \return Transformed result.
 */
Stmt TransformUpdate(const Stage& stage, const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n, Stmt body, Stmt update);
}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_OPERATION_COMPUTE_OP_H_
