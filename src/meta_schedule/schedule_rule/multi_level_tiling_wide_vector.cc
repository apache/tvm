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

#include "../../tir/schedule/analysis.h"
#include "../../tir/schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;
using tir::Schedule;

/*!
 * \brief Extension of MultiLevelTiling for backends with wide vectors.
 * The loop over the innermost spatial axis of the output buffer is always vectorized with the
 * maximum vector length.
 */
class MultiLevelTilingWideVectorNode : public MultiLevelTilingNode {
 public:
  size_t vector_length_in_bits;

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingWideVector";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingWideVectorNode, MultiLevelTilingNode);

 protected:
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingWideVectorNode> n =
        make_object<MultiLevelTilingWideVectorNode>(*this);
    return ScheduleRule(n);
  }

  std::pair<Array<tir::ExprRV>, Array<tir::LoopRV>> SplitLoop(const Schedule& sch, BlockRV block,
                                                              LoopRV loop, int n_tiles) const;
};

std::pair<Array<tir::ExprRV>, Array<tir::LoopRV>> MultiLevelTilingWideVectorNode::SplitLoop(
    const Schedule& sch, BlockRV block_rv, LoopRV loop_rv, int n_tiles) const {
  const tir::ForNode* loop = TVM_SREF_TO_FOR(sch->GetSRef(loop_rv));
  const tir::StmtSRef block_sref = sch->GetSRef(block_rv);
  const tir::BlockNode* block_node = block_sref->StmtAs<tir::BlockNode>();
  const tir::BlockRealize block_realize = tir::GetBlockRealize(sch->state(), block_sref);
  ICHECK(block_node && block_node->writes.size() == 1);

  const auto out_dtype = block_node->writes[0]->buffer->dtype;
  const int vec_len = vector_length_in_bits / out_dtype.bits();

  // Determine if this loop is over the innermost axis of the output buffer.
  // In the example below, we look for a loop whose loop var is bound to the axis co.

  // for (i0, 0, 1) {
  //    for (i1, 0, 56) {
  //      for (i2, 0, 56) {
  //        for (i3, 0, 64) {
  //          for (i4, 0, 3) {
  //            for (i5, 0, 3) {
  //              for (i6, 0, 64) {
  //                block conv2d_nhwc(...) {
  //                  ...
  //                  bind(co, i3)
  //                  ...
  //                  writes([conv2d_nhwc[n, h, w, co]])
  //                  ...
  //                  conv2d_nhwc[n, h, w, co] = ...
  // }
  const size_t innermost_axis = block_node->writes[0]->region.size() - 1;
  const PrimExpr innermost_iter_value = block_realize->iter_values[innermost_axis];

  if (!arith::Analyzer().CanProve(loop->loop_var == innermost_iter_value)) {
    // If this is not the innermost spatial loop, split the loop in the normal way.
    return MultiLevelTilingNode::SplitLoop(sch, block_rv, loop_rv, n_tiles);
  } else {
    // We split the innermost spatial loop in a way that always uses the maximum vector length.
    const int64_t* extent_int = tir::GetLoopIntExtent(loop);
    if (extent_int && *extent_int > vec_len) {
      Array<tir::LoopRV> inner_splits = sch->Split(/*loop=*/loop_rv,
                                                   /*factors=*/{NullOpt, PrimExpr(vec_len)});
      Array<tir::ExprRV> outer_factors = sch->SamplePerfectTile(
          /*loop=*/inner_splits[0],
          /*n=*/n_tiles - 1,
          /*max_innermost_factor=*/max_innermost_factor);
      Array<tir::LoopRV> outer_splits = sch->Split(
          /*loop=*/inner_splits[0], /*factors=*/{outer_factors.begin(), outer_factors.end()});
      outer_splits.push_back(inner_splits[1]);
      outer_factors.push_back(PrimExpr(vec_len));
      return {outer_factors, outer_splits};
    } else {
      Array<tir::ExprRV> factors(n_tiles - 1, PrimExpr(1));
      factors.push_back(loop->extent);
      Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop_rv,
                                             /*factors=*/{factors.begin(), factors.end()});
      return {factors, splits};
    }
  }
}

ScheduleRule ScheduleRule::MultiLevelTilingWideVector(
    String structure, Integer vector_length_in_bits, Optional<Integer> max_innermost_factor,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWideVectorNode>(
      structure, NullOpt, max_innermost_factor, NullOpt, reuse_read, reuse_write);
  node->vector_length_in_bits = vector_length_in_bits->value;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingWideVectorNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingWideVector")
    .set_body_typed(ScheduleRule::MultiLevelTilingWideVector);

}  // namespace meta_schedule
}  // namespace tvm
