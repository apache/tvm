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
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_WITH_INTRIN_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_WITH_INTRIN_H_

#include <string>
#include <vector>

#include "../../tir/schedule/analysis.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic, and annotate
 * the tiled block for tensorization by postproc rewrite.
 */
Optional<tir::BlockRV> TileForIntrin(tir::Schedule sch, tir::BlockRV block,
                                     const std::string& intrin_name);

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingWithIntrinNode : public MultiLevelTilingNode {
 protected:
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) override;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const override {
    ObjectPtr<MultiLevelTilingWithIntrinNode> n =
        make_object<MultiLevelTilingWithIntrinNode>(*this);
    return ScheduleRule(n);
  }

  // Override ApplySubRules to tile the inner loops according to the given tensor intrinsic, then
  // tile the outerloops.
  std::vector<State> ApplySubRules(std::vector<State> states) override;

 public:
  /*! \brief The name of a tensor intrinsic. */
  String intrin_name;

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingWithIntrin";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingWithIntrinNode, MultiLevelTilingNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_WITH_INTRIN_H_
