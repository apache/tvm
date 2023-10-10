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
#include "../utils.h"
#include "tvm/tir/op.h"
#include "tvm/tir/schedule/instruction.h"

namespace tvm {
namespace meta_schedule {

/*! \brief Filters splitting loops according to filter conditions */
class FilterLoopSplitsNode : public PostprocNode {
 public:
  using FFilter = Postproc::FFilter;
  /*! \brief The packed function to the `filter` function.*/
  FFilter f_filter;
  /*! \brief The TODO.*/
  size_t max_continuous_error = 150;
  FFilter GetFilter() { return f_filter; };
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch, const tir::Schedule& orig) final;

  Postproc Clone() const {
    ObjectPtr<FilterLoopSplitsNode> n = make_object<FilterLoopSplitsNode>(*this);
    return Postproc(n);
  }

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.FilterLoopSplits";
  TVM_DECLARE_FINAL_OBJECT_INFO(FilterLoopSplitsNode, PostprocNode);
};

bool FilterLoopSplitsNode::Apply(const tir::Schedule& sch, const tir::Schedule& orig) {
  return true;
}

Postproc Postproc::FilterLoopSplits(FilterLoopSplitsNode::FFilter f_filter) {
  std::cout << "Postproc::FilterLoopSplits" << std::endl << std::flush;
  ObjectPtr<FilterLoopSplitsNode> n = make_object<FilterLoopSplitsNode>();
  n->f_filter = std::move(f_filter);
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(FilterLoopSplitsNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocFilterLoopSplits")
    .set_body_typed(Postproc::FilterLoopSplits);

}  // namespace meta_schedule
}  // namespace tvm
