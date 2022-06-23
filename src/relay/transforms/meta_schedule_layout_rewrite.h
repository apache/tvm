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
#ifndef TVM_RELAY_TRANSFORMS_META_SCHEDULE_LAYOUT_REWRITE_H_
#define TVM_RELAY_TRANSFORMS_META_SCHEDULE_LAYOUT_REWRITE_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/tir/index_map.h>

namespace tvm {
namespace relay {

class MetaScheduleLayoutRewriter : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) final;

  static void LayoutQueuePush(const tir::IndexMap& index_map);
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_META_SCHEDULE_LAYOUT_REWRITE_H_
