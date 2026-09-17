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

#include "../../tirx/ir/specialize.h"

#include <tvm/ffi/function.h>
#include <tvm/s_tir/stmt_functor.h>

namespace tvm {
namespace s_tir {
namespace {
ffi::Optional<VisitInterrupt> PlanBlockBuffers(tirx::StmtExprVisitor* planner,
                                               const SBlockNode* op) {
  // Block allocations were planned before all other block children by the specializer.
  for (const tirx::BufferVar& buffer : op->alloc_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return planner->Visit(buffer); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->VisitBufferMetadata(buffer));
  }
  for (const tirx::IterVar& iter : op->iter_vars) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(iter->dom->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(iter->dom->extent));
  }
  for (const tirx::BufferRegion& region : op->reads) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(region));
  }
  for (const tirx::BufferRegion& region : op->writes) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(region));
  }
  for (const MatchBufferRegion& match : op->match_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return planner->Visit(match->buffer); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->VisitBufferMetadata(match->buffer));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(match->source));
  }
  if (op->init.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(planner->Visit(op->init.value()));
  }
  return planner->Visit(op->body);
}

void InitBufferPlanner(tirx::SpecializeVisitorVTable* vtable) {
  vtable->ClearDispatch<SBlockNode>();
  vtable->SetDispatch<SBlockNode>([](const ffi::Object* node, ObjectVisitor* visitor) {
    return PlanBlockBuffers(static_cast<tirx::StmtExprVisitor*>(visitor),
                            static_cast<const SBlockNode*>(node));
  });
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { tirx::RegisterSpecializeBufferPlannerExtension(InitBufferPlanner); }
}  // namespace s_tir
}  // namespace tvm
