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

#ifndef TVM_S_TIR_IR_VISITOR_WITH_ANALYZER_H_
#define TVM_S_TIR_IR_VISITOR_WITH_ANALYZER_H_

#include <tvm/s_tir/stmt_functor.h>

#include "../../tirx/ir/ir_visitor_with_analyzer.h"

namespace tvm {
namespace s_tir {
class IRVisitorWithAnalyzer : public tirx::IRVisitorWithAnalyzer {
 public:
  using Parent = tirx::IRVisitorWithAnalyzer;
  using Parent::Visit;
  using Parent::Visit_;
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IRVisitorWithAnalyzer, Parent)
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeNode* op) {
    return s_tir::StmtExprVisitor::VisitBlockRealize(this, op);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<IRVisitorWithAnalyzer, SBlockNode>(vtable);
    SetDispatch<IRVisitorWithAnalyzer, SBlockRealizeNode>(vtable);
  }
};
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_IR_VISITOR_WITH_ANALYZER_H_
