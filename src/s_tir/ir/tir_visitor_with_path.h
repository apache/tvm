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

#ifndef TVM_S_TIR_IR_TIR_VISITOR_WITH_PATH_H_
#define TVM_S_TIR_IR_TIR_VISITOR_WITH_PATH_H_
#include <tvm/s_tir/stmt.h>

#include "../../tirx/ir/tir_visitor_with_path.h"
namespace tvm {
namespace s_tir {
class TIRVisitorWithPath : public tirx::TIRVisitorWithPath {
 public:
  using Parent = tirx::TIRVisitorWithPath;
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(TIRVisitorWithPath, Parent)
 protected:
  using AccessPath = ffi::reflection::AccessPath;
  using Parent::VisitStmt_;
  virtual void VisitStmt_(const SBlockNode* op, AccessPath path);
  virtual void VisitStmt_(const SBlockRealizeNode* op, AccessPath path);
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    vtable->SetDispatch<SBlockNode>(
        [](const ffi::ObjectRef& node, StmtVisitor* self, AccessPath path) {
          static_cast<TIRVisitorWithPath*>(self)->VisitStmt_(
              static_cast<const SBlockNode*>(node.get()), path);
        });
    vtable->SetDispatch<SBlockRealizeNode>(
        [](const ffi::ObjectRef& node, StmtVisitor* self, AccessPath path) {
          static_cast<TIRVisitorWithPath*>(self)->VisitStmt_(
              static_cast<const SBlockRealizeNode*>(node.get()), path);
        });
  }
};
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_IR_TIR_VISITOR_WITH_PATH_H_
