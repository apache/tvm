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

#include "./tir.h"

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/script/printer/visit_traced.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace script {
namespace printer {

TIRTopLevelFrame::TIRTopLevelFrame() : TIRFrame(make_object<TIRTopLevelFrameNode>()) {}

TIRGeneralFrame::TIRGeneralFrame() : TIRFrame(make_object<TIRGeneralFrameNode>()) {}

void PostOrderVisitExprTraced(const TracedObject<PrimExpr>& expr,
                              const std::function<void(const TracedObject<PrimExpr>&)>& callback) {
  PostOrderVisitTraced(
      expr, [](const ObjectRef& object) { return object->IsInstance<PrimExprNode>(); },
      [&](const TracedObject<ObjectRef>& object) { callback(object.Downcast<PrimExpr>()); });
}

void PostOrderVisitStmtExprTraced(
    const TracedObject<tir::Stmt>& stmt,
    const std::function<void(const TracedObject<ObjectRef>&)>& callback) {
  PostOrderVisitTraced(
      stmt,
      [](const ObjectRef& object) {
        return object->IsInstance<PrimExprNode>() || object->IsInstance<tir::StmtNode>();
      },
      [&](const TracedObject<ObjectRef>& object) { callback(object); });
}

ExprDoc GetTypeAnnotationDocForVar(const TracedObject<tir::Var>& var, const IRDocsifier& p) {
  auto type_annotation = var.GetAttr(&tir::VarNode::type_annotation);
  if (type_annotation.Get().defined()) {
    return p->AsExprDoc(type_annotation);
  } else {
    auto dtype = var.GetAttr(&tir::VarNode::dtype);
    Type raw_type = GetTypeFromRuntimeDataType(dtype.Get());
    return p->AsExprDoc(MakeTraced(raw_type, dtype.GetPath()));
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<RootNodeContainer>("tir", [](TracedObject<RootNodeContainer> obj, IRDocsifier p) {
      const ObjectRef& root_node = obj.Get()->root_node;

      TIRTopLevelFrame top_level_frame;
      auto frame_ctx = p->WithFrame(top_level_frame);

      // Because we are printing a single element, concise scoping should be allowed by default
      top_level_frame->allow_concise_scoping = true;

      Doc root_doc = p->AsDoc<Doc>(MakeTraced(root_node));

      Array<StmtDoc> doc_to_print = top_level_frame->free_var_definitions;

      if (const auto* stmt_doc_node = root_doc.as<StmtDocNode>()) {
        doc_to_print.push_back(GetRef<StmtDoc>(stmt_doc_node));
      } else if (const auto* expr_doc_node = root_doc.as<ExprDocNode>()) {
        doc_to_print.push_back(ExprStmtDoc(GetRef<ExprDoc>(expr_doc_node)));
      } else if (const auto* stmt_block_node = root_doc.as<StmtBlockDocNode>()) {
        doc_to_print = runtime::Concat(doc_to_print, stmt_block_node->stmts);
      } else if (const auto* slice_doc_node = root_doc.as<SliceDocNode>()) {
        doc_to_print.push_back(ExprStmtDoc(IdDoc("_")[{GetRef<SliceDoc>(slice_doc_node)}]));
      } else {
        ICHECK(false) << "Cannot print " << root_doc->GetTypeKey() << " as top level doc for TIR.";
      }

      return StmtBlockDoc(doc_to_print);
    });
}  // namespace printer
}  // namespace script
}  // namespace tvm
