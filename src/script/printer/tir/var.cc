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
#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

TracedObject<String> GetVarNameHint(const TracedObject<tir::Var>& var) {
  TracedObject<String> name_hint = var.GetAttr(&tir::VarNode::name_hint);
  if (name_hint.Get().empty()) {
    return MakeTraced(String("v"), var.GetPath());
  } else {
    return name_hint;
  }
}

IdDoc CreateFreeVariableDefinition(TracedObject<tir::Var> var, IRDocsifier p) {
  TracedObject<String> name_hint = GetVarNameHint(var);
  // TODO(yelite): When implementing the PrimFunc printing, the logic here
  // needs to change, putting variable def into PrimFuncFrame if it exists.
  TIRTopLevelFrame top_level_frame = p->GetFrame<TIRTopLevelFrame>().value();
  IdDoc doc = p->vars->Define(var.Get(), name_hint, top_level_frame);
  StmtDoc def_doc = AssignDoc(doc, NullOpt, GetTypeAnnotationDocForVar(var, p));
  top_level_frame->free_var_definitions.push_back(def_doc);
  return doc;
}

ExprDoc PrintVariable(TracedObject<tir::Var> var, IRDocsifier p) {
  Optional<ExprDoc> doc = p->vars->GetVarDoc(var);
  if (doc.defined()) {
    return doc.value();
  } else {
    return CreateFreeVariableDefinition(var, p);
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Var>(PrintVariable);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SizeVar>([](TracedObject<tir::SizeVar> var, IRDocsifier p) {
      return PrintVariable(MakeTraced(var.Get(), var.GetPath()), p);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IterVar>([](TracedObject<tir::IterVar> v, IRDocsifier p) -> Doc {
      LOG(FATAL) << "Cannot print IterVar directly. Please use the helper functions in tir.h for "
                    "specific usage of IterVar.";
      throw;
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
