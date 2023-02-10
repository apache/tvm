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
#include <tvm/ir/expr.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

Doc PrintTIRVar(tir::Var n, ObjectPath n_p, IRDocsifier d) {
  ICHECK(n->dtype.is_int() && n->dtype.is_scalar()) << "TypeError: Relax only uses "
                                                       "scalar integer TIR variables, but gets: "
                                                    << n;
  if (!d->IsVarDefined(n)) {
    // Find the outmost Relax function frame. If not exist, the outmost Relax frame.
    RelaxFrameNode* f = nullptr;
    for (const Frame& frame : d->frames) {
      if (const auto* relax_frame = frame.as<RelaxFrameNode>()) {
        if (relax_frame->is_func) {
          f = const_cast<RelaxFrameNode*>(relax_frame);
          break;
        } else if (f == nullptr) {
          f = const_cast<RelaxFrameNode*>(relax_frame);
        }
      }
    }
    // There should be at least one Relax frame
    if (f == nullptr) {
      LOG(FATAL) << "IndexError: No relax environment is found when printing a TIR var under "
                    "relax's dispatch token";
    }
    // If the Relax function frame is collecting func vars
    if (f->func_vars) {
      ICHECK(f->is_func);
      f->func_vars->insert(n.get());
    }
    IdDoc var = d->Define(n, GetRef<Frame>(f), n->name_hint.empty() ? "v" : n->name_hint);
    var->source_paths.push_back(n_p);
    f->stmts.push_back(AssignDoc(var,
                                 TIR(d, "Var")->Call({
                                     LiteralDoc::Str(var->name, n_p->Attr("name_hint")),
                                     LiteralDoc::DataType(n->dtype, n_p->Attr("dtype")),
                                 }),
                                 NullOpt));
  }
  if (Optional<ExprDoc> doc = d->GetVarDoc(n)) {
    return doc.value();
  }
  LOG(FATAL) << "IndexError: Variable is not defined in the environment: " << n;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Var>("relax", PrintTIRVar);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::SizeVar>("relax", PrintTIRVar);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::IntImm>(                                             //
        "relax", [](tvm::IntImm n, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          // TODO(@junrushao): support non-int64 cases
          return LiteralDoc::Int(n->value, n_p);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::GlobalVar>(                                             //
        "relax", [](tvm::GlobalVar n, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          IdDoc ret(n->name_hint);
          ret->source_paths.push_back(n_p);
          return ret;
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
