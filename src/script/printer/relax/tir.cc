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

#include "../tir/utils.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

/*! \brief Find the outmost Relax function frame. If not exist, the outmost Relax frame. */
RelaxFrameNode* GetRelaxFrame(IRDocsifier d) {
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
  return f;
}

Doc PrintTIRVar(tir::Var n, ObjectPath n_p, IRDocsifier d) {
  ICHECK(n->dtype.is_scalar()) << "TypeError: "
                               << "Relax only uses scalar TIR variables,"
                               << "but received TIR variable " << n << " with dtype " << n->dtype;

  if (!d->IsVarDefined(n)) {
    RelaxFrameNode* f = GetRelaxFrame(d);
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
    f->stmts.push_back(AssignDoc(var, PrintVarCreation(n, n_p, d), NullOpt));
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
          if (n->dtype.is_bool()) {
            return LiteralDoc::Boolean(n->value, n_p);
          } else {
            return LiteralDoc::Int(n->value, n_p);
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::GlobalVar>(                                             //
        "relax", [](tvm::GlobalVar n, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          if (Optional<ExprDoc> doc = d->GetVarDoc(n)) {
            return doc.value();
          } else {
            IdDoc ret(n->name_hint);
            ret->source_paths.push_back(n_p);
            return ret;
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::IRModule>(                                               //
        "relax", [](tvm::IRModule mod, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          Optional<ExprDoc> doc = d->GetVarDoc(mod);
          ICHECK(doc) << "Unable to print IRModule before definition in Relax.";
          if (d->cfg->module_alias.empty()) {
            // Use Module Name directly
            return doc.value();
          }
          RelaxFrameNode* f = GetRelaxFrame(d);
          ICHECK(f != nullptr && f->is_func)
              << "IndexError: No relax environment is found when printing a module alias var "
                 "under relax's dispatch token";
          if (!f->module_alias_printed) {
            // If the module_alias is not defined before, define it.
            f->stmts.push_back(AssignDoc(IdDoc(d->cfg->module_alias), doc.value(), NullOpt));
            f->module_alias_printed = true;
          }
          return IdDoc(d->cfg->module_alias);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>("relax", [](Range range, ObjectPath p, IRDocsifier d) -> Doc {
      return Relax(d, "Range")
          ->Call({
              d->AsDoc<ExprDoc>(range->min, p->Attr("min")),
              d->AsDoc<ExprDoc>(range->extent + range->min, p->Attr("extent")),
          });
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
