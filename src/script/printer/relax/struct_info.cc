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
#include <tvm/tir/stmt_functor.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ObjectStructInfo>(  //
        "", [](relax::ObjectStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "Object");
        });

ExprDoc PrintShapeVar(const PrimExpr& e, const ObjectPath& e_p, const IRDocsifier& d) {
  ExprDoc expr_doc = d->AsDoc<ExprDoc>(e, e_p);
  // Step 1. Find if `func_vars` are being collected
  const RelaxFrameNode* f = nullptr;
  for (const Frame& frame : d->frames) {
    if (const auto* relax_frame = frame.as<RelaxFrameNode>()) {
      if (relax_frame->func_vars) {
        f = relax_frame;
        break;
      }
    }
  }
  // Step 2. Figure out if the PrimExpr contains at least a func var
  bool func_var_mode = false;
  if (f != nullptr) {
    tir::PostOrderVisit(e, [f, &func_var_mode](const ObjectRef& obj) -> void {
      if (const auto* var = obj.as<tir::VarNode>()) {
        if (f->func_vars->count(var)) {
          func_var_mode = true;
        }
      }
    });
  }
  // Step 3. Stringify the PrimExpr if func var exists
  if (func_var_mode) {
    return LiteralDoc::Str(DocToPythonScript(expr_doc, d->cfg), e_p);
  }
  return expr_doc;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::PrimStructInfo>(
        "", [](relax::PrimStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          Array<ExprDoc, void> args;
          Array<String> kwargs_keys;
          Array<ExprDoc, void> kwargs_values;

          if (n->value.defined()) {
            kwargs_keys.push_back("value");
            kwargs_values.push_back(PrintShapeVar(n->value.value(), n_p->Attr("value"), d));
          } else {
            args.push_back(LiteralDoc::DataType(n->dtype, n_p->Attr("dtype")));
          }

          return Relax(d, "Prim")->Call(args, kwargs_keys, kwargs_values);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ShapeStructInfo>(
        "", [](relax::ShapeStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          if (n->values.defined()) {
            Array<PrimExpr> shape = n->values.value();
            ObjectPath shape_p = n_p->Attr("values");
            Array<ExprDoc> shape_docs;
            for (int i = 0, ndim = shape.size(); i < ndim; ++i) {
              shape_docs.push_back(PrintShapeVar(shape[i], shape_p->ArrayIndex(i), d));
            }
            return Relax(d, "Shape")->Call({ListDoc(shape_docs)});
          }
          return Relax(d, "Shape")
              ->Call({}, {"ndim"}, {LiteralDoc::Int(n->ndim, n_p->Attr("ndim"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::TensorStructInfo>(  //
        "", [](relax::TensorStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          Array<ExprDoc> args;
          Array<String> kwargs_keys;
          Array<ExprDoc> kwargs_values;
          if (n->shape.defined()) {
            // Need to dig into ShapeExpr to preserve the `R.shape` prefix
            if (const auto* shape = n->shape.value().as<relax::ShapeExprNode>()) {
              auto shape_expr = GetRef<relax::ShapeExpr>(shape);
              ObjectPath shape_p = n_p->Attr("shape")->Attr("values");
              Array<ExprDoc> shape_docs;
              for (int i = 0, ndim = shape_expr->values.size(); i < ndim; ++i) {
                shape_docs.push_back(
                    PrintShapeVar(shape_expr->values[i], shape_p->ArrayIndex(i), d));
              }
              args.push_back(TupleDoc(shape_docs));
            } else {
              args.push_back(d->AsDoc<ExprDoc>(n->shape.value(), n_p->Attr("shape")));
            }
          }
          if (!n->IsUnknownDtype()) {
            kwargs_keys.push_back("dtype");
            kwargs_values.push_back(LiteralDoc::DataType(n->dtype, n_p->Attr("dtype")));
          }
          if (!n->shape.defined() && !n->IsUnknownNdim()) {
            kwargs_keys.push_back("ndim");
            kwargs_values.push_back(LiteralDoc::Int(n->ndim, n_p->Attr("ndim")));
          }
          if (n->vdevice.defined() && n->vdevice.value()->target.defined()) {
            kwargs_keys.push_back("vdevice");
            std::string dev_kind = n->vdevice.value()->target->kind->name;
            int dev_index = FindVDeviceIndexByTargetKind(n->vdevice.value(), d);
            kwargs_values.push_back(
                LiteralDoc::Str(dev_kind + ":" + std::to_string(dev_index), n_p->Attr("vdevice")));
          }
          if (args.empty() && kwargs_keys.empty()) {
            return Relax(d, "Tensor");
          }
          return Relax(d, "Tensor")->Call(args, kwargs_keys, kwargs_values);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::TupleStructInfo>(  //
        "", [](relax::TupleStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          if (n->fields.empty()) {
            return Relax(d, "Tuple");
          }
          Array<ExprDoc> fields_doc;
          ObjectPath fields_p = n_p->Attr("fields");
          for (int i = 0, l = n->fields.size(); i < l; ++i) {
            fields_doc.push_back(d->AsDoc<ExprDoc>(n->fields[i], fields_p->ArrayIndex(i)));
          }
          return Relax(d, "Tuple")->Call(fields_doc);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::FuncStructInfo>(  //
        "", [](relax::FuncStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          auto ret_doc = d->AsDoc<ExprDoc>(n->ret, n_p->Attr("ret"));
          auto purity_doc = LiteralDoc::Boolean(n->purity, n_p->Attr("purity"));

          if (n->IsOpaque()) {
            Array<String> keys;
            Array<ExprDoc, void> values;

            if (!n->ret->IsInstance<relax::ObjectStructInfoNode>()) {
              keys.push_back("ret");
              values.push_back(ret_doc);
            }
            if (n->purity) {
              keys.push_back("purity");
              values.push_back(purity_doc);
            }

            if (keys.size()) {
              return Relax(d, "Callable")->Call({}, keys, values);
            } else {
              return Relax(d, "Callable");
            }
          }
          // TODO(@junrushao): track symbolic shape relation
          Array<ExprDoc> params_doc;
          Array<relax::StructInfo> params = n->params.value();
          ObjectPath params_p = n_p->Attr("params");
          for (int i = 0, n_params = params.size(); i < n_params; ++i) {
            params_doc.push_back(d->AsDoc<ExprDoc>(params[i], params_p->ArrayIndex(i)));
          }
          return Relax(d, "Callable")->Call({TupleDoc(params_doc), ret_doc, purity_doc});
        });

TVM_SCRIPT_REPR(relax::ObjectStructInfoNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::PrimStructInfoNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ShapeStructInfoNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::TensorStructInfoNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::TupleStructInfoNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::FuncStructInfoNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
