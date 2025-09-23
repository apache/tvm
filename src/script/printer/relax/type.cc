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
#include <tvm/ffi/reflection/registry.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ShapeType>(  //
        "", [](relax::ShapeType n, AccessPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "Shape")
              ->Call({}, {"ndim"}, {LiteralDoc::Int(n->ndim, n_p->Attr("ndim"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ObjectType>(  //
        "", [](relax::ObjectType n, AccessPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "Object");
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::TensorType>(  //
        "", [](relax::TensorType n, AccessPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "Tensor")
              ->Call({}, {"ndim", "dtype"},
                     {LiteralDoc::Int(n->ndim, n_p->Attr("ndim")),
                      LiteralDoc::DataType(n->dtype, n_p->Attr("dtype"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::PackedFuncType>(  //
        "", [](relax::PackedFuncType n, AccessPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "PackedFunc");  // TODO(@junrushao): verify if this is correct
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::TupleType>(  //
        "relax", [](tvm::TupleType n, AccessPath n_p, IRDocsifier d) -> Doc {
          if (n->fields.empty()) {
            return Relax(d, "Tuple");
          }
          ffi::Array<ExprDoc> fields_doc;
          AccessPath fields_p = n_p->Attr("fields");
          for (int i = 0, l = n->fields.size(); i < l; ++i) {
            fields_doc.push_back(d->AsDoc<ExprDoc>(n->fields[i], fields_p->ArrayItem(i)));
          }
          return Relax(d, "Tuple")->Call(fields_doc);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::FuncType>(
        "relax", [](tvm::FuncType n, AccessPath n_p, IRDocsifier d) -> Doc {
          ffi::Array<ExprDoc> arg_types_doc;
          ffi::Array<Type> arg_types = n->arg_types;
          AccessPath arg_types_p = n_p->Attr("arg_types");
          for (int i = 0, n_params = arg_types.size(); i < n_params; ++i) {
            arg_types_doc.push_back(d->AsDoc<ExprDoc>(arg_types[i], arg_types_p->ArrayItem(i)));
          }
          return Relax(d, "Callable")
              ->Call({TupleDoc(arg_types_doc),  //
                      d->AsDoc<ExprDoc>(n->ret_type, n_p->Attr("ret_type"))});
        });

TVM_SCRIPT_REPR(relax::ShapeTypeNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ObjectTypeNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::TensorTypeNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::PackedFuncTypeNode, ReprPrintRelax);
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ReprPrintRelax", ReprPrintRelax);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
