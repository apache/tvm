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

#include <dlpack/dlpack.h>
#include <tvm/ir/type.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include "../utils.h"
#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("tir", [](TracedObject<PrimType> ty, IRDocsifier p) -> Doc {
      TracedBasicValue<DataType> dtype = ty.GetAttr(&PrimTypeNode::dtype);
      String ty_str = runtime::DLDataType2String(dtype.Get());
      return TIR(p)->Attr(MakeTraced(ty_str, ty.GetPath()));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PointerType>("tir", [](TracedObject<PointerType> ty, IRDocsifier p) -> Doc {
      TracedObject<Type> element_type = ty.GetAttr(&PointerTypeNode::element_type);
      TracedObject<String> storage_scope = ty.GetAttr(&PointerTypeNode::storage_scope);

      ExprDoc element_type_doc = p->AsDoc<ExprDoc>(element_type);
      if (storage_scope.Get().empty()) {
        return TIR(p)->Attr("Ptr")->Call({element_type_doc});
      } else {
        return TIR(p)->Attr("Ptr")->Call({element_type_doc, LiteralDoc::Str(storage_scope)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TupleType>("tir", [](TracedObject<TupleType> ty, IRDocsifier p) -> Doc {
      auto fields = ty.GetAttr(&TupleTypeNode::fields);

      if (fields.empty()) {
        return LiteralDoc::None(fields.GetPath());
      }
      return TIR(p)->Attr("Tuple")->Call(AsExprDocArray(fields, p));
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
