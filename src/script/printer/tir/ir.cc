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
#include <tvm/target/target.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_REGISTER_NODE_TYPE(TIRFrameNode);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IntImm>("", [](IntImm imm, ObjectPath imm_p, IRDocsifier d) -> Doc {
      DataType dtype = imm->dtype;
      if (dtype == d->cfg->int_dtype) {
        return LiteralDoc::Int(imm->value, imm_p->Attr("value"));
      } else if (dtype == DataType::Bool()) {
        return TIR(d, DType2Str(dtype))
            ->Call({LiteralDoc::Boolean(imm->value, imm_p->Attr("value"))});
      } else {
        return TIR(d, DType2Str(dtype))->Call({LiteralDoc::Int(imm->value, imm_p->Attr("value"))});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FloatImm>("", [](FloatImm imm, ObjectPath imm_p, IRDocsifier d) -> Doc {
      DataType dtype = imm->dtype;
      if (dtype == d->cfg->float_dtype) {
        return LiteralDoc::Float(imm->value, imm_p->Attr("value"));
      } else {
        return TIR(d, DType2Str(dtype))
            ->Call({LiteralDoc::Float(imm->value, imm_p->Attr("value"))});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>("tir", [](Range range, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d, "Range")
          ->Call({
              d->AsDoc<ExprDoc>(range->min, p->Attr("min")),
              d->AsDoc<ExprDoc>(range->extent + range->min, p->Attr("extent")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("", [](PrimType ty, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d, DType2Str(ty->dtype));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PointerType>("", [](PointerType ty, ObjectPath ty_p, IRDocsifier d) -> Doc {
      ExprDoc element_type{nullptr};
      if (const auto* prim_type = ty->element_type.as<PrimTypeNode>()) {
        element_type = LiteralDoc::DataType(prim_type->dtype,  //
                                            ty_p->Attr("element_type")->Attr("dtype"));
      } else {
        element_type = d->AsDoc<ExprDoc>(ty->element_type, ty_p->Attr("element_type"));
      }
      if (ty->storage_scope == "") {
        return TIR(d, "handle")->Call({element_type});
      } else {
        return TIR(d, "handle")
            ->Call({element_type, LiteralDoc::Str(ty->storage_scope, ty_p->Attr("storage_scope"))});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TupleType>("", [](TupleType ty, ObjectPath p, IRDocsifier d) -> Doc {
      if (ty->fields.empty()) {
        return LiteralDoc::None(p);
      }
      return TIR(d, "Tuple")->Call(d->AsDoc<ListDoc>(ty->fields, p->Attr("fields"))->elements);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Target>("", [](Target target, ObjectPath p, IRDocsifier d) -> Doc {
      Map<String, ObjectRef> config = target->Export();
      return TIR(d, "target")->Call({d->AsDoc<ExprDoc>(config, p)});
    });

TVM_SCRIPT_REPR(IntImmNode, ReprPrintTIR);
TVM_SCRIPT_REPR(FloatImmNode, ReprPrintTIR);
TVM_SCRIPT_REPR(PrimTypeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(PointerTypeNode, ReprPrintTIR);
TVM_SCRIPT_REPR(TupleTypeNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
