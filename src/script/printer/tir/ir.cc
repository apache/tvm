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
    .set_dispatch<IntImm>("", [](IntImm imm, ObjectPath p, IRDocsifier d) -> Doc {
      DataType dtype = imm->dtype;
      if (dtype == Default::IntDType()) {
        return LiteralDoc::Int(imm->value);
      } else if (dtype == DataType::Bool()) {
        return LiteralDoc::Boolean(imm->value);
      } else {
        return TIR(d)  //
            ->Attr(runtime::DLDataType2String(dtype))
            ->Call({LiteralDoc::Int(imm->value)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FloatImm>("", [](FloatImm imm, ObjectPath p, IRDocsifier d) -> Doc {
      DataType dtype = imm->dtype;
      if (dtype == Default::FloatDType()) {
        return LiteralDoc::Float(imm->value);
      } else {
        return TIR(d)
            ->Attr(runtime::DLDataType2String(dtype))
            ->Call({LiteralDoc::Float(imm->value)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>("", [](Range range, ObjectPath p, IRDocsifier d) -> Doc {
      return TIR(d)->Attr("Range")->Call({
          d->AsDoc<ExprDoc>(range->min, p->Attr("min")),
          d->AsDoc<ExprDoc>(range->extent, p->Attr("extent")),
      });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("", [](PrimType ty, ObjectPath p, IRDocsifier d) -> Doc {
      std::string dtype = ty->dtype.is_void() ? "void" : runtime::DLDataType2String(ty->dtype);
      return TIR(d)->Attr(dtype);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PointerType>("", [](PointerType ty, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc element_type = d->AsDoc<ExprDoc>(ty->element_type, p->Attr("element_type"));
      if (ty->storage_scope == "") {
        return TIR(d)->Attr("Ptr")->Call({element_type});
      } else {
        return TIR(d)->Attr("Ptr")->Call({element_type, LiteralDoc::Str(ty->storage_scope)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TupleType>("", [](TupleType ty, ObjectPath p, IRDocsifier d) -> Doc {
      if (ty->fields.empty()) {
        return LiteralDoc::None();
      }
      return TIR(d)  //
          ->Attr("Tuple")
          ->Call(d->AsDoc<ListDoc>(ty->fields, p->Attr("fields"))->elements);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Target>("", [](Target target, ObjectPath p, IRDocsifier d) -> Doc {
      Map<String, ObjectRef> config = target->Export();
      return TIR(d)->Attr("target")->Call({d->AsDoc<ExprDoc>(config, p)});
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
