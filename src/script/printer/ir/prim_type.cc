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
#include <tvm/ir/type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<IntImm>(
      "", [](IntImm imm, AccessPath imm_p, IRDocsifier d) -> Doc {
        DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
        if (dtype == d->cfg->int_dtype) {
          return LiteralDoc::Int(imm, imm_p->Attr("value"));
        } else if (dtype == DLDataType{kDLBool, 8, 1}) {
          return TIR(d, DType2Str(dtype))
              ->Call({LiteralDoc::Boolean(static_cast<bool>(imm->value), imm_p->Attr("value"))});
        } else {
          return TIR(d, DType2Str(dtype))->Call({LiteralDoc::Int(imm, imm_p->Attr("value"))});
        }
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<FloatImm>(
      "", [](FloatImm imm, AccessPath imm_p, IRDocsifier d) -> Doc {
        DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
        if (dtype == d->cfg->float_dtype) {
          return LiteralDoc::Float(imm->value, imm_p->Attr("value"));
        } else {
          return TIR(d, DType2Str(dtype))
              ->Call({LiteralDoc::Float(imm->value, imm_p->Attr("value"))});
        }
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<PrimType>(
      "",
      [](PrimType ty, AccessPath p, IRDocsifier d) -> Doc { return TIR(d, DType2Str(ty->dtype)); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<StringType>(
      "", [](StringType ty, AccessPath p, IRDocsifier d) -> Doc {
        return IR(d, "StringType")->Call({});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<TupleType>(
      "", [](TupleType ty, AccessPath p, IRDocsifier d) -> Doc {
        if (ty->fields.empty()) {
          return LiteralDoc::None(p);
        }
        return TIR(d, "Tuple")->Call(d->AsDoc<ListDoc>(ty->fields, p->Attr("fields"))->elements);
      });
}
}  // namespace printer
}  // namespace script
}  // namespace tvm
