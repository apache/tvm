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
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/tensor.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<DataTypeImm>(
      "", [](DataTypeImm n, AccessPath p, IRDocsifier d) -> Doc {
        return TIR(d, "dtype")->Call({LiteralDoc::DataType(n->value, p->Attr("value"))});
      });
  IRDocsifier::vtable().set_dispatch<DataTypeImm>(
      "ir", [](DataTypeImm n, AccessPath p, IRDocsifier d) -> Doc {
        return IR(d, "dtype")->Call({LiteralDoc::DataType(n->value, p->Attr("value"))});
      });
  IRDocsifier::vtable().set_dispatch<GenericConst>(
      "", [](GenericConst n, AccessPath p, IRDocsifier d) -> Doc {
        if (auto dtype = n->value.as<DLDataType>()) {
          return TIR(d, "dtype")->Call({LiteralDoc::DataType(*dtype, p->Attr("value"))});
        }
        if (n->value.as<runtime::Tensor>()) {
          // Tensor-valued constants are Relax constants and keep the Relax spelling.
          return IRDocsifier::vtable()("relax", n, p, d);
        }
        return d->AddMetadata(n);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<GenericConst>(
      "ir", [](GenericConst n, AccessPath p, IRDocsifier d) -> Doc {
        if (auto dtype = n->value.as<DLDataType>()) {
          return IR(d, "dtype")->Call({LiteralDoc::DataType(*dtype, p->Attr("value"))});
        }
        return IRDocsifier::vtable()("", n, p, d);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<ffi::Array<Any>>(  //
      "", [](ffi::Array<Any> array, AccessPath p, IRDocsifier d) -> Doc {
        int n = array.size();
        ffi::Array<ExprDoc> results;
        results.reserve(n);
        for (int i = 0; i < n; ++i) {
          results.push_back(d->AsDoc<ExprDoc>(array[i], p->ArrayItem(i)));
        }
        return ListDoc(results);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<ffi::Map<Any, Any>>(  //
      "", [](ffi::Map<Any, Any> dict, AccessPath p, IRDocsifier d) -> Doc {
        using POO = std::pair<Any, Any>;
        std::vector<POO> items{dict.begin(), dict.end()};
        bool is_str_map = true;
        for (const auto& kv : items) {
          if (!kv.first.as<ffi::String>()) {
            is_str_map = false;
            break;
          }
        }
        if (is_str_map) {
          std::sort(items.begin(), items.end(), [](const POO& lhs, const POO& rhs) {
            return lhs.first.as_or_throw<ffi::String>() < rhs.first.as_or_throw<ffi::String>();
          });
        }
        int n = dict.size();
        ffi::Array<ExprDoc> ks;
        ffi::Array<ExprDoc> vs;
        ks.reserve(n);
        vs.reserve(n);
        for (int i = 0; i < n; ++i) {
          ks.push_back(d->AsDoc<ExprDoc>(items[i].first, p->MapItemMissing(items[i].first)));
          vs.push_back(d->AsDoc<ExprDoc>(items[i].second, p->MapItem(items[i].first)));
        }
        return DictDoc(ks, vs);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<ffi::Shape>(
      "", [](ffi::Shape n, AccessPath n_p, IRDocsifier d) -> Doc {
        int s = n.size();
        ffi::Array<ExprDoc> results;
        results.reserve(s);
        for (int i = 0; i < s; ++i) {
          results.push_back(d->AsDoc<ExprDoc>(IntImm::Int32(n[i]), n_p->ArrayItem(i)));
        }
        return TupleDoc(results);
      });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
