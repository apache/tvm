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
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ffi::Array<Any>>(  //
        "", [](ffi::Array<Any> array, AccessPath p, IRDocsifier d) -> Doc {
          int n = array.size();
          ffi::Array<ExprDoc> results;
          results.reserve(n);
          for (int i = 0; i < n; ++i) {
            results.push_back(d->AsDoc<ExprDoc>(array[i], p->ArrayItem(i)));
          }
          return ListDoc(results);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ffi::Map<Any, Any>>(  //
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
              return Downcast<ffi::String>(lhs.first) < Downcast<ffi::String>(rhs.first);
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

}  // namespace printer
}  // namespace script
}  // namespace tvm
