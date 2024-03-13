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
#include <tvm/relax/distributed/global_info.h>

#include "../relax/utils.h"
#include "./utils.h"
namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<runtime::ShapeTuple>(
        "", [](runtime::ShapeTuple n, ObjectPath n_p, IRDocsifier d) -> Doc {
          int s = n.size();
          Array<ExprDoc> results;
          results.reserve(s);
          for (int i = 0; i < s; ++i) {
            results.push_back(d->AsDoc<ExprDoc>(Integer(n[i]), n_p->ArrayIndex(i)));
          }
          return TupleDoc(results);
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
