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
#include <tvm/ir/memory_pools.h>
#include <tvm/node/functor.h>
#include <tvm/tir/usmp/utils.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::usmp::AllocatedPoolInfo>(
        "", [](tir::usmp::AllocatedPoolInfo node, ObjectPath p, IRDocsifier d) -> Doc {
          return IR(d, "AllocatedPoolInfo")
              ->Call({}, {"pool_info", "allocated_size"},
                     {d->AsDoc<ExprDoc>(node->pool_info, p->Attr("pool_info")),
                      d->AsDoc<ExprDoc>(node->allocated_size, p->Attr("allocated_size"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ConstantPoolInfo>("",
                                    [](ConstantPoolInfo node, ObjectPath p, IRDocsifier d) -> Doc {
                                      return IR(d, "ConstantPoolInfo")
                                          ->Call(
                                              {d->AsDoc<ExprDoc>(node->constant_info_array,
                                                                 p->Attr("constant_info_array"))});
                                    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ConstantInfo>("", [](ConstantInfo node, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "ConstantInfo")
          ->Call({d->AsDoc<ExprDoc>(node->name_hint, p->Attr("name_hint"))},
                 {"byte_offset", "data"},
                 {d->AsDoc<ExprDoc>(node->byte_offset, p->Attr("byte_offset")),
                  d->AddMetadata(node->data)});
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
