
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

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/tir/stmt.h>

#include "../utils.h"
#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>([](TracedObject<tir::BufferStore> stmt, IRDocsifier p) {
      Array<ExprDoc> indices = AsExprDocArray(stmt.GetAttr(&tir::BufferStoreNode::indices), p);
      Array<Doc> index_docs(indices.begin(), indices.end());
      return AssignDoc(p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::buffer))[index_docs],
                       p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::value)), NullOpt);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>([](TracedObject<tir::Evaluate> stmt, IRDocsifier p) {
      return ExprStmtDoc(p->AsExprDoc(stmt.GetAttr(&tir::EvaluateNode::value)));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Store>([](TracedObject<tir::Store> stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::Store cannot be printed. Store is replaced by BufferStore.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](TracedObject<tir::BufferRealize> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL)
          << "tir::BufferRealize cannot be printed. All the BufferRealize should be nested inside "
             "with AttrStmt.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](TracedObject<tir::ProducerStore> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerStore cannot be printed";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](TracedObject<tir::ProducerRealize> stmt,
                                           IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerRealize cannot be printed";
      throw;
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
