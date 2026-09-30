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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tvm/ir/prim/op.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ExprDoc RelaxShapeDim(DocTranslatorObj* d, const PrimExpr& dim) {
  ExprDoc translated = d->Translate(dim).value();
  if (auto mul = dim.as<prim::MulNode>()) {
    ExprDoc compact = OperationDoc(OperationDocNode::Kind::kMult,
                                   {RelaxShapeDim(d, mul->a), RelaxShapeDim(d, mul->b)});
    d->RecordOrigin(compact, dim);
    return compact;
  }
  if (auto integer = dim.as<IntImmNode>();
      integer && integer->ty.as_or_throw<PrimType>()->dtype == (DLDataType{kDLInt, 64, 1})) {
    ExprDoc compact = LiteralDoc::Int(ffi::GetRef<IntImm>(integer), std::nullopt);
    d->RecordOrigin(compact, dim);
    return compact;
  }
  return translated;
}

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
