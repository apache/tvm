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

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace testing {

Function IncorrectImplementationOfBindParamForRelaxFunc(Function func) {
  auto body = Downcast<relax::SeqExpr>(func->body);
  body.CopyOnWrite()->blocks = {
      BindingBlock({VarBinding(func->params[0], relax::PrimValue::Int64(0))})};

  auto write_ptr = func.CopyOnWrite();
  write_ptr->params = {};
  write_ptr->body = body;
  return func;
}

TVM_REGISTER_GLOBAL("relax.testing.IncorrectImplementationOfBindParamForRelaxFunc")
    .set_body_typed(IncorrectImplementationOfBindParamForRelaxFunc);

tir::PrimFunc IncorrectImplementationOfBindParamForPrimFunc(tir::PrimFunc func) {
  auto new_body = tir::LetStmt(func->params[0], IntImm(func->params[0]->dtype, 0), func->body);

  auto write_ptr = func.CopyOnWrite();
  write_ptr->params = {};
  write_ptr->body = new_body;
  return func;
}

TVM_REGISTER_GLOBAL("relax.testing.IncorrectImplementationOfBindParamForPrimFunc")
    .set_body_typed(IncorrectImplementationOfBindParamForPrimFunc);

}  // namespace testing
}  // namespace relax
}  // namespace tvm
