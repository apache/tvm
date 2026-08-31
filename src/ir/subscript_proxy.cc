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

/*!
 * \file subscript_proxy.cc
 * \brief Type-directed realization for the Python frontend's SubscriptProxy.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn("__subscript_expr_realize__");
  refl::GlobalDef().def(
      "ir.SubscriptExprRealize",
      [](Expr value,
         ffi::Array<ffi::Variant<
             ffi::Tuple<ffi::Optional<PrimExpr>, PrimExpr, ffi::Optional<PrimExpr>>, PrimExpr>>
             slice) -> ffi::ObjectRef {
        TVM_FFI_CHECK(value.defined(), TypeError) << "Cannot subscript an undefined expression";
        static refl::TypeAttrColumn realize_column("__subscript_expr_realize__");
        ffi::AnyView packed_realize = realize_column[value->ty->type_index()];
        TVM_FFI_CHECK(packed_realize != nullptr, TypeError)
            << "Type " << value->ty->GetTypeKey()
            << " does not register __subscript_expr_realize__";
        ffi::ObjectRef result =
            packed_realize.cast<ffi::Function>()(value, slice).cast<ffi::ObjectRef>();
        TVM_FFI_CHECK(result.defined(), TypeError)
            << "__subscript_expr_realize__ for type " << value->ty->GetTypeKey()
            << " returned an undefined object";
        return result;
      });
}

}  // namespace tvm
