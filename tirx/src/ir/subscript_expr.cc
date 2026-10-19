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
 * \file subscript_expr.cc
 * \brief Type-directed realization for Python expression subscription.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(tvm::type_attr::kExprSubscriptRealizeByTy);
  refl::TypeAttrDef<TupleTypeNode>().attr(
      tvm::type_attr::kExprSubscriptRealizeByTy,
      [](Expr value, SubscriptSlice slice, Location loc) -> ffi::ObjectRef {
        TVM_FFI_CHECK_EQ(slice.size(), 1, IndexError)
            << "A tuple expression requires exactly one index";
        auto index = slice[0].as<PrimExpr>();
        TVM_FFI_CHECK(index.has_value(), TypeError) << "A tuple expression requires a point index";
        const auto* imm = index.value().as<IntImmNode>();
        TVM_FFI_CHECK(imm != nullptr, TypeError)
            << "A tuple expression requires a constant integer index";
        return TupleGetItem(value, imm->value.as<int>().value(), loc);
      });
  refl::GlobalDef().def(
      "ir.SubscriptExprRealize",
      [](Expr value, SubscriptSlice slice, Location loc) -> ffi::ObjectRef {
        TVM_FFI_CHECK(value.defined(), TypeError) << "Cannot subscript an undefined expression";
        static refl::TypeAttrColumn realize_column(tvm::type_attr::kExprSubscriptRealizeByTy);
        auto realize = tvm::type_attr::kExprSubscriptRealizeByTy.ValueFrom(
            realize_column[value->ty->type_index()]);
        TVM_FFI_CHECK(realize.has_value(), TypeError)
            << "Type " << value->ty->GetTypeKey() << " does not support subscript";
        ffi::ObjectRef result = (*realize)(value, slice, loc);
        TVM_FFI_CHECK(result.defined(), TypeError)
            << "__tvm_expr_subscript_realize__ for type " << value->ty->GetTypeKey()
            << " returned an undefined object";
        return result;
      });
}

}  // namespace tvm
