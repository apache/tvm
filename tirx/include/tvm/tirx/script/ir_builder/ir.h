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
 * \file tvm/tirx/script/ir_builder/ir.h
 * \brief IRBuilder APIs for TIRx functions and statements.
 */
#ifndef TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_
#define TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_

#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/tensor.h>
#include <tvm/script/ir_builder/ir.h>
#include <tvm/tirx/script/ir_builder/frame.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile/thread_scope.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

using tvm::Var;
using tvm::ffi::Tensor;
using tvm::ffi::Tuple;
using tvm::ffi::Variant;
using tvm::tirx::TensorVar;
using tvm::tirx::ThreadScope;

/*!
 * \brief The primitive function statement.
 * \return The FunctionFrame.
 */
FunctionFrame Function(bool is_private = false, bool persistent = false,
                       bool is_declaration = false);

/*!
 * \brief The Function variable arguments adding function.
 * \param name The name of the variable.
 * \param var The variable argument.
 * \return The variable.
 */
Var Arg(ffi::String name, Var var);

/*!
 * \brief The Function tensor arguments adding function.
 * \param name The name of the tensor.
 * \param tensor The tensor argument.
 * \return The tensor.
 */
TensorVar Arg(ffi::String name, TensorVar tensor);

/*!
 * \brief The Function naming statement.
 * \param name The name of the Function.
 */
void FuncName(ffi::String name);

/*!
 * \brief The Function annotation statement.
 * \param attrs The annotations of the Function.
 */
void FuncAttrs(ffi::Map<ffi::String, ffi::Any> attrs);

/*!
 * \brief The Function return type statement.
 * \param ret_type The return type of the Function.
 * \return The return type.
 */
Type FuncRet(Type ret_type);

/*!
 * \brief Store data in a tensor destination.
 * \param dest The destination expression.
 * \param indices The indices location to be stored.
 * \param value The value to be stored.
 * \return The same statement that was added to the parent frame.
 */
tvm::Stmt TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value);

/*!
 * \brief Create a TIR var that represents a pointer
 *
 * \param dtype The optional data type of the pointer.  If omitted, construct
 * an opaque handle.
 *
 * \param storage_scope The storage scope of the pointer.
 *
 * \return The pointer.
 */
inline Var Handle(ffi::Optional<PrimType> dtype = std::nullopt,
                  runtime::MemoryScope storage_scope = {}) {
  Type type_annotation = dtype.has_value() ? Type(PtrType(dtype.value(), storage_scope))
                                           : Type(PtrType::VoidPtrType(storage_scope));
  return tvm::Var("", type_annotation);
}

inline Var TensorMap() { return tvm::Var("", PtrType(tvm::tirx::TensorMapType())); }

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_IR_BUILDER_IR_H_
