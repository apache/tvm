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
#ifndef TVM_SCRIPT_IR_BUILDER_IR_H_
#define TVM_SCRIPT_IR_BUILDER_IR_H_

#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/script/ir_builder/frame.h>

#include <vector>

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

/*!
 * \brief The IRModule declaration statement.
 * \return The IRModuleFrame.
 */
TVM_DLL IRModuleFrame IRModule();

/*!
 * \brief Declare a Function without given the specific function implementation.
 * \note It is usually used in cross-function call. And we can specify the function by `DefFunction`
 * \param func_name The function unique name.
 * \param func_signature A Function w/o body, which used to specify the function signature
 *                       (i.e. func params and func return type/shape).
 * \return The corresponding GlobalVar.
 */
TVM_DLL GlobalVar DeclFunction(const ffi::String& func_name, const BaseFunc& func_signature);

/*!
 * \brief Define the function which is declared before.
 * \param func_name The function unique name.
 * \param func The given function implementation
 */
TVM_DLL void DefFunction(const ffi::String& func_name, const BaseFunc& func);

/*!
 * \brief The serial For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \param dtype The optional dtype of the loop var ("int32" or "uint32"). When omitted
 *              it is inferred from the bounds.
 * \return The ForFrame.
 */
ForFrame Serial(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                ffi::Optional<PrimExpr> step = std::nullopt,
                ffi::Optional<PrimType> dtype = std::nullopt);
/*!
 * \brief The parallel For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \param dtype The optional dtype of the loop var ("int32" or "uint32").
 * \return The ForFrame.
 */
ForFrame Parallel(PrimExpr start, PrimExpr stop,
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                  ffi::Optional<PrimExpr> step = std::nullopt,
                  ffi::Optional<PrimType> dtype = std::nullopt);
/*!
 * \brief The vectorized For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \param dtype The optional dtype of the loop var ("int32" or "uint32").
 * \return The ForFrame.
 */
ForFrame Vectorized(PrimExpr start, PrimExpr stop,
                    ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                    ffi::Optional<PrimExpr> step = std::nullopt,
                    ffi::Optional<PrimType> dtype = std::nullopt);
/*!
 * \brief The unrolled For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \param dtype The optional dtype of the loop var ("int32" or "uint32").
 * \return The ForFrame.
 */
ForFrame Unroll(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                ffi::Optional<PrimExpr> step = std::nullopt,
                ffi::Optional<PrimType> dtype = std::nullopt);
/*!
 * \brief The grid For statement.
 * \param extents The extents of the iteration.
 * \param dtype The optional dtype of every loop var ("int32" or "uint32"). When omitted
 *              each loop var takes the dtype of its own extent.
 * \return The ForFrame.
 */
ForFrame Grid(ffi::Array<ffi::Variant<PrimExpr, ffi::Tuple<PrimExpr, PrimExpr>>> extents,
              ffi::Optional<PrimType> dtype = std::nullopt);

/*!
 * \brief The assertion statement.
 * \param condition The assertion condition.
 * \param error_kind The error kind (e.g. "RuntimeError", "TypeError", "ValueError").
 * \param message_parts The error message parts (stored as separate fragments in the IR).
 * \return The AssertFrame.
 */
AssertFrame Assert(PrimExpr condition, ffi::String error_kind,
                   ffi::Array<ffi::String> message_parts);

/*!
 * \brief Create a Bind (variable binding).
 *
 * Emits a flat Bind statement to the current frame and returns the bound variable.
 *
 * \param value The value to be bound.
 * \param type_annotation  The type annotation of the binding.
 *                         Usually it is used for fine-grained var typing,
 *                         particularly, PtrType.
 * \param var The variable to be bound. If not specified, a new variable will be created.
 * \return The bound Var.
 */
Var Bind(Expr value, ffi::Optional<Type> type_annotation = std::nullopt,
         ffi::Optional<Var> var = std::nullopt);

/*!
 * \brief Create a while loop.
 * \param condition The termination condition of the loop.
 * \return The result WhileFrame.
 */
WhileFrame While(PrimExpr condition);

/*!
 * \brief Create a return statement.
 * \param value The value to return.
 * \return The same statement that was added to the parent frame.
 */
tvm::Stmt Return(Expr value);

/*!
 * \brief Create a break statement.
 * \return The same statement that was added to the parent frame.
 */
tvm::Stmt Break();

/*!
 * \brief Create a continue statement.
 * \return The same statement that was added to the parent frame.
 */
tvm::Stmt Continue();

/*!
 * \brief Create an if statement.
 * \param condition The condition of if statement.
 * \return The result IfFrame.
 */
IfFrame If(PrimExpr condition);

/*!
 * \brief Create a then.
 * \return The result ThenFrame.
 */
ThenFrame Then();

/*!
 * \brief Create an else.
 * \return The result ElseFrame.
 */
ElseFrame Else();

/*!
 * \brief Construct a result-free region with lexical body parameters.
 * \param op The region operation.
 * \param args Operands evaluated outside the region body.
 * \param body_params Existing body parameters, or omitted to generate them from the operation.
 * \param attrs Additional operation attributes.
 */
RegionFrame Region(Op op, ffi::Array<Expr> args,
                   ffi::Optional<ffi::Array<Var>> body_params = std::nullopt,
                   DictAttrs attrs = DictAttrs());

/*!
 * \brief Evaluate the input expression.
 * \param value The input expression to evaluate.
 * \return The same statement that was added to the parent frame.
 */
tvm::Stmt Evaluate(Expr value);

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_IR_H_
