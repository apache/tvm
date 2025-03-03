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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/relax/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

/////////////////////////////// Function ////////////////////////////////

/*!
 * \brief Start a function frame.
 * \param is_pure Whether the function is annotated as pure.
 * \param is_private Whether the function is annotated as private.
 * \return The created ir_builder Function frame.
 */
TVM_DLL FunctionFrame Function(const Bool& is_pure, const Bool& is_private);

/*!
 * \brief Add a parameter to the last function frame.
 * \param name The name of the parameter.
 * \param struct_info The struct_info of the parameter.
 * \return The created function parameter var.
 */
TVM_DLL tvm::relax::Var Arg(const String& name, const tvm::relax::StructInfo& struct_info);

/*!
 * \brief Specify the name of the last function frame.
 * \param name The function name.
 */
TVM_DLL void FuncName(const String& name);

/*!
 * \brief Specify the attrs of the last function frame.
 * \param attrs The function attrs.
 */
TVM_DLL void FuncAttrs(Map<String, ObjectRef> attrs);

/*!
 * \brief Specify the return struct info of the last function frame.
 * \param ret_sinfo The return struct info.
 */
TVM_DLL void FuncRetStructInfo(const tvm::relax::StructInfo& ret_sinfo);

/*!
 * \brief Specify the return value of the last function frame.
 * \param value The return value.
 */
TVM_DLL void FuncRetValue(const tvm::relax::Expr& value);

///////////////////////////// BindingBlock //////////////////////////////

/*!
 * \brief Start a binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame BindingBlock();

/*!
 * \brief Start a dataflow binding block frame.
 * \return The created ir_builder Block frame.
 */
TVM_DLL BlockFrame Dataflow();

/*!
 * \brief Expose the dataflow block output variables as global ones
 * \param vars The output variables of a dataflow block
 */
TVM_DLL void DataflowBlockOutput(const Array<tvm::relax::Var>& vars);

////////////////////////////// Bindings ////////////////////////////////

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param value The right side value of the bindings to be emitted.
 * \param annotate_struct_info The optional struct info annotation for the emitted value.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::relax::Var Emit(
    const tvm::relax::Expr& value,
    const Optional<tvm::relax::StructInfo>& annotate_struct_info = NullOpt);

/*!
 * \brief Emit a match_cast binding to the last binding block frame.
 * \param value The value of the MatchCast to be emitted.
 * \param struct_info The struct info of the MatchCast to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::relax::Var EmitMatchCast(const tvm::relax::Expr& value,
                                      const tvm::relax::StructInfo& struct_info);

/*!
 * \brief Emit a binding to the last binding block frame.
 * \param binding The binding to be emitted.
 * \return The left side var of the emitted binding.
 */
TVM_DLL tvm::relax::Var EmitVarBinding(const tvm::relax::VarBinding& binding);

///////////////////////////// If Then Else /////////////////////////////

/*!
 * \brief Create an if statement.
 * \param condition The condition of if statement.
 * \return The result IfFrame.
 */
IfFrame If(tvm::relax::Expr condition);
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

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_IR_H_
