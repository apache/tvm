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
#ifndef TVM_SCRIPT_IR_BUILDER_IR_IR_H_
#define TVM_SCRIPT_IR_BUILDER_IR_IR_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/node/node.h>
#include <tvm/script/ir_builder/ir/frame.h>

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
TVM_DLL GlobalVar DeclFunction(const String& func_name, const BaseFunc& func_signature);

/*!
 * \brief Define the function which is declared before.
 * \param func_name The function unique name.
 * \param func The given function implementation
 */
TVM_DLL void DefFunction(const String& func_name, const BaseFunc& func);

/*!
 * \brief Add a Relax function or a TIR PrimFunc to the IRModuleFrame.
 * \param func The function to be added.
 * \param func_name_hint The name hint of the function to be added.
 * \note If the function to be added already exists, return its
 * GlobalVar directly.
 * \return The global var bound to the added function.
 */
TVM_DLL GlobalVar AddFunction(const BaseFunc& func, String func_name_hint);

/*!
 * \brief Update a Relax function or a TIR PrimFunc in the IRModuleFrame.
 * \param gv The global var referring the function to be updated.
 * \param function The updated function.
 */
TVM_DLL void UpdateFunction(const GlobalVar& gv, BaseFunc function);

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_IR_IR_H_
