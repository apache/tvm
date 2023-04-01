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
#ifndef TVM_RELAX_OP_BASIC_H_
#define TVM_RELAX_OP_BASIC_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

// (TVM-TOOL) cc_op begin decl/basic/*
/*!
 * Call a VM builtin PackedFunc in destination-passing style (DPS). The difference between
 *     `call_builtin_with_ctx` and `call_dps_packed` is that `call_builtin_with_ctx` takes
 *     an extra argument `ctx` at the beginning of the arguments, which is the context of the
 *     current VM.
 * \param func The function being called.
 * \param args The arguments to the packed func. Always a Relax Tuple expression.
 * \param sinfo_args The StructInfo of the arguments.
 * \return The created call node.
 */
relax::Call call_builtin_with_ctx(relax::ExternFunc func, relax::Tuple args,
                                  relax::StructInfo sinfo_args);
/*!
 * Call a PackedFunc in destination-passing style (DPS).
 * \param func The function being called.
 * \param args The arguments to the packed func. Always a Relax Tuple expression whose length
 * indicates * the number `n + m` in the example. \param out_sinfo The StructInfo of the output.
 * \return The created call node.
 */
relax::Call call_dps_packed(relax::ExternFunc func, relax::Tuple args, relax::StructInfo out_sinfo);
/*!
 * Call a PrimFunc in TensorIR, and return its output using a special calling convention
 *     called destination-passing style (DPS) in TVM.
 * \param gvar The global variable that points to the function being called.
 * \param args The arguments to the function. Always a Relax Tuple expression whose length indicates
 * * the number `n` in the example the number of arguments. \param out_sinfo The StructInfo of the
 * output. It is used to infer the number of outputs, and indicates * the number `m` in the example.
 * \param tir_vars The TIR variables to be used with the call. They are usually used for symbolic
 * shapes. \return The created call node.
 */
relax::Call call_tir(tvm::GlobalVar gvar, relax::Tuple args, relax::StructInfo out_sinfo,
                     Optional<relax::Expr> tir_vars);
/*!
 * Invoke a closure.
 * \param closure The closure being invoked.
 * \param args The arguments to the closure. Always a Relax Tuple expression.
 * \param sinfo_args The StructInfo of the output
 * \return The created call node.
 */
relax::Call invoke_closure(relax::Expr closure, relax::Tuple args, relax::StructInfo sinfo_args);
/*!
 * Create a closure with free variables and return the closure.
 * \param func The function being called.
 * \param args The arguments to the packed func. Always a Relax Tuple expression.
 * \return The created call node.
 */
relax::Call make_closure(tvm::GlobalVar func, relax::Tuple args);
/*!
 * Create a call node that represents a null value object.
 * \return The created call node.
 */
relax::Call null_value();
/*!
 * Get shape of a tensor. It gets TensorStructInfo and returns ShapeStructInfo
 * \param expr The input expression of TensorStructInfo.
 * \return The created call node.
 */
relax::Call shape_of(relax::Expr expr);
// (TVM-TOOL) cc_op end decl/basic/*

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_BASIC_H_
