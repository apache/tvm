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
 * \file tvm/tirx/analysis.h
 * \brief Analysis utilities and passes for TIR.
 */
#ifndef TVM_TIRX_ANALYSIS_H_
#define TVM_TIRX_ANALYSIS_H_

#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt.h>

#include <string>

namespace tvm {

namespace tirx {

/*!
 * \brief Verify if memory accesses are legal for a specific target device type.
 *
 *  In the case that tgt is cuda, if not all workload is bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal. This pass performs verification for this case.
 *
 * \param func The function to be verified.
 * \return Success of memory verification.
 */
TVM_TIRX_DLL bool VerifyMemory(const Function& func);

/*!
 * \brief Calculate the expression complexity based on number of symbols it contains.
 * \param expr The expr to be calculated.
 */
TVM_TIRX_DLL size_t CalculateExprComplexity(const PrimExpr& expr);

/*!
 * \brief Check whether a TIRx function is well formed.
 * \param func The Function to be verified.
 * \param assert_mode Whether to raise on invalid IR (default: true).
 * \return True for valid IR; false for invalid IR when assert_mode is false.
 */
TVM_TIRX_DLL bool CheckWellFormed(const Function& func, bool assert_mode = true);

/*!
 * \brief Check whether a TIRx module is well formed.
 * \param mod The IRModule to be verified.
 * \param assert_mode Whether to raise on invalid IR (default: true).
 * \return True for valid IR; false for invalid IR when assert_mode is false.
 */
TVM_TIRX_DLL bool CheckWellFormed(const IRModule& mod, bool assert_mode = true);

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tirx::attr::kIsEntryFunc.name`, whose name is `main` or being the only Function.
 * \param mod The IRModule to find the entry function.
 * \param result_g_var The result GlobalVar of the entry function.
 * \return The entry function.
 */
TVM_TIRX_DLL const FunctionNode* FindEntryFunc(const IRModule& mod, GlobalVar* result_g_var);

// Pass variants of verification analysis
// directly throws RuntimeError when verification fails.
namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassContext;

/*!
 * \brief Pass variant of VerifyMemory.
 *
 * \returns The pass.
 * \sa tvm::tirx::VerifyMemory
 */
TVM_TIRX_DLL Pass VerifyMemory();

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_ANALYSIS_H_
