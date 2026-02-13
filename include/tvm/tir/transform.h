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
 * \file tvm/tir/transform.h
 * \brief TIR specific transformation passes.
 */
#ifndef TVM_TIR_TRANSFORM_H_
#define TVM_TIR_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include <string>
#include <vector>

namespace tvm {
namespace tir {
namespace transform {

using tvm::transform::CreateModulePass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassContextNode;
using tvm::transform::PassInfo;
using tvm::transform::PassInfoNode;
using tvm::transform::PassNode;
using tvm::transform::Sequential;

/*
 * \brief Create a function pass that optimizes PrimFuncs.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreatePrimFuncPass(std::function<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
                                int opt_level, ffi::String name,
                                tvm::ffi::Array<ffi::String> required, bool traceable = false);

/*!
 * \brief Lower vectorization loops.
 *
 * \param enable_vectorize Whether vectorization is enabled.
 *
 * \return The pass.
 */
TVM_DLL Pass VectorizeLoop(bool enable_vectorize = true);

/*!
 * \brief Rewrite storage allocation pattern.
 *  Moves the allocation to outer most possible scope.
 *  Trying to share space between allocations to make
 *  a static allocation plan when possible.
 *
 * \return The pass.
 */
TVM_DLL Pass StorageRewrite();

/*!
 * \brief unroll the constant loop marked by unroll.
 * This pass also automatically attach pragma unroll tag to loops which meets the standard.
 *
 * \return The pass.
 */
TVM_DLL Pass UnrollLoop();

/*!
 * \brief Remove No Op from the Stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass RemoveNoOp();

/*!
 * \brief Run arithmetic simplifications on the statements and expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass Simplify();

/*!
 * \brief Convert an IRModule to be SSA form.
 *
 * This pass handles cases where the same tir::Var appears in
 * multiple functions within the same module.  For example, after
 * extracting a fragment from one function into another, where the
 * same `tir::Var` may be defined both as within the body of the
 * original function, and as a parameter within the hoisted function.
 *
 * \return The pass.
 */
TVM_DLL Pass ConvertSSA();

/*!
 * \brief Transform the high-level PrimFunc to a low-level version
 *        that can be used as an API function.
 *
 *
 *  The main task of this function is to create code to :
 *   - Map the values in the api_args to Var that is required by body.
 *   - Insert assertions to check type/value of the passed arguments.
 *
 * \note
 *  The function signature have two cases
 *
 *  let num_packed_args = len(api_args);
 *
 *  if num_packed_args is zero:
 *     f()
 *
 *  if num_packed_args is not zero:
 *       f(void *, TVMFFIAny* packed_args, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n,
 *         TVMFFIAny* out_ret_val)
 *
 *       where n == len(api_args), k == num_packed_args
 *
 * \return The pass.
 */
TVM_DLL Pass MakePackedAPI();

/*!
 * \brief Transform the high-level PrimFunc to a C signature that can be used
 *   to call the operator directly.
 *
 *  The main task of this function is to create code that maps the values in the
 *  api_args to Var that is required by body
 *
 * \return The pass.
 */
TVM_DLL Pass MakeUnpackedAPI();

/*!
 * \brief Remap the thread axis
 *
 *  This can be used to get equivalent program which uses
 *  threadIdx.y in place of threadIdx.x by passing
 *  {"threadIdx.x": thread_axis("threadIdx.y")}
 *
 *
 * \return The pass.
 */
TVM_DLL Pass RemapThreadAxis(ffi::Map<ffi::String, IterVar> axis_map);

/*!
 * \brief Lower custom datatypes.
 *
 * See tvm::datatypes::Registry for more information on adding custom datatypes.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerCustomDatatypes();

/*!
 * \brief Annotate locations that should be run on the device
 *
 * Insert `AttrStmt` nodes specifying a target on which regions within
 * the PrimFunc should be executed.  Only modifies functions that have
 * a `tvm::attr::kTarget` attribute, and where that target defines a
 * host.
 *
 * \return The pass.
 */
TVM_DLL Pass AnnotateDeviceRegions();

/*!
 * \brief Split the function into a host function and device functions.
 *
 * The resulting host-side function will keep the same
 * `tvm::attr::kTarget` attribute (e.g. `T.target("cuda",
 * host=T.target("llvm"))`).  This ensures that `MakePackedAPI` knows
 * which device type should be used for the input buffers.
 *
 * The resulting device-side function will
 * have the host stripped from its target attribute
 * (e.g. `T.target("cuda")`).
 *
 * \return The pass.
 */
TVM_DLL Pass SplitHostDevice();

/*!
 * \brief Lower cross-device function calls.
 *
 * Prior to this pass, host to device calls are represented as
 * subroutine calls, with environment parameters (e.g. env_thread)
 * specified internally.  The device function is an internal function,
 * without a `tvm::attr::kGlobalSymbol` attribute.
 *
 * After this pass, host to device calls are represented as
 * tvm_call_packed built-in.  The device function is an
 * externally-exposed function, with a non-empty
 * `tvm::attr::kGlobalSymbol` attribute.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerDeviceKernelLaunch();

/*!
 * \brief skip assert stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass SkipAssert();

/*!
 * \brief This annotation is for nodes to be disabled for builtin lowering
 */
static constexpr const char* kDisableLowerTVMBuiltin = "disable_lower_builtin";

/*!
 * \brief Lower builtin intrinsics.
 * \return The pass.
 */
TVM_DLL Pass LowerTVMBuiltin();

/*!
 * \brief Lower the target specific function intrinsics in each of the function.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerIntrin();

/*!
 * \brief Lower warp memory access to low-level device related function calls.
 * \return The pass.
 */
TVM_DLL Pass LowerWarpMemory();

/*!
 * \brief Lower attached storage access information on device.
 *
 * \note Run this pass after all storage access analysis finish.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerDeviceStorageAccessInfo();

/*!
 * \brief Combine context calls in the host function.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineContextCall();

/*!
 * \brief Narrow down PrimExpr datatype in stmt to target_bits.
 *
 * \param target_bits The target bits
 *
 * \note Run this pass after storage flatten.
 * \return The pass.
 */
TVM_DLL Pass NarrowDataType(int target_bits);

/*!
 * \brief Force to narrow down indexing expressions and integer buffers to int32 dtype.
 *
 * \return The pass.
 * \note This pass should not be used in default cases.
 */
TVM_DLL Pass ForceNarrowIndexToInt32();

/*!
 * \brief Legalize bf16 compute Ops. Add a cast to fp32
 *   before Ops, then add a cast back to bf16.
 * \return The pass.
 */
TVM_DLL Pass BF16ComputeLegalize();

/*!
 * \brief Legalize fp8 compute Ops. Add a cast to fp16/fp32
 *   before Ops, then add a cast back to fp8.
 * \param promote_dtype The data type used for type promotion, defaults to float16
 * \note Must be run after BindTarget, as it relies on target attributes for PrimFuncs
 * \return The pass.
 */
TVM_DLL Pass FP8ComputeLegalize(ffi::String promote_dtype = "float16");

/*!
 * \brief Legalize bf16 storage types to u16.
 * \return The pass.
 */
TVM_DLL Pass BF16StorageLegalize();

/*!
 * \brief Legalize fp8 storage types to u8.
 * \note Must be run after BindTarget, as it relies on target attributes for PrimFuncs
 * \return The pass.
 */
TVM_DLL Pass FP8StorageLegalize();

/*!
 * \brief Inline calls to private functions
 *
 * \return The pass.
 */
TVM_DLL Pass InlinePrivateFunctions();

/*!
 * \brief Rewrite the pointer content type of arguments,
 *  as well as Alloc internal to the function to use
 *  the most frequently accessed type for load/store
 *  to avoid pointer casting in backend when possible.
 *
 * \return The pass.
 */
TVM_DLL Pass PointerValueTypeRewrite();

/*!
 * \brief Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional
 *        BufferLoad/BufferStore for the TIR not contains opaque block.
 * \return The pass.
 */
TVM_DLL Pass FlattenBuffer();

/*!
 * \brief Implements a Common Subexpression Elimination (CSE) for TIR
 *        which introduces let-in bindings for duplicated sub-expressions.
 * \param enable_cse_tir Whether common subexpression elimination is enabled.
 * \param identify_equiv_terms Whether equivalent terms should be identified.
 * \return The pass.
 */
TVM_DLL Pass CommonSubexprElimTIR(bool enable_cse_tir = true, bool identify_equiv_terms = false);

/*!
 * \brief This pass is post-scheduling pass to convert all
 *        Parallel For loops to Serial ones. This is run
 *        to attain lesser memory and/or executor/backend
 *        does not support parallel launch of For loops.
 * \return The pass.
 */
TVM_DLL Pass ConvertForLoopsToSerial();

/*!
 * \brief This is the unified static memory planner pass that will
 * plan for memory intra- and inter- PrimFuncs together. The pass
 * requires all the function to be PrimFuncs including the main.
 * \return The pass.
 */
TVM_DLL Pass UnifiedStaticMemoryPlanner();

/*!
 * \brief Annotate a PrimFunc with a given target.
 * \return The pass.
 */
TVM_DLL Pass BindTarget(Target target);

/*!
 * \brief Set a PrimFunc as the entry point if it is only function in IRModule.
 * \return The pass.
 */
TVM_DLL Pass AnnotateEntryFunc();

/*!
 * \brief Filter PrimFuncs with a given condition.
 * \return The pass.
 */
TVM_DLL Pass Filter(ffi::TypedFunction<bool(PrimFunc)> fcond);

/*!
 * \brief Remove the weight layout rewrite block
 * \param skip_tensor_rewrite If True, exact rewrite of Tensor, according to the given index map,
 *  will be skipped. Only the shape of the Tensor is transformed correctly, and the content of
 *  the destination array will be filled with random values.
 *
 *  When this pass is called many times during MetaSchedule tuning, the raw data of Tensor,
 *  before and after rewrite, does not matter. Since Tensor layout rewrite, using IndexMap's
 *  MapTensor, is currently slow, skipping the exact rewrite is sometimes necessary.
 *
 * \return The pass.
 */
}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_H_
