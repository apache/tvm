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
TVM_DLL Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required);

/*!
 * \brief Inject prefetch instructions into stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectPrefetch();

// TODO(tvm-team): consolidate configs to the PassContext
/*!
 * \brief Flatten the multi-dimensional read/write
 *  to single dimensional Load/Store
 *
 * \param cache_line_size The size of CPU cache line.
 * \param create_bound_attribute Whether to create bound attributes.
 *
 * \return The Pass
 */
TVM_DLL Pass StorageFlatten(int cache_line_size, bool create_bound_attribute = false);

/*!
 * \brief Inject copy intrinsics with optional pad.
 *
 * \param pragma_key The pragma key for hint of copy.
 * \param fintrin The function with signature
 *
 *   Stmt fintrin(Buffer src,
 *                Buffer dst,
 *                Array<Expr> pad_before,
 *                Array<Expr> pad_after,
 *                Expr pad_value)
 * \return The pass.
 */
TVM_DLL Pass InjectCopyIntrin(String pragma_key, runtime::PackedFunc fintrin);

/*!
 * \brief Detect and insert sync points to co-processor.
 *
 * \return The pass.
 */
TVM_DLL Pass CoProcSync();

/*!
 * \brief Lift common attrs with attr_key to outer scope.
 *
 * \param attr_key The attribute key to be checked.
 * \return The pass.
 */
TVM_DLL Pass LiftAttrScope(String attr_key);

/*!
 * \brief partition loops in the stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass LoopPartition();

/*!
 * \brief Lower vectorization loops.
 *
 * \param enable_vectorize Whether vectorization is enabled.
 *
 * \return The pass.
 */
TVM_DLL Pass VectorizeLoop(bool enable_vectorize = true);

/*!
 * \brief Inject virtual thread loops.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectVirtualThread();

/*!
 * \brief Inject double buffer statements.
 *
 * \return The pass.
 */
TVM_DLL Pass InjectDoubleBuffer();

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
 * \brief Detect and rewrite unsafe select that contains memory access.
 *
 * \return The pass.
 */
TVM_DLL Pass RewriteUnsafeSelect();

/*!
 * \brief Run arithmetic simplifications on the statements and expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass Simplify();

/*!
 * \brief Instruments bound checkers.
 *
 * \return The pass.
 */
TVM_DLL Pass InstrumentBoundCheckers();

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
 *       f(TVMArg* packed_args, int* packed_arg_type_ids, int num_packed_args,
 *         api_arg_k, api_arg_k+1, ... api_arg_n,
 *         TVMValue* out_ret_val, int* out_ret_tcode)
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
TVM_DLL Pass RemapThreadAxis(Map<String, IterVar> axis_map);

/*!
 * \brief Lower custom datatypes.
 *
 * See tvm::datatypes::Registry for more information on adding custom datatypes.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerCustomDatatypes();

/*!
 * \brief Decorate all the function's body as device function.
 *
 * \return The pass.
 */
TVM_DLL Pass DecorateDeviceScope();

/*!
 * \brief Split the function into a host function and device functions.
 *
 * \return The pass.
 */
TVM_DLL Pass SplitHostDevice();

/*!
 * \brief skip assert stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass SkipAssert();

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 *
 * \param storage_scope The storage scope considered.
 * \return The pass.
 */
TVM_DLL Pass ThreadSync(String storage_scope);

/*!
 * \brief Lower cross thread alleduce.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerThreadAllreduce();

/*!
 * \brief Infer the TensorCore fragment infomation using tensor intrinsics
 *
 * \return The pass.
 */
TVM_DLL Pass InferFragment();

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
 * \brief Legalize bf16 typed Ops. Add a cast to fp32
 *   before Ops, then add a cast back to bf16.
 * \return The pass.
 */
TVM_DLL Pass BF16Legalize();

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
 * \brief Hoist loop-invariant IfThenElse nodes to
 * outside the elligible loops.
 *
 * \return The pass.
 */
TVM_DLL Pass HoistIfThenElse();

/*!
 * \brief Hoist loop-invariant expressions nodes to
 * outside the elligible loops.
 *
 * Can hoist conditionals used in IfThenElse statements and
 * expressions, bindings of variables in Let statements and
 * expressions, or boolean expressions, configurable to enable/disable
 * each hoistable type.
 *
 * \return The pass.
 */
TVM_DLL Pass HoistExpression();

/*!
 * \brief Lower cross-thread reduction from thread
 * bindings to intrinsic function calls.
 * \return The pass.
 */
TVM_DLL Pass LowerCrossThreadReduction();

/*!
 * \brief Lower block init stmt into IfThenElse stmts
 * \return The pass.
 */
TVM_DLL Pass LowerInitBlock();

/*!
 * \brief Locate the buffer allocation to the exact position (usually is
 *        the lca of buffer access). This pass will inject opaque block
 *        with alloc_buffers at the allocation site.
 * \return The pass.
 */
TVM_DLL Pass PlanAndUpdateBufferAllocationLocation();

/*!
 * \brief Substitute all the block vars with the PrimExprs they are bound to, indicated by the
 *        corresponding iter_values in BlockRealize, for opaque blocks by removing all
 *.        the iter_values in BlockRealize and iter_vars in Block.
 * \return The pass.
 */
TVM_DLL Pass ConvertBlocksToOpaque();

/*!
 * \brief Compact the buffer access region by removing the buffer regions that are not accessed,
 *        i.e. narrowing the buffer shape and adjust the access region if necessary.
 *
 * Before narrowing, `B` is a `[16, 16]` buffer, but only a skinny vector `B[i, 0:16]` is accessed.
 *
 *  \code
 *
 *  for i in range(0, 16):
 *      with T.block():
 *          B = T.alloc_buffer(16, 16)
 *          for j in range(0, 16):
 *              B[i, j] = A[i, j] + 1
 *          for j in range(0, 16):
 *              C[i, j] = B[i, j] + 1
 *
 *  \endcode
 *
 * This pass narrows the buffer shape and adjust its accessed region accordingly.
 * In this particular case, because only a `1 * 16` vector of `B` is accessed,
 * the pass narrows `B` to shape `[1, 16]`, and changes the access to `B[i, j]` to `B[0, j]`.
 *
 *  \code
 *
 *  for i in range(0, 16):
 *      with T.block():
 *          B = T.alloc_buffer(1, 16)
 *          for j in range(0, 16):
 *              B[0, j] = A[i, j] + 1
 *          for j in range(0, 16):
 *              C[i, j] = B[0, j] + 1
 *
 *  \endcode
 *
 *
 * \return The pass.
 */
TVM_DLL Pass CompactBufferAllocation();

/*!
 * This pass legalizes packed calls by wrapping their arguments into TVMValues
 */
TVM_DLL Pass LegalizePackedCalls();

/*!
 * \brief Remove match buffers inside the block. Also, it will validate the binding.
 * \return The pass.
 */
TVM_DLL Pass LowerMatchBuffer();

/*!
 * \brief Remove the block to ensure that the TIR can not be scheduled again.
 * \return The pass.
 */
TVM_DLL Pass LowerOpaqueBlock();

/*!
 * \brief Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional
 *        BufferLoad/BufferStore for the TIR not contains opaque block.
 * \return The pass.
 */
TVM_DLL Pass FlattenBuffer();

/*
 * \brief Flatten the multi-dimensional read/write
 *  to two dimensional texture Load/Store and realize
 *  texture buffer allocations.
 *
 * \return The Pass
 */
TVM_DLL Pass TextureFlatten();

/*
 * \brief Lower VTCM allocations
 *
 * \return The Pass
 */
TVM_DLL Pass LowerVtcmAlloc();

/*!
 * \brief Lower Async TIR primitives to DMA copy and wait builtins
 */
TVM_DLL Pass LowerAsyncDMA();

/*!
 * \brief Implements a Common Subexpression Elimination (CSE) for TIR
 *        which introduces let-in bindings for duplicated sub-expressions.
 * \param enable_cse_tir Whether common subexpression elimination is enabled.
 * \param identify_equiv_terms Whether equivalent terms should be identified.
 * \return The pass.
 */
TVM_DLL Pass CommonSubexprElimTIR(bool enable_cse_tir = true, bool identify_equiv_terms = false);

/*!
 * \brief Unify all the thread bindings for "blockIdx.x/y/z", "threadIdx.x/y/z", and
 *        "vthread.x/y/z". Before the unification, two vars that are bound to a thread axis (e.g.,
 *        "threadIdx.x") use different IterVars and variables in their AttrStmts. After the
 *        unification, we use a consolidated IterVar and a variable for them.
 * \return The pass.
 * \note `vthread` is a legacy behavior that will be deprecated, though thread bindings of `vthread`
 *       are still also unified in this pass. Please use `vthread.x`, `vthread.y` and `vthread.z`
 *       instead.
 */
TVM_DLL Pass UnifyThreadBinding();

/*!
 *  A pass to merge multiple TIR-level dynamic shared memory allocations into one
 */
TVM_DLL Pass MergeDynamicSharedMemoryAllocations();

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
 * \brief This pass transforms annotated loops into pipelined ones where producers and consumers
 * are overlapped with the information provided in loop annotations, which enables optimization
 * techniques like prefetching and pipeline parallelism.
 *
 * The pipeline scope consists of the direct children of the annotated loop (ignoring BlockRealize,
 * Block, SeqStmt), and the number of children is denoted by `n` in the documentation.
 *
 * The following annotations are used to guide the loop transformation:
 *
 * 1) Loop annotation `software_pipeline_stage` defines the pipeline stage.
 * An array of `n` integers, and each element should be in range [0, max_stage],
 * where max_stage is the maximum (inclusive) stage.
 * 2) Loop annotation `software_pipeline_order` defines the pipeline order.
 * An array of `n` integers, a permutation of [0, 1, ..., num_components - 1];
 * 3) Block annotation `double_buffer_scope` controls certain buffer sizes to allow decoupling of
 * read/write dependency. It's an integer index of the write regions of the block.
 *
 * Every annotated loop is transformed into a loop with three blocks as its direct children:
 *
 * 1) Prologue block, where components whose stage is less than `max_stage` is executed;
 *
 * 2) Body block, where all the components are executed;
 *
 * 3) Epilogue block, where only components whose stage is greater than 0 will be executed.
 * The execution order is controlled by the annotation `software_pipeline_order`,
 * and thus could be different than the original order.
 *
 * Note: For nested software pipelines, the inner software pipeline will be generated first,
 * which may affect the number of the direct children of the outer loop.
 * In this case, the annotations for the outer software
 * pipeline should include the result of the inner software pipeline,
 * which is the three blocks as discussed above.
 * Example:
 *
 * Before this pass, the TIR is:
 *
 * \code{.py}
 * @T.prim_func
 * def before_transform(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]) -> None:
 *     for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
 *         for i in T.serial(0, 16,
 *                           annotations={"software_pipeline_stage": [0, 1],
 *                                        "software_pipeline_order": [0, 1]}
 *                          ):
 *             with T.block():
 *                 T.reads(A[tx, i])
 *                 T.writes(C[tx, i])
 *                 B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
 *                 with T.block("B"):
 *                     T.reads(A[tx, i])
 *                     T.writes(B[tx, 0])
 *                     B[tx, 0] = A[tx, i] * T.float32(2)
 *                 with T.block("C"):
 *                     T.reads(B[tx, 0])
 *                     T.writes(C[tx, i])
 *                     C[tx, i] = B[tx, 0] + T.float32(1)
 * \endcode
 *
 * The TIR above annotates the loop as a two-stage pipeline with no reordering.
 * After applying this pass, the TIR is transformed into:
 *
 * \code{.py}
 * @T.prim_func
 * def after_transform(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]) -> None:
 *     for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
 *         with T.block():
 *             T.reads([A[tx, 0:16]])
 *             T.writes([C[tx, 0:16]])
 *             B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
 *             with T.block("prologue"):
 *                 T.reads([A[tx, 0]])
 *                 T.writes([B[0, tx, 0]])
 *                 B[0, tx, 0] = A[tx, 0] * T.float32(2)
 *             with T.block("body"):
 *                 T.reads([A[tx, 1:16], B[0:2, tx, 0]])
 *                 T.writes([B[0:2, tx, 0], C[tx, 0:15]])
 *                 for i in T.serial(0, 15):
 *                     with T.block("B"):
 *                         T.reads([A[tx, i + 1]])
 *                         T.writes([B[(i + 1) % 2, tx, 0]])
 *                         B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
 *                     with T.block("C"):
 *                         T.reads([B[i % 2, tx, 0]])
 *                         T.writes([C[tx, i]])
 *                         C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
 *             with T.block("epilogue"):
 *                 T.reads([B[1, tx, 0]])
 *                 T.writes([C[tx, 15]])
 *                 C[tx, 15] = B[1, tx, 0] + T.float32(1)
 * \endcode
 *
 * The original loop has two blocks, B and C, as its direct children. The loop annotations indicate
 * that block B has stage == 0, order == 0, block C has stage == 1, order == 1. Therefore, block B
 * should be executed in advance of block C by one iteration. The order 0 and 1 specifies the order
 * of block B and C inside the body block inside the result TIR.
 *
 * \return The IR transform pass.
 */
TVM_DLL Pass InjectSoftwarePipeline();

TVM_DLL Pass BindParams(const Array<runtime::NDArray>& constants);

/*!
 * \brief Pass to collect tir non-scalar constants into module's 'Constants' attribute.
 *
 * \return The pass.
 */
TVM_DLL Pass ExtractPrimFuncConstants();

/*!
 * \brief Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())
 * \return The pass.
 */
TVM_DLL Pass RenormalizeSplitPattern();

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
TVM_DLL Pass Filter(runtime::TypedPackedFunc<bool(PrimFunc)> fcond);

/*!
 * \brief Pass to rewrite global to shared memory copy on CUDA with asyncronous copy.
 * \return The pass.
 */
TVM_DLL Pass InjectPTXAsyncCopy();

/*!
 * \brief Remove the weight layout rewrite block
 * \param skip_ndarray_rewrite If True, exact rewrite of NDArray, according to the given index map,
 *  will be skipped. Only the shape of the NDArray is transformed correctly, and the content of
 *  the destination array will be filled with random values.
 *
 *  When this pass is called many times during MetaSchedule tuning, the raw data of NDArray,
 *  before and after rewrite, does not matter. Since NDArray layout rewrite, using IndexMap's
 *  MapNDArray, is currently slow, skipping the exact rewrite is sometimes necessary.
 *
 * \return The pass.
 */
TVM_DLL Pass RemoveWeightLayoutRewriteBlock(bool skip_ndarray_rewrite = false);

/*!
 * \brief Add the explicit local stage for the shared memory access on GPU.
 * \return The pass.
 */
TVM_DLL Pass ManifestSharedMemoryLocalStage();

}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_H_
