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
 * \file tvm/s_tir/transform.h
 * \brief S-TIR specific transformation passes.
 */
#ifndef TVM_S_TIR_TRANSFORM_H_
#define TVM_S_TIR_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace s_tir {

/*!
 * \brief Renew the definition nodes for a TIR, including Var, Buffer and IterVar.
 *        This pass works as a simple DeepCopy to duplicate a function with different Vars and
 *        Buffers but the same behavior
 * \param func The input PrimFunc.
 * \return The renewed func.
 */
TVM_DLL tir::PrimFunc RenewDefs(const tir::PrimFunc& func);

namespace transform {

using tir::transform::CreatePrimFuncPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;

/*!
 * \brief Canonicalize loop to start from zero .
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeLoop();

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
 * \brief Lift the same thread bindings to their LCA loops
 * \return The pass.
 */
TVM_DLL Pass LiftThreadBinding();

/*!
 * \brief Compact the buffer access region by removing the buffer regions that are not accessed,
 *        i.e. narrowing the buffer shape and adjust the access region if necessary.
 *
 * Before narrowing, `B` is a `[16, 16]` buffer, but only a skinny vector `B[i, 0:16]` is accessed.
 *
 *  \code
 *
 *  for i in range(0, 16):
 *      with T.sblock():
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
 *      with T.sblock():
 *          B = T.alloc_buffer(1, 16)
 *          for j in range(0, 16):
 *              B[0, j] = A[i, j] + 1
 *          for j in range(0, 16):
 *              C[i, j] = B[0, j] + 1
 *
 *  \endcode
 *
 * \param is_strict ensure the compacted shape always smaller than the original shape.
 *   otherwise it allows to grow the shape to match actual accessed buffer regions.
 * \return The pass.
 */
TVM_DLL Pass CompactBufferAllocation(bool is_strict = true);

/*!
 * \brief Remove match buffers inside the block. Also, it will validate the binding.
 * \return The pass.
 */
TVM_DLL Pass LowerMatchBuffer();

/*!
 * \brief Inject permuted layout for shared memory.
 * \return The pass.
 */
TVM_DLL Pass InjectPermutedLayout();

/*!
 * \brief Transform Mma scope (m16n8k8.matrixA/B/C) to local scope with layout transformation.
 * \return The pass.
 */
TVM_DLL Pass TransformMmaBufferLayout();

/*!
 * \brief Remove the block to ensure that the TIR can not be scheduled again.
 * \return The pass.
 */
TVM_DLL Pass LowerOpaqueBlock();

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
 * \brief This pass transforms annotated loops into pipelined ones where producers and consumers
 * are overlapped with the information provided in loop annotations, which enables optimization
 * techniques like prefetching and pipeline parallelism.
 *
 * The pipeline scope consists of the direct children of the annotated loop (ignoring SBlockRealize,
 * SBlock, SeqStmt), and the number of children is denoted by `n` in the documentation.
 *
 * The following annotations are used to guide the loop transformation:
 *
 * 1) Loop annotation `software_pipeline_stage` defines the pipeline stage.
 * An array of `n` integers, and each element should be in range [0, max_stage],
 * where max_stage is the maximum (inclusive) stage.
 * 2) Loop annotation `software_pipeline_order` defines the pipeline order.
 * An array of `n` integers, a permutation of [0, 1, ..., num_components - 1];
 * 3) SBlock annotation `double_buffer_scope` controls certain buffer sizes to allow decoupling of
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
 *
 * \return The IR transform pass.
 */
TVM_DLL Pass InjectSoftwarePipeline();

/*!
 * \brief Automatically do memory optimizations for auto copy blocks
 * \return The pass.
 */
TVM_DLL Pass LowerAutoCopy();

/*!
 * \brief Add the explicit local stage for the shared memory access on GPU.
 * \return The pass.
 */
TVM_DLL Pass ManifestSharedMemoryLocalStage();

/*! \brief Annotate irregular loop mark. */
TVM_DLL Pass AnnotateIrregularLoop();

/*!
 * \brief partition loops in the stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass LoopPartition();

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
 * \brief Hoist loop-invariant IfThenElse nodes to
 * outside the eligible loops.
 *
 * \param variant The variant of the pass.
 *        variant can have any one of following values ["basic", ""(Default)].
 * \return The pass.
 */
TVM_DLL Pass HoistIfThenElse(tvm::ffi::String variant = "");

/*!
 * \brief Hoist loop-invariant expressions to outside the eligible loops.
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
 * \brief Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv()).
 * \return The pass.
 */
TVM_DLL Pass RenormalizeSplitPattern();

/*!
 * \brief Detect and rewrite unsafe select that contains memory access.
 * \return The pass.
 */
TVM_DLL Pass RewriteUnsafeSelect();

/*!
 * \brief Instruments bound checkers.
 * \return The pass.
 */
TVM_DLL Pass InstrumentBoundCheckers();

/*!
 * \brief Rewrite global to local memory copy on CUDA with ldg32 instruction.
 * \param enable_inject Whether to enable injection.
 * \return The pass.
 */
TVM_DLL Pass InjectPTXLDG32(bool enable_inject = true);

/*!
 * \brief Insert intrinsic calls to instrument function and loop level profiling.
 * \return The pass.
 */
TVM_DLL Pass InstrumentProfileIntrinsics();

/*!
 * \brief Lower VTCM allocations.
 * \return The pass.
 */
TVM_DLL Pass LowerVtcmAlloc();

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 * \param storage_scope The storage scope considered.
 * \return The pass.
 */
TVM_DLL Pass ThreadSync(tvm::ffi::String storage_scope);

/*!
 * \brief Infer the TensorCore fragment information using tensor intrinsics.
 * \return The pass.
 */
TVM_DLL Pass InferFragment();

/*!
 * \brief Lower cross thread allreduce.
 * \return The pass.
 */
TVM_DLL Pass LowerThreadAllreduce();

/*!
 * \brief Lower Async TIR primitives to DMA copy and wait builtins.
 * \return The pass.
 */
TVM_DLL Pass LowerAsyncDMA();

/*!
 * \brief Rewrite global to shared memory copy on CUDA with asynchronous copy.
 * \return The pass.
 */
TVM_DLL Pass InjectPTXAsyncCopy();

/*!
 * \brief Merge multiple TIR-level shared memory allocations into one.
 * \return The pass.
 */
TVM_DLL Pass MergeSharedMemoryAllocations();

/*!
 * \brief Set default thread bindings for GPU PrimFuncs.
 * \return The pass.
 */
TVM_DLL Pass DefaultGPUSchedule();

/*!
 * \brief Remove weight layout rewrite block before benchmark.
 * \param skip_tensor_rewrite Whether to skip tensor rewrite.
 * \return The pass.
 */
TVM_DLL Pass RemoveWeightLayoutRewriteBlock(bool skip_tensor_rewrite = false);

/*!
 * \brief Remove stores of tir::builtin::undef.
 * \return The pass.
 */
TVM_DLL Pass RemoveStoreUndef();

/*!
 * \brief Decorate all the function's body as device function.
 * \return The pass.
 */
TVM_DLL Pass DecorateDeviceScope();

/*!
 * \brief Eliminate branches by leveraging buffer assumptions (T.assume).
 * \return The pass.
 */
TVM_DLL Pass UseAssumeToReduceBranches();

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_TRANSFORM_H_
