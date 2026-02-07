# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The TIR backend compilation pipeline for Adreno"""

import tvm
from tvm import tir, s_tir
from tvm.tir import pipeline as tir_pipeline


def default_tir_pipeline():
    """The default tir pipeline used in tvm.tir.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tir.backend.adreno.transform.TextureFlatten(),
            s_tir.transform.CanonicalizeLoop(),
            s_tir.transform.LowerCrossThreadReduction(),
            s_tir.transform.LowerInitBlock(),
            s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            s_tir.transform.ConvertBlocksToOpaque(),
            s_tir.transform.LiftThreadBinding(),
            s_tir.transform.ManifestSharedMemoryLocalStage(),
            s_tir.transform.CompactBufferAllocation(),
            s_tir.transform.LowerAutoCopy(),
            s_tir.transform.UnifyThreadBinding(),
            s_tir.transform.LowerMatchBuffer(),
            tir.transform.Simplify(),
            s_tir.transform.InjectPermutedLayout(),
            s_tir.transform.AnnotateIrregularLoop(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.TransformMmaBufferLayout(),
            s_tir.transform.LowerOpaqueBlock(),
            tir.backend.adreno.transform.InjectTextureAlloc(),
            tir.transform.FlattenBuffer(),
            tir.transform.BF16ComputeLegalize(),
            tir.transform.NarrowDataType(32),
            s_tir.transform.LoopPartition(),
            tir.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            s_tir.transform.InjectVirtualThread(),
            s_tir.transform.InjectDoubleBuffer(),
        ]
        if not bool(config.get("tir.disable_storage_rewrite", False)):
            passes.append(tir.transform.StorageRewrite())
        if config.get("tir.use_async_copy", False):
            passes.append(tir.transform.LowerAsyncDMA())
        passes.extend(
            [
                tir.transform.HoistIfThenElse(),
                tir.transform.UnrollLoop(),
                tir.transform.RenormalizeSplitPattern(),
                tir.transform.Simplify(),
                tir.transform.RemoveNoOp(),
                tir.transform.RewriteUnsafeSelect(),
            ]
        )
        # Additional passes based on configuration.
        if bool(config.get("tir.instrument_bound_checkers", False)):
            passes.append(tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tir.ptx_ldg32", False)):
            passes.append(tir.transform.InjectPTXLDG32(True))
        passes.append(
            tir.transform.CommonSubexprElimTIR(
                not bool(config.get("tir.disable_cse_tir", False)),
                bool(config.get("tir.enable_equiv_terms_in_cse_tir", False)),
            )
        )
        if bool(config.get("tir.instrument_lwp", False)):
            passes.append(tir.transform.InstrumentProfileIntrinsics())
        passes.extend(
            [
                # Bind the target first so that target-specific attributes are available.
                tir.transform.FP8ComputeLegalize(),
                # VerifyVTCMLimit must occur before LowerVtcmAlloc.
                tir.transform.VerifyVTCMLimit(),
                tir.transform.LowerVtcmAlloc(),
                tir.transform.VerifyMemory(),
                tir.transform.AnnotateEntryFunc(),
            ]
        )
        if bool(config.get("tir.detect_global_barrier", False)):
            passes.append(tir.transform.ThreadSync("global"))
        passes.extend(
            [
                tir.transform.ThreadSync("shared"),
                tir.transform.ThreadSync("shared.dyn"),
                tir.transform.ThreadSync("warp"),
                tir.transform.InferFragment(),
                tir.transform.LowerThreadAllreduce(),
            ]
        )
        if bool(config.get("tir.use_async_copy", False)):
            passes.append(tir.transform.InjectPTXAsyncCopy())
        if bool(config.get("tir.ptx_ldg32", False)):
            passes.append(tir.transform.InjectPTXLDG32())
        passes.extend(
            [
                tir.transform.AnnotateDeviceRegions(),
                tir.transform.SplitHostDevice(),
                # MergeSharedMemoryAllocations must follow SplitHostDevice.
                tir.transform.MergeSharedMemoryAllocations(),
                tir.transform.MakePackedAPI(),
                tir.transform.FP8StorageLegalize(),
                tir.transform.BF16StorageLegalize(),
                tir.transform.LowerDeviceKernelLaunch(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline


tir_pipeline.PIPELINE_MAP["adreno"] = default_tir_pipeline
