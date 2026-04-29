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
from tvm import s_tir, tirx
from tvm.tirx import pipeline as tir_pipeline


def default_tir_pipeline():
    """The default tirx pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            s_tir.backend.adreno.transform.TextureFlatten(),
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
            tirx.transform.Simplify(),
            s_tir.transform.InjectPermutedLayout(),
            s_tir.transform.AnnotateIrregularLoop(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.TransformMmaBufferLayout(),
            s_tir.transform.LowerOpaqueBlock(),
            s_tir.backend.adreno.transform.InjectTextureAlloc(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            s_tir.transform.LoopPartition(),
            tirx.transform.VectorizeLoop(not bool(config.get("tirx.disable_vectorize", False))),
            s_tir.transform.InjectVirtualThread(),
            s_tir.transform.InjectDoubleBuffer(),
        ]
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tirx.transform.StorageRewrite())
        if config.get("tirx.use_async_copy", False):
            passes.append(s_tir.transform.LowerAsyncDMA())
        passes.extend(
            [
                s_tir.transform.HoistIfThenElse(),
                tirx.transform.UnrollLoop(),
                s_tir.transform.RenormalizeSplitPattern(),
                tirx.transform.Simplify(),
                tirx.transform.RemoveNoOp(),
                s_tir.transform.RewriteUnsafeSelect(),
            ]
        )
        # Additional passes based on configuration.
        if bool(config.get("tirx.instrument_bound_checkers", False)):
            passes.append(s_tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32(True))
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        if bool(config.get("tirx.instrument_lwp", False)):
            passes.append(s_tir.transform.InstrumentProfileIntrinsics())
        passes.extend(
            [
                # Bind the target first so that target-specific attributes are available.
                tirx.transform.FP8ComputeLegalize(),
                # VerifyVTCMLimit must occur before LowerVtcmAlloc.
                s_tir.transform.VerifyVTCMLimit(),
                s_tir.transform.LowerVtcmAlloc(),
                tirx.transform.BindParallelLoopsToThreads(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
            ]
        )
        passes.extend(
            [
                s_tir.transform.ThreadSync("shared"),
                s_tir.transform.ThreadSync("shared.dyn"),
                s_tir.transform.ThreadSync("warp"),
                s_tir.transform.InferFragment(),
                s_tir.transform.LowerThreadAllreduce(),
            ]
        )
        if bool(config.get("tirx.use_async_copy", False)):
            passes.append(s_tir.transform.InjectPTXAsyncCopy())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32())
        passes.extend(
            [
                tirx.transform.AnnotateDeviceRegions(),
                tirx.transform.SplitHostDevice(),
                # MergeSharedMemoryAllocations must follow SplitHostDevice.
                s_tir.transform.MergeSharedMemoryAllocations(),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
                tirx.transform.LowerDeviceKernelLaunch(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline


tir_pipeline.PIPELINE_MAP["adreno"] = default_tir_pipeline
