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
"""The TIR backend compilation pipeline."""

import tvm
from tvm import tir


def default_tir_pipeline():
    """The default tir pipeline used in tvm.tir.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tir.transform.LowerCrossThreadReduction(),
            tir.transform.LowerInitBlock(),
            tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tir.transform.ConvertBlocksToOpaque(),
            tir.transform.LiftThreadBinding(),
            tir.transform.ManifestSharedMemoryLocalStage(),
            tir.transform.CompactBufferAllocation(),
            tir.transform.LowerAutoCopy(),
            tir.transform.UnifyThreadBinding(),
            tir.transform.LowerMatchBuffer(),
            tir.transform.Simplify(),
            tir.transform.InjectPermutedLayout(),
            tir.transform.InjectSoftwarePipeline(),
            tir.transform.TransformMmaBufferLayout(),
            tir.transform.LowerOpaqueBlock(),
            tir.transform.FlattenBuffer(),
            tir.transform.BF16ComputeLegalize(),
            tir.transform.NarrowDataType(32),
            tir.transform.LoopPartition(),
            tir.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tir.transform.InjectVirtualThread(),
            tir.transform.InjectDoubleBuffer(),
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


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tir.transform.LowerTVMBuiltin(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerIntrin(),
        tir.transform.LowerDeviceStorageAccessInfo(),
        tir.transform.CombineContextCall(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tir.transform.LowerWarpMemory(),
        tir.transform.Simplify(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerDeviceStorageAccessInfo(),
        tir.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {
    "default": default_tir_pipeline,
}


def get_tir_pipeline(name: str = "default", **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    """
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name}," f"candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def get_default_tir_pipeline(
    target: tvm.target.Target,  # pylint: disable=unused-argument
) -> tvm.transform.Pass:
    """Get the default TIR pipeline for the given target."""
    return default_tir_pipeline()
