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
"""The S-TIR backend compilation pipeline."""

import tvm
from tvm import s_tir, tirx
from tvm.tirx import compilation_pipeline as tir_pipeline

tir = tirx  # alias for backward compat


def default_s_tir_pipeline(*, prepare_only=False):
    """The default tirx pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
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
            s_tir.transform.StmtSimplify(),
            s_tir.transform.InjectPermutedLayout(),
            s_tir.transform.AnnotateIrregularLoop(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.TransformMmaBufferLayout(),
            s_tir.transform.LowerOpaqueBlock(),
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
        passes.extend(
            [
                s_tir.transform.HoistIfThenElse(),
                tirx.transform.UnrollLoop(),
                s_tir.transform.RenormalizeSplitPattern(),
                tirx.transform.StmtSimplify(),
                tirx.transform.RemoveNoOp(),
                s_tir.transform.RewriteUnsafeSelect(),
            ]
        )
        # Additional passes based on configuration.
        if bool(config.get("tirx.instrument_bound_checkers", False)):
            passes.append(s_tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tirx.s_tir.ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32(True))
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                # Bind the target first so that target-specific attributes are available.
                tirx.transform.FP8ComputeLegalize(),
                # VerifyVTCMLimit must occur before LowerVtcmAlloc.
                s_tir.transform.VerifyVTCMLimit(),
                s_tir.transform.LowerVtcmAlloc(),
                tirx.transform.VerifyMemory(),
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
        if bool(config.get("tirx.s_tir.ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32())
        passes.append(s_tir.transform.MergeSharedMemoryAllocations())
        if not prepare_only:
            passes.extend(
                [
                    tirx.transform.AnnotateEntryFunc(),
                    tirx.transform.SplitHostDevice(),
                    tirx.transform.MakePackedAPI(),
                    tirx.transform.FP8StorageLegalize(),
                    tirx.transform.BF16StorageLegalize(),
                ]
            )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.LowerWarpMemory(),
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


tir_pipeline.PIPELINE_MAP["s_tir"] = default_s_tir_pipeline


def _select_default_pipeline(mod, target):
    """Select S-TIR lowering only for functions constructed in this dialect."""
    scheduled = {
        gv: func
        for gv, func in mod.functions.items()
        if isinstance(func, tirx.PrimFunc) and not func.is_tirx
    }
    if not scheduled:
        return None
    mixed = len(scheduled) != len(mod.functions)
    name = "s_tir"
    if target is not None and target.kind.name == "opencl" and "adreno" in target.keys:
        name = "adreno"
    s_pipeline = tir_pipeline.get_tir_pipeline(name, prepare_only=mixed)
    if not mixed:
        return s_pipeline

    lower_s_tir, finalize_host, finalize_device = s_pipeline
    lower_tirx, _, _ = tir_pipeline.get_tir_pipeline("tirx", prepare_only=True)

    @tvm.transform.module_pass(opt_level=0)
    def _lower_mixed(input_mod, _ctx):
        s_funcs = {gv: func for gv, func in input_mod.functions.items() if gv in scheduled}
        t_funcs = {gv: func for gv, func in input_mod.functions.items() if gv not in scheduled}
        lowered = tvm.IRModule(attrs=input_mod.attrs, global_infos=input_mod.global_infos)
        for funcs, lowering in ((s_funcs, lower_s_tir), (t_funcs, lower_tirx)):
            group = tvm.IRModule(funcs, attrs=input_mod.attrs, global_infos=input_mod.global_infos)
            lowered.update(lowering(group))
        return tir_pipeline.finalize_tir_pipeline()(lowered)

    return _lower_mixed, finalize_host, finalize_device


tir_pipeline.register_default_tir_pipeline_selector(_select_default_pipeline)
