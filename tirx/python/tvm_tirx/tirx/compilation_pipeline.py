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

import inspect

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.backend.cuda import transforms as cuda_transforms


def default_tir_pipeline(*, target=None):
    """The default pipeline for TIRx primitive functions."""
    return tirx_pipeline(target=target)


def tirx_pipeline(*, prepare_only=False, target=None):
    """The TIRX pipeline used in tvm_tirx.tirx.compile"""

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> tvm_tirx.ir.IRModule:
        """The default lowering passes for TIR backend."""
        mod = _bind_target(mod, target)
        pass_ctx = tvm_tirx.ir.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tirx.transform.LowerTIRx(),
            tirx.transform.StmtSimplify(),
            tirx.transform.FlattenTensor(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tirx.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.VerifyMemory(),
                tirx.transform.LowerThreadAllreduce(),
            ]
        )
        mod = tvm_tirx.ir.transform.Sequential(passes)(mod)
        return mod if prepare_only else finalize_tir_pipeline()(mod)

    return _pipeline, finalize_host_passes, finalize_device_passes


def _bind_target(mod, target):
    if target is None:
        return mod
    return tvm_tirx.ir.transform.Sequential(
        [
            tirx.transform.BindTarget(target),
            cuda_transforms.BindBackendConfig(),
            cuda_transforms.SpecializeEntryHelpers(),
        ]
    )(mod)


def _bind_legacy_target(mod, target):
    for prepare in _LEGACY_TARGET_PREPARERS:
        mod = prepare(mod)
    return _bind_target(mod, target)


def _with_target_binding(pipeline, target, *, original_mod=None, prepared_mod=None):
    """Adapt factories that expect their input to have target metadata already."""
    if target is None:
        return pipeline
    lowering, finalize_host, finalize_device = pipeline
    # IRModule supports in-place updates; only reuse an unchanged selection input.
    original_fields = (
        (original_mod.functions, original_mod.attrs, original_mod.global_infos)
        if original_mod is not None
        else ()
    )

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod, _ctx):
        if (
            prepared_mod is not None
            and mod.same_as(original_mod)
            and all(
                current is None if original is None else original.same_as(current)
                for original, current in zip(
                    original_fields, (mod.functions, mod.attrs, mod.global_infos)
                )
            )
        ):
            # Custom module passes may update their input container in place.
            mod = tvm_tirx.IRModule(
                prepared_mod.functions,
                attrs=prepared_mod.attrs,
                global_infos=prepared_mod.global_infos,
            )
        else:
            mod = _bind_legacy_target(mod, target)
        return lowering(mod)

    return _pipeline, finalize_host, finalize_device


def finalize_tir_pipeline():
    """Lower module-wide calling conventions after all dialects are prepared."""
    return tvm_tirx.ir.transform.Sequential(
        [
            tirx.transform.AnnotateEntryFunc(),
            tirx.transform.SplitHostDevice(),
            cuda_transforms.LowerIket(),
            cuda_transforms.SpecializeDeviceHelpers(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.FP8ComputeLegalize(),
            tirx.transform.MakePackedAPI(),
            tirx.transform.FP8StorageLegalize(),
            tirx.transform.BF16StorageLegalize(),
        ]
    )


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm_tirx.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm_tirx.ir.transform.Sequential(device_pass_list)


def finalize_device_passes_tirx():  # pylint: disable=unused-argument
    """The TIRx finalization passes for TIR backend."""
    device_pass_list = [tirx.transform.LowerIntrin()]
    return tvm_tirx.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {"default": default_tir_pipeline, "tirx": tirx_pipeline}


def register_tir_pipeline(name: str, pipeline_factory) -> None:
    """Register a factory returning lowering, host, and device finalization passes.

    Factories accepting an explicit ``target`` keyword own target binding. They
    must lower S-TIR thread bindings before binding the complete module and
    specializing helpers. Other factories receive inputs with targets already
    bound; they cannot classify calls inside unlowered S-TIR thread bindings.
    """

    PIPELINE_MAP[name] = pipeline_factory


def get_tir_pipeline(name: str | None = None, *, target=None, **kwargs):
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline

    target : Optional[tvm_tirx.target.Target]
        Target to bind during lowering, after thread launches become explicit.
    """
    if name is None:
        name = "default"
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name},candidates are {list(PIPELINE_MAP.keys())}"
        )
    factory = PIPELINE_MAP[name]
    if "target" in inspect.signature(factory).parameters:
        return factory(target=target, **kwargs)
    return _with_target_binding(factory(**kwargs), target)


# Dialects register their own default selection without introducing imports here.
_DEFAULT_PIPELINE_SELECTORS = []
# Dialects validate their own scheduling constructs before legacy target binding.
_LEGACY_TARGET_PREPARERS = []


def register_default_tir_pipeline_selector(selector) -> None:
    """Register a callback returning a pipeline tuple or None for (module, target).

    Selectors accepting ``target_to_bind`` pass it to the selected pipeline
    factory. Existing two-argument selectors inspect the bound module, which is
    reused by the selected pipeline without binding it a second time.
    """
    _DEFAULT_PIPELINE_SELECTORS.append(selector)


def get_default_tir_pipeline(
    target: tvm_tirx.target.Target | None,
    mod: tvm_tirx.ir.IRModule | None = None,
    *,
    target_to_bind: tvm_tirx.target.Target | None = None,
):
    """Get the default pipeline for the input module and target."""
    original_mod = mod
    prepared_mod = None
    remaining_target = target_to_bind
    pipeline = None
    if mod is not None:
        for selector in _DEFAULT_PIPELINE_SELECTORS:
            accepts_target = "target_to_bind" in inspect.signature(selector).parameters
            if not accepts_target and remaining_target is not None:
                mod = _bind_legacy_target(mod, remaining_target)
                prepared_mod = mod
                remaining_target = None
            pipeline = (
                selector(mod, target, target_to_bind=remaining_target)
                if accepts_target
                else selector(mod, target)
            )
            if pipeline is not None:
                break
    if pipeline is None:
        pipeline = get_tir_pipeline("default", target=remaining_target)
    if prepared_mod is not None:
        return _with_target_binding(
            pipeline, target_to_bind, original_mod=original_mod, prepared_mod=prepared_mod
        )
    return pipeline
