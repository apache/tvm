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
import os

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.backend.cuda import transform as cuda_transform


def default_tir_pipeline(*, target=None):
    """Return :func:`tirx_pipeline` with its standard lowering stages.

    ``lower, host_factory, device_factory = default_tir_pipeline(target=target)``
    returns the lowering pass and two finalization factories. Apply ``lower(mod)``
    to prepare the module before host/device code generation. For example, the
    simplification stage uses the loop range in this TVMScript body fragment::

        # Before
        for i in T.serial(4):
            T.call_extern("consume", i // 4, ty="void")
        # After
        for i in T.serial(4):
            T.call_extern("consume", 0, ty="void")

    ABI and target annotations may also change; see :func:`tirx_pipeline`.
    """
    return tirx_pipeline(target=target)


def tirx_pipeline(*, prepare_only=False, target=None):
    """Prepare TIRx functions and optionally lower their module-wide ABI.

    Bind ``target`` when supplied, dispatch tile instructions, lower layouts,
    flatten tensors, simplify indices and loops, and verify memory accesses.
    ``prepare_only=True`` stops before :func:`finalize_tir_pipeline`. The return
    value is ``(lowering_pass, host_finalization_factory, device_finalization_factory)``.

    Examples
    --------
    ``lower, _, _ = tirx_pipeline(target=target, prepare_only=True)`` followed by
    ``after = lower(mod)`` prepares a module. With a C target,
    this TVMScript body fragment simplifies as follows::

        # Before
        for i in T.serial(4):
            T.call_extern("consume", i // 4, ty="void")
        # After
        for i in T.serial(4):
            T.call_extern("consume", 0, ty="void")

    Tile programs require the appropriate target, layouts and dispatch rules;
    this scalar example illustrates only the simplification stage.
    """

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(
        mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext
    ) -> tvm_tirx.ir.IRModule:
        """The default lowering passes for TIR backend."""
        mod = _bind_target(mod, target)
        pass_ctx = tvm_tirx.ir.transform.PassContext.current()
        config = pass_ctx.config
        passes = [tirx.transform.TileDispatch()]
        if "TVM_TIRX_PRINT_AFTER_DISPATCH_OPS" in os.environ:
            passes.append(tvm_tirx.ir.transform.PrintIR())
        passes += [
            tirx.transform.LowerTensorLayouts(),
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
            cuda_transform.BindBackendConfig(),
            cuda_transform.SpecializeEntryHelpers(),
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
    """Split host/device functions, legalize low-precision types and lower the ABI.

    Apply ``after = finalize_tir_pipeline()(prepared_mod)`` after dialect-specific
    preparation and target binding. It also annotates the entry function,
    lowers CUDA helpers and creates packed entry wrappers where required.
    For a module containing one private function, the entry-attribute fragment
    changes as follows (the function body is omitted)::

        # Before
        T.func_attr({"target": T.target("c")})
        # After
        T.func_attr({"target": T.target("c"), "tirx.is_entry_func": True})

    Public functions with a global symbol additionally acquire the packed ABI;
    see :func:`~tvm_tirx.tirx.transform.MakePackedAPI`.
    """
    return tvm_tirx.ir.transform.Sequential(
        [
            tirx.transform.AnnotateEntryFunc(),
            tirx.transform.SplitHostDevice(),
            cuda_transform.LowerIket(),
            cuda_transform.SpecializeDeviceHelpers(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.FP8ComputeLegalize(),
            tirx.transform.MakePackedAPI(),
            tirx.transform.FP8StorageLegalize(),
            tirx.transform.BF16StorageLegalize(),
        ]
    )


def finalize_host_passes():  # pylint: disable=unused-argument
    """Lower runtime builtins and target intrinsics in host functions.

    Use ``after = finalize_host_passes()(host_mod)`` after ABI lowering and
    host/device splitting. Functions must carry their host target. For target
    ``c`` and ``x: T.float32``, an expression inside a TVMScript body changes to::

        # Before
        T.exp(x)
        # After
        T.call_pure_extern("expf", x, ty="float32")

    Packed calls and allocation builtins are also lowered before code generation.
    """
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm_tirx.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """Simplify device statements and lower target intrinsics.

    Use ``after = finalize_device_passes()(device_mod)`` after host/device
    splitting, with a target on each device function. For example, the following
    TVMScript body fragment simplifies division using the loop range::

        # Before
        for i in T.serial(4):
            T.call_extern("consume", i // 4, ty="void")
        # After
        for i in T.serial(4):
            T.call_extern("consume", 0, ty="void")

    Intrinsic expansion depends on the device target's registered lowering rules.
    """
    device_pass_list = [
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
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
