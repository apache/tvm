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
"""Pre-defined pipelines.

oRelax enables flexible pipeline optimizations before min build.
This namespace offers a pre-defined collection that can be used
as it is or serves as a basis to do further composition.
"""
# pylint: disable=unused-argument
from typing import Union

import tvm
from tvm import meta_schedule as ms

from . import backend, transform


def zero_pipeline(*, enable_warning: bool = False):
    """Wrapper function that returns the zero pipeline.

    Parameters
    ----------
    enable_warning : bool
        A boolean value indicating if to print warnings
        * in LegalizeOps pass, for CallNode whose op's legalization function is
        not registered,
        * in MetaScheduleApplyDatabase pass, for TIR functions now showing up in
        the database. By default we don't print warning.
    """

    @tvm.transform.module_pass(opt_level=0)
    def f_zero_pipeline(mod: tvm.ir.IRModule, ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """Pipeline that applies pre-tuned logs.

        Parameters
        ----------
        mod : tvm.ir.IRModule
            Input IRModule.

        ctx : tvm.transform.PassContext
            The pass context

        Returns
        -------
        mod: tvm.ir.IRModule
            The result transformed module.
        """
        seq = tvm.transform.Sequential(
            [
                transform.LegalizeOps(enable_warning=enable_warning),
                transform.AnnotateTIROpPattern(),
                transform.FoldConstant(),
                transform.FuseOps(),
                transform.FuseTIR(),
            ]
        )
        mod = seq(mod)
        if ms.Database.current():
            mod = transform.MetaScheduleApplyDatabase(enable_warning=enable_warning)(mod)
        return mod

    return f_zero_pipeline


def default_build_pipeline():
    """The default compilation pipeline used in tvm.compile"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                backend.DispatchSampling(),
                backend.DispatchSortScan(),
                transform.LegalizeOps(),
                transform.RewriteDataflowReshape(),
                transform.ToNonDataflow(),
                transform.RemovePurityChecking(),
                transform.CallTIRRewrite(),
                transform.StaticPlanBlockMemory(),
                transform.RewriteCUDAGraph(),
                transform.LowerAllocTensor(),
                transform.KillAfterLastUse(),
                transform.LowerRuntimeBuiltin(),
                transform.ComputePrimValue(),
                transform.VMShapeLower(),
                transform.AttachGlobalSymbol(),
            ],
        )
        mod = seq(mod)
        return mod

    return _pipeline


def static_shape_tuning_pipeline(
    total_trials: int,
    target: Union[str, tvm.target.Target],
    work_dir: str = "tuning_logs",
    cpu_weight_prepack: bool = False,
):
    """Tune the static shape model and store the log to database.

    Parameters
    ----------
    total_trials : int
        Total number of trials to run.

    target : Union[str, tvm.target.Target]
        The target device to tune the model.

    work_dir : str
        The directory to store the tuning logs.

    cpu_weight_prepack : bool
        Whether to enable the cpu weight prepack feature.

    Note
    ----
    `cpu_weight_prepack` is expected to be `True` when running on CPU for
    better performance. However, it requires an explicit layout transformation
    step by calling the corresponding vm function, which changes the interface
    of deployment. So we disable it by default. Here is an example to enable it:

    .. code-block:: python

        mod = relax.pipeline.static_shape_tuning_pipeline(
            total_trials=1000,
            target="llvm -num-cores 16",
            work_dir="tuning_logs",
            cpu_weight_prepack=True,
        )(mod)

        ex = tvm.compile(mod, target=target)
        vm = relax.VirtualMachine(ex, device=tvm.cpu())

        # Transform the params using the vm function
        # the name should be f"{func_name}_transform_params"
        params = vm["main_transform_params"](params["main"])

        input_data = tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"))
        out = vm["main"](input_data, *params).numpy()
    """

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        if cpu_weight_prepack:
            pre_tuning_layout_rewrite = [transform.AttachAttrLayoutFreeBuffers()]
            post_tuning_layout_rewrite = [
                transform.SplitLayoutRewritePreproc(),
                transform.LiftTransformParams(),
                transform.FoldConstant(),
            ]
        else:
            pre_tuning_layout_rewrite = []
            post_tuning_layout_rewrite = []

        with tvm.target.Target(target):
            mod = tvm.transform.Sequential(
                [
                    transform.DecomposeOpsForInference(),
                    transform.CanonicalizeBindings(),
                    zero_pipeline(),
                    *pre_tuning_layout_rewrite,
                    # Skip tuning if total_trials is 0
                    (
                        transform.MetaScheduleTuneIRMod({}, work_dir, total_trials)
                        if total_trials > 0
                        else tvm.transform.Sequential([])
                    ),
                    transform.MetaScheduleApplyDatabase(work_dir),
                    *post_tuning_layout_rewrite,
                ]
            )(mod)

        return mod

    return _pipeline


# global map of pre-built pipelines
PIPELINE_MAP = {
    "zero": zero_pipeline,
    "default": default_build_pipeline,
    "default_build": default_build_pipeline,
    "static_shape_tuning": static_shape_tuning_pipeline,
}


def get_pipeline(name: str = "zero", **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    kwargs : Dict[str, object]
        Keyword args for configuring the pipeline.

    Returns
    -------
    pipeline: tvm.transform.Pass
       The transformation pipeline.
    """

    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name}," f"candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def register_pipeline(name: str):
    """Register a new pipeline

    Parameters
    ----------
    name : str
        Name of the pipeline
    """
    if name in PIPELINE_MAP:
        raise ValueError(f"Pipeline {name} has already been registered")

    def _register(func):
        PIPELINE_MAP[name] = func
        return func

    return _register


def library_dispatch_passes(target: tvm.target.Target):
    """Get the default library dispatch passes for the given target."""
    if target.kind.name == "cuda":
        return backend.cuda.library_dispatch_passes(target)
    if target.kind.name == "rocm":
        return backend.rocm.library_dispatch_passes(target)
    if target.kind.name == "metal":
        return backend.gpu_generic.library_dispatch_passes(target)
    if target.kind.name == "llvm":
        return backend.cpu_generic.library_dispatch_passes(target)
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return backend.adreno.library_dispatch_passes(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(f"Target {target} is not yet supported by library dispatch passes.")


def legalize_passes(target: tvm.target.Target):
    """Get the default legalization passes for the given target."""
    if target.kind.name == "cuda":
        return backend.cuda.legalize_passes(target)
    if target.kind.name == "rocm":
        return backend.rocm.legalize_passes(target)
    if target.kind.name == "metal":
        return backend.gpu_generic.legalize_passes(target)
    if target.kind.name == "llvm":
        return backend.cpu_generic.legalize_passes(target)
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return backend.adreno.legalize_passes(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(f"Target {target} is not yet supported by library dispatch passes.")


def dataflow_lower_passes(target: tvm.target.Target):
    """Get the default legalization passes for the given target."""
    if target.kind.name == "cuda":
        return backend.cuda.dataflow_lower_passes(target)
    if target.kind.name == "rocm":
        return backend.rocm.dataflow_lower_passes(target)
    if target.kind.name == "metal":
        return backend.gpu_generic.dataflow_lower_passes(target)
    if target.kind.name == "llvm":
        return backend.cpu_generic.dataflow_lower_passes(target)
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return backend.adreno.dataflow_lower_passes(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(f"Target {target} is not yet supported by dataflow lowering passes.")


def finalize_passes(target: tvm.target.Target):
    """Get the default legalization passes for the given target."""
    if target.kind.name == "cuda":
        return backend.cuda.finalize_passes(target)
    if target.kind.name == "rocm":
        return backend.rocm.finalize_passes(target)
    if target.kind.name == "metal":
        return backend.gpu_generic.finalize_passes(target)
    if target.kind.name == "llvm":
        return backend.cpu_generic.finalize_passes(target)
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return backend.adreno.finalize_passes(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(f"Target {target} is not yet supported by finalization passes.")


def get_default_pipeline(target: tvm.target.Target):
    """Get the default Relax compilation pipeline for the given target."""
    if target.kind.name == "cuda":
        return backend.cuda.get_default_pipeline(target)
    if target.kind.name == "rocm":
        return backend.rocm.get_default_pipeline(target)
    if target.kind.name == "metal":
        return backend.gpu_generic.get_default_pipeline(target)
    if target.kind.name == "llvm":
        return backend.cpu_generic.get_default_pipeline(target)
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return backend.adreno.get_default_pipeline(target)
    # Todo(tvm-team): support gpu-generic
    raise ValueError(
        f"Target {target} is not yet supported by default pipeline. "
        "Please lower and build the IRModule manually."
    )
