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
import tvm
from tvm import meta_schedule as ms

from . import transform, backend


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
    """The default compilation pipeline used in relax.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
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
                transform.VMBuiltinLower(),
                transform.VMShapeLower(),
                transform.AttachGlobalSymbol(),
            ],
        )
        mod = seq(mod)
        return mod

    return _pipeline


# global map of pre-built pipelines
PIPELINE_MAP = {
    "zero": zero_pipeline,
    "default_build": default_build_pipeline,
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
