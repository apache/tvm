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
"""The build utils in python."""
import warnings
from typing import Callable, Optional, Union

import tvm
from tvm.ir.module import IRModule
from tvm.runtime import Executable
from tvm.target import Target
from tvm.tir import PrimFunc


def build(
    mod: Union[PrimFunc, IRModule],
    target: Optional[Union[str, Target]] = None,
    pipeline: Optional[Union[str, tvm.transform.Pass]] = "default",
):
    """
    Build a function with a signature, generating code for devices
    coupled with target information.

    This function is deprecated. Use `tvm.compile` or `tvm.tir.build` instead.

    Parameters
    ----------
    mod : Union[PrimFunc, IRModule]
        The input to be built.
    target : Optional[Union[str, Target]]
        The target for compilation.
    pipeline : Optional[Union[str, tvm.transform.Pass]]
        The pipeline to use for compilation.

    Returns
    -------
    tvm.runtime.Module
        A module combining both host and device code.
    """
    warnings.warn(
        "build is deprecated. Use `tvm.compile` or `tvm.tir.build` instead.",
        DeprecationWarning,
    )
    return tvm.tir.build(mod, target, pipeline)


def _contains_relax(mod: Union[PrimFunc, IRModule]) -> bool:
    if isinstance(mod, PrimFunc):
        return False
    if isinstance(mod, IRModule):
        return any(isinstance(func, tvm.relax.Function) for _, func in mod.functions_items())

    raise ValueError(f"Function input must be a PrimFunc or IRModule, but got {type(mod)}")


def compile(  # pylint: disable=redefined-builtin
    mod: Union[PrimFunc, IRModule],
    target: Optional[Target] = None,
    *,
    relax_pipeline: Optional[Union[tvm.transform.Pass, Callable, str]] = "default",
    tir_pipeline: Optional[Union[tvm.transform.Pass, Callable, str]] = "default",
) -> Executable:
    """
    Compile an IRModule to a runtime executable.

    This function serves as a unified entry point for compiling both TIR and Relax modules.
    It automatically detects the module type and routes to the appropriate build function.

    Parameters
    ----------
    mod : Union[PrimFunc, IRModule]
        The input module to be compiled. Can be a PrimFunc or an IRModule containing
        TIR or Relax functions.
    target : Optional[Target]
        The target platform to compile for.
    relax_pipeline : Optional[Union[tvm.transform.Pass, Callable, str]]
        The compilation pipeline to use for Relax functions.
        Only used if the module contains Relax functions.
    tir_pipeline : Optional[Union[tvm.transform.Pass, Callable, str]]
        The compilation pipeline to use for TIR functions.

    Returns
    -------
    Executable
        A runtime executable that can be loaded and executed.
    """
    # TODO(tvm-team): combine two path into unified one
    if _contains_relax(mod):
        return tvm.relax.build(
            mod,
            target,
            relax_pipeline=relax_pipeline,
            tir_pipeline=tir_pipeline,
        )
    lib = tvm.tir.build(mod, target, pipeline=tir_pipeline)
    return Executable(lib)
