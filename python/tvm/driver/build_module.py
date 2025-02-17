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
from typing import Union, Optional


import tvm.tir


from tvm.runtime import ndarray
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.target import Target
from tvm.driver import _ffi_api as _driver_ffi

from . import _ffi_api as ffi


def lower(
    inp: Union[PrimFunc, IRModule],
    name: str = "main",
    simple_mode: bool = False,
) -> IRModule:
    """Lowering step before build into target.

    Parameters
    ----------
    inp : Union[tvm.tir.PrimFunc, IRModule]
        The TE schedule or TensorIR PrimFunc/IRModule to be built

    name : str
        The name of the result function.

    simple_mode : bool
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    Returns
    -------
    m : IRModule
       The result IRModule
    """
    if isinstance(inp, IRModule):
        return ffi.lower_module(inp, simple_mode)
    if isinstance(inp, PrimFunc):
        return ffi.lower_primfunc(inp, name, simple_mode)
    raise ValueError(
        f"Expected input to be an IRModule, PrimFunc or te.Schedule, but got {type(inp)}"
    )


def build(
    inputs: Union[PrimFunc, IRModule],
    target: Optional[Union[str, Target]] = None,
    name: str = "main",
):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    input : Union[tvm.tir.PrimFunc, IRModule]
        The input to be built

    target : Optional[Union[str, Target]]
        The target and option of the compilation.

    name : str
        The name of result function.

    Returns
    -------
    ret : tvm.module
        A module that combines both host and device code.

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(inputs, PrimFunc):
        input_mod = lower(inputs, name=name)
    elif isinstance(inputs, tvm.IRModule):
        assert (
            len(inputs.get_global_vars()) > 0
        ), "Expected a non-empty IRModule, but the IRModule contained no functions."
        input_mod = lower(inputs)
    else:
        raise ValueError("Inputs must be IRModule or PrimFunc")

    target = Target.current() if target is None else target
    if target is None and isinstance(input_mod, tvm.IRModule):
        target_mod = {}
        for gvar, func in input_mod.functions.items():
            tgt = func.attrs["target"] if "target" in func.attrs else "llvm"
            if tgt not in target_mod:
                target_mod[tgt] = {}
            target_mod[tgt][gvar] = func

        target_input_mod = {}
        for tgt in target_mod.keys():
            tir_mod = tvm.IRModule(target_mod[tgt])
            tir_mod = tir_mod.with_attrs(input_mod.attrs)
            target_input_mod[tgt] = tir_mod
    else:
        target_input_mod = {target: input_mod}

    # Because modules can be created from a variety of sources, we annotate them
    # with the relevant attributes here to ensure they propagate
    annotated_mods = {}
    for tgt, mod in target_input_mod.items():
        if not isinstance(tgt, (str, Target)):
            raise ValueError("The key of inputs must be str or " "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be IRModule, " "or dict of str to IRModule.")
        annotated_mods[tgt] = mod

    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods)
    if not target_host:
        for tar, mod in annotated_mods.items():
            device_type = ndarray.device(tar.kind.name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)

    rt_mod_host = _driver_ffi.tir_to_runtime(annotated_mods, target_host)

    return rt_mod_host
