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
from typing import Union, Optional, List, Mapping

import warnings

import tvm.tir

from tvm import te

from tvm.runtime import Module
from tvm.runtime import ndarray
from tvm.ir import container
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.te import tensor
from tvm.target import Target
from tvm.tir.buffer import Buffer
from tvm.tir.expr import Var
from tvm.driver import _ffi_api as _driver_ffi

from . import _ffi_api as ffi


def get_binds(args, compact=False, binds=None):
    """Internal function to get binds and arg_list given arguments.
    Parameters
    ----------
    args : list of Buffer or Tensor or Var
        The argument lists to the function.
    compact : bool
        If the statement has already bound to a compact buffer.
    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.
    Returns
    -------
    binds: dict
        The bind specification
    arg_list: list
        The list of symbolic buffers of arguments.
    """
    binds, arg_list = ffi.get_binds(args, compact, binds)
    return binds, arg_list


def schedule_to_module(
    sch: te.Schedule,
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
) -> IRModule:
    """According to the given schedule, form a function.

    This is a low-level function intended for testing purposes, and
    does not apply any optimization passes.  In general, `tvm.lower`
    and `tvm.build` should be used instead.

    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
        The given scheduler to form the raw body
    args : list of Buffer or Tensor or Var
        The argument lists to the function.
    name : str
        The name of result function, default name is "main"
    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        The binds information
    Returns
    -------
    The body formed according to the given schedule
    """
    return ffi.schedule_to_module(sch, args, name, binds)


def lower(
    inp: Union[te.Schedule, PrimFunc, IRModule],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
    simple_mode: bool = False,
) -> IRModule:
    """Lowering step before build into target.

    Parameters
    ----------
    inp : Union[tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule]
        The TE schedule or TensorIR PrimFunc/IRModule to be built

    args : Optional[List[Union[tvm.tir.Buffer, tensor.Tensor, Var]]]
        The argument lists to the function for TE schedule.

        It should be None if we want to lower TensorIR.
    name : str
        The name of the result function.

    binds : Optional[Mapping[tensor.Tensor, tvm.tir.Buffer]]
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

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
    if isinstance(inp, te.Schedule):
        return ffi.lower_schedule(inp, args, name, binds, simple_mode)
    raise ValueError(
        f"Expected input to be an IRModule, PrimFunc or te.Schedule, but got {type(inp)}"
    )


def build(
    inputs: Union[te.Schedule, PrimFunc, IRModule, Mapping[str, IRModule]],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    target: Optional[Union[str, Target]] = None,
    target_host: Optional[Union[str, Target]] = None,
    runtime: Optional[
        "tvm.relay.backend.Runtime"
    ] = None,  # Type is annotated this way to avoid cyclic dependency
    name: Optional[str] = "default_function",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    inputs : Union[tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule, Mapping[str, IRModule]]
        The input to be built

    args : Optional[List[Union[tvm.tir.Buffer, tensor.Tensor, Var]]]
        The argument lists to the function.

    target : Optional[Union[str, Target]]
        The target and option of the compilation.

    target_host : Optional[Union[str, Target]]
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm interpreter is used.

    runtime : Optional[Runtime]
        Runtime to generate artifacts for

    name : Optional[str]
        The name of result function.

    binds : Optional[Mapping[tensor.Tensor, tvm.tir.Buffer]]
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    Returns
    -------
    ret : tvm.module
        A module that combines both host and device code.

    Examples
    ________
    There are two typical example uses of this function depending on the type
    of the argument `inputs`:
    1. it is an IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.te.create_schedule(C.op)
        m = tvm.lower(s, [A, B, C], name="test_add")
        rt_mod = tvm.build(m, target="llvm")

    2. it is a dict of compilation target to IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.te.create_schedule(C.op)
        with tvm.target.cuda() as cuda_tgt:
          s2 = topi.cuda.schedule_injective(cuda_tgt, [C])
          m1 = tvm.lower(s1, [A, B, C], name="test_add1")
          m2 = tvm.lower(s2, [A, B, C], name="test_add2")
          rt_mod = tvm.build({"llvm": m1, "cuda": m2})

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(inputs, te.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        input_mod = lower(inputs, args, name=name, binds=binds)
    elif isinstance(inputs, (list, tuple, container.Array)):
        merged_mod = tvm.IRModule({})
        for x in inputs:
            merged_mod.update(lower(x))
        input_mod = merged_mod
    elif isinstance(inputs, PrimFunc):
        input_mod = lower(inputs, name=name)
    elif isinstance(inputs, tvm.IRModule):
        input_mod = lower(inputs)
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError(
            f"Inputs must be te.Schedule, IRModule, PrimFunc, "
            f"or dict of target to IRModule, "
            f"but got {type(inputs)}."
        )

    if not isinstance(inputs, (dict, container.Map)):
        target = Target.current() if target is None else target
        if target is None and isinstance(input_mod, tvm.IRModule):
            target_mod = {}
            for gvar, func in input_mod.functions.items():
                tgt = func.attrs["target"] if func.attrs and "target" in func.attrs else "llvm"
                if tgt not in target_mod:
                    target_mod[tgt] = {}
                target_mod[tgt][gvar] = func

            target_input_mod = {}
            for tgt in target_mod.keys():
                tir_mod = tvm.IRModule(target_mod[tgt])
                tir_mod.with_attrs(input_mod.attrs)
                target_input_mod[tgt] = tir_mod
        else:
            target_input_mod = {target: input_mod}
    else:
        target_input_mod = {tgt: lower(mod) for tgt, mod in inputs.items()}

    # Because modules can be created from a variety of sources, we annotate them
    # with the relevant attributes here to ensure they propagate
    annotated_mods = {}
    for tgt, mod in target_input_mod.items():
        if not isinstance(tgt, (str, Target)):
            raise ValueError("The key of inputs must be str or " "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule, " "or dict of str to IRModule.")
        annotated_mods[tgt] = mod.with_attr("runtime", runtime)

    # TODO(mbs): Both CompilationConfig and TIRToRuntime implement the same host target
    #  defaulting logic, but there's currently no way to get back the decided host.
    if target_host is not None:
        warnings.warn(
            "target_host parameter is going to be deprecated. "
            "Please pass in tvm.target.Target(target, host=target_host) instead."
        )

    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)
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

    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)

    if not isinstance(target_host, Target):
        target_host = Target(target_host)

    if str(runtime) == "crt" and runtime["system-lib"]:
        if target_host.kind.name == "c":
            create_csource_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateCSourceCrtMetadataModule"
            )
            to_return = create_csource_crt_metadata_module([rt_mod_host], target_host, runtime)
        elif target_host.kind.name == "llvm":
            create_llvm_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateLLVMCrtMetadataModule"
            )
            to_return = create_llvm_crt_metadata_module([rt_mod_host], target_host, runtime)
    else:
        to_return = rt_mod_host

    return OperatorModule.from_module(to_return, ir_module_by_target=annotated_mods, name=name)


class OperatorModule(Module):
    """Wraps the Module returned by tvm.build() and captures additional outputs of that function."""

    @classmethod
    def from_module(cls, mod, **kwargs):
        # NOTE(areusch): It is generally unsafe to continue using `mod` from this point forward.
        # If an exception occurs in cls.__init__, handle will be deleted. For this reason,
        # set mod.handle to None.
        handle = mod.handle
        mod.handle = None
        return cls(handle, **kwargs)

    def __init__(self, handle, ir_module_by_target=None, name=None):
        super(OperatorModule, self).__init__(handle)
        self.ir_module_by_target = ir_module_by_target
        self.name = name
