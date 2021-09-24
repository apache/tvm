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
"""The build utils in python.
"""

from typing import Union, Optional, List, Mapping
import warnings

import tvm.tir

from tvm.runtime import Module
from tvm.runtime import ndarray
from tvm.ir import container
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.target import codegen
from tvm.te import tensor
from tvm.te import schedule
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
    sch: schedule.Schedule,
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
) -> IRModule:
    """According to the given schedule, form a function.
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
    inp: Union[schedule.Schedule, PrimFunc, IRModule],
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
    # ffi.relay.lower_te_pass()
    if isinstance(inp, IRModule):
        return ffi.lower_module(inp, simple_mode)
    if isinstance(inp, PrimFunc):
        return ffi.lower_primfunc(inp, name, simple_mode)
    if isinstance(inp, schedule.Schedule):
        return ffi.lower_schedule(inp, args, name, binds, simple_mode)
    raise ValueError("Expected input to be an IRModule, PrimFunc or Schedule, but got, ", type(inp))


def _build_for_device(input_mod, target, target_host):
    """Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    input_mod : IRModule
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    host_mod : IRModule
        The host IRModule.

    device_mod : tvm.module
        A module that contains device code.
    """
    mod_mixed = _driver_ffi.get_mod_mixed(input_mod, target)
    device_mod = _driver_ffi.get_device_mod(mod_mixed, target)
    host_mod = _driver_ffi.get_host_mod(mod_mixed, target_host)

    device_type = ndarray.device(target.kind.name, 0).device_type
    if device_type == ndarray.cpu(0).device_type and target_host == target:
        assert len(device_mod.functions) == 0
    if "gpu" in target.keys and len(device_mod.functions) == 0:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do " "bind?" % target
        )

    # rt_mod_dev is runtime::Module so this can be moved out maybe?
    rt_mod_dev = (
        codegen.build_module(device_mod, target) if len(device_mod.functions) != 0 else None
    )
    # TIR module for the host, runtime module for devices?
    return host_mod, rt_mod_dev


def build(
    inputs: Union[schedule.Schedule, PrimFunc, IRModule, Mapping[str, IRModule]],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    target: Optional[Union[str, Target]] = None,
    target_host: Optional[Union[str, Target]] = None,
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
        otherwise a stackvm intepreter is used.
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
          rt_mod = tvm.build({"llvm": m1, "cuda": m2}, target_host="llvm")
    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """

    # Lowering
    if isinstance(inputs, (schedule.Schedule, tvm.IRModule, PrimFunc)):
        # should this be te_lower instead?
        input_mod = lower(inputs, args, name=name, binds=binds)
    elif isinstance(inputs, (list, tuple, container.Array)):
        merged_mod = tvm.IRModule({})
        for x in inputs:
            merged_mod.update(lower(x))
        input_mod = merged_mod
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError(
            f"Inputs must be Schedule, PrimFunc, IRModule or dict of target to IRModule, "
            f"but got {type(inputs)}."
        )

    if not isinstance(inputs, (dict, container.Map)):
        target = Target.current() if target is None else target
        target = target if target else "llvm"
        target_input_mod = {target: input_mod}
    else:
        target_input_mod = inputs

    for tar, mod in target_input_mod.items():
        if not isinstance(tar, (str, Target)):
            raise ValueError("The key of inputs must be str or " "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule," "or dict of str to IRModule.")

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )

    if not target_host:
        for tar, mod in target_input_mod.items():
            tar = Target(tar)
            device_type = ndarray.device(tar.kind.name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )

    mod_host_all = tvm.IRModule({})

    device_modules = []
    for tar, input_mod in target_input_mod.items():
        mod_host, mdev = _build_for_device(input_mod, tar, target_host)
        mod_host_all.update(mod_host)
        device_modules.append(mdev)

    # Generate a unified host module.
    rt_mod_host = codegen.build_module(mod_host_all, target_host)

    # Import all modules.
    for mdev in device_modules:
        if mdev:
            rt_mod_host.import_module(mdev)

    # stop moving to C++ here.
    if not isinstance(target_host, Target):
        target_host = Target(target_host)
    if (
        target_host.attrs.get("runtime", tvm.runtime.String("c++")) == "c"
        and target_host.attrs.get("system-lib", 0) == 1
    ):
        if target_host.kind.name == "c":
            create_csource_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateCSourceCrtMetadataModule"
            )
            to_return = create_csource_crt_metadata_module([rt_mod_host], target_host)

        elif target_host.kind.name == "llvm":
            create_llvm_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateLLVMCrtMetadataModule"
            )
            to_return = create_llvm_crt_metadata_module([rt_mod_host], target_host)
    else:
        to_return = rt_mod_host

    return OperatorModule.from_module(to_return, ir_module_by_target=target_input_mod, name=name)


# What is OperatorModule and how is it different from runtime::Module


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
