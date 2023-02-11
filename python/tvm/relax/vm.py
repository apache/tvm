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
# pylint: disable=invalid-name, redefined-builtin, no-else-return
"""The Relax virtual machine"""
from typing import Callable, List, Optional, Union, Dict, Tuple, Any
import numpy as np  # type: ignore

from tvm._ffi import base as _base
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.runtime import Device, Module, PackedFunc, container
from tvm.runtime.object import Object
from tvm.tir.function import PrimFunc
from . import _ffi_api
from ..rpc.base import RPC_SESS_MASK


class Executable(object):
    """The executable object emitted by the VM compiler or the ExecBuilder."""

    def __init__(self, mod: Module):
        self.mod = mod
        self._stats = self.mod["stats"]
        self._as_text = self.mod["as_text"]
        self._as_python = self.mod["as_python"]

    def stats(self) -> str:
        """print the detailed statistics of the executable."""
        return self._stats()

    def as_text(self) -> str:
        """print the instructions as text format."""
        return self._as_text()

    def as_python(self) -> str:
        """print the instructions as python program."""
        return self._as_python()


class VirtualMachine(object):
    """Relax VM runtime."""

    NAIVE_ALLOCATOR = 1
    POOLED_ALLOCATOR = 2

    def __init__(
        self,
        exec: Union[Executable, Module],
        device: Union[Device, List[Device]],
        memory_cfg: Optional[Union[str, Dict[Device, str]]] = None,
    ) -> None:
        """
        Construct a VirtualMachine wrapper object.

        Parameters
        ----------
        exec: Union[Executable, Module]
            The VM executable or Runtime Module

        device : Union[Device, List[Device]]
            The device to deploy the module.

        memory_cfg : Optional[Union[str, Dict[Device, str]]]
            Config the type of memory allocator. The allocator type can be ["naive",
            "pooled"]. If memory_cfg is None, all devices will use pooled allocator
            by default. If memory_cfg is string, all devices will use the specified
            allocator type. If memory_cfg is a dict, each device uses the allocator
            type specified in the dict, or pooled allocator if not specified in the
            dict.
        """
        self.module = (
            exec.mod["vm_load_executable"]()
            if isinstance(exec, Executable)
            else exec["vm_load_executable"]()
        )
        self._invoke_closure = self.module["invoke_closure"]
        self._save_function = self.module["save_function"]
        self._set_input = self.module["set_input"]
        self._invoke_stateful = self.module["invoke_stateful"]
        self._get_output = self.module["get_output"]
        self._get_output_arity = self.module["get_output_arity"]
        self._get_function_arity = self.module["get_function_arity"]
        self._get_function_param_name = self.module["get_function_param_name"]
        self._setup_device(device, memory_cfg)

    def _setup_device(self, dev: Device, memory_cfg: Union[str, Dict[Device, str]]) -> None:
        """init devices and allocators."""
        devs = dev
        if not isinstance(dev, (list, tuple)):
            if not isinstance(dev, tvm.runtime.Device):
                raise TypeError(
                    "dev is expected to be Device or \
                                List[Device]"
                )
            devs = [dev]

        if any(dev.device_type % RPC_SESS_MASK == tvm.cpu().device_type for dev in devs[:-1]):
            raise RuntimeError(
                "CPU host is required to be the last element of the device list if provided."
            )

        # CPU is required for executing shape functions
        if devs[-1].device_type % RPC_SESS_MASK != tvm.cpu().device_type:
            devs.append(tvm.cpu())

        default_alloc_type = VirtualMachine.POOLED_ALLOCATOR
        if memory_cfg is None:
            memory_cfg = {}
        elif isinstance(memory_cfg, str):
            assert memory_cfg in ["naive", "pooled"]
            if memory_cfg == "naive":
                default_alloc_type = VirtualMachine.NAIVE_ALLOCATOR
            memory_cfg = {}
        elif not isinstance(memory_cfg, dict):
            raise TypeError(
                "memory_cfg is expected be string or dictionary, "
                + "but received {}".format(type(memory_cfg))
            )
        init_args = []
        for device in devs:
            init_args.append(device.device_type % RPC_SESS_MASK)
            init_args.append(device.device_id)
            alloc_type = memory_cfg[device] if device in memory_cfg else default_alloc_type
            init_args.append(alloc_type)
        self.module["vm_initialization"](*init_args)

    def __getitem__(self, key: str) -> PackedFunc:
        return self.module[key]

    def invoke_closure(self, closure: Object, *args: Any) -> Object:
        """Invoke a closure.

        Parameters
        ----------
        closure : Object
            The VMClosure Object.

        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the closure.

        Returns
        -------
        result : Object
            The output.
        """
        return self._invoke_closure(closure, *args)

    def save_function(
        self,
        func_name: str,
        saved_name: str,
        *args: List[Any],
        include_return: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Convenience function. Takes a function from the module and saves
        a `PackedFunc` that, when called, will invoke the function with the given arguments.
        The `PackedFunc` can be accessed from the module using `saved_name`.
        This is included to facilitate timing trials:
        Invoking the returned `PackedFunc` will have less overhead from dictionary lookups
        than normally running through the VM.

        If the saved name is taken, it can be overridden, though it cannot override
        the name of a function defined in the Relax source.

        This is really creating a closure, but the function has a different name
        to avoid confusion with `invoke_closure` (they are not meant to be used together).

        Parameters
        ----------
        func_name : str
            The function that should be packaged up.

        saved_name : str
            The name that the resulting closure should be saved under.

        include_return : bool
            Whether the saved PackedFunc should return its output.
            If timing over RPC, it may not be desirable to send output
            between machines.

        args : List[Any]
            The arguments to package up with the function.

        kwargs : Dict[str, Any]
            Any named arguments to package up with the function
        """
        cargs: List[Any] = []
        if kwargs:
            args = self._convert_func_named_args(func_name, args, **kwargs)
        for arg in args:
            self._convert(arg, cargs)
        self._save_function(func_name, saved_name, int(include_return), *cargs)

    def _convert(self, arg: Any, cargs: List) -> None:
        """helper function to convert arguments to vm function."""

        def _gettype(arg):
            if isinstance(arg, np.float16):
                return "float16"
            elif isinstance(arg, (_base.integer_types, bool)):
                return "int32"
            else:
                return "float32"

        if isinstance(arg, Object):
            cargs.append(arg)
        elif isinstance(arg, np.ndarray):
            nd_arr = tvm.nd.array(arg, device=tvm.cpu(0))
            cargs.append(nd_arr)
        elif isinstance(arg, tvm.runtime.NDArray):
            cargs.append(arg)
        elif isinstance(arg, (tuple, list)):
            field_args: List[Any] = []
            for field in arg:
                self._convert(field, field_args)
            cargs.append(container.tuple_object(field_args))
        elif isinstance(arg, (_base.numeric_types, bool)):
            dtype = _gettype(arg)
            value = tvm.nd.array(np.array(arg, dtype=dtype), device=tvm.cpu(0))
            cargs.append(value)
        elif isinstance(arg, str):
            cargs.append(arg)
        else:
            raise TypeError("Unsupported type: %s" % (type(arg)))

    def _convert_func_named_args(self, func_name: str, args: Any, **kwargs: Any) -> Any:
        """
        Takes named function parameters and returns a list of those needed,
        in the order they should appear
        """
        # kwargs can be a super set of the required function parameters.
        # We only find the ones that are needed.
        func_arity = self._get_function_arity(func_name)
        func_params = [self._get_function_param_name(func_name, i) for i in range(func_arity)]
        new_args = [None] * len(func_params)
        cnt = 0
        for k in kwargs:
            if k in func_params:
                idx = func_params.index(k)
                new_args[idx] = kwargs[k]
                cnt += 1
            else:
                print(f'Warning: Keyword argument "{k}" is unused in {func_name}')
        assert len(args) + cnt == len(func_params)
        idx = 0
        for i, arg in enumerate(new_args):
            if arg is None:
                new_args[i] = args[idx]
                idx += 1
        return new_args

    def set_input(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """Set the inputs to a function.
        This interface works when using VM over RPC by internally converting NDArray in
        the arguments to DLTensor, which is supported in RPC where remote could only
        have a minimal C runtime.

        Note: If `set_input` is used, the function *must* be called using `invoke_stateful`
        and the results must be obtained using `get_outputs`.

        Parameters
        ----------
        func_name : str
            The name of the function.
        args: List[tvm.runtime.NDArray] or List[np.ndarray]
            The arguments to the function.
        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.
        """
        cargs: List[Any] = []

        if kwargs:
            args = self._convert_func_named_args(func_name, args, **kwargs)

        for arg in args:
            self._convert(arg, cargs)

        self._set_input(func_name, *cargs)

    def invoke_stateful(self, func_name: str) -> None:
        """
        Call the named function from the VM module using the arguments set using `set_input`.
        It is an error to call `invoke_stateful` without using `set_input` first
        (even if it's to set 0 inputs); conversely, if `set_input` has been called,
        it is an error to call the function without using `invoke_stateful`.

        The results of the call can be obtained by calling `get_outputs`.

        Parameters
        ----------
        func_name: str
            The name of the function to call.
        """
        self._invoke_stateful(func_name)

    def get_outputs(self, func_name: str) -> Union[tvm.Object, Tuple[Any]]:
        """
        Get the value output by the function by the given name
        after a call of `invoke_stateful`.

        It is an error to call this function without first calling `invoke_stateful`.

        Parameters
        ----------
        func_name: str
            The name of the function whose output should be fetched.

        Returns
        -------
        ret: Union[tvm.Object, Tuple[Any]]
            The result of the earlier call to the function via `invoke_stateful`.
            If the result is a tuple, it returns a list of the fields.
            The fields are potentially also tuples, so these can be arbitrily nested.
        """
        # to deal with potentially nested tuples, we need to query for arity recursively
        def get_output_rec(func_name, *idx):
            arity = self._get_output_arity(func_name, *idx)
            if arity == -1:
                return self._get_output(func_name, *idx)
            # otherwise we need to specify more indices
            idx_list = list(idx)
            return tuple(get_output_rec(func_name, *(idx_list + [i])) for i in range(arity))

        return get_output_rec(func_name)

    def time_evaluator(
        self,
        func_name,
        dev,
        number=10,
        repeat=1,
        min_repeat_ms=0,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
        f_preproc="",
    ) -> Callable[..., tvm.runtime.module.BenchmarkResult]:
        """
        Returns an evaluator that times a function in the module.
        This follows the same convention as time_evaluator in tvm.runtime.module.
        This can be used in combination with save_function() so that the
        timings avoid extra dictionary lookups.

        Parameters
        ----------
        func_name: str
            The name of the function in the module.

        dev: Device
            The device we should run this function on.

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        f_preproc: str, optional
            The preprocess function name we want to execute before executing the time evaluator.

        Note
        ----
        The function will be invoked  (1 + number x repeat) times,
        with the first call discarded in case there is lazy initialization.

        Example
        -------
        Normal use with a VM function (may not work over RPC if the function returns a tuple):

        .. code-block:: python

            target = tvm.target.Target("llvm", host="llvm")
            ex = relax.vm.build(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm.cpu())
            timing_res = vm.time_evaluator("func_name", tvm.cpu())(arg0, arg1, ..., argn)

        Use with the stateful API:

        .. code-block:: python

            target = tvm.target.Target("llvm", host="llvm")
            ex = relax.vm.build(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm.cpu())
            vm.set_input("func_name", arg0, arg1, ..., argn)
            timing_res = vm.time_evaluator("invoke_stateful", tvm.cpu())("func_name")

        With saved closures via `save_function` (this results in
        fewer dictionary lookups in the timed portion):

        .. code-block:: python

            target = tvm.target.Target("llvm", host="llvm")
            ex = relax.vm.build(TestTimeEvaluator, target)
            vm = relax.VirtualMachine(mod, tvm.cpu())
            vm.save_function("func_name", "func_name_saved", arg0, arg1, ..., argn)
            timing_res = vm.time_evaluator("func_name_saved", tvm.cpu())()

        Returns
        -------
        ftimer : function
            The function that takes same argument as func and returns a BenchmarkResult.
            The ProfileResult reports `repeat` time costs in seconds.

        """
        return self.module.time_evaluator(
            func_name,
            dev,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
            f_preproc=f_preproc,
        )


def _vmcodegen(
    builder: "relax.ExecBuilder",
    mod: tvm.IRModule,
    exec_mode: str = "bytecode",
) -> tvm.IRModule:
    """Running VM codegen.

    Parameters
    ----------
    builder: relax.ExecBuilder
        ExecBuilder to collect the vm executable.

    mod: IRModule
        The input IRModule to be built.

    exec_mode: {"bytecode", "compiled"}
        The execution mode.

    Return
    ------
    leftover: IRModule
        Left over IRModule that may contain extra functions.
    """

    if exec_mode == "bytecode":
        return _ffi_api.VMCodeGen(builder, mod)  # type:ignore
    if exec_mode == "compiled":
        return _ffi_api.VMTIRCodeGen(builder, mod)  # type: ignore
    raise ValueError("Unknown exec_mode %s" % exec_mode)


def _vmlink(
    builder: "relax.ExecBuilder",
    target: Union[str, tvm.target.Target],
    tir_mod: Optional[tvm.IRModule] = None,
    ext_libs: List[tvm.runtime.Module] = None,
    params: Optional[Dict[str, list]] = None,
):
    """
    Internal codegen function to make executable.

    This function is only used for unit-testing purpoes.

    Use build instead.

    Parameters
    ----------
    builder: relax.ExecBuilder
        Builder used to collect executables.

    target : Union[str, tvm.target.Target]
        A build target which can have optional host side compilation target.

    tir_mod: IRModule
        The input TIR IRModule to be linked together.

    ext_libs:  List[tvm.runtime.Module]
        List of compiled external modules.

    params: Optional[Dict[str, list]]
        Extra parameter mappings.

    Returns
    -------
    ex: tvm.relax.vm.Executable
        An executable that can be loaded by virtual machine.
    """
    if isinstance(target, str):
        target = tvm.target.Target(target)
    if params is None:
        params = {}
    if ext_libs is None:
        ext_libs = []
    lib = None
    if tir_mod is not None:
        lib = tvm.build(tir_mod, target=target)
    return Executable(_ffi_api.VMLink(builder, target, lib, ext_libs, params))  # type: ignore


def build(
    mod: tvm.IRModule,
    target: Union[str, tvm.target.Target],
    params: Optional[Dict[str, list]] = None,
    exec_mode: str = "bytecode",
) -> Executable:
    """
    Build an IRModule to VM executable.

    Parameters
    ----------
    mod: IRModule
        The input IRModule to be built.

    target : Union[str, tvm.target.Target]
        A build target which can have optional host side compilation target.

        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        to setup the dimensions and parameters correctly.
        host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm interpreter is used.

    params: Optional[Dict[str, list]]
        Parameters for the input IRModule that will be bound.

    exec_mode: {"bytecode", "compiled"}
        The execution mode.

    Returns
    -------
    ex: tvm.relax.vm.Executable
        An executable that can be loaded by virtual machine.

    Example
    -------

    .. code-block:: python
        class InputModule:
            @R.function
            def foo(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")):
                z = R.add(x, y)
                return z

        mod = InputModule
        target = tvm.target.Target("llvm", host="llvm")
        ex = relax.vm.build(mod, target)
    """
    if isinstance(target, str):
        target = tvm.target.Target(target)

    passes = []
    passes.append(relax.transform.RewriteDataflowReshape())
    passes.append(relax.transform.ToNonDataflow())
    passes.append(relax.transform.CallTIRRewrite())
    passes.append(relax.transform.VMBuiltinLower())
    passes.append(relax.transform.VMShapeLower())
    passes.append(relax.transform.AttachGlobalSymbol())
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)

    # Extract external runtime modules if exist.
    ext_libs = []
    if mod.attrs and "external_mods" in mod.attrs:
        ext_libs = mod.attrs["external_mods"]

    # builder collects the executable
    builder = relax.ExecBuilder()
    leftover_mod = _vmcodegen(builder, new_mod, exec_mode=exec_mode)
    tir_mod = _filter_tir(leftover_mod)
    return _vmlink(builder, target, tir_mod, ext_libs, params)


def _filter_tir(mod: tvm.IRModule) -> tvm.IRModule:
    tir_mod = IRModule({})
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], PrimFunc):
            tir_mod[gv] = mod[gv]
    return tir_mod
