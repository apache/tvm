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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
The Relay Virtual Machine runtime.

Implements a Python interface to executing the compiled VM object.
"""
import numpy as np

import tvm
from tvm.runtime import Module
from tvm._ffi.runtime_ctypes import TVMByteArray
from tvm._ffi import base as _base
from .object import Object
from . import _ffi_api, container
from ..rpc.base import RPC_SESS_MASK


def _convert(arg, cargs):
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
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(container.tuple_object(field_args))
    elif isinstance(arg, (_base.numeric_types, bool)):
        dtype = _gettype(arg)
        value = tvm.nd.array(np.array(arg, dtype=dtype), device=tvm.cpu(0))
        cargs.append(value)
    elif isinstance(arg, str):
        cargs.append(arg)
    else:
        raise TypeError("Unsupported type: %s" % (type(arg)))


def convert(args):
    cargs = []
    for arg in args:
        _convert(arg, cargs)

    return cargs


class Executable(object):
    """Relay VM executable"""

    def __init__(self, mod):
        self.mod = mod
        self._function_params = {}
        self._save = self.mod["save"]
        self._get_lib = self.mod["get_lib"]
        self._get_bytecode = self.mod["get_bytecode"]
        self._get_constants = self.mod["get_constants"]
        self._get_virtual_devices = self.mod["get_virtual_devices"]
        self._get_primitives = self.mod["get_primitives"]
        self._get_stats = self.mod["get_stats"]
        self._get_function_arity = self.mod["get_function_arity"]
        self._get_function_param_name = self.mod["get_function_param_name"]
        self._move_late_bound_consts = self.mod["move_late_bound_consts"]
        self._load_late_bound_consts = self.mod["load_late_bound_consts"]

    def save(self):
        """Save the Relay VM Executable.

        Returns
        -------
        code : bytearray
            The binary blob representing a serialized Relay VM executable. It
            can then be saved to disk and later deserialized into a new
            Executable.

        lib : :py:class:`~tvm.runtime.Module`
            The runtime module that contains the generated code. It is
            basically a library that is composed of hardware dependent code.

        Notes
        -----
        The returned code is organized with the following sections in order.
         - Global section. This section contains the globals used by the
         virtual machine.

         - Constant section. This section is used to store the constant pool of
         a virtual machine.

         - Primitive name section. This section is introduced to accommodate
         the list of primitive operator names that will be invoked by the
         virtual machine.

         - Code section. The VM functions, including bytecode, are sitting in
         this section.

        Examples
        --------

        .. code-block:: python

            import numpy as np
            import tvm
            from tvm import te
            from tvm import relay
            # define a simple network.
            x = relay.var('x', shape=(10, 10))
            f = relay.Function([x], x + x)
            mod = tvm.IRModule({"main": f})
            # create a Relay VM.
            dev = tvm.cpu()
            target = "llvm"
            executable = relay.vm.compile(mod, target)
            code, lib = executable.save()
            # save and load the code and lib file.
            tmp = tvm.contrib.utils.tempdir()
            path_lib = tmp.relpath("lib.so")
            lib.export_library(path_lib)
            with open(tmp.relpath("code.ro"), "wb") as fo:
                fo.write(code)
            loaded_lib = tvm.runtime.load_module(path_lib)
            loaded_code = bytearray(open(tmp.relpath("code.ro"), "rb").read())
            # deserialize.
            des_exec = tvm.runtime.vm.Executable.load_exec(loaded_code, loaded_lib)
            # execute the deserialized executable.
            x_data = np.random.rand(10, 10).astype('float32')
            des_vm = tvm.runtime.vm.VirtualMachine(des_exec, dev)
            res = des_vm.run(x_data)
            print(res.numpy())
        """
        return self._save(), self._get_lib()

    @staticmethod
    def load_exec(bytecode, lib):
        """Construct an executable from saved artifacts.

        Parameters
        ----------
        bytecode : bytearray
            The binary blob representing a the Relay VM bytecode.

        lib : :py:class:`~tvm.runtime.Module`
            The runtime module that contains the generated code.

        Returns
        -------
        exec: Executable
            An executable constructed using the provided artifacts.
        """
        if isinstance(bytecode, (bytes, str)):
            bytecode = bytearray(bytecode)
        elif not isinstance(bytecode, (bytearray, TVMByteArray)):
            raise TypeError(
                "bytecode is expected to be the type of bytearray "
                + "or TVMByteArray, but received {}".format(type(bytecode))
            )

        if lib is not None and not isinstance(lib, tvm.runtime.Module):
            raise TypeError(
                "lib is expected to be the type of tvm.runtime.Module"
                + ", but received {}".format(type(lib))
            )

        return Executable(_ffi_api.Load_Executable(bytecode, lib))

    @property
    def lib(self):
        """Get the library that contains hardware dependent code.

        Returns
        -------
        ret : :py:class:`~tvm.runtime.Module`
            The runtime module that contains hardware dependent code.
        """
        return self._get_lib()

    @property
    def stats(self):
        """Get the statistics of the Relay VM executable.

        Returns
        -------
        ret : String
            The statistic information of the VM executable.
        """
        return self._get_stats()

    @property
    def primitive_ops(self):
        """Get the name of the primitive ops contained in the executable.

        Returns
        -------
        ret : List[String]
            The list of primitive ops.
        """
        ret = []
        num_primitives = _ffi_api.GetNumOfPrimitives(self.module)
        for i in range(num_primitives):
            ret.append(_ffi_api.GetPrimitiveFields(self.module, i))
        return ret

    @property
    def bytecode(self):
        """Get the bytecode of the Relay VM executable.

        Returns
        -------
        ret : String
            The bytecode of the executable.

        Notes
        -----
        The bytecode is in the following format:
          func_name reg_file_size num_instructions

          param1 param2 ... paramM

          instruction1

          instruction2

          ...

          instructionN

        Each instruction is printed in the following format:
          hash opcode field1 ... fieldX # The text format.

        The part starting from # is only used for visualization and debugging.
        The real serialized code doesn't contain it, therefore the deserializer
        doesn't need to deal with it as well.
        """
        return self._get_bytecode()

    @property
    def constants(self):
        """Returns a human-readable description of all the constants in the executable.
        Useful for debugging and diffing generated executables in unit tests."""
        return self._get_constants()

    @property
    def virtual_devices(self):
        """Returns a human-readable description of all the (virtual) devices in the executable."""
        return self._get_virtual_devices()

    @property
    def primitives(self):
        """Returns a human-readable description of all the primitives (ie PackedFuncs) in the
        executable"""
        return self._get_primitives()

    @property
    def globals(self):
        """Get the globals used by the Relay VM executable.

        Returns
        -------
        ret : List[String]
            The globals contained in the executable.
        """
        ret = []
        num_globals = _ffi_api.GetNumOfGlobals(self.module)
        for i in range(num_globals):
            ret.append(_ffi_api.GetGlobalFields(self.module, i))
        return ret

    @property
    def module(self):
        """Return the runtime module contained in a virtual machine executable."""
        return self.mod

    def get_function_params(self, func_name):
        """Get VM Function parameters"""
        if func_name in self._function_params:
            return self._function_params[func_name]
        arity = self._get_function_arity(func_name)
        assert arity >= 0
        params = []
        for i in range(arity):
            p = self._get_function_param_name(func_name, i)
            assert p
            params.append(p)
        self._function_params[func_name] = params
        return params

    def move_late_bound_consts(self, path, byte_limit):
        """Move all constants of byte size greater or equal to byte_limit to file at path"""
        return self._move_late_bound_consts(path, byte_limit)

    def load_late_bound_consts(self, path):
        """Re-load constants previously saved to file at path"""
        return self._load_late_bound_consts(path)


class VirtualMachine(object):
    """Relay VM runtime.

    Parameters
    ----------
    exe : Executable
        The VM executable.

    device : tvm.runtime.Device or List[tvm.runtime.Device]
        The device(s) on which the model will run.
        Currently at most one device per device type is supported.

    memory_cfg : str or Dict[tvm.runtime.Device, str], optional
        Config the type of memory allocator. The allocator type can be ["naive",
        "pooled"]. If memory_cfg is None, all devices will use pooled allocator
        by default. If memory_cfg is string, all devices will use the specified
        allocator type. If memory_cfg is a dict, each device uses the allocator
        type specified in the dict, or pooled allocator if not specified in the
        dict.
    """

    NAIVE_ALLOCATOR = 1
    POOLED_ALLOCATOR = 2

    def __init__(self, exe, device, memory_cfg=None):
        """
        Construct a VirtualMachine wrapper class which provides a simple
        interface over the raw C++ Module based API.

        Parameters
        ----------
        exe: Union[Executable, Module]
            The executable either with the wrapper Python type or the raw runtime.Module.

            In most cases this will be the Python wrapper class tvm.runtime.vm.Executable but
            if you instead get the underlying runtime.Module subclass (i.e `exe.mod`) you
            can directly pass it to this method.

            This case can occur when doing things such as RPC where TVM's module APIs
            return the raw modules, not the wrapped modules. This constructor will
            handle this internally.

        device: Union[Device, List[Device]]
            The device, or devices on which to execute the VM code.

        memory_cfg: Optional[str]
            The allocator behavior to use for the VM.

        Returns
        -------
        vm: VirtualMachine
            A VM wrapper object.
        """
        if not isinstance(exe, Executable) and not isinstance(exe, Module):
            raise TypeError(
                "exe is expected to be the type of Executable, "
                + "but received {}".format(type(exe))
            )

        if not isinstance(exe, Executable):
            exe = Executable(exe)

        self.module = exe.mod["vm_load_executable"]()
        self._exec = exe
        self._init = self.module["init"]
        self._invoke = self.module["invoke"]
        self._invoke_stateful = self.module["invoke_stateful"]
        self._get_output = self.module["get_output"]
        self._get_num_outputs = self.module["get_num_outputs"]
        self._get_input_index = self.module["get_input_index"]
        self._set_input = self.module["set_input"]
        self._set_one_input = self.module["set_one_input"]
        self._setup_device(device, memory_cfg)

    def _setup_device(self, dev, memory_cfg):
        """Init devices and allocators."""
        devs = dev
        if not isinstance(dev, (list, tuple)):
            if not isinstance(dev, tvm.runtime.Device):
                raise TypeError("dev is expected to be Device or List[Device]")
            devs = [dev]

        # CPU is required for executing shape functions
        if not any(c.device_type % RPC_SESS_MASK == tvm.cpu().device_type for c in devs):
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
        self._init(*init_args)

    def set_input(self, func_name, *args, **kwargs):
        """Set the input to a function.
        If device type and device id for input tensor are the same as
        for target one the zero copy is used. It means that internal
        tensor is reference to memory allocated by input one.
        Otherwise new internal NDarray is created and data is copied

        Parameters
        ----------
        func_name : str
            The name of the function.

        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the function.

        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.
        """
        if kwargs:
            # kwargs is a super set of the required function parameters. We
            # only find the ones that are needed.
            func_params = self._exec.get_function_params(func_name)
            new_args = [None] * len(func_params)
            cnt = 0
            for k in kwargs:
                if k in func_params:
                    idx = func_params.index(k)
                    new_args[idx] = kwargs[k]
                    cnt += 1
            assert len(args) + cnt == len(func_params)
            idx = 0
            for i, arg in enumerate(new_args):
                if arg is None:
                    new_args[i] = args[idx]
                    idx += 1
            args = new_args
        cargs = convert(args)
        self._set_input(func_name, *cargs)

    def set_one_input(self, func_name, *args, **kwargs):
        """Set the one input tensor with tag to a function.

        Parameters
        ----------
        func_name : str
            The name of the function.
        args : [str or int, tvm.runtime.NDArray]
            name or index of tensor and input tensor, optional
        kwargs: dict of str or int to tvm.runtime.NDArray, optional
            taged arguments to the function.
        Only args or kwargs should exist
        """
        if kwargs:
            assert len(kwargs) == 1
            tag = next(iter(kwargs))
            if isinstance(tag, str):
                func_params = self._exec.get_function_params(func_name)
                assert tag in func_params
            self._set_one_input(func_name, tag, kwargs[tag])
        else:
            assert len(args) == 2
            self._set_one_input(func_name, args[0], args[1])

    def invoke(self, func_name, *args, **kwargs):
        """Invoke a function.

        Parameters
        ----------
        func_name : str
            The name of the function.

        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the function.

        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        result : Object
            The output.
        """
        if args or kwargs:
            self.set_input(func_name, *args, **kwargs)
        return self._invoke(func_name)

    def run(self, *args, **kwargs):
        """Run the main function.

        Parameters
        ----------
        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the function.

        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        result : Object
            The output.
        """
        return self.invoke("main", *args, **kwargs)

    def invoke_stateful(self, func_name, *args, **kwargs):
        """Invoke a function and ignore the returned result.

        Use this function when running over rpc because it is currently
        impossible to return a ADT object over rpc. To get the outputs, use
        :py:func`get_outputs`.

        Parameters
        ----------
        func_name : str
            The name of the function.

        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the function.

        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.
        """
        if args or kwargs:
            self.set_input(func_name, *args, **kwargs)
        self._invoke_stateful(func_name)

    def get_outputs(self):
        """Get the outputs from a call to :py:func`invoke_stateful`.

        Returns
        -------
        outputs : List[NDArray]
        """
        return [self._get_output(i) for i in range(self._get_num_outputs())]

    def get_input_index(self, input_name, func_name="main"):
        """Get inputs index via input name.
        Parameters
        ----------
        name : str
          The input key name
        func_name : str
          The function name

        Returns
        -------
        index: int
          The input index. -1 will be returned if the given input name is not found.
        """
        return self._get_input_index(input_name, func_name)

    def benchmark(
        self,
        device,
        *args,
        func_name="main",
        repeat=5,
        number=5,
        min_repeat_ms=None,
        end_to_end=False,
        **kwargs,
    ):
        """Calculate runtime of a function by repeatedly calling it.

        Use this function to get an accurate measurement of the runtime of a function. The function
        is run multiple times in order to account for variability in measurements, processor speed
        or other external factors.  Mean, median, standard deviation, min and max runtime are all
        reported. On GPUs, CUDA and ROCm specifically, special on-device timers are used so that
        synchonization and data transfer operations are not counted towards the runtime. This allows
        for fair comparison of runtimes across different functions and models. The `end_to_end` flag
        switches this behavior to include data transfer operations in the runtime.

        The benchmarking loop looks approximately like so:

        .. code-block:: python

            for r in range(repeat):
                time_start = now()
                for n in range(number):
                    func_name()
                time_end = now()
                total_times.append((time_end - time_start)/number)


        Parameters
        ----------
        func_name : str
            The function to benchmark

        repeat : int
            Number of times to run the outer loop of the timing code (see above). The output will
            contain `repeat` number of datapoints.

        number : int
            Number of times to run the inner loop of the timing code. This inner loop is run in
            between the timer starting and stopping. In order to amortize any timing overhead,
            `number` should be increased when the runtime of the function is small (less than a 1/10
            of a millisecond).

        min_repeat_ms : Optional[float]
            If set, the inner loop will be run until it takes longer than `min_repeat_ms`
            milliseconds. This can be used to ensure that the function is run enough to get an
            accurate measurement.

        end_to_end : bool
            If set, include time to transfer input tensors to the device and time to transfer
            returned tensors in the total runtime. This will give accurate timings for end to end
            workloads.

        args : Sequence[Object]
            Arguments to the function. These are cached before running timing code, so that data
            transfer costs are not counted in the runtime.

        kwargs : Dict[str, Object]
            Named arguments to the function. These are cached like `args`.

        Returns
        -------
        timing_results : BenchmarkResult
            Runtimes of the function. Use `.mean` to access the mean runtime, use `.results` to
            access the individual runtimes (in seconds).
        """
        min_repeat_ms = 0 if min_repeat_ms is None else min_repeat_ms
        if end_to_end:
            # We need to unpack keyword arguments into positional arguments
            packed_args = list(args)
            for k, v in kwargs.items():
                i = self.get_input_index(k, func_name)
                if i < 0:
                    raise TypeError(f"{func_name}() got an unexpected keyword argument '{k}'")
                while i >= len(packed_args):
                    packed_args.append(None)
                packed_args[i] = v
            return self.module.time_evaluator(
                "invoke_return_to_device",
                device,
                repeat=repeat,
                number=number,
                min_repeat_ms=min_repeat_ms,
            )(func_name, device.device_type % RPC_SESS_MASK, device.device_id, *packed_args)
        if args or kwargs:
            self.set_input(func_name, *args, **kwargs)
        return self.module.time_evaluator(
            "invoke", device, repeat=repeat, number=number, min_repeat_ms=min_repeat_ms
        )(func_name)
