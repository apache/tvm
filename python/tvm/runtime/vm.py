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
from tvm._ffi.runtime_ctypes import TVMByteArray
from tvm._ffi import base as _base
from .object import Object
from . import _ffi_api, container


def _convert(arg, cargs):
    if isinstance(arg, Object):
        cargs.append(arg)
    elif isinstance(arg, np.ndarray):
        nd_arr = tvm.nd.array(arg, ctx=tvm.cpu(0))
        cargs.append(nd_arr)
    elif isinstance(arg, tvm.runtime.NDArray):
        cargs.append(arg)
    elif isinstance(arg, (tuple, list)):
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(container.tuple_object(field_args))
    elif isinstance(arg, (_base.numeric_types, bool)):
        dtype = "int32" if isinstance(arg, (int, bool)) else "float32"
        value = tvm.nd.array(np.array(arg, dtype=dtype), ctx=tvm.cpu(0))
        cargs.append(value)
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
        self._get_stats = self.mod["get_stats"]
        self._get_function_arity = self.mod["get_function_arity"]
        self._get_function_param_name = self.mod["get_function_param_name"]

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
            mod = relay.Module({"main": f})
            # create a Relay VM.
            ctx = tvm.cpu()
            target = "llvm"
            executable = relay.vm.compile(mod, target)
            code, lib = executable.save()
            # save and load the code and lib file.
            tmp = tvm.contrib.util.tempdir()
            path_lib = tmp.relpath("lib.so")
            lib.export_library(path_lib)
            with open(tmp.relpath("code.ro"), "wb") as fo:
                fo.write(code)
            loaded_lib = tvm.runtime.load_module(path_lib)
            loaded_code = bytearray(open(tmp.relpath("code.ro"), "rb").read())
            # deserialize.
            des_exec = tvm.runtime.vm.Executable.load_exec(loaded_code, loaded_code)
            # execute the deserialized executable.
            x_data = np.random.rand(10, 10).astype('float32')
            des_vm = tvm.runtime.vm.VirtualMachine(des_exec, ctx)
            res = des_vm.run(x_data)
            print(res.asnumpy())
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
            code = bytearray(bytecode)
        elif not isinstance(bytecode, (bytearray, TVMByteArray)):
            raise TypeError(
                "bytecode is expected to be the type of bytearray "
                + "or TVMByteArray, but received {}".format(type(code))
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


class VirtualMachine(object):
    """Relay VM runtime.

    Parameters
    ----------
    exe : Executable
        The VM executable.

    ctx : tvm.runtime.TVMContext or List[tvm.runtime.TVMContext]
        The context to deploy the module

    memory_cfg : str or Dict[tvm.runtime.TVMContext, str], optional
        Config the type of memory allocator. The allocator type can be ["naive",
        "pooled"]. If memory_cfg is None, all contexts will use pooled allocator
        by default. If memory_cfg is string, all contexts will use the specified
        allocator type. If memory_cfg is a dict, each context uses the allocator
        type specified in the dict, or pooled allocator if not specified in the
        dict.
    """

    NAIVE_ALLOCATOR = 1
    POOLED_ALLOCATOR = 2

    def __init__(self, exe, ctx, memory_cfg=None):
        if not isinstance(exe, Executable):
            raise TypeError(
                "exe is expected to be the type of Executable, "
                + "but received {}".format(type(exe))
            )
        self.module = _ffi_api._VirtualMachine(exe.module)
        self._exec = exe
        self._init = self.module["init"]
        self._invoke = self.module["invoke"]
        self._set_input = self.module["set_input"]
        self._setup_ctx(ctx, memory_cfg)

    def _setup_ctx(self, ctx, memory_cfg):
        """Init context and allocators."""
        ctxs = ctx
        if not isinstance(ctx, (list, tuple)):
            if not isinstance(ctx, tvm.runtime.TVMContext):
                raise TypeError(
                    "ctx is expected to be TVMContext or \
                                List[TVMContext]"
                )
            ctxs = [ctx]

        # CPU is required for executing shape functions
        if not any(c.device_type == tvm.cpu().device_type for c in ctxs):
            ctxs.append(tvm.cpu())

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
        for context in ctxs:
            init_args.append(context.device_type)
            init_args.append(context.device_id)
            alloc_type = memory_cfg[context] if context in memory_cfg else default_alloc_type
            init_args.append(alloc_type)
        self._init(*init_args)

    def set_input(self, func_name, *args, **kwargs):
        """Set the input to a function.

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
