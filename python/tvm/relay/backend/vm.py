# License .to the Apache Software Foundation (ASF) under one
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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name
"""
The Relay Virtual Machine.

Implements a Python interface to compiling and executing on the Relay VM.
"""
import numpy as np

import tvm
from tvm._ffi.runtime_ctypes import TVMByteArray
from . import _vm
from . import vmobj as _obj
from .interpreter import Executor


def _update_target(target):
    target = target if target else tvm.target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")

    tgts = {}
    if isinstance(target, (str, tvm.target.Target)):
        dev_type = tvm.expr.IntImm("int32", tvm.nd.context(str(target)).device_type)
        tgts[dev_type] = tvm.target.create(target)
    elif isinstance(target, dict):
        for dev, tgt in target.items():
            dev_type = tvm.expr.IntImm("int32", tvm.nd.context(dev).device_type)
            tgts[dev_type] = tvm.target.create(tgt)
    else:
        raise TypeError("target is expected to be str, tvm.target.Target, " +
                        "or dict of str to str/tvm.target.Target, but received " +
                        "{}".format(type(target)))
    return tgts

def _convert(arg, cargs):
    if isinstance(arg, (np.ndarray, tvm.nd.NDArray)):
        cargs.append(_obj.tensor_object(arg))
    elif isinstance(arg, (tuple, list)):
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(_obj.tuple_object(field_args))
    else:
        raise "unsupported type"

def convert(args):
    cargs = []
    for arg in args:
        _convert(arg, cargs)

    return cargs


class VirtualMachine(object):
    """Relay VM runtime."""
    def __init__(self, mod):
        self.mod = mod
        self._init = self.mod["init"]
        self._load_params = self.mod["load_params"]
        self._invoke = self.mod["invoke"]

    def init(self, ctx):
        """Initialize the context in the VM.

        Parameters
        ----------
        ctx : :py:class:`TVMContext`
            The runtime context to run the code on.
        """
        args = [ctx.device_type, ctx.device_id]
        self._init(*args)

    def load_params(self, params):
        """Load parameters for the VM.

        Parameters
        ----------
        params : Union[bytearray, Dict]
            The dictionary that contains serialized parameters.
        """
        if isinstance(params, dict):
            params = tvm.relay.save_param_dict(params)
        elif isinstance(params, (bytes, str)):
            params = bytearray(params)
        if not isinstance(params, (bytearray, TVMByteArray)):
            raise TypeError("params must be a bytearray")

        self._load_params(bytearray(params))

    def invoke(self, func_name, *args):
        """Invoke a function.

        Parameters
        ----------
        func_name : str
            The name of the function.

        args : list[NDArray] or list[np.ndarray]
            The arguments to the function.

        Returns
        -------
        result : Object
            The output.
        """
        cargs = convert(args)
        return self._invoke(func_name, *cargs)

    def run(self, *args):
        """Run the main function.

        Parameters
        ----------
        args : list[NDArray] or list[np.ndarray]
            The arguments to the function.

        Returns
        -------
        result : Object
            The output.
        """
        return self.invoke("main", *args)

    @property
    def module(self):
        """Return the runtime module contained in a virtual machine."""
        return self.mod


class VMCompiler(object):
    """Build Relay module to run on VM runtime."""
    def __init__(self):
        self.mod = _vm._VMCompiler()
        self._compile = self.mod["compile"]
        self._get_vm = self.mod["get_vm"]

    def compile(self, mod, target=None, target_host=None):
        """
        Parameters
        ----------
        mod : relay.Module
            The Relay module to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
            device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        target_host : str or :any:`tvm.target.Target`, optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm intepreter is used.

        Returns
        -------
        vm : VirtualMachine
            The VM runtime.
        """
        target = _update_target(target)
        self._compile(mod, target, target_host)
        return VirtualMachine(self._get_vm())


class VMExecutor(Executor):
    """
    An implementation of the executor interface for
    the Relay VM.

    Useful interface for experimentation and debugging
    the VM can also be used directly from the API.
    supported by `tvm.relay.vm`.

    Parameters
    ----------
    mod : :py:class:`~tvm.relay.module.Module`
        The module to support the execution.

    ctx : :py:class:`TVMContext`
        The runtime context to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """
    def __init__(self, mod, ctx, target):
        if mod is None:
            raise RuntimeError("Must provide module to get VM executor.")
        self.mod = mod
        self.ctx = ctx
        self.target = target
        compiler = VMCompiler()
        self.vm = compiler.compile(mod, target)
        self.vm.init(ctx)

    def _make_executor(self, expr=None):
        main = self.mod["main"]

        def _vm_wrapper(*args, **kwargs):
            args = self._convert_args(main, args, kwargs)
            return self.vm.run(*args)

        return _vm_wrapper
