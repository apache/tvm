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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""
The Relay Virtual Vachine.

Implements a Python interface to compiling and executing on the Relay VM.
"""
import numpy as np

import tvm
from .. import transform
from ..expr import GlobalVar, Expr
from . import _vm
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
        raise TypeError("target is expected to be str or " +
                        "tvm.target.Target, but received " +
                        "{}".format(type(target)))
    return tgts


class VirtualMachine(object):
    """Relay VM runtime."""
    def __init__(self, mod):
        self.mod = mod
        self._init = self.mod["init"]
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



class BuildModule(object):
    """Build Relay module to run on VM runtime."""
    def __init__(self):
        self.mod = _vm._BuildModule()
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


def optimize(mod):
    """Perform several optimizations on a module before executing it in the
    Relay virtual machine.

    Parameters
    ----------
    mod : tvm.relay.Module
        The module to optimize.

    Returns
    -------
    ret : tvm.relay.Module
        The optimized module.
    """
    main_func = mod["main"]

    opt_passes = []
    if not main_func.params and isinstance(main_func.body, GlobalVar):
        opt_passes.append(transform.EtaExpand())

    opt_passes = opt_passes + [
        transform.SimplifyInference(),
        transform.FuseOps(),
        transform.InferType()
    ]

    seq = transform.Sequential(opt_passes)
    return seq(mod)

def _convert(arg, cargs):
    if isinstance(arg, np.ndarray):
        tensor = _vm._Tensor(tvm.nd.array(arg))
        cargs.append(tensor)
    elif isinstance(arg, tvm.nd.NDArray):
        tensor = _vm._Tensor(arg)
        cargs.append(tensor)
    elif isinstance(arg, tuple):
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(_vm._Tuple(*field_args))
    else:
        raise "unsupported type"

def convert(args):
    cargs = []
    for arg in args:
        _convert(arg, cargs)

    return cargs

def _eval_vm(mod, ctx, *args):
    """
    Evaluate a module on a given context with the provided arguments.

    Parameters
    ----------
    mod: relay.Module
        The module to optimize, will execute its entry_func.

    ctx: tvm.Context
        The TVM context to execute on.

    args: List[tvm.NDArray, np.ndarray]
        The arguments to evaluate.
    """
    mod = optimize(mod)
    args = list(args)
    assert isinstance(args, list)
    cargs = convert(args)

    result = _vm._evaluate_vm(mod, ctx.device_type, ctx.device_id, *cargs)
    return result

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
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, expr=None):
        expr = expr if expr else self.mod
        assert expr, "either expr or self.mod should be not null."
        if isinstance(expr, Expr):
            self.mod["main"] = expr
        main = self.mod["main"]

        def _vm_wrapper(*args, **kwargs):
            args = self._convert_args(main, args, kwargs)
            print(type(args[0]))
            return _eval_vm(self.mod, self.ctx, *args)

        return _vm_wrapper
