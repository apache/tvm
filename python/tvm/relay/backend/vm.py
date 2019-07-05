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
from tvm._ffi.function import Object
from .. import transform
from ..backend.interpreter import Executor
from ..expr import GlobalVar, Expr
from . import _vm

Object = Object

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
            return _eval_vm(self.mod, self.ctx, *args)

        return _vm_wrapper
