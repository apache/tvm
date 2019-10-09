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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
The Relay Virtual Machine.

Implements a Python interface to compiling and executing on the Relay VM.
"""
import numpy as np

import tvm
from tvm import autotvm
from tvm import TVMContext
from tvm.relay import expr as _expr
from . import _vm
from . import vmobj as _obj
from .interpreter import Executor

Tensor = _obj.Tensor
Datatype = _obj.Datatype

def _convert(arg, cargs):
    if isinstance(arg, (np.ndarray, tvm.nd.NDArray)):
        cargs.append(_obj.Tensor(arg))
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


class Executable(object):
    """Relay VM executable"""
    def __init__(self, mod):
        self.mod = mod
        self._set_context = self.mod["set_context"]
        self._get_lib = self.mod["get_lib"]
        self._get_bytecode = self.mod["get_bytecode"]
        self._get_stats = self.mod["get_stats"]

    def set_context(self, ctx):
        """Initialize the context of the VM executable.

        Parameters
        ----------
        ctx : Union[:py:class:`tvm.TVMContext`, List[py:class:`tvm.TVMContext`]]
            The runtime context to run the code on.
        """

        if isinstance(ctx, TVMContext):
            ctx = [ctx]
        elif not isinstance(ctx, (list, tuple)):
            raise ValueError("ctx has to be the type of TVMContext or a list of "
                             "TVMContext")
        # args[0], args[1] are used as the primary/fallback context type and id
        # for heterogeneous execution.
        args = []
        for cur_ctx in ctx:
            if not isinstance(cur_ctx, TVMContext):
                raise ValueError("ctx has to be the type of TVMContext or a list "
                                 "of TVMContext")
            args.append(cur_ctx.device_type)
            args.append(cur_ctx.device_id)

        self._set_context(*args)

    @property
    def lib(self):
        """Get the library that contains hardware dependent code.

        Returns
        -------
        ret : :py:class:`~tvm.Module`
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
        num_primitives = _vm.GetNumOfPrimitives(self.module)
        for i in range(num_primitives):
            ret.append(_vm.GetPrimitiveFields(self.module, i))
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
        num_globals = _vm.GetNumOfGlobals(self.module)
        for i in range(num_globals):
            ret.append(_vm.GetGlobalFields(self.module, i))
        return ret

    @property
    def module(self):
        """Return the runtime module contained in a virtual machine executable."""
        return self.mod


class VirtualMachine(object):
    """Relay VM runtime."""
    def __init__(self, mod):
        if not isinstance(mod, (Executable, tvm.module.Module)):
            raise TypeError("mod is expected to be the type of Executable or " +
                            "tvm.Module, but received {}".format(type(mod)))
        m = mod.module if isinstance(mod, Executable) else mod
        self.mod = _vm._VirtualMachine(m)
        self._invoke = self.mod["invoke"]

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


def compile(mod, target=None, target_host=None, params=None):
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

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    exec : Executable
        The VM executable that contains both library code and bytecode.
    """
    compiler = VMCompiler()

    target = compiler.update_target(target)
    target_host = compiler.update_target_host(target, target_host)
    if params:
        compiler.set_params(params)
    tophub_context = compiler.tophub_context(target)
    with tophub_context:
        compiler._compile(mod, target, target_host)
    return Executable(compiler._get_exec())

class VMCompiler(object):
    """Build Relay module to run on VM runtime."""
    def __init__(self):
        self.mod = _vm._VMCompiler()
        self._compile = self.mod["compile"]
        self._get_exec = self.mod["get_executable"]
        self._set_params_func = self.mod["set_params"]

    def set_params(self, params):
        """Set constant parameters for the model"""
        inputs = {}
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                param = _nd.array(param)
            inputs[name] = _expr.const(param)
        self._set_params_func(inputs)

    def update_target(self, target):
        """Update target"""
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

    def update_target_host(self, target, target_host):
        """Update target host"""
        target_host = None if target_host == "" else target_host
        if not target_host:
            for device_type, tgt in target.items():
                if device_type.value == tvm.nd.cpu(0).device_type:
                    target_host = tgt
                    break
        if not target_host:
            target_host = "llvm" if tvm.module.enabled("llvm") else "stackvm"
        return tvm.target.create(target_host)

    def tophub_context(self, target):
        # If current dispatch context is fallback context (the default root context),
        # then load pre-tuned parameters from TopHub
        if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
            tophub_context = autotvm.tophub.context(list(target.values()))
        else:
            tophub_context = autotvm.util.EmptyContext()
        return tophub_context

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

    ctx : :py:class:`~tvm.TVMContext`
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
        self.executable = compile(mod, target)
        self.executable.set_context(ctx)
        self.vm = VirtualMachine(self.executable)

    def _make_executor(self, expr=None):
        main = self.mod["main"]

        def _vm_wrapper(*args, **kwargs):
            args = self._convert_args(main, args, kwargs)
            return self.vm.run(*args)

        return _vm_wrapper
