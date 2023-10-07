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
"""Hybrid Programming APIs of TVM Python Package.

This package maps a subset of python to HalideIR so that:
1. Users can write some preliminary versions of the computation patterns
have not been supported yet and verify it across the real execution and
python semantic emulation.
2. So far, it is a text format dedicated to HalideIR Phase 0. Refer tvm.lower
for more details. A larger ambition of this module is to support all levels of
HalideIR.
"""

# TODO(@were): Make this module more complete.
# 1. Support HalideIR dumping to Hybrid Script
# 2. Support multi-level HalideIR
import inspect
import tvm._ffi
import tvm.te.schedule
from tvm._ffi.base import decorate

from .module import HybridModule
from .parser import source_to_op
from .utils import _pruned_source


def script(pyfunc):
    """Decorate a python function as hybrid script.

    The hybrid function support emulation mode and parsing to
    the internal language IR.

    Returns
    -------
    hybrid_func : function
        A decorated hybrid script function.
    """
    # pylint: disable=import-outside-toplevel, missing-docstring
    def wrapped_func(func, *args, **kwargs):
        from .utils import _is_tvm_arg_types

        if _is_tvm_arg_types(args):
            src = _pruned_source(func)
            closure_vars = inspect.getclosurevars(func).nonlocals
            closure_vars.update(inspect.getclosurevars(func).globals)
            return source_to_op(src, args, func.__globals__, closure_vars)

        from .runtime import _enter_hybrid_runtime, _restore_runtime

        intersect = _enter_hybrid_runtime(func)
        value = func(*args, **kwargs)
        _restore_runtime(func, intersect)
        return value

    return decorate(pyfunc, wrapped_func)


def build(sch, inputs, outputs, name="hybrid_func"):
    """Dump the current schedule to hybrid module

    Parameters
    ----------
    sch: tvm.te.Schedule
        The schedule to be dumped

    inputs: An array of Tensors or Vars
        The inputs of the function body

    outputs: An array of Tensors
        The outputs of the function body

    Returns
    -------
    module: HybridModule
        The built results is wrapped in a HybridModule.
        The usage of HybridModule is roughly the same as normal TVM-built modules.
    """
    sch = sch.normalize()
    bounds = tvm.te.schedule.InferBound(sch)
    stmt = tvm.te.schedule.ScheduleOps(sch, bounds)

    src = _Dump(stmt, inputs, outputs, name)

    return HybridModule(src, name)


tvm._ffi._init_api("tvm.hybrid", __name__)
