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
"""Helper functions in TVM Script Parser"""

import inspect
from tvm import IRModule

from . import _ffi_api
from .parser import from_source


def create_module(functions=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    functions: Optional[dict].
        Map of GlobalVar or str to PrimFunc

    Returns
    -------
    mod : IRModule
        An IRModule containing the passed definitions
    """

    return IRModule(functions=functions)


def asscript(input_ir, show_meta=False):
    """Transform a PrimFunc or IRModule to python syntax script

    Parameters
    ----------
    input_ir : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be dumped

    show_meta : bool
        Whether show meta

    Returns
    -------
    script : str
        The Python script
    """

    return _ffi_api.AsTVMScript(input_ir, show_meta)


def tir(script_in):
    """Decorate a python function or class as tvm script.

    The tvm function or parsing support parsing to the internal TIR.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """

    if inspect.isfunction(script_in):
        return _parse(script_in)

    if inspect.isclass(script_in):
        return TVMScriptClass(script_in)

    raise TypeError("Only function and class are supported")


def module(script_in):
    """Decorate a python function or class as tvm script.

    Alias for tvm.script.tir for now.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """
    return tir(script_in)


class TVMScriptClass:
    """Helper class for decorating a class"""

    def __init__(self, script_in):
        self.script = script_in

    def __call__(self, *args, **kwargs):
        # call the parser to transform tvm script into TIR
        return _parse(self.script)


def _parse(script_in):
    """Helper function to parse TVM script into TIR"""
    return from_source(inspect.getsource(script_in), inspect.getsourcelines(script_in)[1])
