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
"""Helper functions in Hybrid Script Parser"""

import inspect
from tvm import IRModule

from . import _ffi_api
from .parser import source_to_op


def create_module(functions=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    functions: Optional[dict].
        Map of global var or str to PrimFunc or HybridFunction

    Returns
    -------
    mod : IRModule
        A IRModule containing the passed definitions
    """

    functions = {key: _parse(func) if isinstance(func, HybridFunction) else func
                 for key, func in functions.items()}
    return IRModule(functions=functions)


def ashybrid(input_ir, show_meta=False):
    """Transform a Function or Module to python syntax script

    Parameters
    ----------
    input_ir : Union[Function, Module, HybridFunction]
        The Function or Module to be dumped

    show_meta : bool
        Whether show meta

    Returns
    -------
    script : str
        The Python script
    """

    if isinstance(input_ir, HybridFunction):
        # transform HybridFunction to Function
        input_ir = _parse(input_ir)
    return _ffi_api.AsHybrid(input_ir, show_meta)


def script(origin_script):
    """Decorate a python function or class as hybrid script.

    The hybrid function or parsing support parsing to the internal TIR.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """

    if inspect.isfunction(origin_script):
        return HybridFunction(origin_script)

    if inspect.isclass(origin_script):
        return HybridClass(origin_script)

    raise TypeError("Only function and class are supported")


class HybridClass:
    """Helper class for decorating a class"""

    def __init__(self, origin_script):
        self.origin_script = origin_script

    def __call__(self, *args, **kwargs):
        # call the parser to transform hybrid script into TIR
        return _parse(self)


class HybridFunction:
    """Helper class for decorating a function"""

    def __init__(self, origin_script):
        self.origin_script = origin_script


def _parse(hybrid_script):
    """Helper function to parse hybrid_script into TIR"""
    return source_to_op(inspect.getsource(hybrid_script.origin_script),
                        inspect.getsourcelines(hybrid_script.origin_script)[1])
