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
"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
import numpy as np

from .._ffi.base import string_types
from .._ffi.function import get_global_func
from .._ffi.runtime_ctypes import TVMContext
from ..rpc import base as rpc_base

def create(tflite_fname, ctx):
    """Create a runtime executor module given a graph and module.
    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.
    ctx : TVMContext or list of TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext. Otherwise, the first context in the list will
        be used as this purpose. All context should be given for heterogeneous
        execution.
    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(tflite_fname, string_types):
        raise ValueError("Type %s is not supported" % type(tflite_fname))

    fcreate = get_global_func("tvm.tflite_runtime.create")
    return TfliteModule(fcreate(tflite_fname, ctx))


class TfliteModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._invoke = module["invoke"]
        self._get_output = module["get_output"]
        self._allocate_tensors = module["allocate_tensors"]

    def set_input(self, index, value):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments
        """
        self._set_input(index, value)

    def invoke(self):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        self._invoke()

    def allocate_tensors(self):
        self._allocate_tensors()


    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)
