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
"""TFLite runtime that load and run tflite models."""
from .._ffi.function import get_global_func
from ..rpc import base as rpc_base

def create(tflite_model_bytes, ctx):
    """Create a runtime executor module given a tflite model and context.
    Parameters
    ----------
    tflite_model_byte : bytes
        The tflite model to be deployed in bytes string format.
    ctx : TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext.
    Returns
    -------
    tflite_runtime : TFLiteModule
        Runtime tflite module that can be used to execute the tflite model.
    """
    device_type = ctx.device_type
    if device_type >= rpc_base.RPC_SESS_MASK:
        fcreate = ctx._rpc_sess.get_function("tvm.tflite_runtime.create")
        return TFLiteModule(fcreate(bytearray(tflite_model_bytes), ctx))
    fcreate = get_global_func("tvm.tflite_runtime.create")
    return TFLiteModule(fcreate(bytearray(tflite_model_bytes), ctx))


class TFLiteModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual tflite functions.

    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual tflite functions.
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
        """Invoke forward execution of the model

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        self._invoke()

    def allocate_tensors(self):
        """Allocate space for all tensors.
        """
        self._allocate_tensors()


    def get_output(self, index):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index
        """
        return self._get_output(index)
