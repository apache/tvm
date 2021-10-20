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
import tvm._ffi
from ..rpc import base as rpc_base


def create(tflite_model_bytes, device, runtime_target="cpu"):
    """Create a runtime executor module given a tflite model and device.
    Parameters
    ----------
    tflite_model_byte : bytes
        The tflite model to be deployed in bytes string format.
    device : Device
        The device to deploy the module. It can be local or remote when there
        is only one Device.
    runtime_target: str
        Execution target of TFLite runtime: either `cpu` or `edge_tpu`.
    Returns
    -------
    tflite_runtime : TFLiteModule
        Runtime tflite module that can be used to execute the tflite model.
    """
    device_type = device.device_type

    if runtime_target == "edge_tpu":
        runtime_func = "tvm.edgetpu_runtime.create"
    else:
        runtime_func = "tvm.tflite_runtime.create"

    if device_type >= rpc_base.RPC_SESS_MASK:
        fcreate = device._rpc_sess.get_function(runtime_func)
    else:
        fcreate = tvm._ffi.get_global_func(runtime_func)

    return TFLiteModule(fcreate(bytearray(tflite_model_bytes), device))


class TFLiteModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual tflite functions.

    Attributes
    ----------
    module : Module
        The internal tvm module that holds the actual tflite functions.
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._invoke = module["invoke"]
        self._get_output = module["get_output"]
        self._set_num_threads = module["set_num_threads"]

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

    def get_output(self, index):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index
        """
        return self._get_output(index)

    def set_num_threads(self, num_threads):
        """Set the number of threads via kwargs
        Parameters
        ----------
        num_threads : int
           The number of threads
        """
        self._set_num_threads(num_threads)
