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
"""A Python wrapper for the Module-based Model Runtime Interface for Ahead-of-Time compilation."""

import numpy as np


class AotModule(object):
    """Wraps the AOT executor runtime.Module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the implemented model functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the implemented model functions.

    Examples
    --------

    .. code-block:: python

        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor

        # build the library using graph executor
        lib = relay.build(...)
        lib.export_library("compiled_lib.so")
        # load it back as a runtime
        lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
        # Call the library factory function for default and create
        # a new runtime.Module, wrap with aot module.
        gmod = tvm.runtime.executor.AotModule(lib["default"](dev))
        # use the aot  module.
        gmod.set_input("x", data)
        gmod.run()
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_input_index = module["get_input_index"]
        self._get_num_inputs = module["get_num_inputs"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if key is not None:
            v = self._get_input(key)
            if v is None:
                raise RuntimeError("Could not find '%s' in model's inputs" % key)
            v.copyfrom(value)

        if params:
            # upload big arrays first to avoid memory issue in rpc mode
            keys = list(params.keys())
            keys.sort(key=lambda x: -np.prod(params[x].shape))
            for k in keys:
                # TODO(zhiics) Skip the weights for submodule in a better way.
                # We should use MetadataModule for initialization and remove
                # params from set_input
                val = self._get_input(k)
                if val:
                    self._get_input(k).copyfrom(params[k])

    def run(self, **input_dict):
        """Run forward execution of the model

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the model

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the model

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_input_index(self, name):
        """Get inputs index via input name.

        Parameters
        ----------
        name : str
           The input key name

        Returns
        -------
        index: int
            The input index. -1 will be returned if the given input name is not found.
        """
        return self._get_input_index(name)

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
