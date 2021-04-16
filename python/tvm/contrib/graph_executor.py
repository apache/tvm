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
"""Minimum graph executor that executes graph containing TVM PackedFunc."""
import numpy as np
import tvm._ffi

from tvm.rpc import _ffi_api as _rpc_ffi_api
from tvm.rpc import base as rpc_base
from tvm._ffi.base import string_types
from tvm._ffi.runtime_ctypes import Device


def create(graph_json_str, libmod, device):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str
        The graph to be deployed in json format output by json graph.
        The graph can contain operator(tvm_op) that points to the name
        of PackedFunc in the libmod.

    libmod : tvm.runtime.Module
        The module of the corresponding function

    device : Device or list of Device
        The device to deploy the module. It can be local or remote when there
        is only one Device. Otherwise, the first device in the list will
        be used as this purpose. All device should be given for heterogeneous
        execution.

    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.

    Note
    ----
    See also :py:class:`tvm.contrib.graph_executor.GraphModule`
    for examples to directly construct a GraphModule from an exported
    relay compiled library.
    """
    assert isinstance(graph_json_str, string_types)

    dev, num_rpc_dev, device_type_id = get_device(libmod, device)

    if num_rpc_dev == len(dev):
        fcreate = dev[0]._rpc_sess.get_function("tvm.graph_executor.create")
    else:
        fcreate = tvm._ffi.get_global_func("tvm.graph_executor.create")

    return GraphModule(fcreate(graph_json_str, libmod, *device_type_id))


def get_device(libmod, device):
    """Parse and validate all the device(s).

    Parameters
    ----------
    libmod : tvm.runtime.Module
        The module of the corresponding function

    device : Device or list of Device

    Returns
    -------
    device : list of Device
    num_rpc_dev : Number of rpc devices
    device_type_id : List of device type and device id
    """

    if isinstance(device, Device):
        device = [device]
    elif not isinstance(device, (list, tuple)):
        raise ValueError("dev has to be the type of Device or a list of Device")
    for cur_dev in device:
        if not isinstance(cur_dev, Device):
            raise ValueError("dev has to be the type of Device or a list of Device")

    # device_type_id[0], device_type_id[1] are used as the primary/fallback
    # device type and id. All other ones are used as device for
    # heterogeneous execution.
    num_rpc_dev = 0
    device_type_id = []
    for cur_dev in device:
        device_type = cur_dev.device_type
        if device_type >= rpc_base.RPC_SESS_MASK:
            assert libmod.type_key == "rpc"
            assert _rpc_ffi_api.SessTableIndex(libmod) == cur_dev._rpc_sess._tbl_index
            num_rpc_dev += 1
            device_type = cur_dev.device_type % rpc_base.RPC_SESS_MASK
        device_type_id.append(device_type)
        device_type_id.append(cur_dev.device_id)

    if 0 < num_rpc_dev < len(device):
        raise ValueError("Either all or none of the devices should be rpc.")
    return device, num_rpc_dev, device_type_id


class GraphModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

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
        # a new runtime.Module, wrap with graph module.
        gmod = graph_executor.GraphModule(lib["default"](dev))
        # use the graph module.
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
        self._get_num_inputs = module["get_num_inputs"]
        self._load_params = module["load_params"]
        self._share_params = module["share_params"]

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
                raise RuntimeError("Could not find '%s' in graph's inputs" % key)
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
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the graph

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

    def debug_get_output(self, node, out):
        """Run graph up to node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        raise NotImplementedError("Please use debugger.debug_executor as graph_executor instead.")

    def load_params(self, params_bytes):
        """Load parameters from serialized byte array of parameter dict.

        Parameters
        ----------
        params_bytes : bytearray
            The serialized parameter dict.
        """
        self._load_params(bytearray(params_bytes))

    def share_params(self, other, params_bytes):
        """Share parameters from pre-existing GraphExecutor instance.

        Parameters
        ----------
        other: GraphExecutor
            The parent GraphExecutor from which this instance should share
            it's parameters.
        params_bytes : bytearray
            The serialized parameter dict (used only for the parameter names).
        """
        self._share_params(other.module, bytearray(params_bytes))

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]
