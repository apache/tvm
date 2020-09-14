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
# pylint: disable=invalid-name
"""Helper utility to save parameter dicts."""
import tvm
import tvm._ffi


_save_param_dict = tvm._ffi.get_global_func("tvm.relay._save_param_dict")
_load_param_dict = tvm._ffi.get_global_func("tvm.relay._load_param_dict")


def save_param_dict(params):
    """Save parameter dictionary to binary bytes.

    The result binary bytes can be loaded by the
    GraphModule with API "load_params".

    Parameters
    ----------
    params : dict of str to NDArray
        The parameter dictionary.

    Returns
    -------
    param_bytes: bytearray
        Serialized parameters.

    Examples
    --------
    .. code-block:: python

       # compile and save the modules to file.
       graph, lib, params = tvm.relay.build(func, target=target, params=params)
       module = graph_runtime.create(graph, lib, tvm.gpu(0))
       # save the parameters as byte array
       param_bytes = tvm.relay.save_param_dict(params)
       # We can serialize the param_bytes and load it back later.
       # Pass in byte array to module to directly set parameters
       module.load_params(param_bytes)
    """
    args = []
    for k, v in params.items():
        args.append(k)
        args.append(tvm.nd.array(v))
    return _save_param_dict(*args)


def load_param_dict(param_bytes):
    """Load parameter dictionary to binary bytes.

    Parameters
    ----------
    param_bytes: bytearray
        Serialized parameters.

    Returns
    -------
    params : dict of str to NDArray
        The parameter dictionary.
    """
    if isinstance(param_bytes, (bytes, str)):
        param_bytes = bytearray(param_bytes)
    load_arr = _load_param_dict(param_bytes)
    return {v.name: v.array for v in load_arr}
