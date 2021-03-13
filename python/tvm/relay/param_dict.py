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
import tvm.runtime


def save_param_dict(params):
    """Save parameter dictionary to binary bytes.

    The result binary bytes can be loaded by the
    GraphModule with API "load_params".

    .. deprecated:: 0.9.0
        Use :py:func:`tvm.runtime.save_param_dict` instead.

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

       # set up the parameter dict
       params = {"param0": arr0, "param1": arr1}
       # save the parameters as byte array
       param_bytes = tvm.runtime.save_param_dict(params)
       # We can serialize the param_bytes and load it back later.
       # Pass in byte array to module to directly set parameters
       tvm.runtime.load_param_dict(param_bytes)
    """
    return tvm.runtime.save_param_dict(params)


def load_param_dict(param_bytes):
    """Load parameter dictionary to binary bytes.

    .. deprecated:: 0.9.0
        Use :py:func:`tvm.runtime.load_param_dict` instead.

    Parameters
    ----------
    param_bytes: bytearray
        Serialized parameters.

    Returns
    -------
    params : dict of str to NDArray
        The parameter dictionary.
    """
    return tvm.runtime.load_param_dict(param_bytes)
