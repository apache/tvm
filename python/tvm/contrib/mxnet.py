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
"""MXNet bridge wrap Function MXNet's async function."""
from __future__ import absolute_import as _abs

import tvm._ffi.registry
import tvm.runtime._ffi_api
from tvm.runtime import Module

# pylint: disable=invalid-name
_wrap_async = None


def to_mxnet_func(func, const_loc=None):
    """Wrap a TVM function as MXNet function

    MXNet function runs asynchrously via its engine.

    Parameters
    ----------
    func : Function
        A TVM function that can take positional arguments

    const_loc : list of int
        List of integers indicating the argument position
        of read only NDArray argument.
        The NDArray argument location that are not annotated
        will be viewed as mutable arrays in MXNet's engine.

    Returns
    -------
    async_func : Function
        A function that can take MXNet NDArray as argument
        in places that used to expect TVM NDArray.
        Run asynchrously in MXNet's async engine.
    """
    # only import mxnet when wrap get called.
    # pylint: disable=import-self, import-outside-toplevel
    import mxnet

    if isinstance(func, Module):
        func = func.entry_func

    def _get_bridge_func():
        """Get MXNet bridge function"""
        if not mxnet.base._LIB.MXTVMBridge:
            raise RuntimeError(
                "MXTVMBridge not exist in mxnet package," " please update to latest version"
            )

        fdict = tvm._ffi.registry.extract_ext_funcs(mxnet.base._LIB.MXTVMBridge)
        ret = fdict["WrapAsyncCall"]
        ret.is_global = True
        return ret

    global _wrap_async

    if _wrap_async is None:
        # Register extension type in first time
        _wrap_async = _get_bridge_func()
        tvm._ffi.registry.register_extension(mxnet.nd.NDArray)

    const_loc = const_loc if const_loc else []
    return _wrap_async(func, tvm.runtime._ffi_api.TVMSetStream, len(const_loc), *const_loc)
