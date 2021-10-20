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
"""The interface of expr function exposed from C++."""
import tvm._ffi
import tvm.driver
from tvm.target import Target


@tvm._ffi.register_func("relay.backend.build")
def build(mod, target, target_host=None):
    """Backend build function.

    Parameters
    ----------
    mod : tvm.IRModule or Dict[str, tvm.IRModule]
        Input module

    target : tvm.Target
        The target to run the code on.

    target_host : tvm.Target
        The host target.

    Returns
    -------
    module : tvm.Module
        The runtime module.
    """
    target_host = None if target_host == "" else target_host
    target, target_host = Target.check_and_update_host_consist(target, target_host)
    return tvm.driver.build(mod, target=target)


@tvm._ffi.register_func("relay._tensor_value_repr")
def _tensor_value_repr(tvalue):
    return str(tvalue.data.numpy())


@tvm._ffi.register_func("relay._constant_repr")
def _tensor_constant_repr(tvalue):
    dtype = tvm.runtime.DataType(tvalue.data.dtype)
    if tvm.target.datatype.get_type_registered(dtype.type_code):
        return "custom tensor of type " + dtype.type_code
    return str(tvalue.data.numpy())


tvm._ffi._init_api("relay.backend", __name__)
