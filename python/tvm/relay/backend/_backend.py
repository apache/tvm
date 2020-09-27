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


@tvm._ffi.register_func("relay.backend.lower")
def lower(sch, inputs, func_name, source_func):
    """Backend function for lowering.

    Parameters
    ----------
    sch : tvm.te.Schedule
        The schedule.

    inputs : List[tvm.te.Tensor]
        The inputs to the function.

    func_name : str
        The name of the function.

    source-func : tvm.relay.Function
        The source function to be lowered.

    Returns
    -------
    mod : tvm.IRModule
        The result of lowering.
    """
    # pylint: disable=broad-except, import-outside-toplevel
    import traceback

    try:
        f = tvm.driver.lower(sch, inputs, name=func_name)
        # logging.debug("lower function %s", func_name)
        # logging.debug("%s", _build.lower(sch, inputs, simple_mode=True))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile function\n"
        msg += "-----------------------------\n"
        msg += source_func.astext()
        raise RuntimeError(msg)
    return f


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
    if target_host == "":
        target_host = None
    return tvm.driver.build(mod, target=target, target_host=target_host)


@tvm._ffi.register_func("relay._tensor_value_repr")
def _tensor_value_repr(tvalue):
    return str(tvalue.data.asnumpy())


@tvm._ffi.register_func("relay._constant_repr")
def _tensor_constant_repr(tvalue):
    dtype = tvm.runtime.DataType(tvalue.data.dtype)
    if tvm.target.datatype.get_type_registered(dtype.type_code):
        return "custom tensor of type " + dtype.type_code
    return str(tvalue.data.asnumpy())


tvm._ffi._init_api("relay.backend", __name__)
