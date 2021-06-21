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
"""Utilities for ComputeDevices."""
import tvm
import tvm.relay
from .._utils import get_node_size  # pylint: disable=unused-import


def get_function_output_buffer(func, device):
    """Get a NDArray for buffering the function output.

    Parameters
    ----------
    func: tvm.relay.Function
        The function for which the buffer is generated.

    device: tvm.runtime.Device
        The device on which the generated buffer is allocated.

    Returns
    -------
    buf: tvm.runtime.NDArray
        The generated NDArray buffer.
    """
    assert isinstance(func, tvm.relay.Function)

    def _get_ndarray(ttype):
        assert isinstance(ttype, tvm.relay.TensorType)
        return tvm.nd.empty(
            shape=tuple([int(i) for i in ttype.shape]), dtype=ttype.dtype, device=device
        )

    ret_type = func.ret_type
    if isinstance(ret_type, tvm.relay.TensorType):
        return _get_ndarray(ret_type)
    if isinstance(ret_type, tvm.relay.TupleType):
        return tvm.runtime.container.tuple_object([_get_ndarray(t) for t in ret_type.fields])
    raise NotImplementedError(ret_type)
