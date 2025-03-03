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
# pylint: disable=redefined-builtin, invalid-name
"""Global Info Data structures for distributed tensor."""
from typing import List, Union, Tuple

import tvm
from tvm.ir import Range
from tvm.ir.global_info import GlobalInfo
from tvm.runtime import ShapeTuple

from . import _ffi_api as ffi


@tvm._ffi.register_object("relax.distributed.DeviceMesh")
class DeviceMesh(GlobalInfo):
    """Device mesh express a view of topology of devices,
       represented by an n-d matrix of device ids.

    Parameters
    ----------
    shape: Union[ShapeTuple, List[int], Tuple[int]]
        Logical shape of device mesh
    device_ids: Union[List[int], Range]
        Represents the device id in the mesh
    """

    def __init__(
        self, shape: Union[ShapeTuple, List[int], Tuple[int]], device_ids: Union[List[int], Range]
    ):
        if isinstance(shape, (list, tuple)):
            shape = ShapeTuple(shape)
        device_range = None
        if isinstance(device_ids, Range):
            device_range = device_ids
            device_ids = []
        self.__init_handle_by_constructor__(
            ffi.DeviceMesh, shape, device_ids, device_range
        )  # type: ignore


def device_mesh(shape: ShapeTuple, device_ids: Union[List[int], Range]) -> DeviceMesh:
    """Create a device mesh expression.
    Parameters
    ----------
    shape : ShapeTuple
        The shape of the device mesh.
    device_ids: Union[List[int], Range]
        Represents the device id in the mesh

    Returns
    -------
    res : DeviceMesh
        The device mesh.
    """
    return DeviceMesh(shape, device_ids)  # pylint: disable=no-member # type: ignore
