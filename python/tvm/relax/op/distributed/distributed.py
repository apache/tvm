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
# pylint: disable=redefined-builtin
"""Operators for distributed Relax.
"""
from tvm.relax.distributed.struct_info import DeviceMesh, Placement
from . import _ffi_api
from ...expr import Expr


def annotate_sharding(input: Expr, device_mesh: DeviceMesh, placement: Placement) -> Expr:
    """Annotate sharding plan for tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh of the sharding plan
    placement: Placement
      The placement of the sharding plan

    Returns
    -------
    result : relax.Expr
      The tensor unmodified.
    """
    return _ffi_api.annotate_sharding(input, device_mesh, placement)  # type: ignore


def redistribute(input: Expr, device_mesh: DeviceMesh, placement: Placement) -> Expr:
    """Redistribute tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh after redistribution
    placement: Placement
      The placement after redistribution
    Returns
    -------
    result : relax.Expr
      The tensor after redistribution.
    """
    return _ffi_api.redistribute(input, device_mesh, placement)  # type: ignore
