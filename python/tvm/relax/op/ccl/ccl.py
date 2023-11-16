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
"""Relax Collective Communications Library (CCL) operators"""
from typing import Union
from tvm.relax import PrimValue

from . import _ffi_api
from ...expr import Expr
from ....ir import PrimExpr


def allreduce(x, op_type: str = "sum"):  # pylint: disable=invalid-name
    """Allreduce operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.
    op_type: str
      The type of reduction operation to be applied to the input data.
      Now "sum", "prod", "min", "max" and "avg" are supported.

    Returns
    -------
    result : relax.Expr
      The result of allreduce.
    """
    supported_op_types = ["sum", "prod", "min", "max", "avg"]
    assert op_type in supported_op_types, (
        "Allreduce only supports limited reduction operations, "
        f"including {supported_op_types}, but got {op_type}."
    )
    return _ffi_api.allreduce(x, op_type)  # type: ignore # pylint: disable=no-member


def allgather(x, num_workers: Union[int, PrimExpr, PrimValue]):  # pylint: disable=invalid-name
    """AllGather operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.

    num_worker : Union[int, PrimExpr, PrimValue]
      The number of workers to gather data from.

    Returns
    -------
    result : relax.Expr
      The result of allgather.
    """
    if not isinstance(num_workers, PrimValue):
        num_workers = PrimValue(num_workers)
    return _ffi_api.allgather(x, num_workers)  # type: ignore # pylint: disable=no-member


def broadcast_from_worker0(x: Expr) -> Expr:
    """Broadcast data from worker-0 to all other workers.

    Parameters
    ----------
    x : relax.Expr
      The tensor to be broadcast.

    Returns
    -------
    result : relax.Expr
      The same tensor, which has been broadcast to all other workers.
    """
    return _ffi_api.broadcast_from_worker0(x)


def scatter_from_worker0(x: Expr, num_workers: int, axis: int = 0) -> Expr:
    """Perform a scatter operation from worker-0, chunking the given buffer into equal parts.

    Parameters
    ----------
    x : relax.Expr
      The buffer to be divided into equal parts and sent to each worker accordingly.

    num_worker : int
      The number of workers, i.e. the number of parts the given buffer should be chunked into.

    axis : int
      The dimension of the tensor to be scattered. Default is 0.

    Returns
    -------
    result : relax.Expr
      Chunked Tensor received by different workers.
    """
    return _ffi_api.scatter_from_worker0(x, num_workers, axis)
