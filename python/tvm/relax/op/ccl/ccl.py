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
from . import _ffi_api


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
