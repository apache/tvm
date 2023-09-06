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
"""Annotate operators."""
from . import _ffi_api
from ..expr import Expr


def smooth(data: Expr, scale: Expr, kind: int, mode: str) -> Expr:
    """Helper op, that can work as "identity" or "multiply".

    Parameters
    ----------
    data : relax.Expr
        The input data

    scale : relax.Expr
        Scale multiplier

    kind : int
        Kind of argument to be annotated. Can be one of: activation or weight.

    mode : str
        Execution mode for the op. Can be one of: "identity", "multiply" or "quantize".

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.smooth(data, scale, kind, mode)


def absmax(data: Expr, kind: int) -> Expr:
    """Helper op, that does the following transform: abs()->max()->squeeze() for the input tensor.

    Parameters
    ----------
    data : relax.Expr
        The input data

    kind : int
        Kind of argument to be annotated. Can be one of: activation or weight.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.absmax(data, kind)
